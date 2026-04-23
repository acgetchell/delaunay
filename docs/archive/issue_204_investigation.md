# Issue #204 — 4D 500-point retry collapse investigation

**Date:** 2026-04-23
**Branch:** `fix/204-large-scale-debug` (post PR #339 instrumentation)
**Seed:** 4D, ball radius 100, `DELAUNAY_LARGE_DEBUG_CASE_SEED_4D=0xD225B8A07E274AE6`, 500 points, `DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1`
**Raw log:** `logs/2026-04-23-4d-ridge-fan-500.log` (2238 lines; run aborted at 240s wall clock)

## Scope

Capture the structure of the 500-point 4D ridge-fan retry collapse described in
`docs/KNOWN_ISSUES_4D.md` and `docs/TODO.md §2` so we can pick the smallest fix
that actually closes it, rather than one that only addresses the visible symptom.

## What the instrumentation said

Counts across the partial run (the timeout interrupted attempt 0 before it
completed):

- `kind=disconnected_boundary` retryable-skip lines: **501**
- `kind=ridge_fan` retryable-skip lines: **31**
- `kind=non_manifold` / `kind=open_boundary`: 0 each
- `normalize_and_promote_positive_orientation: N cells still appear negative`
  warnings: dozens, with N ranging from 2 up to 33 (seen repeatedly from
  insertion index ~60 onward)
- `negative geometric orientation detected during validation` (cell passing
  validation with `orientation=-1`): dozens, same pattern
- `bulk D≥4: per-insertion repair non-convergent; continuing`: hundreds, for
  both `Delaunay repair failed to converge after 40/50 flips` and
  `Delaunay repair postcondition failed: local k=2 violation remains`

Representative first ridge fan (line 42, insertion index ≈ 61):

- `D=4`, `conflict_cells=156`, `boundary_facets=190`
- `facet_count=4`, `ridge_vertex_count=3`
- `extra_cells=[CellKey(482v9), CellKey(785v1)]`
- `participating_boundary_indices=[30, 52, 103, 150]`
- Ridge vertices: `VertexKey(35v1)`, `VertexKey(40v1)`, `VertexKey(49v1)` (and
  `VertexKey(61v1)` appears as the fourth vertex of boundary_idx=30's facet)

Representative retryable-skip sequence for `bulk_index=158`:

- Attempts 1–4, all with `conflict=kind=disconnected_boundary visited=5
  total=10 disconnected_cells=1`, identical counts across perturbation retries.
- `cells_before_attempt=cells_after_rollback=2389`,
  `vertices_before_attempt=vertices_after_rollback=157`. Rollback works, so
  state across retries really is the same.

## Mapping traces onto the plan's hypotheses

The plan in `#204 fix` proposed four hypotheses. The traces refute or re-weight
them as follows.

### H1 — Cospherical inclusion produces ridge fans

Partially true but not the dominant mode. Ridge-fan is only 6% of retryable
skips; the cavity-reduction log (`ridge_fan_shrink` then `reextract`) succeeds
at unwinding most ridge fans, thanks to PR #339's cross-fan accumulation. The
31 remaining ridge-fan skips are all that survive reduction; the other 94% of
skips are `DisconnectedBoundary`.

### H2 — Cavity reduction cannot converge

Partially true, but the reason is different. On the first ridge-fan sample,
`cavity_reduction` emits exactly two events:

1. iteration=0 `initial_ok boundary_facets=40`
2. iteration=1 `no_reduction_rule_matched`

So the first cavity was fine for that insertion. The problem is that
subsequent insertions keep hitting `DisconnectedBoundary`, which can only
reduce via EXPAND-with-non-conflict-neighbors or SHRINK-fallback, and the trace
shows it often escapes neither.

### H3 — Perturbation step too small

**False.** Attempts 1–4 for the same `bulk_index` produce identical
`visited/total/disconnected_cells` counts. That is the canonical signature of
perturbation not changing the conflict-region topology at all, which only
happens when the surrounding triangulation is itself non-manifold — the
cavity BFS walks the same broken neighbor graph regardless of the vertex's
exact coordinates. Fix C from the plan would have no effect here.

### H4 — Skip-driven post-construction violation

**Partially true, but not the root cause.** The final
`Cell violates Delaunay property: cell contains vertex that is inside
circumsphere` error is downstream of something earlier: by the time skips
start accumulating, hundreds of cells with orientation = −1 have already been
retained by `normalize_and_promote_positive_orientation`, and the flip-repair
path has accepted hundreds of unresolved k=2 violations. The cavity BFS walks
into these cells and rightly reports `DisconnectedBoundary`, because the
triangulation is no longer a valid manifold at that point.

## Actual root cause

**Repair acceptance of broken state, compounded over insertions.**

Two code paths in the bulk-construction loop swallow violations that should
be hard failures in 4D:

1. `normalize_and_promote_positive_orientation` accepts "residual negative
   cells" after its bounded promotion passes, logging them as "likely
   near-degenerate FP noise". In D≥4 the residual can be tens of cells per
   insertion; these survive and break the geometric invariants
   downstream.
2. `bulk D≥4: per-insertion repair non-convergent; continuing` soft-fails both
   `Delaunay repair failed to converge after N flips` and
   `Delaunay repair postcondition failed: local k=2 violation remains` — with
   a 50-flip per-attempt ceiling and queues that routinely show
   `max_queue=271`. The queue is growing faster than it drains.

Once the triangulation has negative-orientation cells and unresolved local
violations in it, the conflict-region BFS for the next insertion walks an
inconsistent neighbor graph, producing `DisconnectedBoundary` skips that
perturbation cannot repair.

## Revised fix direction

The fix must live entirely inside the repair and retry layers. The #307
orientation relaxation (accepting residual negative-orientation cells after
bounded promotion) stays in place so the flip-repair path still has its
chance at eventual consistency; the problem is that that chance isn't
actually being granted today.

- **Fix 2 — Raise the per-insertion flip budget for D≥4 and escalate
  before soft-fail.** The observed queue sizes (180–271) dwarf the 50-flip
  ceiling. Quadruple the D≥4 budget, and before the soft-fail logs and
  continues, escalate once to a 4× budget with the full TDS as seed set.
- **Fix 3 — Abort the retry loop early when perturbation yields identical
  conflict-region counts.** If attempt `n` rolls back to the same cell/vertex
  counts and produces the same `conflict=kind=...` detail as attempt `n−1`,
  further perturbation is pointless; surface it as a non-retryable skip
  instead of burning the remaining attempts that always fail.
- **Fix 4 — Triggered global-repair cadence when local repair stalls.**
  Count consecutive D≥4 soft-fails; when the counter crosses a threshold,
  synchronously invoke the existing global flip-repair entry point with a
  bounded budget and reset. This is what finally gives eventual consistency
  a chance to materialize on the stream of cumulative violations that the
  #307 relaxation intentionally permits.

Tightening `normalize_and_promote_positive_orientation` itself is out of
scope: the relaxation exists by design (resolved #307), and the fix target
is making the flip-repair path actually drain what the relaxation allows
to accumulate. Fixes A/B/C from the original plan are deferred: they either
target the wrong layer (A, B) or are a no-op on the observed data (C).

## Non-findings worth recording

- The one-shot `ridge-fan-dump` emitted exactly one entry before the test
  aborted; the dump is firing at the first detected fan, as intended.
- `bulk-progress` traces show the first attempt reaches
  `processed=150 total_vertices=500 inserted=149 skipped=1 cells=2282` in
  about 11.8s. The timeout hit mid-attempt, not during shuffled retries, so
  the "all 7 shuffled retries fail" summary in the old KNOWN_ISSUES_4D note
  is still unverified on this branch at 240s; longer runs are needed to
  confirm, though the per-attempt shape clearly degrades.
- No `non_manifold_facet` / `open_boundary` retryable skips fired. The
  cavity-reduction path handles those cleanly when they do occur.

## Step 2a measurement — Actual flip-budget demand (2026-04-23)

Extracted from `logs/2026-04-23-4d-ridge-fan-500.log` using read-only
grep/awk pipelines:

### Failure-mode breakdown (total 922 `bulk D` soft-fail events in the run)

- `Delaunay repair failed to converge after N flips`: **211** (23%).
- `Delaunay repair postcondition failed: local k=2 violation remains`: **711** (77%).

The postcondition-failure plurality is significant: raising the flip budget
addresses the 23% convergence-failure slice; the 77% postcondition slice is
where Fix 4's triggered global repair was expected to contribute.

### `max_queue` at convergence failure (n=211)

```text
min=91  p50=207  p90=281  p95=312  p99=409  max=416  mean=210.7
```

Interpretation: even the minimum queue observed (91) is nearly 2× the
typical 50-flip local-repair budget (`seed_cells.len() * (D+1) * 2` with
`seed_cells.len()≈5`, so `5*5*2 = 50`). At p95 the queue reaches 312 — the
repair is provably never able to drain the backlog inside the budget.

### `checked_facets` at convergence failure (n=211)

```text
min=104  p50=1180  p90=1338  p95=1379  max=7585  mean=1168.1
```

Shows the work the repair is doing before the budget kills it. These are
traversal counts, not flip counts.

### Which budget did each failure hit? (`failed to converge after N flips`)

```text
N=10: 6   N=20: 14   N=30: 4   N=40: 3
N=50: 179  (85% of convergence failures)
N=60: 3   N=280: 1   N=310: 1
```

Budget ∈ {10, 20, 30, 40, 50, 60, 280, 310} reflects the
`seed_cells.len() * (D+1) * 2, floor=8` formula across varying cavity sizes.
The dominant 85% hit N=50 (typical 5-cell seed set).

### Chosen Fix 2 constants

Declared as `pub(crate) const` in `src/triangulation/delaunay.rs`:

- `LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4 = 12` (was inline `2`)
- `LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4 = 96` (was inline `8`)
- `LOCAL_REPAIR_ESCALATION_BUDGET_FACTOR_D_GE_4 = 4`
- `LOCAL_REPAIR_ESCALATION_MIN_GAP = 8`

Budget evaluated for a typical 5-cell seed set becomes `5*5*12 = 300`,
which covers p50 (207) and p90 (281) and brushes p95 (312). The tail
(p95–p99, 312–409) is handled by the escalation path (4× budget with the
full TDS as seed set, rate-limited by `ESCALATION_MIN_GAP`).

The `_D_GE_4` suffix keeps the 2D/3D paths unchanged. D<4 retains the
existing `* 4, floor=16` formula which is already adequate for its repair
queues.

### Follow-up measurement (deferred)

The data above is from a budget-limited run: every sample is a failure.
We do not know the distribution of `flips_performed` at successful local
repair exit (nobody logs it today). If Fix 2 does not hold up on a larger
seed, add a `tracing::debug!` at successful exit from
`repair_delaunay_local_*` logging `(flips, max_queue, seed_cells)` and
re-run to get the *success* distribution.

## Step 2 result — Fix 2 alone closed the 500-point 4D case (2026-04-23)

Run recorded in `logs/2026-04-23-4d-fix2-full-500.log`. With the budget
constants above plus the escalation path in place, the 500-point 4D seed
`0xD225B8A07E274AE6` now:

- Inserts **500 of 500 vertices with 0 skips**.
- Uses `max_attempts=1` (no perturbation retries triggered across the entire
  run).
- Removes 0 cells during insertion repair.
- Completes batch insertion in **229.8 s**, final flip-repair in 2.37 s
  (`flips=0`: triangulation already Delaunay), and validation in 1.24 s, for
  a total wall time of **233.4 s**.
- Passes full `validation_report` (Levels 1–4).

During the 500-point insertion the shorter 90-second probe observed only two
local-repair budget hits, both resolved by the escalation path
(`escalation succeeded: 2`, `escalation also non-convergent: 0`,
`failed to converge (local budget hit): 2`). No `DisconnectedBoundary`,
`RidgeFan`, or `postcondition k=2` soft-fails fired — a complete collapse of
the pre-fix failure-mode distribution (501 / 31 / 711 respectively).

Implications for the remaining plan:

- **Fix 3 (no-progress perturbation detection)** is not needed for this case:
  perturbation is never invoked. The no-progress detector would be useful for
  future seeds that still hit perturbation exhaustion, but it is no longer
  on the critical path for #204.
- **Fix 4 (triggered global-repair cadence)** is not needed for this case:
  the local repair's raised budget plus one escalation drained the backlog
  entirely, so consecutive soft-fails never accumulate to the threshold.
- Both remain documented in the plan as contingent fallbacks if a future
  seed (e.g. larger point counts or a different distribution) surfaces a
  residual failure that the budget + escalation cannot close.

## Next step

Proceed to Step 3 of the plan (regression test) and Step 4 (documentation
and TODO refresh).
