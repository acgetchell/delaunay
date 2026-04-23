# Known Issues: Dimensional and Large-Scale Limitations

## Status (v0.7.5)

### Re-verified on 2026-04-23 (release mode)

These release-mode reruns supersede the old 35-point 3D, 100-point 4D, and
500-point 4D correctness failures described below:

- 3D seed `0xE30C78582376677C` now passes at 35 vertices and at 1000 vertices.
- The 3D 1000-prefix bisect reports no failing prefix for that seed.
- 4D seed `0x9B7786C999C56A16` now completes the 100-point batch: attempt 0
  finishes with `inserted=86`, `skipped=14`, and the shuffled retry 1
  (`perturbation_seed=0x34D84963BCC98F21`) inserts 100/100 vertices with zero
  skips and passes validation in about 15.4s total wall time.
- 4D seed `0xD225B8A07E274AE6` now inserts **500/500** vertices with zero
  skips on the first attempt (no perturbation retries triggered) and passes
  Level 1–4 validation in ~233s total wall time. See the
  "Historical 4D 500-point retry-collapse reproducer (now fixed)" section
  below and `docs/archive/issue_204_investigation.md` for the Fix 2 details.
- The remaining open part of #204 is the default 4D 3000-point batch run,
  which now has progress instrumentation and is clearly a scale/observability
  problem rather than the earlier correctness repros.

### Current issues

#### 4D+ bulk construction at very large scale

The historical correctness failures at 35–100 vertices in 3D and 100–500
vertices in 4D are now fixed. What remains is a scale/runtime concern: the
default 3000-point 4D large-scale debug harness is expensive to investigate
and may still degrade at very large input counts without a bounded test fixture.

**Severity:** Medium (4D batch-construction runtime / observability)
**Affects:** the default 3000-point large-scale debug harness and any
similarly large seeded 4D batch inputs.
**Recommended workaround:** use release-mode runs and smaller seeded probes
when you need quick iteration; prefer incremental insertion for production
4D workloads if large batch runtimes are unacceptable.

#### Historical 4D 500-point retry-collapse reproducer (now fixed)

Before Fix 2 of the #204 plan, 4D seed `0xD225B8A07E274AE6` (ball radius 100,
allow skips) exhausted all 7 shuffled retries. Each attempt finished with
`inserted≈266–300`, `skipped≈200–234`, and the same final error:
`Cell violates Delaunay property: cell contains vertex that is inside
circumsphere`. Representative skip samples were dominated by
`Conflict region error: Ridge fan detected: 4 facets share ridge with 3
vertices`.

**Root cause:** the D≥4 per-insertion local-repair flip budget was too tight
(50-flip ceiling vs. observed `max_queue` p95 = 312), so repair never drained
its backlog. The D≥4 soft-fail arm then silently continued after each failed
local repair, accumulating unresolved k=2 postcondition violations and
negative-orientation cells into the next insertion's conflict BFS. See
`docs/archive/issue_204_investigation.md` for the full measurement and
root-cause analysis.

**Fix (2026-04-23):** Fix 2 of the #204 plan raised the D≥4 flip budget
(`LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4 = 12`,
`LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4 = 96`) and added one escalation pass
(`LOCAL_REPAIR_ESCALATION_BUDGET_FACTOR_D_GE_4 = 4`, rate-limited by
`LOCAL_REPAIR_ESCALATION_MIN_GAP = 8`) with the full TDS as seed set before
the soft-fail arm accepts a non-convergent repair. The #307 orientation
relaxation stays in place so flip repair still has its chance at eventual
consistency; the budget bump is what lets that chance materialize. Regression
coverage lives in
`tests/regressions.rs::regression_issue_204_4d_500_local_repair_budget`
(gated behind `slow-tests`).

**Current recheck (2026-04-23):**

- 4D 500-point batch construction (release mode, seed `0xD225B8A07E274AE6`,
  ball radius=100) inserts **500 of 500** vertices, skips **0**, and passes
  `validation_report` (Levels 1–4) in ~233.4s total wall time.
- Only 2 local-repair budget hits were observed, both resolved by the new
  escalation path. No `DisconnectedBoundary`, `RidgeFan`, or
  `postcondition k=2` retryable-skip traces fired (down from 501 / 31 / 711
  respectively on the pre-fix run).
- 4D 3000-point batch construction (release mode, seed `0xE7E6701F918B07FA`,
  ball radius=100) still emits periodic batch-progress summaries. Large-scale
  runtime characterisation at 3000+ points remains open under the
  "4D+ bulk construction at very large scale" item above.

#### Historical 3D flip-cycle reproducer (now fixed)

The historical 3D flip-cycle seed used by #204/#306 no longer reproduces on
the current branch in release mode.

**Current recheck (2026-04-23):**

- 35-point release run: passes with 35/35 inserted and validation OK
- 1000-point release run: passes with 1000/1000 inserted and validation OK in
  ~69.6s total wall time
- 1000-prefix bisect: reports no failing prefix for the same seed

**#204 findings (v0.7.4):** the incremental-prefix bisect found a **minimal
failing prefix of 35 vertices** (seed `0xE30C78582376677C`, ball radius=100).
Failure occurs at insertion index 22:

- Local repair: 64 flips, 53 cycles detected, ambiguous=0
- Global fallback: 192 flips, 187 cycles detected, ambiguous=0
- Replay: see "3D minimal reproducer" in the reproduction section below

Previously documented threshold was ~130+ points; the bisect shows the actual
threshold is much lower with this seed/distribution.

### What has been fixed

#### Orientation enforcement (v0.7.0–v0.7.1)

- Coherent combinatorial orientation is now explicitly validated and normalized at the TDS layer.
- Flip paths now enforce orientation invariants (debug assertions + post-flip normalization/validation).
- Construction and repair paths canonicalize stored cell ordering to positive orientation.

#### Exact predicates (v0.7.1–v0.7.2)

- Exact orientation predicates via `la_stack::Matrix::det_sign_exact` (Bareiss algorithm
  in `BigRational` arithmetic). Provably correct sign for finite matrix entries.
- Exact insphere predicates using the same three-stage evaluation (f64 fast filter →
  exact Bareiss → BOUNDARY fallback).
- Both `insphere()`, `insphere_lifted()`, and `robust_insphere()` now use exact sign
  classification, eliminating false BOUNDARY/DEGENERATE results from floating-point
  rounding on well-separated inputs.

#### SoS orientation normalization (v0.7.2+)

- `AdaptiveKernel::orientation()` now applies SoS when the exact determinant is zero,
  matching the existing insphere SoS behavior (#263). Both predicates return ±1 for
  distinct points; only truly identical f64 coordinates yield 0.
- Callers that need true geometric degeneracy detection (e.g. initial simplex validation,
  flip degenerate-cell guards) now use `robust_orientation` (exact, no SoS) directly.
- Hilbert-sort quantized dedup integrated into `order_vertices_hilbert` and runs
  unconditionally when Hilbert ordering is active.  Guards against SoS failures on
  identical quantized points with zero extra allocation (reuses quantized coordinates
  from the sorting phase).

#### Provable error bounds (v0.7.3)

- Replaced the heuristic adaptive-tolerance fast filter with provable `det_errbound()`
  (Shewchuk-style permanent-based bounds) in both `insphere_from_matrix` and
  `orientation_from_matrix` (#228, PR #255).  For D ≤ 4, the f64 fast filter now has a
  mathematically guaranteed error bound; ambiguous cases fall through to exact Bareiss.
  (The wrapper function `adaptive_tolerance_insphere()` retains its name but now delegates
  entirely to the provable `insphere_from_matrix` path.)
- For D ≥ 5, `det_errbound()` returns `None`, so every call goes directly to exact
  arithmetic.  Extending bounds to higher dimensions is tracked in #256.
- Trade-off: the provable bounds correctly reject more cases to the exact path, which
  is slower.  ~47 proptests were disabled due to CI timeouts (#256).

### What remains

- **Predicate performance (#256):** the provable error bounds in `det_errbound()` reject
  more cases to exact Bareiss than the old heuristic, causing ~47 proptests to exceed CI
  timeouts.  Re-enabling them requires Shewchuk-style multi-stage adaptive expansion or
  faster exact arithmetic in la-stack.
- **Stack-matrix dimension limit:** `MAX_STACK_MATRIX_DIM = 7` limits exact insphere
  to D ≤ 5 (the insphere matrix is (D+2)×(D+2)).  For D ≥ 6, `robust_insphere`
  falls back to symbolic perturbation and centroid-based tie-breaking.

### Reproduction / verification commands

Use the debug large-scale test to verify current behavior on a given branch.
**Important:** use `--release` for runs above ~30 vertices; debug-mode overhead
makes larger runs appear to hang.

```bash
# 3D minimal reproducer (35 vertices, fails at insertion 22)
DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE=new \
  DELAUNAY_LARGE_DEBUG_N_3D=35 \
  DELAUNAY_LARGE_DEBUG_CASE_SEED_3D=0xE30C78582376677C \
  cargo test --release --test large_scale_debug debug_large_scale_3d \
  -- --ignored --nocapture

# 3D incremental-prefix bisect (finds minimal failing prefix)
DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL=1000 \
  cargo test --release --test large_scale_debug \
  debug_large_scale_3d_incremental_prefix_bisect -- --ignored --nocapture

# 4D 100-point — permissive (allows skips)
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --release --test large_scale_debug debug_large_scale_4d \
  -- --ignored --nocapture

# 4D 100-point — strict (no skips)
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=0 \
  cargo test --release --test large_scale_debug debug_large_scale_4d \
  -- --ignored --nocapture

# 4D 500-point seeded repro (all shuffled retries still fail)
DELAUNAY_BULK_PROGRESS_EVERY=50 \
  DELAUNAY_LARGE_DEBUG_N_4D=500 \
  DELAUNAY_LARGE_DEBUG_CASE_SEED_4D=0xD225B8A07E274AE6 \
  DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --release --test large_scale_debug debug_large_scale_4d \
  -- --ignored --nocapture

# 4D prefix bisect (targets the seeded 500-point repro by default)
DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --release --test large_scale_debug \
  debug_large_scale_4d_incremental_prefix_bisect -- --ignored --nocapture
```

### Recommendations

- **2D:** robust at all tested sizes.
- **3D:** the historical #306/#204 seed now passes in release mode; continue to
  use the large-scale harness as a monitoring tool rather than assuming a 35-point
  correctness failure still exists.
- **4D:** the historical 100-point skip repro is fixed, but seeded 500-point
  and larger batch runs can still fail after all shuffled retries. Use release
  mode for investigation, prefer smaller seeded probes to debug the
  `Ridge fan detected` path, and use incremental insertion when you need more
  predictable progress at large N.
- **5D:** experimental; incremental insertion strongly recommended. Exact insphere
  predicates are available (5D uses a 7×7 matrix, within the stack limit).
- **6D+:** exact insphere is not available (matrix exceeds stack limit); falls back
  to symbolic perturbation.  Use with caution.

### Related

- Test file: `tests/large_scale_debug.rs`
- Flip/repair implementation: `src/core/algorithms/flips.rs`
- Triangulation integration: `src/core/triangulation.rs`
- Orientation specification: `docs/ORIENTATION_SPEC.md`
- Orientation tests: `tests/tds_orientation.rs`
