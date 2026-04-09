# Known Issues: Dimensional and Large-Scale Limitations

## Status (v0.7.5)

### Current issues

#### 4D+ bulk construction failures

Large-scale 4D bulk construction can produce Delaunay-validation failures on
adversarial/degenerate point sets, even when local repair steps appear to succeed.

**Severity:** High (correctness)
**Affects:** primarily large 4D bulk runs (typically 100+ vertices)
**Recommended workaround:** prefer incremental insertion for production 4D workloads

**#204 findings (v0.7.4):** 4D 100-point batch construction (release mode,
seed `0x9B7786C999C56A16`, ball radius=100) inserts only **12 of 100 vertices**;
88 are skipped as degeneracies.  All 88 skips hit the same cell
(`CellKey(29v7)`, vertices `[6v1, 2v1, 9v1, 11v1, 7v1]`) which has negative
geometric orientation.  In debug mode, per-insertion PLManifoldStrict validation
of this cell produces repeated warnings that cause extreme slowness (appears as
a hang but is not an algorithmic deadlock).  The resulting 12-vertex
triangulation passes L1–L4 validation.

#### 3D large-scale flip convergence

Flip-based Delaunay repair can enter cycles (oscillating flip sequences that
never converge).  The triangulation is topologically valid but may have local
Delaunay violations that flips cannot resolve.

**Severity:** High (correctness)
**Affects:** 3D bulk construction at moderate scale (35+ vertices with default seed)
**Root cause (updated):** predicate degeneracies have been ruled out — the #204
debug runs show `ambiguous=0, predicate_failures=0` in every cycle report.  SoS
is working correctly.  The remaining cycles are caused by **cavity/topology
interactions** where a sequence of locally legal flips forms a cycle.

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
```

### Recommendations

- **2D:** robust at all tested sizes.
- **3D:** flip-cycle failures start at 35+ vertices with the default seed.
  SoS eliminates predicate degeneracies but cavity/topology flip cycles persist.
  This is the primary open correctness issue.
- **4D:** batch construction produces a negative-orientation cell early, causing
  most subsequent insertions to be skipped.  Use incremental insertion for
  critical correctness paths.
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
