# Known Issues: Dimensional and Large-Scale Limitations

## Status (v0.7.2)

### Current issues

#### 3D large-scale flip convergence

At ~130+ points in 3D, flip-based Delaunay repair can enter cycles (oscillating
flip sequences that never converge). The triangulation is topologically valid but
may have local Delaunay violations that flips cannot resolve.

**Severity:** Medium (correctness)
**Affects:** 3D bulk construction at moderate-to-large scale
**Root cause:** exact degeneracies (cospherical configurations) where the insphere
predicate returns BOUNDARY, leaving the flip heuristic unable to choose a direction.
With `AdaptiveKernel` (default), SoS now breaks these ties for both insphere and
orientation predicates (#233, #263). Flip convergence at large scale may still be
affected by cavity/topology interactions rather than predicate degeneracies.

### What has been fixed

#### 4D+ negative geometric orientation (#230, v0.7.2+)

- **Root cause:** Per-cell vertex swaps in `fill_cavity()` and flip code broke
  sibling coherence. BFS normalization then restored coherence at the cost of
  negative geometric orientation, causing mass vertex skipping in 4D+.
- **Fix:** Removed per-cell orientation swaps from `fill_cavity()` and merged
  `NEGATIVE`/`POSITIVE` orientation match arms in flip code. Orientation is now
  handled exclusively by `normalize_and_promote_positive_orientation()` after
  wiring, which runs BFS normalization followed by global sign canonicalization.
- **Result:** 4D N=100 regression test now inserts all 100 vertices and passes
  L1–L3 validation with `PLManifoldStrict` topology.
- Removed `canonicalize_positive_orientation_for_cells` (both call sites and method).

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

### What remains

- **Heuristic fast-filter tolerance:** the f64 fast filter in `insphere_from_matrix` /
  `orientation_from_matrix` uses an adaptive tolerance that is a heuristic, not a
  provable error bound.  la-stack #44 tracks adding Shewchuk-style bounds.
- **Stack-matrix dimension limit:** `MAX_STACK_MATRIX_DIM = 7` limits exact insphere
  to D ≤ 5 (the insphere matrix is (D+2)×(D+2)).  For D ≥ 6, `robust_insphere`
  falls back to symbolic perturbation and centroid-based tie-breaking.

### Current behavior (v0.7.2+, post-#230/#233/#266)

With the #230 fix, exact predicates, and identity-based SoS perturbation:

- **4D N=100 (release mode, ~30s):** batch construction inserts all 100
  vertices and passes L1–L3 validation with `PLManifoldStrict` topology.
  The previous mass-skipping behavior (only ~12/100 inserted) is resolved.
- **4D N=200+:** larger 4D runs may still encounter flip convergence or
  Delaunay validation failures due to the combinatorial complexity of
  higher-dimensional flip repair. Use the debug harness with
  `DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS` to cap repair cost.

### Reproduction / verification command

```bash
# 4D regression test (exact seed from #230)
cargo test --release --test large_scale_debug --features slow-tests \
  regression_issue_230 -- --ignored --nocapture

# General 4D debug harness — N=100 (strict, no skips)
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=0 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture

# Larger 4D run with capped repair — N=200
DELAUNAY_LARGE_DEBUG_N_4D=200 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS=500 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
```

### Recommendations

- **2D–3D:** generally robust for moderate sizes. SoS (now applied to both orientation
  and insphere) eliminates most predicate-degeneracy flip cycles.
- **4D:** batch construction now works for moderate sizes (tested up to 100 points).
  Larger runs may benefit from incremental insertion or capped repair budgets.
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
