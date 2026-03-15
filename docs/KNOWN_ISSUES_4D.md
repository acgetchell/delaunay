# Known Issues: Dimensional and Large-Scale Limitations

## Status (v0.7.2)

### Current issues

#### 4D+ bulk construction failures

Large-scale 4D bulk construction can produce Delaunay-validation failures on
adversarial/degenerate point sets, even when local repair steps appear to succeed.

**Severity:** High (correctness)
**Affects:** primarily large 4D bulk runs (typically 100+ vertices)
**Recommended workaround:** prefer incremental insertion for production 4D workloads

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

### Reproduction / verification command

Use the debug large-scale test to verify current behavior on a given branch.
The examples below cover both the historical `N=100` case and a larger `N=200`
case aligned with the "100+ vertices" scope:

```bash
# Permissive debug run (allows intentional skips) — N=100
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
# Strict debug run (no skips allowed) — N=100
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=0 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture

# Permissive debug run (allows intentional skips) — N=200
DELAUNAY_LARGE_DEBUG_N_4D=200 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture

# Strict debug run (no skips allowed) — N=200
DELAUNAY_LARGE_DEBUG_N_4D=200 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=0 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
```

### Recommendations

- **2D–3D:** generally robust for moderate sizes. SoS (now applied to both orientation
  and insphere) eliminates most predicate-degeneracy flip cycles.
- **4D:** use incremental insertion for critical correctness paths.
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
