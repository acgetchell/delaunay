# Known Issues: 4D Bulk Construction

## Status (targeting v0.7.2)

### Current issue

Large-scale 4D bulk construction can still produce final Delaunay-validation failures on
adversarial/degenerate point sets, even when local repair steps appear to succeed.

**Severity:** High (correctness)  
**Affects:** primarily large 4D bulk runs (typically 100+ vertices)  
**Recommended workaround:** prefer incremental insertion for production 4D workloads

### What has been fixed

The previous orientation root cause has been addressed:

- Coherent combinatorial orientation is now explicitly validated and normalized at the TDS layer.
- Flip paths now enforce orientation invariants (debug assertions + post-flip normalization/validation).
- Construction and repair paths canonicalize stored cell ordering to positive orientation.

So, this issue is no longer attributed to missing TDS orientation enforcement.

### What remains

The remaining risk is in high-dimensional Delaunay-repair convergence/numerical robustness for
large or degenerate inputs (not orientation bookkeeping itself).

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

- **3D:** generally robust for moderate sizes; large degenerate sets may still stress repair.
- **4D:** use incremental insertion for critical correctness paths.
- **5D+:** experimental; incremental insertion strongly recommended.

### Related

- Test file: `tests/large_scale_debug.rs`
- Flip/repair implementation: `src/core/algorithms/flips.rs`
- Triangulation integration: `src/core/triangulation.rs`
- Orientation specification: `docs/ORIENTATION_SPEC.md`
- Orientation tests: `tests/tds_orientation.rs`
