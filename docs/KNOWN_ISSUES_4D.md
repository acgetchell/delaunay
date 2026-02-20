# Known Issues: 4D Bulk Construction

## Status: v0.7.0

### Issue: 4D bulk construction can produce invalid triangulations

**Severity:** High (correctness)  
**Affects:** 4D bulk construction with 100+ vertices  
**Workaround:** Use incremental insertion for production 4D triangulations

### Description

Large-scale 4D bulk construction (100+ vertices) can complete without convergence failures but
produce triangulations that fail final Delaunay validation. The repair logic succeeds locally but
leaves global violations.

### Test Evidence

```bash
# This passes (79 inserted, 21 skipped degenerate)
# but fails final validation: "Cell violates Delaunay property"
DELAUNAY_LARGE_DEBUG_N_4D=100 DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
  cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
```

Error: `Delaunay property violated after construction: Cell violates Delaunay property: cell contains vertex that is inside circumsphere`

Runtime: ~110 seconds before failure  
Degenerate points encountered: 21/100 (ridge fans, disconnected cavities)

### 3D Large-Scale Status

3D construction works reliably up to 100 vertices. At 1000 vertices, repair convergence issues appear:

```bash
# Fails at vertex 18 with repair cycling
DELAUNAY_LARGE_DEBUG_N_3D=1000 cargo test --test large_scale_debug \
  debug_large_scale_3d -- --ignored --exact --nocapture
```

Error: `Delaunay repair failed to converge after 32 flips (cycles=6)`

### Root Cause Hypothesis

The TDS does not enforce positive orientation invariants. Many assumptions throughout the codebase
rely on positive orientation, but it's not guaranteed at the data structure level. This creates
subtle correctness issues in higher dimensions where:

1. Repair logic may flip to negative-orientation states
2. Local repairs succeed but create global inconsistencies
3. Validation catches the violations after construction completes

### Recommendations for Users (v0.7.0)

- **3D:** Use bulk construction for datasets up to ~100 vertices; expect potential issues beyond that
- **4D:** Use incremental insertion instead of bulk construction for production use
- **5D+:** Experimental; incremental insertion recommended

### Future Work

Fixing this properly requires architectural changes:

1. **Short-term:** Add more robust repair convergence detection and fallback strategies
2. **Long-term:** Enforce positive orientation invariants at the TDS level (major refactor)

The long-term fix is tracked separately as it would affect the entire TDS architecture and is appropriate for v0.8 or v1.0.

### Related

- Test file: `tests/large_scale_debug.rs`
- Repair logic: `src/core/delaunay_triangulation.rs` (repair_delaunay_local_single_pass)
- Validation: `src/core/util/delaunay_validation.rs`
