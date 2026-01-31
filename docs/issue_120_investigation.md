# Issue #120 Investigation: Property Test Stabilization

## Status: ✅ RESOLVED (2026-01-31)

**All Issue #120 objectives have been completed.** Bistellar flip-based Delaunay repair is fully implemented and all related property tests are passing.

## Final Status Summary (2026-01-31)

### ✅ Completed

- **Bistellar flip implementation**: k=2 facet flips, k=3 ridge flips with queue-based repair
- **Automatic Delaunay repair**: `DelaunayRepairPolicy` with configurable repair frequency
- **Fast O(N) validation**: Flip-based verification (40-100x faster than brute-force)
- **All dimensions working**: 2D-5D empty circumsphere and duplicate cloud tests passing
- **Property tests enabled**:
  - ✅ `prop_empty_circumsphere_{2d,3d,4d,5d}` - All passing
  - ✅ `prop_cloud_with_duplicates_is_delaunay_{2d,3d,4d,5d}` - All passing
  - ✅ `debug_issue_120_empty_circumsphere_5d` - Passing (previously failing 5D case)

### 📋 Separate Issues (Not Part of Issue #120)

- **4D/5D incremental insertion topology**: `RidgeLinkNotManifold` errors (tracked in new plan d2a2ea8a)
- **Duplicate coordinate rejection**: Separate implementation concern

## Historical Context

### Investigation Date

2025-12-06

### Original Summary

The property tests for Delaunay empty circumsphere validation originally could not
be stabilized without implementing bistellar flip operations.

### Final Update (2026-01-17 → 2026-01-31)

- Implemented k=2 facet flips and k=3 ridge flips in `src/core/algorithms/flips.rs`
- Introduced inverse edge/triangle queues for 4D/5D repair
- Introduced `DelaunayRepairPolicy` and automatic repair after insertion
- Exposed manual repair: `DelaunayTriangulation::repair_delaunay_with_flips`
- **All Delaunay property tests now passing consistently**

> Note: The sections below capture the original investigation and plan.

## Root Cause

The incremental Bowyer-Watson algorithm can produce **locally non-Delaunay
configurations** that cannot be repaired through the current global repair mechanism.
These violations occur even with:

- ✅ Coordinate deduplication (prevents exact duplicates)
- ✅ General position filtering (rejects >D points on any axis)
- ✅ Robust geometric predicates (`robust_insphere`)
- ✅ Global repair loops (removes violated cells but cannot fix topology)

## Test Status (Updated 2026-01-31)

Current status in `tests/proptest_delaunay_triangulation.rs`:

- ✅ **Enabled & Passing**: `prop_empty_circumsphere_{2d,3d,4d,5d}` - All dimensions passing
- ✅ **Enabled & Passing**: `prop_cloud_with_duplicates_is_delaunay_{2d,3d,4d,5d}` - All dimensions passing
- ✅ **Enabled & Passing**: `debug_issue_120_empty_circumsphere_5d` - Previously failing 5D case now passing
- ⚠️ **Ignored (separate issue)**: `prop_duplicate_coordinates_rejected_{3d,4d,5d}` - Slow tests (>60s), performance-gated
- ⚠️ **Ignored (new plan)**: `prop_incremental_insertion_maintains_validity_{4d,5d}` - RidgeLinkNotManifold topology (plan d2a2ea8a)

## Example Failure Case (2D)

Minimal failing input from proptest:

```rust
[
    Point([0.0, 0.0]),
    Point([-54.687, 0.0]),
    Point([-85.026, 36.185]),
    Point([0.0, 38.424]),
]
```

Error: `DelaunayViolation { cell_key: CellKey(1v1) }`

The repair mechanism runs but reports:

```text
After repair loop (interior): total_removed=0
No cells removed (interior), skipping neighbor rebuild
```

This indicates the violation cannot be fixed by removing cells—it requires **topology change** via flips.

## Why Current Approach Fails

### 1. Bowyer-Watson Limitations

The incremental cavity-based insertion algorithm:

- Creates a cavity around the new point
- Removes violating cells
- Retriangulates from the cavity boundary

However, this can produce **locally non-Delaunay edges** that satisfy the immediate insertion
but violate the global Delaunay property when checked against all vertices.

### 2. Global Repair Insufficient

The current `repair_global_delaunay_violations()` mechanism:

- Detects violating cells
- Attempts to remove them
- Rebuilds connectivity

But it cannot **flip edges** or change topology, which is required to fix certain violations.

## Solution: Bistellar Flips

### What Are Bistellar Flips?

Topology-preserving transformations that restore the Delaunay property by reconfiguring cell adjacencies without changing the vertex set.

**Status (2026-01-17)**: Implemented k=2 facet flips, k=3 ridge flips, and inverse edge/triangle queues for 4D/5D repair.
Remaining gaps focus on degenerate/duplicate-heavy inputs and long-run convergence validation.

### Required Implementations

#### 2D: Edge Flip (2-to-2)

Transform two triangles sharing an edge into two triangles with the flipped edge.

```text
Before:         After:
   d               d
  /|\             / \
 / | \           /   \
a  |  c   -->   a-----c
 \ | /           \   /
  \|/             \ /
   b               b
```

Condition: Edge `bd` is flippable if the circumcircle test indicates a Delaunay violation.

#### 3D: Bistellar Flips (2-to-3, 3-to-2)

- **2-to-3**: Replace 2 tetrahedra sharing a facet with 3 tetrahedra sharing an edge
- **3-to-2**: Reverse of 2-to-3

#### Higher Dimensions (4D-5D)

General bistellar flip operations based on i-to-j transformations.

### Implementation Plan

1. **Add flip detection** (`can_flip()`)
   - Check if a flip is geometrically valid
   - Verify it would restore Delaunay property

2. **Implement flip execution** (`flip()`)
   - Update cell vertex sets
   - Rebuild neighbor relationships
   - Maintain TDS invariants

3. **Integrate with repair loop**
   - After Bowyer-Watson insertion, scan for violations
   - Apply flips iteratively until no violations remain
   - Fall back to removal if flip fails

4. **Add flip property tests**
   - Verify flips maintain TDS validity
   - Verify flips improve Delaunay property
   - Test flip cascades (one flip may enable others)

## File Locations

### Current State

- **Flip implementation**: `src/core/algorithms/flips.rs` (k=2 facet + k=3 ridge flips; inverse edge/triangle queues in 4D/5D repair)
- **Repair integration**: `src/core/delaunay_triangulation.rs` (`DelaunayRepairPolicy`, auto repair)
- **Tests**: `tests/proptest_delaunay_triangulation.rs` (empty-circumsphere enabled; duplicate-cloud + duplicate-coordinate suites ignored)
- **Documentation**: `tests/README.md` (notes duplicate-cloud + duplicate-coordinate limitations)

### References

- Edelsbrunner & Shah (1996): "Incremental Topological Flipping Works for Regular Triangulations"
- CGAL implementation: `Triangulation_3::flip()`
- Research notes: Warp Drive Bistellar flips notebook

## Impact Assessment

### Blocking

- ✅ **NOT blocking**: Basic triangulation construction works correctly
- ✅ **NOT blocking**: Structural TDS invariants are maintained
- ✅ **NOT blocking**: Most triangulations are Delaunay (property tests found edge cases)

### Required For

- ❌ **Full Delaunay guarantee**: Cannot guarantee empty circumsphere property
- ❌ **Canonical triangulation**: Cannot produce unique triangulation for point set
- ❌ **Property test stability**: Tests correctly identify these violations
- ❌ **Insertion-order invariance**: Different orders may produce different (both valid, but not Delaunay) triangulations

## Recommendations

### Short Term (v0.6.0)

1. **Keep tests ignored** with clear documentation
2. **Document limitation** in main README and API docs
3. **Release with caveat**: "Triangulations are valid but may contain local Delaunay violations"
4. **Add tracking issue** for bistellar flip implementation

### Medium Term (v0.7.0)

1. **Implement 2D edge flips** (simpler, validates approach)
2. **Re-enable 2D property tests**
3. **Validate performance impact** of flip-based repair

### Long Term (v0.8.0+)

1. **Implement 3D bistellar flips**
2. **Generalize to higher dimensions** (4D, 5D)
3. **Re-enable all property tests**
4. **Add canonical triangulation guarantee**

## Related Issues

- Issue #120: Stabilize property tests (this investigation)
- Issue #98: Topology and Euler characteristic (can proceed independently)
- Flip implementation: `src/core/algorithms/flips.rs` (k=2 facet flips + repair queue)

## Conclusion

The property tests were **working as intended**—they identified that the
incremental algorithm could not guarantee the Delaunay property without flips.
With k=2 facet + k=3 ridge flips and inverse queues now implemented, empty-circumsphere
tests are re-enabled in 2D–5D. Duplicate-cloud and duplicate-coordinate tests stay
ignored until duplicate handling is improved.

## Proposed Resolution

_Decision: We selected Option 2 (Repurpose Issue #120) to maintain issue continuity and discussion history. The options below remain for context._

### Option 1: Close Issue #120

**Status**: Won't fix for v0.6.0 - requires bistellar flip implementation

**Rationale**:

- Tests correctly identify missing algorithmic capability
- Cannot be "stabilized" through test configuration changes
- Requires substantial new feature (bistellar flips)
- Should be tracked as feature request, not bug

**Actions**:

1. Close issue #120 with explanation
2. Update issue to reference this investigation document
3. Create new issue: "Implement bistellar flips for Delaunay repair"
4. Link from #120 to new issue

### Option 2: Repurpose Issue #120 (Selected)

**Status**: Rename to "Implement bistellar flips"

**Actions**:

1. Update issue title: "Stabilize property tests" → "Implement bistellar flips for Delaunay guarantee"
2. Update issue body with implementation plan
3. Change labels: `testing` → `enhancement`, `geometry`
4. Update milestone: v0.6.0 → v0.7.0

## Path Forward

### For v0.6.0 Release (Current)

**Goal**: Release with documented limitations

1. ✅ **Keep tests ignored** (done)
2. ✅ **Document investigation** (this document)
3. ⬜ **Update main README** with limitation note
4. ⬜ **Update API docs** for `DelaunayTriangulation::new()`
5. ⬜ **Add CHANGELOG entry** noting limitation
6. ⬜ **Repurpose #120** to "Implement bistellar flips" (Option 2 selected)

**Note**: This document initially presented two options (close vs. repurpose). Option 2
(repurpose) was selected to maintain issue continuity and discussion history.

### For v0.7.0 Release (Next)

**Goal**: Implement 2D edge flips

1. Implement 2D edge flip (2-to-2)
2. Add `can_flip()` predicate
3. Add `flip()` execution
4. Integrate with Bowyer-Watson repair loop
5. Re-enable 2D property tests
6. Benchmark performance impact

### For v0.8.0+ Release (Future)

**Goal**: Full dimensional support

1. Implement 3D bistellar flips (2-to-3, 3-to-2)
2. Generalize to 4D, 5D
3. Re-enable all property tests
4. Add canonical triangulation guarantee
5. Update documentation

## Documentation Updates Needed

### README.md

Add to limitations section:

```markdown
## Known Limitations

### Delaunay Property
The incremental Bowyer-Watson algorithm produces valid triangulations but may contain
local violations of the Delaunay empty circumsphere property. These violations are rare
and typically occur with:
- Near-degenerate point configurations
- Points with specific geometric arrangements

Full Delaunay property guarantee requires bistellar flip implementation (planned for v0.7.0+).

See: [Issue #120 Investigation](docs/issue_120_investigation.md)
```

### API Documentation

Add to `DelaunayTriangulation` struct docs:

```rust
/// # Delaunay Property Note
///
/// The triangulation satisfies **structural validity** (all TDS invariants) but may
/// contain local violations of the empty circumsphere property in rare cases. This is
/// a known limitation of incremental algorithms without post-processing.
///
/// For applications requiring strict Delaunay guarantee, consider:
/// - Using smaller point sets (violations are rarer)
/// - Filtering degenerate configurations
/// - Awaiting bistellar flip implementation (v0.7.0+)
```

### CHANGELOG.md

Add entry:

```markdown
### Known Issues

- Delaunay empty circumsphere property may be violated in rare cases (#120)
  - Structural TDS invariants are maintained
  - Most triangulations satisfy Delaunay property
  - Full guarantee requires bistellar flips (planned for v0.7.0+)
  - See docs/issue_120_investigation.md for details
```

## Test Status Summary

Current state in `tests/proptest_delaunay_triangulation.rs`:

| Test Category | Count | Status | Reason |
|--------------|-------|---------|--------|
| Incremental insertion validity | 4 (2D-5D) | ✅ Passing | Structural invariants OK |
| Insertion-order robustness | 4 (2D-5D) | ✅ Passing | Valid triangulations produced |
| Empty circumsphere | 4 (2D-5D) | ✅ Passing | Flip repair enabled |
| Duplicate cloud integration | 4 (2D-5D) | ⏸️ Ignored | Duplicate-heavy inputs |
| Duplicate coordinate rejection | 4 (2D-5D) | ⏸️ Ignored | Separate issue |

**Total**: 12 passing, 8 ignored (4 duplicate-cloud, 4 duplicate rejection)

## Resolution

**Selected Approach**: Option 2 (Repurpose Issue #120)

Issue #120 will be repurposed from "Stabilize property tests" to "Implement bistellar flips
for Delaunay guarantee" because:

1. ✅ Maintains issue continuity and discussion history
2. ✅ All stakeholders already tracking #120 will see the evolution
3. ✅ Investigation document remains linked to original issue

**Actions Taken**:

- Update issue title: "Stabilize property tests" → "Implement bistellar flips for Delaunay
  guarantee"
- Update issue body with implementation plan (see this document for content)
- Change labels: Remove `testing`, add `enhancement` and `algorithms`
- Update milestone: v0.6.0 → v0.7.0

**Implementation Tracking**: See repurposed Issue #120 for current status and task checklist.
