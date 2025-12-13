# Issue #120 Investigation: Property Test Stabilization

## Investigation Date

2025-12-06

## Summary

The property tests for Delaunay empty circumsphere validation cannot be stabilized
without implementing bistellar flip operations. The tests are correctly identifying
algorithmic limitations rather than test configuration issues.

## Root Cause

The incremental Bowyer-Watson algorithm can produce **locally non-Delaunay
configurations** that cannot be repaired through the current global repair mechanism.
These violations occur even with:

- ✅ Coordinate deduplication (prevents exact duplicates)
- ✅ General position filtering (rejects >D points on any axis)
- ✅ Robust geometric predicates (`robust_insphere`)
- ✅ Global repair loops (removes violated cells but cannot fix topology)

## Failing Tests

All tests in `tests/proptest_delaunay_triangulation.rs`:

- `prop_empty_circumsphere_{2d,3d,4d,5d}` - Empty circumsphere property validation
- `prop_cloud_with_duplicates_is_delaunay_{2d,3d,4d,5d}` - Duplicate cloud integration tests

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

- **Stub implementation**: `src/core/algorithms/flips.rs` (placeholder with TODOs, no functional
  implementation)
- **Failing tests**: `tests/proptest_delaunay_triangulation.rs` (empty circumsphere and duplicate
  cloud test sections)
- **Documentation**: `tests/README.md` (updated to reflect bistellar flip dependency)

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
- Stub file: `src/core/algorithms/flips.rs` (ready for implementation)

## Conclusion

The property tests are **working as intended**—they correctly identify that the
current algorithm cannot guarantee the Delaunay property without bistellar flips.
This is a known limitation of incremental algorithms without post-processing.

The tests should remain ignored until bistellar flips are implemented, as they
represent a fundamental algorithmic capability gap rather than a test configuration
issue.

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
| Empty circumsphere | 4 (2D-5D) | ⏸️ Ignored | Requires bistellar flips |
| Duplicate cloud integration | 4 (2D-5D) | ⏸️ Ignored | Requires bistellar flips |
| Duplicate coordinate rejection | 4 (2D-5D) | ⏸️ Ignored | Separate issue |

**Total**: 8 passing, 12 ignored (8 for bistellar flips, 4 for duplicate rejection)

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
