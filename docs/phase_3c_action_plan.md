# Phase 3C Action Plan: Complete Facet Migration

**Status**: 🔴 Required for Compilation  
**Created**: 2025-10-14  
**Prerequisites**: Phase 3A (TDS/Cell/Facet core refactor) - ✅ Complete

---

## Executive Summary

Phase 3A successfully refactored the core TDS, Cell, and Facet types to use key-based storage. However, **the library does not currently compile** due to trait definitions and algorithm modules still referencing the deprecated `Facet` type.

**Phase 3C** completes the migration by updating:
1. Trait method signatures (InsertionAlgorithm)
2. ConvexHull module implementation
3. All code that constructs or consumes Facet objects

---

## Current Compilation Status

### Library Compilation (cargo check --lib)
- **Errors**: 14 compilation errors
- **Breakdown**:
  - 8 errors: `cannot find type Facet in this scope`
  - 2 errors: `no method named vertices found for reference &(CellKey, u8)`
  - 1 error: `use of undeclared type Facet`
  - 3 errors: unused type parameters in ConvexHull

### Test Compilation (cargo test --lib)
- **Errors**: 208 compilation errors (cascade from library errors)

### Affected Files
1. `src/core/traits/insertion_algorithm.rs` (trait definitions)
2. `src/geometry/algorithms/convex_hull.rs` (algorithm implementation)
3. Various test files

---

## Phase 3C Tasks

### Task 1: Update InsertionAlgorithm Trait (2-3 hours)

**File**: `src/core/traits/insertion_algorithm.rs`

**Problem**: Trait methods return `Vec<Facet<T, U, V, D>>` which no longer exists.

**Methods to Update**:
1. `find_cavity_boundary_facets()` (line 1127)
   - Current: `Result<Vec<Facet<T, U, V, D>>, InsertionError>`
   - Target: `Result<Vec<(CellKey, u8)>, InsertionError>`

2. `create_cells_from_boundary_facets()` (line ~2000)
   - Update parameter from `&[Facet<T, U, V, D>]` to `&[(CellKey, u8)]`

3. Any helper methods that construct/consume Facet objects

**Implementation Strategy**:
```rust
// Before (deprecated):
fn find_cavity_boundary_facets(
    &self,
    tds: &Tds<T, U, V, D>,
    bad_cells: &[CellKey],
) -> Result<Vec<Facet<T, U, V, D>>, InsertionError>

// After (Phase 3C):
fn find_cavity_boundary_facets(
    &self,
    tds: &Tds<T, U, V, D>,
    bad_cells: &[CellKey],
) -> Result<Vec<(CellKey, u8)>, InsertionError>
```

**Migration Steps**:
1. Update trait method signatures
2. Update default trait implementation (lines 1136-1260)
3. Update RobustBowyerWatson implementation
4. Update all call sites to handle `(CellKey, u8)` instead of `Facet`

**Note**: `RobustBowyerWatson` already has `robust_find_cavity_boundary_facets_lightweight()` that returns the correct type. This can serve as reference implementation.

---

### Task 2: Refactor ConvexHull Module (3-4 hours)

**File**: `src/geometry/algorithms/convex_hull.rs`

**Problems**:
1. `ConvexHull` struct stores `Vec<Facet<T, U, V, D>>` (deprecated)
2. Type parameters T, U, V are unused (because Facet was removed)
3. Methods return `&Facet` or `Iterator<Facet>` (both invalid)

**Current Structure** (lines ~177-180):
```rust
pub struct ConvexHull<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    facets: Vec<Facet<T, U, V, D>>,  // ❌ Deprecated type
    // ... other fields
}
```

**Target Structure**:
```rust
pub struct ConvexHull<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Lightweight facet handles: (CellKey, facet_index)
    facet_handles: Vec<(CellKey, u8)>,
    
    /// Reference to TDS for resolving facet data on-demand
    /// Store this if ConvexHull needs to provide facet access
    /// OR make methods take &Tds parameter
    // ... other fields
}
```

**Methods to Update**:
1. `get_facet()` - line 900
   - Current: `Option<&Facet<T, U, V, D>>`
   - Options:
     - Return `Option<(CellKey, u8)>` (lightweight)
     - Return `Option<FacetView>` (requires TDS parameter)
     - Remove method entirely

2. `facets()` - line 933
   - Current: `std::slice::Iter<'_, Facet<T, U, V, D>>`
   - Target: Iterator over `(CellKey, u8)` or `FacetView`

3. Construction/insertion methods that create Facets

**Design Decision Needed**:

**Option A**: Store TDS reference in ConvexHull
```rust
pub struct ConvexHull<'tds, T, U, V, const D: usize> {
    facet_handles: Vec<(CellKey, u8)>,
    tds: &'tds Tds<T, U, V, D>,
}
```
- **Pros**: Can return `FacetView` from methods
- **Cons**: Adds lifetime parameter, more complex

**Option B**: Pass TDS as parameter to methods
```rust
pub struct ConvexHull<T, U, V, const D: usize> {
    facet_handles: Vec<(CellKey, u8)>,
}

impl ConvexHull {
    pub fn facet_views<'tds>(&self, tds: &'tds Tds<T, U, V, D>) 
        -> impl Iterator<Item = FacetView<'tds, T, U, V, D>>
    {
        // ...
    }
}
```
- **Pros**: No lifetime parameter, simpler struct
- **Cons**: All methods need TDS parameter

**Recommendation**: Use Option B (pass TDS as parameter) to keep struct simple and follow existing patterns in the codebase.

**Migration Steps**:
1. Change `facets` field to `facet_handles: Vec<(CellKey, u8)>`
2. Update all construction code to store handles instead of Facets
3. Add `&Tds` parameter to methods that need to access facet data
4. Update public API methods to work with handles or FacetView
5. Fix all call sites

**Breaking Changes**: Yes, public API changes
- `get_facet()` signature changes
- `facets()` iterator type changes
- Some methods may need additional `&Tds` parameter

---

### Task 3: Update Tests (1-2 hours)

After Tasks 1 and 2 are complete:

1. Update test code that constructs Facets
2. Update test code that calls affected trait methods
3. Update test assertions to work with `(CellKey, u8)` handles
4. Add tests for new `FacetView`-based APIs

---

## Estimated Timeline

| Task | Estimated Time | Complexity |
|------|---------------|------------|
| Task 1: InsertionAlgorithm trait | 2-3 hours | Medium |
| Task 2: ConvexHull module | 3-4 hours | High |
| Task 3: Test updates | 1-2 hours | Low |
| **Total** | **6-9 hours** | **Medium-High** |

---

## Implementation Order

1. **Start with InsertionAlgorithm trait** (Task 1)
   - Less complex than ConvexHull
   - Already have working reference implementation
   - Fixes ~8 compilation errors

2. **Then tackle ConvexHull** (Task 2)
   - More complex, architectural decisions needed
   - Fixes ~6 compilation errors
   - Requires careful API design

3. **Finally update tests** (Task 3)
   - Straightforward once library compiles
   - Validates all changes work correctly

---

## Success Criteria

- ✅ `cargo check --lib` passes with 0 errors
- ✅ `cargo test --lib` compiles successfully
- ✅ All existing tests pass
- ✅ No deprecated `Facet` references remain
- ✅ Public API maintains backward compatibility where possible
- ✅ Performance is maintained or improved (lightweight handles)

---

## References

- **Phase 3A Guide**: `docs/phase_3a_implementation_guide.md`
- **Existing Lightweight Implementation**: `RobustBowyerWatson::robust_find_cavity_boundary_facets_lightweight()`
- **FacetView API**: `src/core/facet.rs` (lines 170-518)

---

## Notes

- This work was identified during Phase 3A implementation
- ConvexHull refactor started in earlier conversation but not completed
- Type parameters in ConvexHull may need `PhantomData` if not used elsewhere
