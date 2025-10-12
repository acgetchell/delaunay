# Phase 3A: Cell Key-Based Storage Migration Plan

**Date:** 2025-10-12  
**Status:** IN PROGRESS - Analysis Complete, Partial Implementation  
**Goal:** Convert Cell struct to store VertexKey/CellKey instead of full Vertex objects

---

## Executive Summary

Phase 3A aims to improve memory efficiency and cache locality by having Cell store keys to
vertices/neighbors instead of full objects. Initial implementation revealed this is a **major
architectural change** affecting:

- Cell construction patterns (CellBuilder, `cell!` macro)
- All Cell API methods
- Deserialization logic
- ~50+ call sites across the codebase

**Current Status:** Cell struct updated with key-based fields + PhantomData. Compilation broken with ~20 errors. Needs systematic completion.

---

## Architecture Analysis

### The Core Challenge: Cell Construction Without TDS Context

The fundamental issue is a **chicken-and-egg problem**:

1. **Before Phase 3A:**
   - Cells stored full `Vec<Vertex<T, U, D>>` objects
   - Cells could be created independently via `CellBuilder` or `cell!` macro
   - No TDS context needed for construction

2. **After Phase 3A:**
   - Cells store `SmallBuffer<VertexKey, 8>` keys
   - Converting vertices → keys requires TDS context
   - Cell construction now needs TDS involvement

### What We've Changed So Far

#### ✅ Completed Changes

1. **Cell Struct Definition** (`src/core/cell.rs:244-292`)

   ```rust
   pub struct Cell<T, U, V, const D: usize> {
       vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
       uuid: Uuid,
       neighbor_keys: Option<SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE>>,
       data: Option<V>,
       _phantom: PhantomData<(T, U)>,  // ← Added for unused type params
   }
   ```

2. **Updated Methods:**
   - `vertex_keys()` - Returns `&[VertexKey]` instead of vertices
   - `number_of_vertices()` - Now uses `vertex_keys.len()`
   - `clear_neighbors()` - Uses `neighbor_keys` instead of `neighbors`
   - `vertex_uuids(&Tds)` - Now requires TDS parameter
   - `vertex_uuid_iter(&Tds)` - Now requires TDS parameter

3. **Documentation:**
   - Updated doc comments to reflect Phase 3A changes
   - Added migration notes and `rust,ignore` for broken examples

#### ⚠️ Partially Fixed / Broken

1. **CellBuilder Validation** (`src/core/cell.rs:418-447`)
   - Still references `self.vertices` (field doesn't exist)
   - Needs to validate vertex keys or accept vertices temporarily

2. **Cell Deserialization** (`src/core/cell.rs:293-397`)
   - Receives vertices from serde but needs to produce keys
   - Current placeholder creates empty `vertex_keys`
   - Proper fix requires TDS reconstruction after deserialization

3. **Removed Methods:**
   - `vertices()` → **REMOVED** (no direct vertex access without TDS)
   - Methods now require `&Tds` parameter to resolve keys

---

## Compilation Errors Inventory

**Total Errors:** ~20 across 4 files

### By File

#### `src/core/cell.rs` (12 errors)

1. Line 420: `self.vertices` in CellBuilder::validate - field doesn't exist
2. Line 660: `self.vertices.contains(vertex)` - field doesn't exist
3. Line 690: `self.vertices.iter()` (2 occurrences) - field doesn't exist
4. Line 762: Cell construction with `vertices` field - field doesn't exist
5. Line 764: Cell construction with `neighbors` field - should be `neighbor_keys`
6. Line 875: `for vertex in &self.vertices` - field doesn't exist
7. Line 884: `self.vertices.iter()` - field doesn't exist
8. Line 889-891: `self.vertices.len()` (2 occurrences) - field doesn't exist
9. Line 898: `self.neighbors` - should be `neighbor_keys`
10. Line 1020: `self.vertices` - field doesn't exist
11. Line 1088: `cell.vertices().len()` - method removed
12. Line 1272: `cell.vertices().len()` - method removed

#### `src/core/facet.rs` (1 error)

- Line 366: `cell.vertices()` - method removed

#### `src/core/algorithms/robust_bowyer_watson.rs` (3 errors)

- Line 574: `cell.vertices().iter()` - method removed
- Line 643: `bad_cell.vertices().len()` - method removed  
- Line 1100: `adjacent_cell.vertices()` - method removed

#### `src/core/triangulation_data_structure.rs` (implied)

- Many references to `cell.vertices` and `cell.neighbors` fields
- Not yet checked comprehensively

---

## Migration Strategy Options

### Option A: TDS-Centric Construction (Recommended)

**Philosophy:** Cells are internal TDS data structures and should only be created by TDS.

**Changes Required:**

1. Make Cell fields fully private or add internal constructor
2. TDS methods create cells with keys directly
3. Remove/deprecate `cell!` macro for standalone use
4. CellBuilder becomes internal TDS helper

**Pros:**

- Clean architecture - cells are TDS implementation details
- Best memory efficiency and cache locality
- Enforces proper key management

**Cons:**

- Breaking API change for users creating cells directly
- All examples/tests need updating
- Larger initial refactor

**Implementation Steps:**

1. Add `Cell::new_with_keys()` internal constructor
2. Update TDS cell insertion methods to use key-based construction
3. Fix all TDS internal usage (triangulation_data_structure.rs)
4. Update algorithms (robust_bowyer_watson.rs, etc.)
5. Update facet construction
6. Update tests and examples
7. Deprecate standalone cell construction

### Option B: Hybrid Approach (Pragmatic)

**Philosophy:** Keep both vertices and keys during transition period.

**Changes Required:**

1. Add back `vertices: Vec<Vertex<T, U, D>>` field to Cell
2. Keep `vertex_keys` and populate both
3. Gradually migrate methods to use keys
4. Eventually remove vertices field in Phase 3B

**Pros:**

- Incremental migration possible
- Less disruptive to existing code
- Can migrate performance-critical paths first

**Cons:**

- Memory overhead during transition
- Maintenance burden of dual storage
- Potential synchronization bugs

### Option C: Visitor Pattern (Advanced)

**Philosophy:** Cell provides keys; external resolver function provides vertex access.

**Changes Required:**

1. Cell methods that need vertices take `impl Fn(VertexKey) -> &Vertex` closure
2. TDS provides resolver closure in its methods
3. Clean separation of concerns

**Pros:**

- Flexible design
- Cell remains decoupled from TDS
- Testable with mock resolvers

**Cons:**

- More complex API
- Performance overhead of closures
- Steeper learning curve

---

## Recommended Implementation Plan (Option A)

### Phase 3A.1: Core Cell API (1-2 hours)

**Priority:** HIGH - Foundational

1. **Add Internal Cell Constructor**

   ```rust
   impl<T, U, V, const D: usize> Cell<T, U, V, D> {
       /// Internal constructor for TDS use only
       pub(crate) fn new_with_keys(
           vertex_keys: impl Into<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
           data: Option<V>,
       ) -> Self {
           Self {
               vertex_keys: vertex_keys.into(),
               uuid: make_uuid(),
               neighbor_keys: None,
               data,
               _phantom: PhantomData,
           }
       }
   }
   ```

2. **Fix Cell Methods Requiring Vertices** (with `&Tds` parameter)
   - `contains_vertex(&self, vkey: VertexKey) -> bool` - compare keys directly
   - `has_vertex_in_common(&self, other: &Cell, tds: &Tds) -> bool`
   - `is_valid(&self, tds: &Tds) -> Result<(), CellValidationError>`
   - Any comparison/hashing that needs vertex data

3. **Update Serialization/Deserialization**
   - Serialize: Store vertex UUIDs (via TDS lookup)
   - Deserialize: Mark as requiring TDS reconstruction
   - Or: Serialize full vertices, reconstruct keys on TDS load

### Phase 3A.2: CellBuilder Refactor (1 hour)

**Priority:** MEDIUM - For TDS construction helpers

**Option 1:** Keep builder internal to TDS

```rust
// In TDS impl
pub fn insert_cell(&mut self, vertices: Vec<Vertex<T, U, D>>, data: Option<V>) -> CellKey {
    let vertex_keys = vertices.iter()
        .map(|v| self.insert_vertex(v.clone()))
        .collect();
    let cell = Cell::new_with_keys(vertex_keys, data);
    self.cells.insert(cell)
}
```

**Option 2:** Builder accepts both vertices and keys

```rust
pub struct CellBuilder<T, U, V, const D: usize> {
    // Either vertices (for construction) or keys (for internal use)
    vertex_source: Either<Vec<Vertex<T, U, D>>, SmallBuffer<VertexKey, 8>>,
    data: Option<V>,
}
```

### Phase 3A.3: TDS Integration (2-3 hours)

**Priority:** HIGH - Most errors are here

Files to update:

1. `src/core/triangulation_data_structure.rs`
   - Update all `cell.vertices` → `cell.vertex_keys()`
   - Add TDS parameter to cell methods that need it
   - Fix neighbor assignment logic (keys instead of UUIDs)

2. `src/core/algorithms/robust_bowyer_watson.rs`
   - Line 574: `cell.vertex_keys()` + TDS lookup
   - Line 643: `bad_cell.number_of_vertices()`
   - Line 1100: `adjacent_cell.vertex_keys()` + TDS lookup

3. `src/core/facet.rs`
   - Line 366: Update facet construction with keys

### Phase 3A.4: Comparison & Hashing (1 hour)

**Priority:** MEDIUM - Needed for collections

Update trait implementations:

```rust
impl<T, U, V, const D: usize> PartialEq for Cell<T, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        // Compare by UUID (unique identifier)
        // Or compare sorted vertex_keys if semantic equality needed
        self.uuid == other.uuid
    }
}

impl<T, U, V, const D: usize> Hash for Cell<T, U, V, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}
```

### Phase 3A.5: Tests & Examples (2-3 hours)

**Priority:** HIGH - Validate correctness

1. Update unit tests in `src/core/cell.rs`
2. Update integration tests in `tests/`
3. Update examples in `examples/`
4. Update benchmarks in `benches/`
5. Run full test suite: `just test-all`

### Phase 3A.6: Documentation (1 hour)

**Priority:** MEDIUM - User-facing

1. Update `docs/code_organization.md`
2. Update CHANGELOG.md
3. Fix doc test examples (marked `rust,ignore`)
4. Update README if needed

---

## Testing Strategy

### Unit Tests

- Cell construction via TDS
- Key-based vertex access
- Neighbor management with keys
- Serialization round-trip

### Integration Tests

- Triangulation construction with key-based cells
- Bowyer-Watson insertion
- Convex hull extraction
- Cross-dimensional tests (2D-5D)

### Performance Tests

- Memory usage comparison (keys vs full objects)
- Cache locality improvements
- Benchmark critical paths

---

## Rollback Plan

If issues arise, revert changes:

```bash
git checkout main -- src/core/cell.rs
cargo check
```

Current changes are isolated to cell.rs. Easy rollback point.

---

## Key Decisions to Make

1. **CellBuilder Future:**
   - [ ] Keep as public API with vertices input?
   - [ ] Make internal TDS-only helper?
   - [ ] Remove entirely?

2. **Serialization Strategy:**
   - [ ] Serialize vertex UUIDs, reconstruct keys via TDS?
   - [ ] Serialize full vertices, convert to keys on load?
   - [ ] Custom serde format?

3. **Public API Changes:**
   - [ ] Mark as breaking change (0.6.0)?
   - [ ] Provide compatibility layer?
   - [ ] Update all examples immediately?

4. **Comparison Semantics:**
   - [ ] UUID-based equality (identity)?
   - [ ] Vertex-based equality (structural)?
   - [ ] Both via different traits?

---

## Time Estimates

| Phase | Description | Time | Priority |
|-------|-------------|------|----------|
| 3A.1 | Core Cell API | 1-2h | HIGH |
| 3A.2 | CellBuilder | 1h | MEDIUM |
| 3A.3 | TDS Integration | 2-3h | HIGH |
| 3A.4 | Comparison/Hash | 1h | MEDIUM |
| 3A.5 | Tests/Examples | 2-3h | HIGH |
| 3A.6 | Documentation | 1h | MEDIUM |
| **Total** | | **8-11h** | |

---

## Next Session Checklist

- [ ] Review this plan and decide on Option A/B/C
- [ ] Make key decisions (CellBuilder, serialization, API)
- [ ] Start with Phase 3A.1 (Core Cell API)
- [ ] Commit incrementally after each working phase
- [ ] Run `just quality` after each phase

---

## References

- Original Discussion: Conversation history (Phase 3A planning)
- Related Files:
  - `src/core/cell.rs` - Main changes
  - `src/core/triangulation_data_structure.rs` - TDS integration needed
  - `src/core/algorithms/robust_bowyer_watson.rs` - Algorithm updates needed
  - `src/core/facet.rs` - Facet construction
- Similar Patterns: Vertex already uses key-based approach for some operations

---

## Notes

- SmallBuffer<VertexKey, 8> provides zero-heap-allocation for D ≤ 7
- PhantomData<(T, U)> required for unused type parameters
- Current errors are expected and catalogued above
- This is a foundational change enabling future optimizations
