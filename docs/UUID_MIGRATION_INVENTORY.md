# UUID Migration Inventory

This document tracks all UUID usages in the delaunay crate and categorizes them for migration to SlotMap keys.

## Executive Summary

- **Total Files with UUID Usage**: 10 core files + 3 documentation files
- **External API/Persistence (Must Keep UUID)**: ~15-20% of usages
- **Internal Logic (Migrate to Keys)**: ~80-85% of usages

## Status Legend

- ✅ **KEEP**: External API/persistence - must remain UUID
- 🔄 **MIGRATE**: Internal logic - replace with VertexKey/CellKey  
- ⚠️ **REVIEW**: Needs detailed analysis

---

## File-by-File Analysis

### 1. `/src/core/triangulation_data_structure.rs`

**Impact**: HIGH - Core data structure

#### External API/Persistence (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 256, 282, 284, 296 | `TriangulationValidationError::InvalidCell { cell_id: Uuid }` | Error reporting to user |
| 415 | `pub uuid_to_vertex_key: UuidToVertexKeyMap` | Mapping structure (but content is internal) |
| 421 | `pub uuid_to_cell_key: UuidToCellKeyMap` | Mapping structure (but content is internal) |
| 2058, 2070 | Cell validation error creation with UUID | User-facing error messages |
| 2429, 2452 | Neighbor validation with UUID in error messages | Error reporting |

#### Internal Logic (🔄 MIGRATE)

| Line Range | Usage | Replace With |
|------------|-------|--------------|
| 848, 851, 858 | Internal UUID → Key lookups during validation | Direct key usage |
| 908, 911, 918, 931 | Vertex mapping operations | Direct VertexKey operations |
| 991, 1004, 1064 | Cell mapping operations | Direct CellKey operations |
| 1185-1186, 1229 | Internal neighbor assignment | Key-based neighbor references |
| 1359, 1700, 1820 | Internal algorithm logic | SlotMap key operations |
| 2299 | Internal cell creation/validation | Direct key validation |
| 2694-2715 | UUID mapping maintenance | Replace with key-only operations |
| 2872, 2894, 2958 | Validation logic internals | Direct key comparisons |
| 3364, 3673 | Algorithm implementation details | Key-based logic |
| 4188, 4204-4210, 4233 | Internal data structure operations | SlotMap key operations |
| 4447, 4524-4528 | Test and internal utility methods | Direct key usage |
| 4990-4992, 5085 | Test code UUID operations | Key-based testing |
| 5480, 5898 | Additional test scenarios | Key validation tests |

### 2. `/src/core/collections.rs`

**Impact**: LOW - Type definitions only

#### External API (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 115 | `pub use uuid::Uuid;` | Public re-export for API compatibility |
| 245, 376, 409, 417, 420 | UUID in type aliases for public collections | External interface |
| 455, 486, 544, 546 | UUID documentation examples | Public API examples |

#### Internal Logic (🔄 MIGRATE)

| Line | Usage | Replace With |
|------|-------|--------------|
| 163, 165-166 | Internal UUID usage in examples | Replace with key-based examples |
| 232 | Internal mapping examples | Key-based collection patterns |

### 3. `/src/core/util.rs`  

**Impact**: MEDIUM - Utility functions

#### External API (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 46, 49 | UUID parameter in public utility functions | External API |
| 681, 761, 766-767 | Public UUID-based validation functions | User-facing validation |

#### Internal Logic (🔄 MIGRATE)  

| Line | Usage | Replace With |
|------|-------|--------------|
| 104, 108-109, 115 | Internal UUID operations | Direct key operations |
| 122, 125, 128, 134, 140, 144-145 | UUID generation for internal use | Replace with direct key insertion |
| 155-156 | UUID-based internal lookups | SlotMap key access |
| 775, 782, 793, 798 | Internal utility UUID operations | Key-based operations |
| 806-815 | UUID manipulation in internals | Key operations |

### 4. `/src/core/vertex.rs`

**Impact**: HIGH - Core entity

#### External API (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 36, 50 | `uuid: Uuid` field in Vertex struct | Core identity for serialization/API |
| 68, 71 | `uuid()` getter method | Public API |
| 181-182, 184 | UUID construction and access | External interface |
| 355, 364, 381 | Public UUID-based methods | API compatibility |
| 1551-1552, 1560, 1565-1566, 1574 | UUID in public methods | External API |

#### Internal Logic (🔄 MIGRATE)

| Line | Usage | Replace With |
|------|-------|--------------|
| 439, 446, 451, 458 | Internal UUID operations during construction | Direct vertex management |
| 471, 476, 478 | UUID-based internal lookups | Key-based operations |  
| 514, 519, 526, 534-535 | Internal vertex operations | SlotMap operations |
| 673 | UUID usage in internal logic | Key operations |
| 1777, 1785, 1787 | Internal vertex management | Key-based management |
| 1871-1872, 1881-1885 | UUID operations in algorithms | Key-based algorithm logic |
| 1929, 1937-1938, 1940 | Internal UUID comparisons | Key comparisons |
| 1954, 1965, 1967 | UUID-based internal operations | Direct key operations |

### 5. `/src/core/cell.rs`

**Impact**: HIGH - Core entity  

#### External API (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 47, 59 | `uuid: Uuid` field in Cell struct | Core identity |
| 77, 80 | `uuid()` getter method | Public API |
| 221, 224-225 | UUID construction | External interface |
| 243, 257 | Public UUID-based methods | API compatibility |
| 1384, 1398-1399, 1420, 1433-1434 | UUID in public methods | External API |
| 2592, 2603, 2606, 2618, 2622 | Public UUID operations | API methods |
| 2640-2641, 2648, 2652 | UUID-based public interface | External compatibility |
| 3051-3052, 3056 | Public cell operations with UUID | API consistency |

#### Internal Logic (🔄 MIGRATE)

| Line | Usage | Replace With |
|------|-------|--------------|
| 483, 490, 499, 502 | Internal UUID operations during construction | Direct cell management |
| 517, 533 | UUID-based internal lookups | Key-based operations |
| 573, 584, 598, 604 | Internal cell operations | SlotMap operations |
| 792, 801, 838, 870 | UUID usage in internal logic | Key operations |
| 1622, 1636, 1691, 1746, 1786, 1829 | Internal cell management | Key-based management |

### 6. `/src/core/facet.rs`

**Impact**: MEDIUM - Related entity

#### External API (✅ KEEP)

| Line | Usage | Justification |
|------|-------|---------------|
| 86 | UUID in facet construction from external data | API compatibility |
| 619, 630, 642, 677 | UUID parameters in public methods | External interface |
| 1430, 1469, 1495 | Public facet operations with UUID | API methods |

#### ⚠️ **REVIEW NEEDED**: Lines need detailed analysis for internal vs external usage

### 7. `/src/core/boundary.rs`

**Impact**: MEDIUM - Boundary analysis

#### Usage Patterns (⚠️ REVIEW)

| Line | Usage | Analysis Needed |
|------|-------|-----------------|
| 195, 401 | UUID in boundary operations | Determine if public API or internal |

### 8. Algorithm Files

**Impact**: MEDIUM - Algorithm implementations

#### `/src/core/algorithms/robust_bowyer_watson.rs`

- Line 1832: Internal UUID usage - likely **🔄 MIGRATE**

#### `/src/geometry/algorithms/convex_hull.rs`  

- Line 944: UUID in hull algorithm - needs **⚠️ REVIEW**

---

## Migration Strategy by Category

### Phase 1: Internal Data Structure Core

**Priority**: CRITICAL
**Files**: `triangulation_data_structure.rs`, `collections.rs`
**Scope**: Replace internal UUID operations with direct key operations

### Phase 2: Entity Internal Logic  

**Priority**: HIGH
**Files**: `vertex.rs`, `cell.rs`
**Scope**: Migrate internal UUID operations while preserving public UUID API

### Phase 3: Algorithm Implementations

**Priority**: MEDIUM  
**Files**: `util.rs`, `facet.rs`, algorithm files
**Scope**: Replace internal UUID usage in algorithms

### Phase 4: Specialized Components

**Priority**: LOW
**Files**: `boundary.rs`, test files
**Scope**: Clean up remaining internal UUID usage

---

## API Compatibility Plan

### Public API Preservation

1. **Keep UUID fields**: `vertex.uuid()`, `cell.uuid()` methods remain
2. **Keep UUID parameters**: Public methods accepting UUID parameters unchanged
3. **Keep error UUIDs**: Error messages continue to include UUIDs for debugging

### Internal Optimization  

1. **Direct key operations**: Replace UUID→Key lookups with direct key usage
2. **Remove UUID mappings**: Eliminate internal UUID-to-key mapping where possible
3. **Key-based algorithms**: Convert internal algorithms to work with keys directly

### Backward Compatibility

- Serialization/deserialization maintains UUID fields
- Public API surface remains unchanged
- Error messages still include UUIDs for user debugging

---

## Testing Strategy

### Test Migration

1. **Convert internal tests**: Use keys instead of UUIDs where appropriate
2. **Preserve API tests**: Keep UUID-based API tests unchanged  
3. **Add key-based tests**: Ensure key operations work correctly

### Validation Points

1. **Serialization round-trip**: UUIDs preserved in serialization
2. **API compatibility**: Public methods work with UUIDs as before
3. **Performance**: Key-based operations show expected performance gains
4. **Memory usage**: Reduced internal UUID storage

---

## Success Metrics

### Performance Goals

- [ ] Eliminate internal UUID→Key lookups (O(1) to O(0))
- [ ] Reduce memory usage by removing redundant UUID storage
- [ ] Maintain sub-microsecond key operations

### Code Quality Goals  

- [ ] Zero internal UUID→Key mapping operations in hot paths
- [ ] Clean separation between public UUID API and internal key operations
- [ ] Comprehensive test coverage for key-based operations

### Compatibility Goals

- [ ] 100% backward compatibility for public API
- [ ] All serialization tests pass
- [ ] All existing benchmarks maintain or improve performance
