# Triangulation Validation Guide

This document explains the validation hierarchy in the delaunay library and provides guidance on when and how to use each validation level.

## Overview

The library provides **four levels of validation**, each building on the previous level to provide increasingly comprehensive correctness guarantees:

1. **Element Validity** - Basic data integrity
2. **TDS Structural Validity** - Combinatorial correctness
3. **Manifold Topology** - Topological properties
4. **Delaunay Property** - Geometric optimality

## Validation Hierarchy

```text
Level 1: Element Validity
    ↓ (called by)
Level 2: TDS Structural Validity
    ↓ (called by)
Level 3: Manifold Topology
    ↓ (independent)
Level 4: Delaunay Property
```

---

## Level 1: Element Validity

### Purpose

Validates basic data integrity of individual vertices and cells.

### Methods

- `Cell::is_valid()` - Check if a cell has valid structure
- `Vertex::is_valid()` - Check if a vertex has valid coordinates

### What It Checks

- **Vertices**: Coordinate validity, UUID presence, dimension consistency
- **Cells**: Correct number of vertices (D+1), no duplicate vertices, valid UUID

### Complexity

- **Time**: O(1) per element
- **Space**: O(1)

### When to Use

- Building blocks for higher-level validation
- Rarely called directly by users
- Automatically called by Level 2

### Example

```rust
use delaunay::prelude::*;

let v = vertex!([0.0, 0.0, 0.0]);
assert!(v.is_valid().is_ok());
```

---

## Level 2: TDS Structural Validity

### Purpose

Validates the combinatorial structure of the Triangulation Data Structure.

### Method

- `Tds::is_valid()` - Comprehensive TDS validation
- `Tds::validation_report()` - Detailed validation with all violations

### What It Checks

1. **UUID ↔ Key Mappings**: Bidirectional consistency for vertices and cells
2. **No Duplicate Cells**: No cells with identical vertex sets
3. **Facet Sharing Invariant**: Each facet shared by at most 2 cells
4. **Neighbor Consistency**: Mutual neighbor relationships are correct
5. **Cell Validity**: All cells pass Level 1 validation

### Complexity

- **Time**: O(N×D²) where N = number of cells, D = dimension
- **Space**: O(N×D) for facet-to-cells map

### When to Use

- **Production**: After construction or major modifications
- **Tests**: In test suites to catch structural bugs
- **Debug builds**: Use `debug_assert!(dt.is_valid().is_ok())`

### Example

```rust
use delaunay::prelude::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let dt = DelaunayTriangulation::new(&vertices).unwrap();

// Quick structural check
assert!(dt.is_valid().is_ok());

// Detailed report showing all violations
match dt.tds().validation_report() {
    Ok(()) => println!("✓ All TDS invariants satisfied"),
    Err(report) => {
        for violation in report.violations {
            eprintln!("Invariant violation: {:?}", violation);
        }
    }
}
```

---

## Level 3: Manifold Topology

### Purpose

Validates that the triangulation forms a valid topological manifold.

### Method

- `Triangulation::validate_manifold()` - Full manifold validation

### What It Checks

1. **All TDS Structural Invariants** (calls Level 2)
2. **Manifold Facet Property**: Each facet belongs to exactly 1 cell (boundary) or exactly 2 cells (interior)
   - Stronger than Level 2's "≤2 cells per facet"
   - Ensures no facets with 0 cells (disconnected components)
3. **Euler Characteristic**: χ matches expected topology
   - 0D: χ = 1 (single vertex)
   - 1D: χ = V - E = 1 (path with boundary)
   - 2D: χ = V - E + F ∈ {1, 2} (disk or sphere)
   - 3D: χ = V - E + F - C ∈ {0, 1} (ball or sphere)
   - 4D+: Currently allows all values (TODO: full k-simplex counting)

### Complexity

- **Time**: O(N×D²) for edge/facet extraction and Euler calculation
- **Space**: O(N×D) for edge/facet sets

### When to Use

- **Tests**: Verify topological correctness in test suites
- **Debug builds**: After complex operations that might break manifold structure
- **Production**: Only when topological guarantees are critical
- **Not recommended**: Hot paths or large triangulations (expensive)

### Example

```rust
use delaunay::prelude::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
    vertex!([0.25, 0.25, 0.25]), // Interior point
];
let dt = DelaunayTriangulation::new(&vertices).unwrap();

// Thorough manifold validation (includes all TDS checks)
match dt.triangulation().validate_manifold() {
    Ok(()) => println!("✓ Valid manifold with correct Euler characteristic"),
    Err(e) => eprintln!("✗ Manifold validation failed: {}", e),
}
```

---

## Level 4: Delaunay Property

### Purpose

Validates the geometric optimality of the triangulation.

### Method

- `DelaunayTriangulation::validate_delaunay()` - Check empty circumsphere property

### What It Checks

- **Empty Circumsphere Property**: For every D-dimensional cell, no vertex lies strictly inside its circumsphere
- Uses geometric predicates from the kernel (`insphere` test)
- **Independent of Levels 1-3**: Checks geometric property, not structural/topological

### Complexity

- **Time**: O(N×V) where N = cells, V = total vertices
- **Space**: O(1) per test

### When to Use

- **Critical Applications**: When Delaunay guarantees are essential (interpolation, mesh quality)
- **Tests**: After construction to verify correctness
- **Debug**: Investigating geometric issues or suspected violations
- **Rarely**: Expensive for large triangulations

### Example

```rust
use delaunay::prelude::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let dt = DelaunayTriangulation::new(&vertices).unwrap();

// Full Delaunay property validation
match dt.validate_delaunay() {
    Ok(()) => println!("✓ All cells satisfy empty circumsphere property"),
    Err(e) => eprintln!("✗ Delaunay violation: {}", e),
}
```

---

## Decision Tree: Which Validation Level to Use?

```text
Start: Do you need to validate?
    │
    ├─ Just built triangulation? → Skip (construction ensures validity)
    │
    ├─ After manual TDS mutation? → Level 2 (Tds::is_valid)
    │
    ├─ Debugging geometric issues? → Level 4 (validate_delaunay)
    │
    ├─ Writing tests? → Level 2 or 3 depending on what you're testing
    │
    ├─ Production validation?
    │   ├─ Performance critical? → Level 2 (Tds::is_valid)
    │   ├─ Topological correctness critical? → Level 3 (validate_manifold)
    │   └─ Geometric correctness critical? → Level 4 (validate_delaunay)
    │
    └─ Paranoid mode? → All levels (2, 3, 4)
```

---

## Performance Comparison

For a 3D triangulation with 1000 vertices (~5000-6000 cells):

| Level | Time | What It Does |
|-------|------|--------------|
| 1 | ~1μs | Single element check |
| 2 | ~10-50ms | Full structural validation |
| 3 | ~50-100ms | Structural + topological (Euler) |
| 4 | ~100-500ms | Empty circumsphere for all cells |

**Recommendation**: Use Level 2 in production, reserve Level 3+ for tests/debug.

---

## Common Patterns

### Pattern 1: Test Suite Validation

```rust
#[test]
fn test_my_triangulation_operation() {
    let mut dt = create_test_triangulation();
    
    // Perform operation
    my_operation(&mut dt);
    
    // Validate at appropriate level
    assert!(dt.is_valid().is_ok());                          // Level 2: Always
    assert!(dt.triangulation().validate_manifold().is_ok()); // Level 3: For topology tests
    assert!(dt.validate_delaunay().is_ok());                 // Level 4: For geometric tests
}
```

### Pattern 2: Debug Build Validation

```rust
use delaunay::prelude::*;

pub fn my_algorithm(dt: &mut DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    // Do work...
    
    #[cfg(debug_assertions)]
    {
        dt.is_valid().expect("TDS structure violated");
        dt.triangulation().validate_manifold().expect("Manifold property violated");
    }
}
```

### Pattern 3: Conditional Deep Validation

```rust
use delaunay::prelude::*;

pub fn validate_with_level(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>, level: u8) -> Result<(), String> {
    match level {
        2 => dt.is_valid().map_err(|e| e.to_string()),
        3 => dt.triangulation().validate_manifold().map_err(|e| e.to_string()),
        4 => dt.validate_delaunay().map_err(|e| e.to_string()),
        _ => Err("Invalid validation level".to_string()),
    }
}
```

---

## Troubleshooting

### Validation Fails at Level 2

**Problem**: Structural invariants violated
**Likely Cause**: Bug in construction or mutation code
**Fix**: Check for duplicate cells, incorrect neighbor assignments, or mapping inconsistencies

### Validation Passes Level 2, Fails at Level 3

**Problem**: Manifold property violated (facet has 0 or >2 cells) or Euler characteristic wrong
**Likely Cause**: Non-manifold topology or disconnected components
**Fix**: Check facet-to-cells mapping, ensure no isolated cells

### Validation Passes Level 3, Fails at Level 4

**Problem**: Delaunay property violated (vertex inside circumsphere)
**Likely Cause**: Geometric degeneracy, numerical precision, or flipping not implemented
**Fix**: Check for near-coplanar/collinear points, consider using RobustKernel instead of FastKernel

---

## API Reference Summary

| Level | Method | Module | Complexity |
|-------|--------|--------|------------|
| 1 | `Cell::is_valid()` | `core::cell` | O(1) |
| 1 | `Vertex::is_valid()` | `core::vertex` | O(1) |
| 2 | `Tds::is_valid()` | `core::triangulation_data_structure` | O(N×D²) |
| 2 | `Tds::validation_report()` | `core::triangulation_data_structure` | O(N×D²) |
| 3 | `Triangulation::validate_manifold()` | `core::triangulation` | O(N×D²) |
| 4 | `DelaunayTriangulation::validate_delaunay()` | `core::delaunay_triangulation` | O(N×V) |

---

## See Also

- [Topology Documentation](topology.md) - Topological concepts and Euler characteristic
- [Code Organization](code_organization.md) - Where to find validation code
- [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html) - Inspiration for validation design
