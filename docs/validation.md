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

## Automatic validation during incremental insertion (`ValidationPolicy`)

The library always provides **explicit** validation APIs (Levels 1–4) that you can call when you need them.

Separately, incremental construction (`new()` / `insert*()`) can run an **automatic**
*Level 3* topology validation pass after an insertion attempt, controlled by a
`ValidationPolicy` on the triangulation.

This is a performance vs certainty knob: Level 3 (`Triangulation::is_valid()`) is
relatively expensive, so the default behavior is to validate only when something
looks “off”.

### What is validated automatically?

Only **Level 3** (`Triangulation::is_valid()`), using the triangulation’s current
`TopologyGuarantee` (default: `Pseudomanifold`):

- Codimension-1 manifoldness (facet degree: 1 or 2 incident cells per facet)
- Codimension-2 boundary manifoldness (the boundary is closed; "no boundary of boundary")
- (Optional) PL-manifold ridge-link condition (when `TopologyGuarantee::PLManifold`)
- Connectedness (single component)
- No isolated vertices
- Euler characteristic

Note: neighbor-pointer consistency is a **Level 2** structural invariant checked by
`Tds::is_valid()` / `Tds::validate()`, and is intentionally not part of Level 3.

Automatic validation does **not** run Level 4 (the Delaunay empty-circumsphere property).
If you need geometric verification, call `dt.is_valid()` or `dt.validate()` explicitly.

### Default: `OnSuspicion`

The default policy is `ValidationPolicy::OnSuspicion`: we validate Level 3 only when the
insertion deviates from the happy-path and trips internal **suspicion flags**, e.g.:

- A perturbation retry was required (geometric degeneracy).
- The insertion fell back to a conservative “star-split” of the containing cell.
- Non-manifold facet issues were detected and repaired (cells removed).
- Neighbor pointers had to be repaired **and at least one pointer actually changed** (running the repair routine is not, by itself, considered suspicious).

### Available policies

- `ValidationPolicy::Never`: never run Level 3 automatically (fastest, least guarded).
- `ValidationPolicy::OnSuspicion` *(default)*: run Level 3 only when insertion is suspicious.
- `ValidationPolicy::Always`: run Level 3 after every insertion attempt (slowest, best for tests).
- `ValidationPolicy::DebugOnly`: always run Level 3 in debug builds; in release behaves like `OnSuspicion`.

### Example: configuring validation policy

```rust
use delaunay::prelude::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

// Default: validate topology only when insertion is suspicious.
assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);

// For test/debug: validate topology after every insertion.
dt.set_validation_policy(ValidationPolicy::Always);

// For maximum performance (you can still validate explicitly when you choose).
dt.set_validation_policy(ValidationPolicy::Never);
```

---

## Choosing Level 3 topology guarantee (`TopologyGuarantee`)

Level 3 topology validation can be configured to enforce either:

- **Pseudomanifold / manifold-with-boundary** invariants (default), or
- **PL-manifold** invariants (strict mode, adds ridge-link validation).

This is separate from [`ValidationPolicy`](#automatic-validation-during-incremental-insertion-validationpolicy),
which controls *when* Level 3 is run automatically during incremental insertion.

### Default: `Pseudomanifold`

```rust
use delaunay::prelude::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
assert_eq!(dt.manifold_validation_mode(), TopologyGuarantee::Pseudomanifold);

// Opt into stricter PL-manifold validation.
dt.set_manifold_validation_mode(TopologyGuarantee::PLManifold);

// Now Level 3 includes ridge-link validation.
dt.as_triangulation().is_valid().unwrap();
```

---

## Error Types by Layer

The library separates **construction-time** failures from **validation-time** invariant violations, and also separates errors by layer.

### Construction errors (building a triangulation)

- `TdsConstructionError` (Level 2 construction): internal TDS insertion/mapping failures
  (e.g. duplicate UUIDs).
- `TriangulationConstructionError` (Level 3 construction): wraps `TdsConstructionError` and adds
  triangulation-layer failures (e.g. `GeometricDegeneracy`, `DuplicateCoordinates`,
  `InsufficientVertices`).
- `DelaunayTriangulationConstructionError` (Level 4 construction): wraps
  `TriangulationConstructionError`.

### Validation errors (checking invariants)

- `TdsValidationError` (Levels 1–2): element + structural invariants.
- `TriangulationValidationError` (Level 3): wraps `TdsValidationError` and adds
  codimension-1 manifoldness + codimension-2 boundary manifoldness (closed boundary) +
  (optional) ridge-link PL-manifold checks + connectedness + isolated-vertex + Euler characteristic checks.
- `DelaunayTriangulationValidationError` (Level 4): wraps `TriangulationValidationError` and adds
  the empty-circumsphere (Delaunay) checks.

### Reporting (full diagnostics)

`DelaunayTriangulation::validation_report()` returns `Result<(), TriangulationValidationReport>`.
On failure, the `Err(TriangulationValidationReport)` contains a `Vec<InvariantViolation>`; each
`InvariantViolation` stores an `InvariantKind` plus an `InvariantError` **enum** that wraps the
structured error from the failing layer (`TdsValidationError`, `TriangulationValidationError`, or
`DelaunayTriangulationValidationError`).

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

### Methods

- `Tds::is_valid()` - Level 2 (structural) checks only (fast-fail).
- `Tds::validate()` - Levels 1–2 (elements + structural).
- `DelaunayTriangulation::validation_report()` - Cumulative diagnostic report across Levels 1–4.

### What It Checks

`Tds::is_valid()` (Level 2) checks:

1. **UUID ↔ Key Mappings**: Bidirectional consistency for vertices and cells
2. **No Duplicate Cells**: No cells with identical vertex sets
3. **Facet Sharing Invariant**: Each facet shared by at most 2 cells
4. **Neighbor Consistency**: Mutual neighbor relationships are correct (boundary facets have no neighbor; interior facets have reciprocal neighbors)

`Tds::validate()` (Levels 1–2) additionally checks:

- **Vertex Validity**: All vertices pass `Vertex::is_valid()`
- **Cell Validity**: All cells pass `Cell::is_valid()`

### Complexity

- **Time**: O(N×D²) where N = number of cells, D = dimension
- **Space**: O(N×D) for facet-to-cells map

### When to Use

- **Production**: After construction or major modifications
- **Tests**: In test suites to catch structural bugs
- **Debug builds**: Use `debug_assert!(dt.tds().is_valid().is_ok())`

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

// Quick structural check (Level 2)
assert!(dt.tds().is_valid().is_ok());

// Detailed report showing all violations across Levels 1–4 (on failure)
match dt.validation_report() {
    Ok(()) => println!("✓ All invariants satisfied"),
    Err(report) => {
        for violation in report.violations {
            eprintln!("Invariant violation: {:?}", violation);
        }
    }
}
```

### Diagnostics

For most users, start with `dt.tds().is_valid()` (fast-fail) or `dt.validation_report()` (full diagnostics across Levels 1–4).

---

## Level 3: Manifold Topology

### Purpose

Validates that the triangulation forms a valid topological manifold.

### Methods

- `Triangulation::is_valid()` - Level 3 topology validation only.
- `Triangulation::validate()` - Levels 1–3 (elements + structure + topology).

### What It Checks

`Triangulation::is_valid()` (Level 3) checks:

1. **Codimension-1 manifoldness (facet degree)**: Each facet belongs to exactly 1 cell (boundary) or exactly 2 cells (interior)
   - Stronger than Level 2's "≤2 cells per facet"
2. **Codimension-2 boundary manifoldness (closed boundary)**: Each (d−2)-ridge on the boundary must be incident to exactly 2 boundary facets
   - This is the "no boundary of boundary" condition
   - Interior ridges can have higher degree; only boundary ridges are constrained
3. **PL-manifold ridge-link condition** (when `TopologyGuarantee::PLManifold`):
   Each (d−2)-ridge must have a link that is a connected 1-manifold (cycle or path)
4. **Connectedness**: All cells form a single connected component in the cell neighbor graph
   - Detected via a graph traversal over neighbor pointers (O(N·D))
5. **No isolated vertices**: Every vertex must be incident to at least one cell
6. **Euler Characteristic**: χ matches expected topology (when an expectation is defined)
   - Empty: χ = 0
   - Single simplex / Ball(D): χ = 1
   - Closed sphere S^D: χ = 1 + (-1)^D
   - Unknown: χ is computed but not enforced

`Triangulation::validate()` (Levels 1–3) additionally runs `Tds::validate()` first.

### Complexity

- **Time**: O(N×D²) dominated by simplex counting (connectedness adds O(N·D))
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

// Thorough topology validation (includes Levels 1–2 TDS checks)
match dt.as_triangulation().validate() {
    Ok(()) => println!("✓ Valid manifold with correct Euler characteristic"),
    Err(e) => eprintln!("✗ Topology validation failed: {}", e),
}
```

---

## Level 4: Delaunay Property

### Purpose

Validates the geometric optimality of the triangulation.

### Methods

- `DelaunayTriangulation::is_valid()` - Level 4 Delaunay property only (empty circumsphere).
- `DelaunayTriangulation::validate()` - Levels 1–4 (elements + structure + topology + Delaunay property).

### What It Checks

- **Empty Circumsphere Property**: For every D-dimensional cell, no vertex lies strictly inside its circumsphere
- Uses geometric predicates from the kernel (`insphere` test)
- **Independent of Levels 1-3**: Checks geometric property, not structural/topological
- **Known limitation (Issue #120)**: Construction is designed to satisfy this property, but rare local
  violations have been observed for near-degenerate inputs. See [Issue #120 Investigation](issue_120_investigation.md).

### Complexity

- **Time**:
  - `DelaunayTriangulation::is_valid()` (Level 4 only): O(N×V) in the worst case.
  - `DelaunayTriangulation::validate()` (Levels 1–4): O(N×D² + N×V)
    (typically dominated by O(N×V)).
  - `DelaunayTriangulation::validation_report()` (Levels 1–4): O(N×D² + N×V)
    (typically dominated by O(N×V)).
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

// Delaunay property validation (Level 4)
match dt.is_valid() {
    Ok(()) => println!("✓ All cells satisfy empty circumsphere property"),
    Err(e) => eprintln!("✗ Delaunay violation: {}", e),
}
```

---

## Decision Tree: Which Validation Level to Use?

```text
Start: Do you need to validate?
    │
    ├─ Writing tests / CI? → Always validate (start with Level 2; add Level 3/4 as needed)
    │
    ├─ Just built triangulation?
    │   ├─ Production hot path? → Usually skip (but validate during integration testing / when debugging)
    │   └─ Need certainty? → Validate (Level 2 or 3; add Level 4 if geometry matters)
    │
    ├─ After manual TDS mutation? → Level 2 (`dt.tds().is_valid()`)
    │
    ├─ Debugging geometric issues? → Level 4 (`dt.is_valid()`)
    │
    ├─ Production validation?
    │   ├─ Performance critical? → Level 2 (`dt.tds().is_valid()`)
    │   ├─ Topological correctness critical? → Level 3 (`dt.as_triangulation().is_valid()`)
    │   └─ Geometric correctness critical? → Level 4 (`dt.is_valid()`)
    │
    └─ Paranoid mode? → All levels (`dt.validate()`)
```

---

## Performance Comparison

The numbers below are **order-of-magnitude** examples to help you choose a validation level.

**Measurement conditions (baseline):**

- Single-threaded, `--release`, on typical modern hardware
- 3D triangulation with ~1000 vertices (~5000–6000 cells)
- Using the default kernel for most examples (`FastKernel<f64>`)

**Caveats:** Actual times can vary significantly with hardware and compiler settings,
dimension, and especially with **input degeneracy** (near-coplanar/collinear points),
and **kernel choice** (`FastKernel` vs `RobustKernel`).

| Level | Time | What It Does |
|-------|------|--------------|
| 1 | ~1μs | Single element check |
| 2 | ~10–50ms | Full structural validation |
| 3 | ~50–100ms | Structural + topological (connectedness + Euler) |
| 4 | ~100–500ms | Empty circumsphere for all cells (many `insphere` tests) |

**Why Level 4 is usually 2–5× slower than Level 3:** Level 3 is dominated by
combinatorial bookkeeping (roughly O(N×D²)), while Level 4 checks the
empty-circumsphere property and can be **O(cells × vertices)** in the worst case
(millions of `insphere` predicate evaluations for the sizes above). This is also where
`RobustKernel` can be substantially slower than `FastKernel`.

**Recommendation**: Use Level 2 in production (or skip entirely on hot paths once
thoroughly tested), and reserve Level 3+ for tests/debug and for situations where
topological or geometric guarantees are critical.

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
    assert!(dt.tds().is_valid().is_ok());        // Level 2: Structural
    assert!(dt.as_triangulation().is_valid().is_ok()); // Level 3: Topology
    assert!(dt.is_valid().is_ok());                 // Level 4: Delaunay property
    assert!(dt.validate().is_ok());                 // Levels 1–4: Full validation
}
```

### Pattern 2: Debug Build Validation

```rust
use delaunay::prelude::*;

pub fn my_algorithm(dt: &mut DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    // Do work...
    
    #[cfg(debug_assertions)]
    {
        dt.tds().is_valid().expect("TDS structure violated");
        dt.as_triangulation().is_valid().expect("Topology invariant violated");
    }
}
```

### Pattern 3: Conditional Deep Validation

```rust
use delaunay::prelude::*;

pub fn validate_with_level(dt: &DelaunayTriangulation<FastKernel<f64>, (), (), 3>, level: u8) -> Result<(), String> {
    match level {
        2 => dt.tds().is_valid().map_err(|e| e.to_string()),
        3 => dt.as_triangulation().is_valid().map_err(|e| e.to_string()),
        4 => dt.is_valid().map_err(|e| e.to_string()),
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

**Problem**: Codimension-1 manifoldness violated (facet has 0 or >2 cells), boundary is not closed ("boundary of boundary"),
triangulation disconnected, isolated vertex present, or Euler characteristic wrong
**Likely Cause**: Non-manifold topology, missing/broken neighbor wiring, boundary topology corruption,
or disconnected components
**Fix**: Check facet-to-cells mapping, ensure boundary ridges have degree 2 within boundary facets,
ensure no isolated vertices, and verify the cell neighbor graph is connected

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
| 2 | `Tds::validate()` | `core::triangulation_data_structure` | O(N×D²) |
| 3 | `Triangulation::is_valid()` | `core::triangulation` | O(N×D²) |
| 3 | `Triangulation::validate()` | `core::triangulation` | O(N×D²) |
| 4 | `DelaunayTriangulation::is_valid()` | `core::delaunay_triangulation` | O(N×V) |
| 4 | `DelaunayTriangulation::validate()` | `core::delaunay_triangulation` | O(N×D² + N×V) |
| — | `DelaunayTriangulation::validation_report()` | `core::delaunay_triangulation` | O(N×D² + N×V) |

---

## See Also

- [Topology Documentation](topology.md) - Topological concepts and Euler characteristic
- [Code Organization](code_organization.md) - Where to find validation code
- [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html) - Inspiration for validation design
