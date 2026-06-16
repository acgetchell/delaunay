# Triangulation Validation Guide

This document explains the validation hierarchy in the delaunay library and provides guidance on when and how to use each validation level.

For the theoretical background, rationale, and implementation pointers behind the invariants, see
[`invariants.md`](invariants.md).

Examples that derive `thiserror::Error` assume the example crate includes
`thiserror`; run `cargo add thiserror` alongside `delaunay` when copying those
snippets into an application.

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
`TopologyGuarantee` (default: `PLManifold`):

- Codimension-1 manifoldness (facet degree: 1 or 2 incident simplices per facet)
- Codimension-2 boundary manifoldness (the boundary is closed; "no boundary of boundary")
- Ridge-link validation (when `TopologyGuarantee::PLManifold` or `TopologyGuarantee::PLManifoldStrict`)
- Vertex-link validation during insertion (when `TopologyGuarantee::PLManifoldStrict`)
- Connectedness (single component)
- No isolated vertices
- Euler characteristic

Note: neighbor-pointer consistency is a **Level 2** structural invariant checked by
`Tds::is_valid()` / `Tds::validate()`, and is intentionally not part of Level 3.

Automatic validation does **not** run Level 4 (the Delaunay empty-circumsphere property).
If you need geometric verification, call `dt.is_valid()` or `dt.validate()` explicitly.

### Default: derived from `TopologyGuarantee`

The initial policy is derived from the active `TopologyGuarantee`: `PLManifold`
uses `ValidationPolicy::ExplicitOnly`, `PLManifoldStrict` uses
`ValidationPolicy::Always`, and `Pseudomanifold` uses
`ValidationPolicy::OnSuspicion`.

With `ValidationPolicy::OnSuspicion`, Level 3 validation runs only when insertion
deviates from the happy-path and trips internal **suspicion flags**, e.g.:

- A perturbation retry was required (geometric degeneracy).
- The insertion fell back to a conservative “star-split” of the containing simplex.
- Non-manifold facet issues were detected and repaired (simplices removed).
- Neighbor pointers had to be repaired **and at least one pointer actually changed** (running the repair routine is not, by itself, considered suspicious).

### Available policies

- `ValidationPolicy::Never`: never run full Level 3 automatically; compatible only with
  `TopologyGuarantee::Pseudomanifold`.
- `ValidationPolicy::ExplicitOnly` *(default for `PLManifold`)*: run full Level 3
  only through explicit validation calls while still keeping topology checks required
  by the active `TopologyGuarantee`.
- `ValidationPolicy::OnSuspicion` *(default for `Pseudomanifold`)*: run Level 3
  only when insertion is suspicious.
- `ValidationPolicy::Always`: run Level 3 after every insertion attempt (slowest, best for tests).
- `ValidationPolicy::DebugOnly`: always run Level 3 in debug builds; in release behaves like `OnSuspicion`.

### Example: configuring validation policy

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::validation::{ValidationConfigurationError, ValidationPolicy};

#[derive(Debug, thiserror::Error)]
enum ValidationExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Configuration(#[from] ValidationConfigurationError),
}

fn main() -> Result<(), ValidationExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Default PL-manifold mode: caller-owned full validation checkpoints.
    assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);

    // For test/debug: validate topology after every insertion.
    dt.set_validation_policy(ValidationPolicy::Always);

    // For caller-owned full validation checkpoints with the default PL-manifold guarantee.
    dt.try_set_validation_policy(ValidationPolicy::ExplicitOnly)?;

    // `Never` is reserved for the relaxed pseudomanifold guarantee.
    dt.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)?;
    dt.try_set_validation_policy(ValidationPolicy::Never)?;
    Ok(())
}
```

---

## Choosing Level 3 topology guarantee (`TopologyGuarantee`)

Level 3 topology validation can be configured to enforce either:

- **PL-manifold** invariants (default, uses ridge-link checks during insertion and
  requires completion-time vertex-link validation), or
- **Pseudomanifold / manifold-with-boundary** invariants (relaxed mode).

This is separate from [`ValidationPolicy`](#automatic-validation-during-incremental-insertion-validationpolicy),
which controls *when* Level 3 is run automatically during incremental insertion.
The builder keeps these axes coherent by deriving the initial validation policy
from `TopologyGuarantee`: `PLManifoldStrict` starts with `ValidationPolicy::Always`,
`PLManifold` starts with `ValidationPolicy::ExplicitOnly`, and
`Pseudomanifold` starts with `ValidationPolicy::OnSuspicion`.

### Default: `PLManifold`

`PLManifold` uses fast ridge-link validation during insertion and requires a
vertex-link validation pass at construction completion to certify full
PL-manifoldness. You can trigger that final certification via
`Triangulation::validate_at_completion()` (or `Triangulation::validate()`).

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);

    // Optional: relax topology checks for speed (weaker guarantees).
    dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

    // Now Level 3 skips vertex-link validation entirely.
    assert!(dt.as_triangulation().is_valid().is_ok());
    Ok(())
}
```

### Strict: `PLManifoldStrict`

`PLManifoldStrict` runs full vertex-link validation after every insertion. This
matches the legacy `PLManifold` behavior (slowest, maximum safety).

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

- `TdsError` (Levels 1–2): element + structural invariants.
- `TriangulationValidationError` (Level 3): wraps `TdsError` and adds
  codimension-1 manifoldness + codimension-2 boundary manifoldness (closed boundary) +
  (optional) vertex-link PL-manifold checks + connectedness + isolated-vertex + Euler characteristic checks.
- `DelaunayTriangulationValidationError` (Level 4): wraps `TriangulationValidationError` and adds
  the empty-circumsphere (Delaunay) checks.

### Reporting (full diagnostics)

`DelaunayTriangulation::validation_report()` returns `Result<(), TriangulationValidationReport>`.
On failure, the `Err(TriangulationValidationReport)` contains a `Vec<InvariantViolation>`; each
`InvariantViolation` stores an `InvariantKind` plus an `InvariantError` **enum** that wraps the
structured error from the failing layer (`TdsError`, `TriangulationValidationError`, or
`DelaunayTriangulationValidationError`).

---

## Level 1: Element Validity

### Purpose

Validates basic data integrity of individual vertices and simplices.

### Methods

- `Simplex::is_valid()` - Check if a simplex has valid structure
- `Vertex::is_valid()` - Check if a vertex has valid coordinates

### What It Checks

- **Vertices**: Coordinate validity, UUID presence, dimension consistency
- **Simplices**: Correct number of vertices (D+1), no duplicate vertices, valid UUID

### Complexity

- **Time**: O(1) per element
- **Space**: O(1)

### When to Use

- Building blocks for higher-level validation
- Rarely called directly by users
- Automatically called by Level 2

### Example

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulation, TopologyGuarantee, Vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

let v = delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?;
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

1. **UUID ↔ Key Mappings**: Bidirectional consistency for vertices and simplices
2. **No Duplicate Simplices**: No simplices with identical vertex sets
3. **Facet Sharing Invariant**: Each facet shared by at most 2 simplices
4. **Neighbor Consistency**: Mutual neighbor relationships are correct (boundary facets have no neighbor; interior facets have reciprocal neighbors)

`Tds::validate()` (Levels 1–2) additionally checks:

- **Vertex Validity**: All vertices pass `Vertex::is_valid()`
- **Simplex Validity**: All simplices pass `Simplex::is_valid()`
- **Simplex Coordinate Uniqueness**: No simplex contains two vertices with identical coordinates
  (exact `OrderedFloat` comparison). Duplicate-coordinate vertices produce zero-volume
  simplices that break SoS and Pachner moves.
  **Note**: `is_valid()` does **not** check coordinate uniqueness. Use `validate()` (or
  `validation_report()`) for the stronger guarantee.

### Complexity

- **Time**: O(N×D²) where N = number of simplices, D = dimension
- **Space**: O(N×D) for facet-to-simplices map

### When to Use

- **Production**: After construction or major modifications
- **Tests**: In test suites to catch structural bugs
- **Debug builds**: Use `debug_assert!(dt.tds().is_valid().is_ok())`

### Example

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
    Ok(())
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

1. **Codimension-1 manifoldness (facet degree)**: Each facet belongs to exactly 1 simplex (boundary) or exactly 2 simplices (interior)
   - Stronger than Level 2's "≤2 simplices per facet"
2. **Codimension-2 boundary manifoldness (closed boundary)**: Each (d−2)-ridge on the boundary must be incident to exactly 2 boundary facets
   - This is the "no boundary of boundary" condition
   - Interior ridges can have higher degree; only boundary ridges are constrained
3. **PL-manifold vertex-link condition** (when `TopologyGuarantee::PLManifold`):
   For every vertex `v`, the link `Lk(v)` must be a (D−1)-sphere (interior vertex) or (D−1)-ball (boundary vertex).
4. **Connectedness**: All simplices form a single connected component in the simplex neighbor graph
   - Detected via a graph traversal over neighbor pointers (O(N·D))
5. **No isolated vertices**: Every vertex must be incident to at least one simplex
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
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.25, 0.25, 0.25])?, // Interior point
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Thorough topology validation (includes Levels 1–2 TDS checks)
    match dt.as_triangulation().validate() {
        Ok(()) => println!("✓ Valid manifold with correct Euler characteristic"),
        Err(e) => eprintln!("✗ Topology validation failed: {}", e),
    }
    Ok(())
}
```

---

## Level 4: Delaunay Property

### Purpose

Validates the geometric optimality of the triangulation.

### Methods

- `DelaunayTriangulation::is_valid()` - Level 4 Delaunay property only (fast verification via local flip predicates).
- `DelaunayTriangulation::validate()` - Levels 1–4 (elements + structure + topology + Delaunay property).

### What It Checks

- **Delaunay property**: Verified via local flip predicates (k=2/k=3 and
  inverses), equivalent to the empty-circumsphere condition for properly
  constructed triangulations
- Uses geometric predicates from the kernel (`insphere` test)
- **Independent of Levels 1-3**: Checks geometric property, not structural/topological
- **Flip-based repair**: Insertions run k=2/k=3 flip repairs with inverse edge/triangle queues in
  higher dimensions by default. Delaunay validation can still fail if repair is disabled, if repair
  fails to converge, or if inputs are highly degenerate/duplicate-heavy. See
  [Issue #120 Investigation](archive/issue_120_investigation.md).
- **Heuristic fallback**: If flip-based repair does not converge, you can opt into a heuristic
  rebuild fallback via `DelaunayTriangulation::repair_delaunay_with_flips_advanced`.
  This requires `TopologyGuarantee::PLManifold` and `K: ExactPredicates`, and records the
  shuffle/perturbation seeds used. See [Numerical Robustness Guide](numerical_robustness_guide.md).

### Complexity

- **Time**:
  - `DelaunayTriangulation::is_valid()` (Level 4 only): O(simplices) (local flip-predicate verification, for fixed D)
  - `DelaunayTriangulation::validate()` (Levels 1–4): O(simplices × D²) + O(simplices) (typically dominated by Levels 1–3)
  - `DelaunayTriangulation::validation_report()` (Levels 1–4): O(simplices × D²) + O(simplices)
- **Space**: O(1) additional space (aside from temporary working sets)

### When to Use

- **Critical Applications**: When Delaunay guarantees are essential (interpolation, mesh quality)
- **Tests**: After construction to verify correctness
- **Debug**: Investigating geometric issues or suspected violations
- **Avoid**: Hot loops (still O(simplices); use for spot checks / tests)

### Example

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Delaunay property validation (Level 4)
    match dt.is_valid() {
        Ok(()) => println!("✓ All simplices satisfy empty circumsphere property"),
        Err(e) => eprintln!("✗ Delaunay violation: {}", e),
    }
    Ok(())
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

## Performance notes

- Level 2 and Level 3 validation are dominated by combinatorial bookkeeping (roughly O(simplices × D²)).
- Level 4 `DelaunayTriangulation::is_valid()` verifies the Delaunay property via local flip predicates and is
  roughly O(simplices) for fixed `D`.
- A brute-force empty-circumsphere check would be O(simplices × vertices) and is not used by `is_valid()`.

In practice, `DelaunayTriangulation::validate()` is usually dominated by Level 3 (topology) work.
As a post-construction acceptance check, the current 7,500-vertex 3D large-scale
debug harness is the default near-one-minute `validation_report` run for Levels
1–4; on maintainer Apple M4 Max hardware the final report itself is a
low-single-digit-second step.
The explicit 10,000-vertex 3D run is a heavier characterization probe that has
also passed Levels 1–4 validation, but it is not the default local acceptance
helper.

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
use delaunay::prelude::query::*;

pub fn my_algorithm(dt: &mut DelaunayTriangulation<FastKernel<f64>, (), (), 3>) {
    // Do work...

    #[cfg(debug_assertions)]
    {
        debug_assert!(dt.tds().is_valid().is_ok(), "TDS structure violated");
        debug_assert!(
            dt.as_triangulation().is_valid().is_ok(),
            "Topology invariant violated"
        );
    }
}
```

### Pattern 3: Conditional Deep Validation

```rust
use delaunay::prelude::query::*;

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
**Fix**: Check for duplicate simplices, incorrect neighbor assignments, or mapping inconsistencies

### Validation Passes Level 2, Fails at Level 3

**Problem**: Codimension-1 manifoldness violated (facet has 0 or >2 simplices), boundary is not closed ("boundary of boundary"),
triangulation disconnected, isolated vertex present, or Euler characteristic wrong
**Likely Cause**: Non-manifold topology, missing/broken neighbor wiring, boundary topology corruption,
or disconnected components
**Fix**: Check facet-to-simplices mapping, ensure boundary ridges have degree 2 within boundary facets,
ensure no isolated vertices, and verify the simplex neighbor graph is connected

### Validation Passes Level 3, Fails at Level 4

**Problem**: Delaunay property violated (vertex inside circumsphere)
**Likely Cause**: Repair disabled or non-convergent, geometric degeneracy, numerical precision,
or missing higher-dimensional flip coverage
**Fix**: Keep flip repair enabled, handle insertion errors, check for near-coplanar/collinear points,
and consider using `RobustKernel` or `AdaptiveKernel` instead of `FastKernel` (explicit repair
methods require `K: ExactPredicates`, which `FastKernel` does not implement). If repair fails to
converge, consider the opt-in heuristic rebuild fallback via
`dt.repair_delaunay_with_flips_advanced(...)` (requires PL-manifold + `ExactPredicates`).

---

## API Reference Summary

| Level | Method | Module | Complexity |
|-------|--------|--------|------------|
| 1 | `Simplex::is_valid()` | `tds` | O(1) |
| 1 | `Vertex::is_valid()` | `tds` | O(1) |
| 2 | `Tds::is_valid()` | `tds` | O(N×D²) |
| 2 | `Tds::validate()` | `tds` | O(N×D²) |
| 3 | `Triangulation::is_valid()` | `triangulation` | O(N×D²) |
| 3 | `Triangulation::validate()` | `triangulation` | O(N×D²) |
| 4 | `DelaunayTriangulation::is_valid()` | `delaunay` | O(simplices) |
| 4 | `DelaunayTriangulation::validate()` | `delaunay` | O(simplices × D²) + O(simplices) |
| — | `DelaunayTriangulation::validation_report()` | `delaunay` | O(simplices × D²) + O(simplices) |

---

## See Also

- [Invariants](invariants.md) - Theoretical background and rationale for the invariants
- [Topology](topology.md) - Level 3 topology invariants and combinatorial checks
- [Code Organization](code_organization.md) - Where to find validation code
- [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html) - Inspiration for validation design
