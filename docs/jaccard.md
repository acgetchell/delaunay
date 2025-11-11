# Jaccard Set Similarity: Usage and Adoption Plan

This document outlines how and where to apply the Jaccard set similarity utilities
recently added to `src/core/util.rs`, and a concrete plan to adopt them across tests
in the next PR.

Utilities (available via `delaunay::core::util`):

- `jaccard_index<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64` — similarity in [0, 1]
- `jaccard_distance<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64` — dissimilarity in [0, 1]

References:

- Jaccard, P. (1901). Étude comparative de la distribution florale… Bulletin de la Société Vaudoise des Sciences Naturelles, 37(142), 547–579.
- Tanimoto, T. T. (1958). An Elementary Mathematical Theory of Classification and Prediction. IBM Report.

---

## Current usage

- `tests/proptest_delaunay_condition.rs`: insertion-order invariance now uses `jaccard_index` to compare edge sets.

---

## Additional usage opportunities

Below are high-value spots where Jaccard can turn brittle equality checks into robust similarity checks or provide clearer metrics.

1) Serialization vertex preservation

- File: `tests/serialization_vertex_preservation.rs`
- Use case: Compare sets of vertex coordinates before/after serde round-trip.
- Metric: `jaccard_index(original_coords, deserialized_coords)` ≥ 0.99 (ideally 1.0 for unique coordinates).

2) Storage backend behavioral equivalence

- File: `tests/storage_backend_compatibility.rs` (when present/active)
- Use case: Compare triangulation topology (edge sets) across backends.
- Metric: `jaccard_index(edges_backend_a, edges_backend_b)` ≥ 0.999 (expect near-identical results).

3) Convex hull consistency comparisons

- Files: `tests/convex_hull_bowyer_watson_integration.rs`, `tests/proptest_convex_hull.rs`
- Use cases:
  - Compare hull facet sets from repeated construction (stability).
  - Compare hull topology before/after transformations when logically invariant.
- Metric: Jaccard similarity on facet-identifier sets close to 1.0 where equality is expected.

4) Neighbor-set and facet-sharing validations

- Files: `tests/proptest_triangulation.rs`, `tests/proptest_cell.rs`
- Use cases:
  - Validate reciprocity and consistency using set similarity as a diagnostic.
  - Optional: replace brittle equality with similarity during randomized testing.

5) Re-enabling insertion-order invariance properties

- File: `tests/proptest_delaunay_condition.rs`
- Use case: When stabilizing and re-enabling the ignored properties, keep Jaccard as principal similarity metric for edge set comparison.
- Metric: Threshold tuning (see Issue #120 plan) — start around 0.85–0.95 after other robustness improvements.

---

## Adoption plan

Goal: Consolidate set-comparison logic across tests using `core::util::{jaccard_index, jaccard_distance}` and introduce robust, well-instrumented similarity checks.

### Implementation status

- [x] **Extraction helpers** (`src/core/util.rs`): Implemented canonical set extraction utilities
  - `extract_vertex_coordinate_set()` - Extract `HashSet<Point<T, D>>` from TDS
  - `extract_edge_set()` - Extract canonical edge pairs as `HashSet<(u128, u128)>`
  - `extract_facet_identifier_set()` - Extract boundary facet keys as `HashSet<u64>`
  - `extract_hull_facet_set()` - Extract convex hull facet keys as `HashSet<u64>`
  - Uses existing `FacetView::key()` API for facet identification
  - All helpers include comprehensive documentation and examples

- [x] **Testing utilities** (`src/core/util.rs`): Implemented Jaccard assertion macro and diagnostics
  - `JaccardComputationError` - Proper error type for set size overflow (> 2^53)
  - `format_jaccard_report()` - Rich diagnostic reporting with:
    - Set sizes, intersection/union counts, Jaccard index
    - Sample symmetric differences (first 5 unique elements per set)
    - Safe f64 conversion with overflow detection
  - `assert_jaccard_gte!` macro - Assertion with automatic diagnostics on failure
  - Import with `use delaunay::assert_jaccard_gte;`

- [x] **Serialization tests** (`tests/serialization_vertex_preservation.rs`): Migrated to Jaccard similarity
  - Replaced coordinate-by-coordinate equality checks with `extract_vertex_coordinate_set()` + `assert_jaccard_gte!()`
  - Threshold: ≥ 0.99 (99% similarity required for vertex preservation)
  - All 4 tests passing

- [ ] **Storage backend compatibility** (`tests/storage_backend_compatibility.rs`): Not yet updated
  - Note: All tests currently ignored (Phase 4 evaluation tests)
  - Will extract edge sets and compare via Jaccard when activated
  - Planned threshold: ≥ 0.999

- [ ] **Convex hull tests**: Not yet updated
  - `tests/convex_hull_bowyer_watson_integration.rs`
  - `tests/proptest_convex_hull.rs`
  - Will use `extract_hull_facet_set()` for topology comparison
  - Planned thresholds: 0.95–1.0 depending on scenario

- [ ] **Triangulation invariants**: Not yet updated
  - `tests/proptest_triangulation.rs`
  - Will add optional Jaccard diagnostics for neighbor reciprocity failures
  - Retain strict assertions; use Jaccard only for enhanced error reporting

- [ ] **Documentation updates**: Partially complete
  - This file updated with implementation progress
  - Still TODO: Update `tests/README.md` with usage examples and threshold conventions

- [ ] **Prepare for insertion-order invariance re-enablement** (coordinate with Issue #120)

### Threshold conventions (as implemented)

- **Serialization tests**: ≥ 0.99 (strict preservation expected)
- **Storage backend compatibility**: ≥ 0.999 (near-exact equivalence expected)
- **Convex hull comparisons**:
  - Identity/roundtrip: 1.0 where appropriate
  - Numerically sensitive: 0.98–0.99
  - Randomized/stress tests: ≥ 0.95
- **Property tests**: Diagnostics only (retain strict invariants)

Thresholds are initial values subject to tuning based on CI feedback.

### Design decisions

**Facet key computation**: Uses existing `FacetView::key()` method which computes order-invariant 64-bit keys by:

1. Sorting vertex keys
2. Hashing with deterministic FNV-based algorithm (no random seeding)
3. Avoiding external dependencies (local implementation)

**Safe conversions**: All usize→f64 casts checked against 2^53 limit (f64 mantissa precision) with proper error handling via `JaccardComputationError`.

**Canonicalization**:

- Edges: `canonical_edge(u, v)` returns `(min(u,v), max(u,v))`
- Facets: `FacetView::key()` sorts vertex UUIDs before hashing
- Ensures stable, order-independent comparisons

### Acceptance criteria (partially met)

- [x] Tests compile and pass in debug and release
- [x] New helpers centralized in `src/core/util.rs`; no duplicate inline Jaccard logic
- [x] Comprehensive documentation with examples in helper functions
- [ ] `tests/README.md` updated with Jaccard usage and thresholds
- [ ] All test files migrated

### Risk mitigation

- ✅ Canonicalization enforced via helper functions
- ✅ Equality assertions retained where logically required
- ✅ Thresholds documented in code comments with rationale
- ✅ Safe f64 conversion prevents silent precision loss

---

## Example usage patterns

### Basic Jaccard computation

```rust
use std::collections::HashSet;
use delaunay::core::util::{jaccard_index, jaccard_distance};

let a: HashSet<_> = /* build set A */;
let b: HashSet<_> = /* build set B */;

let sim = jaccard_index(&a, &b);      // [0.0, 1.0]
let dist = jaccard_distance(&a, &b);  // [0.0, 1.0]

assert!(sim >= 0.95, "unexpected divergence: similarity={sim:.3}");
```

### Using the assertion macro (recommended)

```rust
use delaunay::assert_jaccard_gte;
use delaunay::core::util::extract_vertex_coordinate_set;
use delaunay::core::Tds;

let tds_before: Tds<f64, (), (), 3> = /* ... */;
let before_coords = extract_vertex_coordinate_set(&tds_before);

// ... perform operation (serialization, transformation, etc.) ...

let tds_after: Tds<f64, (), (), 3> = /* ... */;
let after_coords = extract_vertex_coordinate_set(&tds_after);

// Assert with automatic diagnostics on failure
assert_jaccard_gte!(
    &before_coords,
    &after_coords,
    0.99,
    "Vertex preservation through operation"
);
```

### Extracting canonical sets

```rust
use delaunay::core::util::{
    extract_vertex_coordinate_set,
    extract_edge_set,
    extract_facet_identifier_set,
    extract_hull_facet_set,
};
use delaunay::core::Tds;
use delaunay::geometry::algorithms::convex_hull::ConvexHull;

let tds: Tds<f64, (), (), 3> = /* ... */;

// Vertex coordinates
let vertices = extract_vertex_coordinate_set(&tds);

// Edges (canonical UUID pairs)
let edges = extract_edge_set(&tds);

// Boundary facets
let facets = extract_facet_identifier_set(&tds).unwrap();

// Convex hull facets
let hull = ConvexHull::from_triangulation(&tds).unwrap();
let hull_facets = extract_hull_facet_set(&hull, &tds);
```

---

## Cross-references

- Implementation: `src/core/util.rs` (SET SIMILARITY UTILITIES section)
- Tests integration overview: `tests/README.md`
- Stabilization plan for properties: see issue tracking (e.g., property tests re-enablement plan)
