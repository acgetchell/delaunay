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

## Adoption plan (Next PR)

Goal: Consolidate set-comparison logic across tests using `core::util::{jaccard_index, jaccard_distance}` and introduce robust, well-instrumented similarity checks.

Tasks:

- [ ] Serialization tests: compute `HashSet<Point<_, D>>` (or canonicalized coordinate keys) pre/post round-trip and assert high Jaccard index.
- [ ] Storage backend compatibility: extract canonical edge sets (sorted `(u128, u128)` UUID pairs) per backend and assert high Jaccard index.
- [ ] Convex hull tests: introduce helper to extract canonical facet identifiers (e.g., sorted vertex-key tuples) and compare via Jaccard where applicable.
- [ ] Triangulation invariants: add optional Jaccard diagnostics for neighbor reciprocity and facet overlaps
      (retain current strict assertions; use Jaccard to improve failure messages and enable fuzz-tolerant checks where needed).
- [ ] Document the new helpers and thresholds in `tests/README.md` and reference this document.
- [ ] Prepare thresholds and notes for re-enabling insertion-order invariance properties (coordinate with Issue #120).

Milestones:

1. Implement helpers for canonical edge/facet set extraction reused across tests.
2. Apply Jaccard in serialization and backend compatibility tests.
3. Integrate hull comparisons and optional triangulation diagnostics.
4. Update documentation and finalize threshold guidance.

Acceptance criteria:

- Tests compile and pass in debug and release.
- New helpers centralized; no duplicate inline Jaccard logic in tests.
- `tests/README.md` references Jaccard usage and thresholds.

Risk/considerations:

- Ensure canonicalization (sorted tuples, stable identifiers) before set comparison.
- Keep equality assertions where logically required; use Jaccard to prevent flakiness around near-degenerate cases.
- Track thresholds in comments with rationale to avoid silent regressions.

---

## Example usage pattern

```rust
use std::collections::HashSet;
use delaunay::core::util::{jaccard_index, jaccard_distance};

let a: HashSet<_> = /* build set A */;
let b: HashSet<_> = /* build set B */;

let sim = jaccard_index(&a, &b);      // [0.0, 1.0]
let dist = jaccard_distance(&a, &b);  // [0.0, 1.0]

assert!(sim >= 0.95, "unexpected divergence: similarity={sim:.3}");
```

---

## Cross-references

- Implementation: `src/core/util.rs` (SET SIMILARITY UTILITIES section)
- Tests integration overview: `tests/README.md`
- Stabilization plan for properties: see issue tracking (e.g., property tests re-enablement plan)
