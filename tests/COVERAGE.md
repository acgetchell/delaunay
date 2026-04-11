# Test Coverage Report

Generated on 2026-03-24 using `cargo tarpaulin --lib` (v0.7.3, lib unit tests only; doctests excluded due to tarpaulin MSRV mismatch)

## Overall Coverage: 68.48% (9,393/13,716 lines covered)

## Files by Coverage (Sorted by Coverage Percentage)

### Full Coverage (100%)

| File | Lines Covered | Total Lines |
|------|---------------|-------------|
| `src/core/adjacency.rs` | 24/24 | 24 |
| `src/core/boundary.rs` | 14/14 | 14 |
| `src/core/collections/helpers.rs` | 10/10 | 10 |
| `src/core/edge.rs` | 12/12 | 12 |
| `src/core/util/canonical_points.rs` | 17/17 | 17 |
| `src/core/util/facet_keys.rs` | 56/56 | 56 |
| `src/core/util/hashing.rs` | 12/12 | 12 |
| `src/core/util/uuid.rs` | 9/9 | 9 |
| `src/topology/spaces/euclidean.rs` | 9/9 | 9 |
| `src/topology/spaces/spherical.rs` | 9/9 | 9 |
| `src/topology/spaces/toroidal.rs` | 22/22 | 22 |

### High Coverage (≥70%)

| File | Coverage | Lines Covered | Total Lines |
|------|----------|---------------|-------------|
| `src/core/vertex.rs` | 96.19% | 101/105 | 105 |
| `src/core/util/deduplication.rs` | 95.12% | 39/41 | 41 |
| `src/core/util/facet_utils.rs` | 93.94% | 31/33 | 33 |
| `src/topology/traits/global_topology_model.rs` | 93.37% | 169/181 | 181 |
| `src/core/cell.rs` | 92.86% | 221/238 | 238 |
| `src/core/util/jaccard.rs` | 92.68% | 76/82 | 82 |
| `src/geometry/traits/coordinate.rs` | 92.31% | 24/26 | 26 |
| `src/geometry/sos.rs` | 91.67% | 121/132 | 132 |
| `src/core/facet.rs` | 91.20% | 114/125 | 125 |
| `src/geometry/point.rs` | 88.66% | 86/97 | 97 |
| `src/topology/manifold.rs` | 87.39% | 513/587 | 587 |
| `src/geometry/util/norms.rs` | 86.67% | 26/30 | 30 |
| `src/core/collections/spatial_hash_grid.rs` | 85.29% | 58/68 | 68 |
| `src/topology/characteristics/validation.rs` | 84.62% | 22/26 | 26 |
| `src/geometry/util/point_generation.rs` | 83.33% | 145/174 | 174 |
| `src/core/tds.rs` | 81.34% | 1,077/1,324 | 1,324 |
| `src/geometry/algorithms/convex_hull.rs` | 80.29% | 224/279 | 279 |
| `src/geometry/util/circumsphere.rs` | 79.69% | 51/64 | 64 |
| `src/core/operations.rs` | 76.32% | 29/38 | 38 |
| `src/topology/characteristics/euler.rs` | 76.47% | 143/187 | 187 |
| `src/geometry/util/conversions.rs` | 71.93% | 41/57 | 57 |
| `src/core/util/hilbert.rs` | 71.35% | 122/171 | 171 |
| `src/core/triangulation.rs` | 69.94% | 1,197/1,710 | 1,710 |

### Medium Coverage (40–69%)

| File | Coverage | Lines Covered | Total Lines |
|------|----------|---------------|-------------|
| `src/triangulation/delaunay.rs` | 66.56% | 1,192/1,791 | 1,791 |
| `src/geometry/robust_predicates.rs` | 66.36% | 73/110 | 110 |
| `src/core/algorithms/flips.rs` | 66.33% | 1,440/2,171 | 2,171 |
| `src/geometry/quality.rs` | 65.88% | 56/85 | 85 |
| `src/geometry/util/measures.rs` | 64.92% | 161/248 | 248 |
| `src/core/traits/facet_cache.rs` | 62.50% | 35/56 | 56 |
| `src/geometry/matrix.rs` | 61.54% | 8/13 | 13 |
| `src/core/algorithms/locate.rs` | 60.43% | 252/417 | 417 |
| `src/triangulation/builder.rs` | 55.89% | 441/789 | 789 |
| `src/geometry/kernel.rs` | 55.45% | 61/110 | 110 |
| `src/geometry/predicates.rs` | 54.84% | 102/186 | 186 |
| `src/core/algorithms/incremental_insertion.rs` | 54.19% | 569/1,050 | 1,050 |
| `src/core/util/delaunay_validation.rs` | 47.62% | 90/189 | 189 |
| `src/geometry/util/triangulation_generation.rs` | 47.65% | 81/170 | 170 |

### Low Coverage (<40%) — Priority for Improvement

| File | Coverage | Lines Covered | Total Lines | Notes |
|------|----------|---------------|-------------|-------|
| `src/core/util/measurement.rs` | 33.33% | 2/6 | 6 | Feature-gated (`count-allocations`) |
| `src/lib.rs` | 28.57% | 2/7 | 7 | Re-exports only |
| `src/triangulation/flips.rs` | 12.50% | 4/32 | 32 | Public trait; delegates to `core::algorithms::flips` |
| `src/topology/traits/topological_space.rs` | 0.00% | 0/20 | 20 | Type definitions, minimal executable code |

## Coverage by Module

### Core Module: 69.09% (6,960/10,072 lines)

- **Data structures** (`cell`, `vertex`, `facet`, `edge`, `boundary`): 91–100%
- **TDS** (`tds`): 81%
- **Triangulation layer** (`triangulation`): 70%
- **Delaunay layer** (`delaunay_triangulation`): 67%
- **Algorithms** (`flips`, `incremental_insertion`, `locate`): 54–66%
- **Builder**: 56%
- **Utilities** (`canonical_points`, `facet_keys`, `hashing`, `uuid`): 95–100%
- **Collections** (`spatial_hash_grid`, `helpers`): 85–100%

### Geometry Module: 69.89% (1,346/1,926 lines)

- **Point/Coordinate**: 89–92%
- **SoS** (`sos`): 92%
- **Convex hull**: 80%
- **Predicates** (`predicates`, `robust_predicates`): 55–66%
- **Kernel**: 55%
- **Quality metrics**: 66%
- **Utilities** (`circumsphere`, `conversions`, `norms`): 72–87%

### Topology Module: 85.22% (887/1,041 lines)

- **Manifold**: 87%
- **Euler characteristic**: 76%
- **Validation**: 85%
- **Spaces** (euclidean, spherical, toroidal): 100%
- **Topology model trait**: 93%

### Triangulation Public API: 12.50% (4/32 lines)

- `src/triangulation/flips.rs`: public flip trait (mostly delegates to `core::algorithms::flips`)

## Notes

- Coverage is from `cargo tarpaulin --lib` (unit tests only). Integration tests
  (`tests/`) and property tests (`proptest_*.rs`) exercise significant additional
  code paths not reflected here.
- Benchmark files (`benches/`) are excluded (0% is expected).
- `src/core/util/measurement.rs` is feature-gated (`count-allocations`) and
  intentionally low.
- `src/topology/traits/topological_space.rs` contains public type definitions
  with minimal executable code.
