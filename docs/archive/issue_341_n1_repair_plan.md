# Issue #341: N=1 Repair Performance Plan

This note captures the current plan for resolving #341 after defaulting batch
construction repair to every insertion (`N=1`). The goal is reasonable
performance on 10K vertices in 3D without compromising correctness,
orthogonality, or valid Delaunay output.

## Priority Order

1. Numerical correctness
2. Topological correctness
3. Orthogonality and maintainability
4. Performance within scope

## Current Direction

The branch now treats `EveryInsertion` / `N=1` as the default batch repair
cadence. Because of that, the next performance work should not optimize around
the `N=2` slowdown directly unless it exposes the same hotspot that affects the
default path.

The local flip-repair path has already been improved enough that it is no
longer the dominant 10K cost. The post-insertion topology validation path has
also been scoped to the cells touched by each ordinary insertion. The hot
insertion path now avoids full-TDS orientation normalization when the local
mutation scope is known. The remaining dominant costs are local repair, hull
extension, and ordinary insertion overhead.

## Latest Measurements

### 3K 3D, `N=1`

#### Before Scoped Post-Insertion Topology Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 54.571s.
- Insertion wall time: 52.908s.
- Local repairs: 485 calls, 5.499s total, 74.848ms max.
- Final repair: 1.082s, 0 flips.
- Final validation report: 580.243ms, OK.

#### After Scoped Post-Insertion Topology Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 27.991s.
- Insertion wall time: 26.356s.
- Local repairs: 485 calls, 5.561s total, 74.713ms max.
- Final repair: 1.068s, 0 flips.
- Final validation report: 565.459ms, OK.

#### After Local Insertion Orientation Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 14.599s.
- Insertion wall time: 12.950s.
- Cavity insertions: 2511 calls, 143.671ms total, 0.238ms max.
- Local repairs: 485 calls, 5.653s total, 76.645ms max.
- Final repair: 1.072s, 0 flips.
- Final validation report: 576.049ms, OK.

### 10K 3D, `N=1`

#### Before Scoped Post-Insertion Topology Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 630.582s.
- Insertion loop: 622.605s.
- Local repairs: 1037 calls, 35.162s total, 384.335ms max.
- Final repair: 3.784s, 0 flips.
- Final validation report: 2.004s, OK.
- Sampling showed the current hotspot in `validate_after_insertion`, especially
  `validate_ridge_links`, ridge-link graph construction, and temporary
  facet/ridge key work.

#### After Scoped Post-Insertion Topology Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 261.368s.
- Insertion loop: 253.540s.
- Transactional insertion wall: 190.238s.
- Cavity insertions: 8959 calls, 145.820s total, 36.662ms max.
- Hull extensions: 1037 calls, 14.403s total, 43.503ms max.
- Local repairs: 1037 calls, 46.638s total, 383.214ms max.
- Final repair: 3.689s, 0 flips.
- Final validation report: 2.008s, OK.

#### After Local Insertion Orientation Validation

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 99.466s.
- Insertion loop: 91.593s.
- Transactional insertion wall: 29.593s.
- Cavity insertions: 8959 calls, 743.262ms total, 0.496ms max.
- Hull extensions: 1037 calls, 14.276s total, 40.226ms max.
- Local repairs: 1037 calls, 45.478s total, 370.416ms max.
- Final repair: 3.713s, 0 flips.
- Final validation report: 2.015s, OK.

## Plan

1. Preserve the correctness model: every mutation must remain locally
   topology-safe, and final seeded repair, final global fallback, orientation
   canonicalization, and final validation must remain enabled.
2. Keep the scoped post-insertion topology validation path and compare it
   against full validation in focused tests whenever its scope changes.
3. Keep full validation available for final validation, explicit public
   validation, and any path where the mutation scope cannot be represented
   precisely.
4. Profile and optimize local repair without changing correctness: prioritize
   repeated facet/ridge checks, queue deduplication, and frontier narrowing.
5. Re-run the 3K and 10K large-scale debug cases with `N=1`, then compare
   local repair time, hull-extension time, final repair, and final validation.
6. Reconsider #364 only if profiling shows snapshot/rollback or postcondition
   replay dominates after topology validation is scoped.

## Immediate Next Step

Decide whether the ~99s 10K result is sufficient for #341. If not, profile the
local repair facet/ridge queues at 10K scale and reduce repeated checks without
weakening the final repair or final validation safety nets.
