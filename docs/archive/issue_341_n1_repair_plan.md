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
longer the dominant 10K cost. The next performance target is the topology
validation layer reached after insertion, especially ridge-link validation,
which currently rebuilds/checks more global structure than the local mutation
appears to require.

## Latest Measurements

### 3K 3D, `N=1`

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 54.571s.
- Insertion wall time: 52.908s.
- Local repairs: 485 calls, 5.499s total, 74.848ms max.
- Final repair: 1.082s, 0 flips.
- Final validation report: 580.243ms, OK.

### 10K 3D, `N=1`

- Result: valid Delaunay triangulation, no skipped vertices.
- Total wall time: 630.582s.
- Insertion loop: 622.605s.
- Local repairs: 1037 calls, 35.162s total, 384.335ms max.
- Final repair: 3.784s, 0 flips.
- Final validation report: 2.004s, OK.
- Sampling showed the current hotspot in `validate_after_insertion`, especially
  `validate_ridge_links`, ridge-link graph construction, and temporary
  facet/ridge key work.

## Plan

1. Preserve the correctness model: every mutation must remain locally
   topology-safe, and final seeded repair, final global fallback, orientation
   canonicalization, and final validation must remain enabled.
2. Replace the expensive post-insertion global topology check with a scoped
   topology validation path that checks only the changed cavity and its
   immediate boundary where correctness permits.
3. Keep full validation available for final validation, explicit public
   validation, and any path where the mutation scope cannot be represented
   precisely.
4. Validate the scoped topology checker against the existing full checker in
   focused tests, including interior insertions and hull extensions.
5. Re-run the 3K and 10K large-scale debug cases with `N=1`, then compare
   insertion wall time, local repair time, final repair, and final validation.
6. Reconsider #364 only if profiling shows snapshot/rollback or postcondition
   replay dominates after topology validation is scoped.

## Immediate Next Step

Implement a narrow scoped topology validation path for post-insertion checks,
then validate it against the full topology checker before rerunning the 10K
benchmark.
