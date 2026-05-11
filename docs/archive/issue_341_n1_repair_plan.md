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

## Plan

1. Capture a fresh 3K 3D baseline with the default `N=1` repair policy and
   existing telemetry. Confirm wall time, skipped vertices, local repair calls,
   max queue, facet/ridge timing, and final validity.
2. Add diagnostics only if existing telemetry cannot identify the hotspot. Keep
   any new diagnostics narrow and focused on slow local repair samples:
   insertion index, trigger, seed-cell count, checked items, flips, max queue,
   elapsed time, and phase timing.
3. Optimize the contents of a single local repair pass. Since `N=1` avoids
   intentional seed backlog, focus inside the repair queue rather than batching
   pending seed frontiers. Candidate targets include redundant facet/ridge/edge
   queue entries, stale unchanged handles, and cheaper facet-first processing
   before ridge escalation where correctness allows.
4. Keep safety nets unchanged: final seeded repair, final global fallback,
   orientation canonicalization, and final validation must remain enabled.
5. Validate after each patch with focused flip/repair tests, large-scale debug
   tests, and repository checks.
6. Scale from 3K to 10K once the default `N=1` path is stable and measurably
   better. Reconsider #364 only if profiling shows snapshot/rollback or
   postcondition replay dominates after queue-content optimizations.

## Immediate Next Step

Run a fresh 3K 3D baseline with `N=1` and inspect whether the existing local
repair telemetry is enough to identify the dominant cost.
