# Production Review Remediation Checklist

This checklist tracks the May 2026 `rust-production-review` baseline audit.
Treat partial items as still open until their acceptance notes are satisfied.

## Status Legend

- [x] Closed in the current patch.
- [ ] Open, including partial mitigations that need follow-up work.

## Critical Issues

- [x] **1. Replace fixed duplicate-coordinate tolerance.**
  Duplicate detection now estimates tolerance from nearby geometry and ULP scale.
  Follow-up only if a public `DuplicatePolicy` becomes desirable.
- [x] **2. Remove `unreachable!()` from matrix element access.**
  `matrix_get` and `matrix_set` now use debug assertions for internal indexing
  mistakes without release `unreachable!()` / `assert!()` panic paths.
- [x] **3. Audit non-try stack-matrix dispatch panics.**
  Production matrix dispatch uses the fallible macro directly; the panicking
  convenience macro is test-only.
- [x] **4. Stop silently degrading on missing predicate matrix entries.**
  Predicate matrix helpers now return typed dimension-mismatch errors instead
  of interpreting structural matrix mismatches as boundary/degenerate geometry.
- [x] **5. Preserve TDS runtime identity across transactional rollback.**
  Rollback snapshots now use `clone_for_rollback()` so cache and handle
  provenance identity survives failed attempts.

## High-Value Improvements

- [ ] **6. Split very large source files.**
  Start with `core/algorithms/flips.rs`, then `triangulation/delaunay.rs`,
  `core/triangulation.rs`, and `core/tds.rs`. The triangulation-facing module
  split is tracked for v0.7.8 in #381.
- [ ] **7. Replace full-TDS clone rollback with journaled or localized rollback.**
  This remains the largest performance opportunity. Tracked for v0.8.0 in
  #364.
- [x] **8. Harden robust insphere against squared-norm overflow.**
  Share the relative-coordinate formulation with `AdaptiveKernel::in_sphere`
  or return a typed error for non-finite squared norms.
- [x] **9. Strengthen `OnSuspicion` validation coverage.**
  Normal-path validation now runs pseudomanifold checks, with PL guarantees
  layering ridge and vertex-link checks as required.
- [ ] **10. Replace nested `Option` neighbors with a typed enum.**
  Make unassigned neighbors distinct from assigned boundary slots and hide raw
  mutation behind accessors. Tracked for v0.7.8 in #387.
- [x] **11. Gate `cells_mut()` out of production builds.**
  The raw storage accessor was deleted; tests now use narrower mutation paths
  or exercise lower-level helpers directly.
- [ ] **12. Confirm clone semantics for linear-algebra error variants.**
  Verify `LaError: Clone` and keep parent error enums honestly cloneable.
  Tracked for v0.7.8 in #384.
- [ ] **13. Make strict insphere consistency test control isolated.**
  Rename the once-init env flag for process-wide semantics or use an atomic
  test hook. Tracked for v0.7.8 in #383.
- [ ] **14. Consolidate focused preludes.**
  Reduce overlap and make import surfaces more orthogonal. Folded into the
  v0.7.8 triangulation-module cleanup in #381.
- [x] **15. Audit FastHashMap exposure to attacker-controlled hash keys.**
  Coordinate-derived hash-grid and epsilon-dedup buckets now use randomized
  `SecureHashMap`; remaining `FastHashMap` keys are slot keys, UUID identities,
  or stable facet/ridge hashes derived from slot keys.

## Performance Opportunities

- [x] **16. Avoid duplicate-detection linear-scan fallback where possible.**
  Duplicate spatial indexes now survive topology-only repair and outer
  transactional rollback instead of being cloned/restored or discarded; committed
  removals prune the deleted key, and query paths continue validating grid hits
  against the live TDS.
- [ ] **17. Add a leaner adaptive orientation fast path.**
  Avoid paying the diagnostic plus exact predicate path when SoS is unnecessary.
  Tracked for v0.7.8 in #256.
- [x] **18. Avoid fresh UUID allocation for rollback snapshots.**
  Rollback snapshots now preserve identity with `Arc::clone`; ordinary `Clone`
  intentionally keeps fresh runtime identity.
- [x] **19. Make cell vertex-existence checks debug-only where proven upstream.**
  `Tds::insert_cell_with_mapping` remains fully checked and returns typed
  errors. Audited hot paths now call
  `insert_cell_with_mapping_trusted_vertices`, which checks vertex provenance
  in debug builds and skips the O(D) slotmap lookup in release.
- [x] **20. Gate per-attempt timing on diagnostics or benchmark features.**
  Insertion telemetry now records counters by default and starts per-attempt
  timers only for construction-statistics paths that consume elapsed-time
  telemetry.
- [x] **21. Confirm `AdaptiveKernel::in_sphere` cold-path allocation behavior.**
  Adaptive and robust SoS fallback paths now keep the temporary f64 simplex in
  `SmallBuffer`; the non-degenerate fast path still returns before any fallback
  point conversion.

## API Design Feedback

- [ ] **22. Reconcile topology guarantee and validation policy combinations.**
  Reject incoherent builder combinations or merge the overlapping policy axes.
  Tracked for v0.7.8 in #385.
- [ ] **23. Reconsider skipped insertions as success outcomes.**
  Make skipped duplicate and degeneracy outcomes harder for callers to ignore.
  Tracked for v0.7.8 in #386.
- [ ] **24. Make `Cell` encapsulation consistent.**
  Private neighbor storage plus accessors is the likely direction. Tracked for
  v0.7.8 in #387.
- [ ] **25. Protect `Vertex::incident_cell` mutation.**
  Introduce a checked setter or newtype so invalid incident-cell links are
  harder to construct. Tracked for v0.7.8 in #387.
- [x] **26. Revisit public `core` module naming.**
  Keep `crate::core` as the internal implementation namespace, and expose the
  public low-level surface through curated modules and focused preludes such as
  `tds`, `collections`, `algorithms`, and `query`. Tracked for v0.7.8 in #388.
- [ ] **27. Normalize boxing policy in Delaunay repair error variants.**
  Pick a consistent enum-size and payload strategy. Tracked for v0.7.8 in #384.

## Testing Gaps

- [x] **28. Add adversarial `OnSuspicion` insertion proptests.**
  Exact cocircular and cospherical insertion sequences now assert Level 4
  validation after each attempt and at sequence completion.
- [x] **29. Test non-finite predicate fall-through at topology level.**
  Non-finite insertions now return typed invalid-vertex errors and proptests
  assert the triangulation remains Level 4 valid afterward.
- [x] **30. Add rollback identity regression for facet caches.**
  Convex hull facet-cache tests now prove a failed insertion rollback preserves
  the cached `(identity, generation)` provenance key.
- [x] **31. Add allocation-bounded hot-path tests.**
  `tests/allocation_api.rs` now asserts zero allocations for TDS/public
  `cells()`/`vertices()` iterator paths, `Tds::cell_vertices`, and
  `facet_key_from_vertices`, plus an explicit allocation budget for the hinted
  locate fast path under `--features count-allocations`.
- [x] **32. Benchmark `Tds::clone` cost versus triangulation size.**
  `benches/tds_clone.rs` now measures full `Tds::clone()` snapshot cost across
  deterministic 2D-5D triangulations before any rollback redesign.
- [x] **33. Regression-test robust insphere overflow boundaries.**
  Include inputs near `||x|| ~= 1e154`.
- [ ] **34. Harden doctests that unwrap degenerate construction.**
  Use provably non-degenerate inputs or hidden `Result` wrappers. Tracked by
  the v0.7.8 doctest cleanup in #365 and builder-doc migration in #214.

## Optional And Nitpicks

- [ ] **A. Clarify cfg-only feature flags in `Cargo.toml`.**
  Group empty features under a cfg-only banner. Tracked for v0.7.8 in #382.
- [ ] **B. Remove stale deprecated-warning comment in `lib.rs`.**
  Delete or update the comment if no `allow(deprecated)` follows. Tracked for
  v0.7.8 in #382.
- [ ] **C. Relabel safe-code `SAFETY` comments in `tds.rs`.**
  Use `INVARIANT` where no unsafe reasoning is involved. Tracked for v0.7.8 in
  #382.
- [x] **D. Remove allocation from `geometry/sos.rs` predicate helper.**
  Orientation and insphere SoS helpers now use stack-backed small buffers for
  coordinate and lifted-column scratch storage in supported dimensions.
- [ ] **E. Verify `core/util/uuid.rs` panic helpers are test-only.**
  Keep panic and unreachable paths out of production code. Tracked for v0.7.8
  in #382.
- [x] **F. Remove scale-unit tolerance from the `remove_vertex` doctest.**
  The doctest now finds the vertex by UUID instead of coordinate epsilon.

## Tooling Follow-Up From This Patch

- [x] **G. Fix changelog utilities instead of hand-editing generated Markdown.**
  `postprocess-changelog` is reusable as a text transform, archive generation
  reuses it, and `just changelog` output now passes `just markdown-check`.
