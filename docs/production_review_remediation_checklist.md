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

- [x] **6. Split very large source files.**
  `core/triangulation.rs` and the Delaunay-facing layer have been split into
  orthogonal construction, insertion, query, repair, orientation, validation,
  and serialization modules. The remaining large-file targets are `core/algorithms/flips.rs`
  and, if future review warrants another split, the TDS validation/mutation modules.
- [x] **7. Audit full-TDS clone rollback and centralize transactional mutation.**
  Scoped rollback guards now centralize failure-atomic TDS and triangulation
  mutation, with detached scratch workspaces retained where copy-on-success is
  the intended algorithm. Closed in #364; valid 2D removal rollback behavior
  was completed in #448.
- [x] **8. Harden robust insphere against squared-norm overflow.**
  Share the relative-coordinate formulation with `AdaptiveKernel::in_sphere`
  or return a typed error for non-finite squared norms.
- [x] **9. Strengthen `OnSuspicion` validation coverage.**
  Normal-path validation now runs pseudomanifold checks, with PL guarantees
  layering ridge and vertex-link checks as required.
- [x] **10. Replace nested `Option` neighbors with a typed enum.**
  `NeighborSlot` distinguishes unassigned, boundary, and neighboring-simplex
  states, while raw neighbor mutation remains behind TDS-owned accessors.
  Closed in #387.
- [x] **11. Gate `simplices_mut()` out of production builds.**
  The raw storage accessor was deleted; tests now use narrower mutation paths
  or exercise lower-level helpers directly.
- [x] **12. Confirm clone semantics for linear-algebra error variants.**
  `LaError` clone behavior and the parent error enums were audited together;
  clone bounds now reflect their payload semantics. Closed in #384.
- [x] **13. Make strict insphere consistency test control isolated.**
  The once-init strict-insphere env snapshot is now named and documented as
  process-wide, while unit tests use a thread-local override guard instead of
  mutating process environment state. Closed for v0.7.8 in #383.
- [x] **14. Consolidate focused preludes.**
  Delaunay-facing workflow preludes now live directly under
  `delaunay::prelude::{construction,insertion,flips,repair,delaunayize,diagnostics,validation}`,
  while `delaunay::prelude::triangulation` is scoped to the generic
  `Triangulation` layer.
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
- [x] **17. Add a leaner adaptive orientation fast path.**
  The adaptive kernel uses the filtered orientation sign before exact/SoS
  fallback, and the affected 2D-5D property tests are enabled. Closed in #256.
- [x] **18. Avoid fresh UUID allocation for rollback snapshots.**
  Rollback snapshots now preserve identity with `Arc::clone`; ordinary `Clone`
  intentionally keeps fresh runtime identity.
- [x] **19. Make simplex vertex-existence checks debug-only where proven upstream.**
  `Tds::insert_simplex_with_mapping` remains fully checked and returns typed
  errors. Audited hot paths now call
  `insert_simplex_with_mapping_trusted_vertices`, which checks vertex provenance
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

- [x] **22. Reconcile topology guarantee and validation policy combinations.**
  Incoherent runtime combinations are rejected through typed `try_set_*` policy
  setters, compatibility setters no longer commit rejected state, and builder
  construction continues to derive the initial validation policy from the
  selected topology guarantee. Closed for v0.7.8 in #385.
- [x] **23. Reconsider skipped insertions as success outcomes.**
  Make skipped duplicate and degeneracy outcomes harder for callers to ignore.
  Closed for v0.7.8 in #386.
- [x] **24. Make `Simplex` encapsulation consistent.**
  Neighbor storage is private and mutation is routed through the validated TDS
  surface. Closed in #387.
- [x] **25. Protect `Vertex::incident_simplex` mutation.**
  The field and setter are crate-private, with public read access and TDS-owned
  repair/assignment paths. Closed in #387.
- [x] **26. Revisit public `core` module naming.**
  Keep `crate::core` as the internal implementation namespace, and expose the
  public low-level surface through curated modules and focused preludes such as
  `tds`, `collections`, `algorithms`, and `query`. Closed for v0.7.8 in #388.
- [x] **27. Normalize boxing policy in Delaunay repair error variants.**
  Large recursive/diagnostic payloads follow the audited boxing policy while
  lightweight classification remains inline. Closed in #384.

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
- [x] **31. Add allocation-bounded hot-path benchmarks.**
  `benches/allocation_hot_paths.rs` now asserts zero allocations over calibrated 2D-5D triangulations for TDS/public
  `simplices()`/`vertices()` iterator paths, `Tds::simplex_vertices`, and
  `facet_key_from_vertices`, plus an explicit allocation budget for the hinted
  locate fast path under `--features count-allocations`. `tests/allocation_api.rs`
  remains a narrow wiring smoke test for the allocation measurement API.
- [x] **32. Benchmark `Tds::clone` cost versus triangulation size.**
  `benches/tds_clone.rs` now measures full `Tds::clone()` snapshot cost across
  deterministic 2D-5D triangulations before any rollback redesign.
- [x] **33. Regression-test robust insphere overflow boundaries.**
  Include inputs near `||x|| ~= 1e154`.
- [x] **34. Harden doctests that unwrap degenerate construction.**
  Public doctests now use typed `Result` wrappers or non-degenerate examples,
  and `just verify-expect-counts` tracks a zero doc-comment expect-call baseline.

## Optional And Nitpicks

- [x] **A. Clarify cfg-only feature flags in `Cargo.toml`.**
  Dependency-backed and cfg-only feature gates are now separated in the manifest
  so empty features are intentionally documented.
- [x] **B. Remove stale deprecated-warning comment in `lib.rs`.**
  The crate-level migration comment was removed along with the deprecated
  mutable triangulation escape hatch.
- [x] **C. Relabel safe-code `SAFETY` comments in `tds/storage.rs`.**
  UUID index-map consistency comments now use `INVARIANT` where no unsafe
  reasoning is involved.
- [x] **D. Remove allocation from `geometry/sos.rs` predicate helper.**
  Orientation and insphere SoS helpers now use stack-backed small buffers for
  coordinate and lifted-column scratch storage in supported dimensions.
- [x] **E. Verify `core/util/uuid.rs` panic helpers are test-only.**
  The remaining UUID validation panic paths are confined to the `#[cfg(test)]`
  module; production UUID validation returns typed `UuidValidationError` values.
- [x] **F. Remove scale-unit tolerance from the `delete_vertex` doctest.**
  The doctest now finds the vertex by UUID instead of coordinate epsilon.

## Tooling Follow-Up From This Patch

- [x] **G. Fix changelog utilities instead of hand-editing generated Markdown.**
  `postprocess-changelog` is reusable as a text transform, archive generation
  reuses it, and `just changelog` output now passes `just markdown-check`.
