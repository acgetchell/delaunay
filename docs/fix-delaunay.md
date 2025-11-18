# Fixing Delaunay Insertion Pipeline

Working notes and checklist for redesigning the vertex insertion pipeline so that:

- We have a **unified two-stage algorithm** (fast + robust) for triangulation.
- The **Delaunay property is maintained by construction**, with configurable global checks.
- We can **skip unsalvageable vertices** safely while exposing rich diagnostics.
- The system remains **transactional** and **backwards compatible**.

This is an internal design/implementation document; it is not part of the public API.

---

## Stage 1: Robust Initial Simplex

Goal: build an initial D-simplex from the input vertex set, handling duplicates and degeneracies.

- [x] Implement a search for **D+1 affinely independent vertices**:
  - [x] Skip exact duplicates.
  - [x] Skip near-duplicates within a configurable tolerance.
  - [x] Use robust orientation predicates to reject degenerate configurations (collinear in 2D, coplanar in 3D, etc.).
- [x] If D+1 independent vertices are found:
  - [x] Call the trait helper `create_initial_simplex` with that subset.
- [x] If no valid simplex can be found:
  - [x] Insert all unique vertices into the TDS **without creating any cells** (zero-cell triangulation state).
  - [x] Return a detailed construction error summarizing:
    - [x] Count of unique vertices inserted.
    - [x] Count of duplicates skipped.
    - [x] Count of degenerate candidates.
  - [x] Ensure TDS validity with zero cells so users can later add vertices and recover.
- [x] Wire this behavior into:
  - [x] The default `InsertionAlgorithm::triangulate` implementation.
  - [x] `RobustBowyerWatson::triangulate`.

---

## Stage 2: Unified Per-Vertex Insertion (Fast → Robust → Skip)

Goal: for each vertex, attempt fast insertion, then robust insertion, and finally skip with diagnostics if both fail.

### Vertex classification

- [x] Introduce an internal classifier for incoming vertices:
  - [x] `Unique`.
  - [x] `DuplicateExact`.
  - [x] `DuplicateWithinTolerance(eps)`.
  - [x] `Degenerate*` variants (collinear/coplanar/other) based on robust orientation tests.

    The current implementation uses `robust_orientation` on small sampled subsets
    of existing vertices to conservatively tag new vertices as collinear/coplanar
    degeneracies in 2D/3D. Degenerate classifications flow through the same
    fast → robust insertion pipeline as `Unique` vertices, but the classification
    is preserved end-to-end in `UnifiedPerVertexInsertionOutcome` and
    `UnsalvageableVertexReport` for diagnostics.
- [x] Use a configurable tolerance for near-duplicates (per-coordinate or scalar) via the internal
  `NearDuplicateTolerance` configuration used by `classify_vertex_against_tds`.
- [x] Integrate Jaccard-based similarity heuristics for diagnostics (backed by `core::util::{jaccard_index, extract_*_set}`),
  while always falling back to tolerance-based detection for actual classification.

### Fast-path insertion

- [x] Implement the **fast path** using the existing incremental Bowyer–Watson machinery:
  - [x] Choose strategy per vertex: cavity-based (interior) vs hull extension (exterior).
  - [x] Use the current transactional cavity-based insertion with local iterative Delaunay refinement.
- [x] Ensure fast-path attempts are **transactional**:
  - [x] On success: commit all changes.
  - [x] On error: roll back entirely, leaving the TDS unchanged.

### Robust-path insertion

- [x] On fast-path failure (for recoverable geometric/precision issues):
  - [x] Attempt a **robust insertion** using `RobustBowyerWatson`:
    - [x] Robust cavity-based insertion.
    - [x] Robust hull extension, with strategy order informed by vertex classification when helpful.
  - [x] Maintain the same transactional guarantees (full commit or full rollback).

### Skip semantics and diagnostics

- [x] If both fast and robust attempts fail for a vertex:
  - [x] Do not modify the TDS at all for that vertex.
  - [x] Record the vertex as **unsalvageable**, including:
    - [x] Coordinates (and optionally the full `Vertex<T, U, D>`).
    - [x] The list of strategies attempted (e.g., fast cavity, fast hull, robust cavity, robust hull, global repair).
    - [x] The error chain for each attempt.
  - [x] Continue processing remaining vertices in `triangulate`.
- [x] Expose this information via a **public API** (e.g., on `RobustBowyerWatson` and/or a shared orchestrator):
  - [x] `unsalvageable_vertices()` / `take_unsalvageable_vertices()` returning rich reports rather than bare vertices.
- [x] Ensure invariants:
  - [x] Any unsalvageable vertex **has no footprint** in the final TDS.

---

## Unified Pipeline Orchestration

Goal: share a unified control flow between `IncrementalBowyerWatson` and `RobustBowyerWatson` without breaking the public API.

- [x] Introduce an internal orchestrator (or equivalent) that:
  - [x] Implements Stage 1 (initial simplex) + Stage 2 (per-vertex pipeline).
  - [x] Calls the fast algorithm first (Incremental Bowyer–Watson / trait defaults), then robust, then skip.
- [x] Hook `InsertionAlgorithm::triangulate` and `InsertionAlgorithm::insert_vertex` defaults into this orchestrator.
- [x] Ensure `IncrementalBowyerWatson` and `RobustBowyerWatson` both:
  - [x] Participate in the unified pipeline.
  - [x] Preserve their existing public methods and types.

---

## Configurable Delaunay Validation Cadence

Goal: avoid expensive global Delaunay checks on every insertion, but allow configurable validation schedules.

- [x] Add a configuration type (e.g., `DelaunayCheckPolicy`):
  - [x] `EndOnly` variant (current default; behavior matches the existing single final global validation).
  - [x] `EveryN(NonZeroUsize)` variant used by insertion pipelines to schedule periodic validation.
- [x] Integrate policy into the insertion pipeline:
  - [x] Maintain a count of successful insertions (tracked per `UnifiedInsertionPipeline`).
  - [x] For `EveryN(k)`, invoke global validation/repair every k insertions (wired via `run_global_delaunay_validation_with_policy` in the unified Stage 2 loop).
  - [x] Always run at least one final validation/repair at the end.
  - [x] No-op if there are no cells (zero-cell TDS).
- [ ] For robust configurations, global repair/validation should reuse `repair_global_delaunay_violations` and/or `validate_no_delaunay_violations`.

---

## Statistics and Error Semantics

Goal: keep strong observability and clear behavior on failure.

- [ ] Extend `InsertionStatistics` and related counters as needed to track:
  - [x] Number of vertices processed.
  - [x] Cells created/removed.
  - [ ] Fast vs robust attempts and successes.
  - [x] Number of skipped unsalvageable vertices (tracked via `InsertionStatistics::skipped_vertices` and `UnsalvageableVertexReport`).
  - [ ] Number of global validation/repair runs (currently tracked via test-only counters such as `GLOBAL_DELAUNAY_VALIDATION_CALLS`).
- [ ] Ensure `triangulate` behavior:
  - [ ] Continues on per-vertex geometric/precision failures by marking vertices unsalvageable.
  - [ ] Aborts only on structural TDS errors or unrecoverable invariant violations.
- [ ] Ensure `insert_vertex` behavior:
  - [ ] Remains transactional.
  - [ ] Returns a useful `InsertionError` when a single vertex is unsalvageable, with no TDS changes.
- [ ] Ensure any new public functions in this pipeline return `Result<…>` with an
  appropriate error variant (rather than panicking), reusing existing error
  enums where possible.

---

## Tests and Documentation

Goal: validate correctness and expose the new behavior clearly.

- [ ] Unit tests:
  - [x] Initial simplex selection, including duplicate-only and degenerate-only inputs.
  - [x] Vertex classification (Unique / DuplicateExact / DuplicateWithinTolerance(eps) / Degenerate*).
  - [x] Zero-cell TDS behavior and later recovery via incremental insertion.
    Current behavior: zero-cell fallback retains vertices; incremental `add`
    preserves validity and vertex count without forcing an automatic rebuild.
- [ ] Integration tests:
  - [x] Fast -> robust -> skip pipeline on carefully chosen point sets
    (including existing regression cases).
- [ ] Unsalvageable vertex tracking: verify all unsalvageable vertices come from the input set
  and that their union with kept vertices covers the input, up to duplicates.
- [ ] Validation cadence tests:
  - [ ] `EndOnly` vs `EveryN(k)` behavior.
- [ ] Property-based tests:
  - [ ] Random clouds with duplicates and near-duplicates, various dimensions and coordinate types.
  - [ ] Ensure final triangulations are globally Delaunay for the kept subset.
- [ ] Documentation updates:
  - [ ] Describe the two-stage pipeline in crate-level docs.
  - [ ] Document zero-cell triangulation state and how to recover.
  - [ ] Document Delaunay validation cadence configuration and defaults.
  - [ ] Document unsalvageable vertex reporting for debugging and testing.

---

## Non-Goals / Constraints

- Do **not** break public API types (`IncrementalBowyerWatson`, `RobustBowyerWatson`, `InsertionAlgorithm`, error enums), but extensions are allowed.
- Maintain or improve robustness; do not weaken Delaunay guarantees in robust modes.
- Preserve transactional semantics: no partially applied insertions.

---

## Current Status

- Stage 1 ("Robust Initial Simplex") and Stage 2 ("Unified Per-Vertex Insertion")
  are implemented and wired through `InsertionAlgorithm::triangulate` and the
  internal `UnifiedInsertionPipeline`.
- `RobustBowyerWatson` and `IncrementalBowyerWatson` both participate in the
  unified per-vertex pipeline, and unsalvageable vertices are tracked via
  `UnsalvageableVertexReport`.
- `DelaunayCheckPolicy` exists and is threaded into the unified pipeline; `EndOnly`
  provides a single final global validation and `EveryN(k)` additionally triggers
  periodic validation during Stage 2. Global repair hooks (e.g., reusing
  `repair_global_delaunay_violations`) remain future work.
- Global Delaunay validation is available via `core::util::{is_delaunay, find_delaunay_violations}` and `Tds::validate_delaunay`, and is used at the end of triangulation.
- Structural invariant validation (`Tds::is_valid` / `Tds::validation_report`) and
  property-test suites are in place (`tests/proptest_triangulation.rs`,
  `tests/proptest_delaunay_condition.rs`, `tests/proptest_robust_bowyer_watson.rs`,
  `tests/integration_robust_bowyer_watson.rs`), but some high-dimensional Delaunay
  properties (especially 4D–5D) still exhibit violations or test drift.

## Focused Plan to Debug Remaining Delaunay Violations (2D–5D)

### Phase 1 – Establish a small, reproducible failure set

- [x] Run `just test` once to confirm the current global state.
- [x] Run the Delaunay-heavy suites with verbose output:
  - `cargo test --test proptest_delaunay_condition -- --nocapture`
  - `cargo test --test proptest_robust_bowyer_watson -- --nocapture`
  - `cargo test --test integration_robust_bowyer_watson -- --nocapture`
- [ ] For each failing proptest, capture:
  - test binary + test name,
  - dimension (2D–5D),
  - exact reproduction hints (`PROPTEST_SEED` / `--exact-seed`).
- [ ] Re-run those failures with `PROPTEST_CASES=1` and exact seeds to obtain a small set of canonical failing configurations per dimension.

#### Current Phase 1 status

- `proptest_delaunay_condition` still exhibits intermittent failures in 5D, with
  `validate_no_delaunay_violations` reporting multiple Delaunay violations for
  some randomly generated 5D point sets.
- `proptest_robust_bowyer_watson` is currently passing with bounded
  `PROPTEST_CASES` (e.g., 64), indicating the robust per-vertex insertion logic
  is structurally sound on the exercised inputs.
- `integration_robust_bowyer_watson` passes for 2D–4D clustered/scattered and
  grid-style test cases; failures are presently localized to the global
  Delaunay condition checks exercised in `proptest_delaunay_condition`.

#### Captured seeds (canonical)

- 5D empty circumsphere property:
  - Test: `proptest_delaunay_condition::prop_empty_circumsphere_5d` (D = 5).
  - Canonical configuration: a 7-point 5D vertex set reconstructed and
    hard-coded in `tests/debug_delaunay_violation_5d.rs` and in the
    stepwise unified insertion test in
    `src/core/algorithms/unified_insertion_pipeline.rs`.
  - Replay: use the deterministic debug test harness instead of a raw
    `PROPTEST_SEED`:

    ```text
    cargo test --test debug_delaunay_violation_5d -- --ignored --nocapture
    cargo test -p delaunay src::core::algorithms::unified_insertion_pipeline::tests::debug_5d_stepwise_insertion_of_seventh_vertex -- --ignored --nocapture
    ```

  - Status: both harnesses currently reproduce a Delaunay validation failure for
    this configuration and log detailed diagnostics via
    `debug_print_first_delaunay_violation`.

### Phase 2 – Strengthen validators and add diagnostics

- [x] Review and, if necessary, tighten `core::util::is_delaunay` and `find_delaunay_violations` so that:
  - Delaunay violations are defined as **strictly inside** the circumsphere (on-sphere is allowed).
  - Robust predicates (`robust_insphere`) are used consistently for classification.
  - Structural problems (missing vertices, invalid cells) are reported via `TriangulationValidationError`, not conflated with Delaunay violations.
- [x] Add a debug-only helper in `core::util` (behind `cfg(any(test, debug_assertions))`) that:
  - Calls `find_delaunay_violations` (optionally on a subset of cells),
  - Dumps violating cells, their vertices, and offending external vertices via `eprintln!`,
  - Prints relevant neighbor information to help localize problems.

### Phase 3 – Debug `Tds::new` and incremental insertion paths

- [x] For at least one canonical failing 5D configuration, build the vertex set
  and call `Tds::<f64, Option<()>, Option<()>, D>::new(&vertices)`, capturing the
  resulting `TriangulationConstructionError::ValidationError` and using the debug
  helper to inspect violating cells and points (see `tests/debug_delaunay_violation_5d.rs`).
  - Outcome (5D): `Tds::new` fails with a Delaunay-related validation error for
    the canonical 7-point 5D configuration.
- [ ] Generalize this to additional canonical failing seeds in 2D–4D.
- [x] For incremental failures (5D canonical configuration):
  - Start from a small base triangulation (first 6 vertices of the canonical set).
  - Insert the 7th point via the unified insertion pipeline using
    `UnifiedInsertionPipeline::with_policy(DelaunayCheckPolicy::EndOnly)` and the
    internal stepwise helper in `unified_insertion_pipeline.rs`.
  - After insertion, run `validate_delaunay` before finalization and again after
    `finalize_triangulation`.
  - Outcome (5D): the base 6-vertex triangulation is globally Delaunay, but the
    7th insertion already produces Delaunay violations *before* finalization,
    indicating that the per-vertex unified insertion pipeline itself can leave
    non-Delaunay cells for this configuration.

### Phase 4 – Integrate and test `DelaunayCheckPolicy` semantics

- [x] Ensure `DelaunayCheckPolicy` is fully wired through the unified insertion pipeline:
  - `EndOnly`: run global Delaunay validation once at the end (current default).
  - `EveryN(k)`: run `validate_no_delaunay_violations` every `k` successful insertions **and** once at the end.
  - Skip validation when there are no cells (zero-cell triangulations).
- [x] Add targeted tests that:
  - Use small 3D inputs to verify `EveryN(1)`/`EveryN(k)` exercise the periodic validation path (see `test_bowyer_watson_with_diagnostics_every_n_policy_triggers_validation`).
  - Confirm `EndOnly` still triggers exactly one final validation (see `test_bowyer_watson_with_diagnostics_end_only_policy_single_validation`).
  - Verify zero-cell triangulations are a validation no-op (see `test_run_global_delaunay_validation_with_policy_zero_cells_noop`).
- [ ] For any future repair mechanism (e.g., a `repair_global_delaunay_violations`
  helper), guard it behind a feature flag and document expected behavior without
  enabling it by default.

### Phase 5 – Regression tests, test adjustments, and docs

- [ ] Promote important failing seeds to deterministic regression tests (no proptest) in `tests/regression_delaunay_*.rs`:
  - Hard-code small 2D–5D point sets that previously violated the Delaunay property,
  - Assert both `tds.is_valid()` and `tds.validate_delaunay()` succeed.
- [ ] For tests whose expectations have drifted relative to the current algorithm:
  - If `validate_delaunay()` passes, but the test expects an exact triangulation
    shape, relax expectations to:
    - structural invariants and global Delaunay property, plus
    - coarse metrics (e.g., Jaccard similarity on edge sets) when needed.
  - If `validate_delaunay()` fails, treat that as a genuine algorithm bug and keep
    the test strict until fixed.
- [ ] Update this document and `tests/README.md` with:
  - Instructions for running small seeded Delaunay tests,
  - Guidance on using `just dev` and Delaunay-specific tests during development.

---

### Execution Plan Snapshot (current multi-phase TODO)

This section mirrors the current working plan so it can be recovered even if external TODO state is lost.

- [ ] **Phase 1 – Baseline and failing-seed capture**
  - [x] Run `just test` once to reconfirm the global failure state.
  - [x] Run Delaunay-heavy suites with verbose output:
    - `cargo test --test proptest_delaunay_condition -- --nocapture`
    - `cargo test --test proptest_robust_bowyer_watson -- --nocapture`
    - `cargo test --test integration_robust_bowyer_watson -- --nocapture`
  - [ ] For each failing proptest, record binary, test name, dimension (2D–5D), and exact seed (`PROPTEST_SEED` / `--exact-seed`).
  - [ ] Re-run each failing case with `PROPTEST_CASES=1` and the captured seed to obtain a small canonical failing configuration per dimension.

  #### Current Phase 1 status

  - `proptest_delaunay_condition` still exhibits intermittent 5D failures where
    the empty circumsphere property is violated for some cells after
    triangulation.
  - `proptest_robust_bowyer_watson` is currently passing with bounded
    `PROPTEST_CASES` (e.g., 64), suggesting robust per-vertex insertion is
    behaving correctly on tested inputs.
  - `integration_robust_bowyer_watson` passes in 2D–4D for clustered/scattered
    and grid-based datasets; remaining issues are tied to the global Delaunay
    condition rather than basic structural invariants.

- [ ] **Phase 2 – Validator semantics and diagnostics**
  - [x] Tighten `core::util::{is_delaunay, find_delaunay_violations}` so that:
    - Only `INSIDE` (strictly inside circumsphere) is treated as a Delaunay violation.
    - `BOUNDARY` is allowed.
    - Structural issues are reported as `TriangulationValidationError` variants, not `DelaunayViolation`.
  - [x] Add a debug-only helper (behind `cfg(any(test, debug_assertions))`) to print the first Delaunay violation, including:
    - Violating cell key, its vertices, and coordinates.
    - Offending external vertex key and coordinates.
    - Neighbor information for each facet.
  - [x] Add a basic unit test that exercises the validator and helper on a simple 3D tetrahedron.

- [ ] **Phase 3 – Localize violations in `Tds::new` and per-vertex insertion**
  - [x] For at least one canonical failing 5D configuration, construct vertices and call
    `Tds::<f64, Option<()>, Option<()>, D>::new(&vertices)`, asserting that construction
    currently fails with a Delaunay-related validation error and using the debug helper
    to inspect violating cells and points (see `tests/debug_delaunay_violation_5d.rs`).
  - [ ] Generalize this to additional canonical failing configurations and dimensions.
  - [x] For incremental cases (5D canonical configuration), start from a small valid base
    triangulation and insert remaining vertices one-by-one using the unified pipeline.
    - [x] After inserting the final vertex (7th), assert that Delaunay is already violated
      before `finalize_triangulation` and remains violated afterwards, using the
      stepwise helper in `unified_insertion_pipeline.rs`.
    - [ ] When additional canonical configurations are available (2D–4D), repeat this
      process and record, for each, whether the first violation appears during
      per-vertex insertion or only during the finalization step.
  - [ ] Fix issues in insertion and finalization paths
        (`find_bad_cells`, cavity boundary construction/deduplication,
        `filter_boundary_facets_by_valid_facet_sharing`, `finalize_after_insertion`,
        hull extension, and fallback paths) so that per-vertex insertion
        preserves Delaunay for valid inputs.

- [ ] **Phase 4 – Align tests with the unified Delaunay pipeline**
  - [ ] Update robust Bowyer–Watson tests
        (`src/core/algorithms/robust_bowyer_watson.rs`) so they assert structural
        validity + Delaunay without over-constraining triangulation shape unless
        necessary.
  - [ ] Ensure structural tests in `src/core/{cell,facet,boundary}.rs` still pass
        given any changes to neighbors, incident cells, and boundary extraction.
  - [ ] Reconcile `prop_filter_never_increases_count` in
        `src/core/traits/insertion_algorithm.rs` with
        `filter_boundary_facets_by_valid_facet_sharing`, ensuring the filter never
        increases facet count.
  - [ ] Adjust `src/geometry/util.rs` random triangulation tests so `generate_random_triangulation` returns:
    - Valid, globally Delaunay triangulations when a simplex exists.
    - Clear `GeometricDegeneracy` (or equivalent) for truly unsalvageable inputs.

- [ ] **Phase 5 – Statistics and error semantics**
  - [ ] Extend `InsertionStatistics` to track fast/robust attempts and successes, skipped vertices, and non-test-only global validation runs.
  - [ ] Wire these counters through the unified pipeline and `run_global_delaunay_validation_with_policy`.
  - [ ] Ensure `triangulate` continues past per-vertex geometric/precision failures by marking vertices unsalvageable, aborting only on structural errors.
  - [ ] Ensure `insert_vertex` remains transactional (no TDS changes on error) and returns informative `InsertionError` variants.
  - [ ] Add tests asserting statistics monotonicity and transactional behavior.

- [ ] **Phase 6 – Regression tests, docs, quality tools, and acceptance criteria**
  - [ ] Promote important failing seeds to deterministic regression tests (`tests/regression_delaunay_*.rs`) that assert both structural and Delaunay validity.
  - [ ] Update crate-level docs, this document, and `tests/README.md` to describe:
    - The two-stage (fast + robust) insertion pipeline.
    - Zero-cell triangulation states and recovery.
    - Delaunay validation cadence and `DelaunayCheckPolicy` usage.
    - Unsalvageable vertex reporting.
  - [ ] Run quality and configuration checks (`just fmt`, `just clippy`, `just markdown-lint`, `just spell-check`, `just validate-json`, `just validate-toml`).
  - [ ] Re-run the main test suites (`just test`, `just test-release`,
        Delaunay-heavy tests, examples, and allocation/benchmark sanity checks) and
        treat any remaining `DelaunayViolation` as a bug to fix before completion.
