# Fixing Delaunay Insertion Pipeline

## Status: COMPLETED AND ARCHIVED (November 2025)

This document contains working notes and checklists for the completed redesign of the vertex insertion pipeline.

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
- [x] For robust configurations, global repair/validation reuses
  `repair_global_delaunay_violations` (Stage 4 of the unified pipeline) followed
  by `validate_no_delaunay_violations` via `run_global_delaunay_validation_with_policy`.

---

## Statistics and Error Semantics

Goal: keep strong observability and clear behavior on failure.

- [x] Extend `InsertionStatistics` and related counters as needed to track:
  - [x] Number of vertices processed.
  - [x] Cells created/removed.
  - [x] Fast vs robust attempts and successes.
- [x] Number of skipped unsalvageable vertices (tracked via
  `InsertionStatistics::skipped_vertices` and `UnsalvageableVertexReport`).
- [x] Number of global validation/repair runs (exposed via
  `TriangulationStatistics::global_delaunay_validation_runs` and reinforced by
  `GLOBAL_DELAUNAY_VALIDATION_CALLS` in tests).
- [x] Ensure `triangulate` behavior:
  - [x] Continues on per-vertex geometric/precision failures by marking vertices unsalvageable.
  - [x] Aborts only on structural TDS errors or unrecoverable invariant violations.
- [x] Ensure `insert_vertex` behavior:
  - [x] Remains transactional.
  - [x] Returns a useful `InsertionError` when a single vertex is unsalvageable, with no TDS changes.
- [x] Ensure any new public functions in this pipeline return `Result<…>` with an
  appropriate error variant (rather than panicking), reusing existing error
  enums where possible.

---

## Tests and Documentation

Goal: validate correctness and expose the new behavior clearly.

- [x] Unit tests:
  - [x] Initial simplex selection, including duplicate-only and degenerate-only inputs.
  - [x] Vertex classification (Unique / DuplicateExact / DuplicateWithinTolerance(eps) / Degenerate*).
  - [x] Zero-cell TDS behavior and later recovery via incremental insertion.
    Current behavior: zero-cell fallback retains vertices; incremental `add`
    preserves validity and vertex count without forcing an automatic rebuild.
  - [x] Unified recoverability semantics for Stage 2 (see
    `test_insertion_error_is_recoverable_in_unified_pipeline` and
    `test_unified_pipeline_hard_geometric_failure_skips_vertex` in
    `src/core/traits/insertion_algorithm.rs`).
- [x] Integration tests:
  - [x] Fast -> robust -> skip pipeline on carefully chosen point sets
    (including existing regression cases).
- [x] Unsalvageable vertex tracking: verify all unsalvageable vertices come from the input set
  and that their union with kept vertices covers the input, up to duplicates.
- [x] Validation cadence tests:
  - [x] `EndOnly` vs `EveryN(k)` behavior.
- [x] Property-based tests:
  - [x] Random clouds with duplicates and near-duplicates in 2D and 3D, using f64 coordinates.
  - [x] Ensure final triangulations are globally Delaunay for the kept subset.
- [x] Documentation updates:
- [x] Describe the two-stage pipeline in crate-level docs.
- [x] Document zero-cell triangulation state and how to recover.
- [x] Document Delaunay validation cadence configuration and defaults.
- [x] Document unsalvageable vertex reporting for debugging and testing.

---

## Non-Goals / Constraints

- Do **not** break public API types (`IncrementalBowyerWatson`, `RobustBowyerWatson`, `InsertionAlgorithm`, error enums), but extensions are allowed.
- Maintain or improve robustness; do not weaken Delaunay guarantees in robust modes.
- Preserve transactional semantics: no partially applied insertions.

---

## Current Status

**COMPLETED**: All phases of the Delaunay insertion pipeline redesign are complete.

- Stage 1 ("Robust Initial Simplex") and Stage 2 ("Unified Per-Vertex Insertion")
  are implemented and wired through `InsertionAlgorithm::triangulate` and the
  internal `UnifiedInsertionPipeline`.
- `RobustBowyerWatson` and `IncrementalBowyerWatson` both participate in the
  unified per-vertex pipeline, and unsalvageable vertices are tracked via
  `UnsalvageableVertexReport`.
- `DelaunayCheckPolicy` exists and is threaded into the unified pipeline; `EndOnly`
  provides a single final global validation and `EveryN(k)` additionally triggers
  periodic validation during Stage 2. Robust configurations run a global repair
  pass via `repair_global_delaunay_violations` before the final
  `validate_no_delaunay_violations` check.
- Global Delaunay validation is available via `core::util::{is_delaunay, find_delaunay_violations}` and `Tds::validate_delaunay`, and is used at the end of triangulation.
- Structural invariant validation (`Tds::is_valid` / `Tds::validation_report`) and
  property-test suites are in place (`tests/proptest_triangulation.rs`,
  `tests/proptest_delaunay_condition.rs`, `tests/proptest_robust_bowyer_watson.rs`,
  `tests/integration_robust_bowyer_watson.rs`).
- All Delaunay-heavy test suites pass (2D–5D) with canonical regression tests
  for known configurations in `tests/regression_delaunay_known_configs.rs`.
- Quality checks (fmt, clippy, markdown-lint, spell-check, validate-json, validate-toml)
  and comprehensive test suites (test, test-release, examples) all pass.

## Focused Plan to Debug Remaining Delaunay Violations (2D–5D)

### Phase 1 – Establish a small, reproducible failure set

- [x] Run `just test` once to confirm the current global state.
- [x] Run the Delaunay-heavy suites with verbose output:
  - `cargo test --test proptest_delaunay_condition -- --nocapture`
  - `cargo test --test proptest_robust_bowyer_watson -- --nocapture`
  - `cargo test --test integration_robust_bowyer_watson -- --nocapture`
When new Delaunay-focused proptest failures appear (2D–5D), use the
following playbook instead of treating this as an open TODO:
  - Capture the failing test binary and test name.
  - Record the dimension (2D–5D) and exact reproduction hints
    (`PROPTEST_SEED` / `--exact-seed`).
  - Re-run each failing case with `PROPTEST_CASES=1` and the captured
    seed to obtain a small canonical failing configuration per dimension.
  - Promote each stable configuration to a deterministic regression in
    `tests/regression_delaunay_known_configs.rs` (see that file for the
    multi-dimension regression macro).

#### Current Phase 1 status

- `proptest_delaunay_condition` is currently passing for the configured
  `PROPTEST_CASES`. The previously failing 5D configuration has been promoted
  to a deterministic regression in
  `tests/regression_delaunay_known_configs.rs`.
- `proptest_robust_bowyer_watson` is currently passing with bounded
  `PROPTEST_CASES` (e.g., 64), indicating the robust per-vertex insertion logic
  is structurally sound on the exercised inputs.
- `integration_robust_bowyer_watson` passes for 2D–4D clustered/scattered and
  grid-style test cases; any future failures are expected to surface as
  explicit `DelaunayViolation` errors in regression/property tests rather than
  silent topology corruption.

#### Captured seeds (canonical)

- 5D empty circumsphere property:
  - Test: `proptest_delaunay_condition::prop_empty_circumsphere_5d` (D = 5).
  - Canonical configuration: a 7-point 5D vertex set reconstructed from earlier
    proptest/debug output and now hard-coded in:
    - `tests/regression_delaunay_known_configs.rs` (canonical regression test),
    - the stepwise unified insertion test in
      `src/core/algorithms/unified_insertion_pipeline.rs`.
  - Replay: use the deterministic regression and stepwise debug harnesses instead
    of a raw `PROPTEST_SEED`:

    ```text
    cargo test --test regression_delaunay_known_configs -- --nocapture
    cargo test -p delaunay src::core::algorithms::unified_insertion_pipeline::tests::debug_5d_stepwise_insertion_of_seventh_vertex -- --ignored --nocapture
    ```

  - Status: the regression test now asserts both structural validity and the
    global Delaunay property for this configuration, while the stepwise unified
    insertion debug test (still `#[ignore]`) continues to reproduce the local
    per-vertex Delaunay violation and logs detailed diagnostics via
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
- and call `Tds::<f64, Option<()>, Option<()>, D>::new(&vertices)`, capturing the
- resulting behavior and using the debug helper to inspect violating cells and
- points when present (see `tests/regression_delaunay_known_configs.rs` and the
- stepwise 5D test in `src/core/algorithms/unified_insertion_pipeline.rs`).
- Outcome (5D): `Tds::new` for the canonical 7-point 5D configuration now
- constructs successfully and passes global Delaunay validation; this case is
- kept as a regression test for future changes.
Currently only the canonical 7-point 5D configuration is encoded as a
regression and stepwise debug case. If additional canonical failing
seeds are discovered in 2D–4D, generalize this workflow by adding
similarly small deterministic configurations and re-running the same
`Tds::new` + stepwise unified-insertion analysis for each dimension.
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
- [x] Integrate the global repair helper
  `RobustBowyerWatson::repair_global_delaunay_violations` into the unified
  pipeline and run it by default before `validate_no_delaunay_violations`,
  preferring always-on Delaunay guarantees over an optional faster but weaker
  configuration.

### Phase 5 – Regression tests, test adjustments, and docs

- [x] Promote important failing seeds to deterministic regression tests (no proptest) in `tests/regression_delaunay_*.rs`:
  - Hard-code small 2D–5D point sets that previously violated the Delaunay property,
  - Assert both `tds.is_valid()` and `tds.validate_delaunay()` succeed.
- [x] For tests whose expectations have drifted relative to the current algorithm:
  - If `validate_delaunay()` passes, but the test expects an exact triangulation
    shape, relax expectations to:
    - structural invariants and global Delaunay property, plus
    - coarse metrics (e.g., Jaccard similarity on edge sets) when needed.
  - If `validate_delaunay()` fails, treat that as a genuine algorithm bug and keep
    the test strict until fixed.
- [x] Update this document and `tests/README.md` with:
  - Instructions for running small-seeded Delaunay tests (see the
    "Captured seeds (canonical)" section here and the
    `regression_delaunay_known_configs.rs` entry in `tests/README.md`),
  - Guidance on using `just dev` and Delaunay-specific tests during development.

---

### Execution Plan Snapshot (current multi-phase TODO)

This section mirrors the current working plan so it can be recovered even if external TODO state is lost.

- **Phase 1 – Baseline and failing-seed capture (conditional playbook)**
  - [x] Run `just test` once to reconfirm the global failure state.
  - [x] Run Delaunay-heavy suites with verbose output:
    - `cargo test --test proptest_delaunay_condition -- --nocapture`
    - `cargo test --test proptest_robust_bowyer_watson -- --nocapture`
    - `cargo test --test integration_robust_bowyer_watson -- --nocapture`
  - When new Delaunay-focused proptest failures appear (2D–5D), follow
    this seed-capture procedure instead of treating it as an open TODO:
    - Record the failing binary, test name, dimension (2D–5D), and
      exact seed (`PROPTEST_SEED` / `--exact-seed`).
    - Re-run each failing case with `PROPTEST_CASES=1` and the captured
      seed to obtain a small canonical failing configuration per
      dimension.
    - Promote each stable configuration to a deterministic regression in
      `tests/regression_delaunay_known_configs.rs` using the
      multi-dimension regression macro.

#### Current Phase 1 status

- `proptest_delaunay_condition` is currently passing for the configured
    `PROPTEST_CASES`. The previously failing 5D configuration has been promoted
    to a deterministic regression in
    `tests/regression_delaunay_known_configs.rs`.
- `proptest_robust_bowyer_watson` is currently passing with bounded
    `PROPTEST_CASES` (e.g., 64), suggesting robust per-vertex insertion is
    behaving correctly on tested inputs.
- `integration_robust_bowyer_watson` passes in 2D–4D for clustered/scattered
    and grid-based datasets; any future failures are expected to surface as
    explicit `DelaunayViolation` errors in regression/property tests rather than
    silent topology corruption.

- [x] **Phase 2 – Validator semantics and diagnostics**
  - [x] Tighten `core::util::{is_delaunay, find_delaunay_violations}` so that:
    - Only `INSIDE` (strictly inside circumsphere) is treated as a Delaunay violation.
    - `BOUNDARY` is allowed.
    - Structural issues are reported as `TriangulationValidationError` variants, not `DelaunayViolation`.
  - [x] Add a debug-only helper (behind `cfg(any(test, debug_assertions))`) to print the first Delaunay violation, including:
    - Violating cell key, its vertices, and coordinates.
    - Offending external vertex key and coordinates.
    - Neighbor information for each facet.
  - [x] Add a basic unit test that exercises the validator and helper on a simple 3D tetrahedron.

- **Phase 3 – Localize violations in `Tds::new` and per-vertex insertion (conditional playbook)**
- [x] For the canonical 7-point 5D configuration, construct vertices and call
- `Tds::<f64, Option<()>, Option<()>, D>::new(&vertices)`, asserting that
- construction succeeds and `tds.validate_delaunay()` passes (see
- `tests/regression_delaunay_known_configs.rs`).
- Currently this 5D configuration is the only known canonical
  Delaunay-violation case; it is encoded both as a deterministic
  regression test and as a stepwise unified-insertion debug test.
- When additional canonical failing configurations are discovered in
  2D–4D, generalize this workflow by:
  - Adding each configuration as a deterministic regression in
    `tests/regression_delaunay_known_configs.rs`.
  - Repeating the `Tds::new` + stepwise unified-insertion analysis to
    determine whether violations arise during per-vertex insertion or
    only during finalization.
- If future localized violations point back to insertion/finalization
  code paths, focus debugging on:
        (`find_bad_cells`, cavity boundary construction/deduplication,
        `filter_boundary_facets_by_valid_facet_sharing`, `finalize_after_insertion`,
        hull extension, and fallback paths) to ensure per-vertex
        insertion preserves Delaunay for valid inputs.

- [x] **Phase 4 – Align tests with the unified Delaunay pipeline**
  - [x] Update robust Bowyer–Watson tests
        (`src/core/algorithms/robust_bowyer_watson.rs`) so they assert structural
        validity + Delaunay without over-constraining triangulation shape unless
        necessary.
  - [x] Ensure structural tests in `src/core/{cell,facet,boundary}.rs` still pass
        given any changes to neighbors, incident cells, and boundary extraction.
  - [x] Reconcile `prop_filter_never_increases_count` in
        `src/core/traits/insertion_algorithm.rs` with
        `filter_boundary_facets_by_valid_facet_sharing`, ensuring the filter never
        increases facet count.
  - [x] Adjust `src/geometry/util.rs` random triangulation tests so `generate_random_triangulation` returns:
    - Valid, globally Delaunay triangulations when a simplex exists.
    - Clear `GeometricDegeneracy` (or equivalent) for truly unsalvageable inputs.

- [x] **Phase 5 – Statistics and error semantics**
  - [x] Extend `InsertionStatistics` to track fast/robust attempts and successes, skipped vertices, and non-test-only global validation runs.
  - [x] Wire these counters through the unified pipeline and `run_global_delaunay_validation_with_policy`.
  - [x] Ensure `triangulate` continues past per-vertex geometric/precision failures by marking vertices unsalvageable, aborting only on structural errors.
  - [x] Ensure `insert_vertex` remains transactional (no TDS changes on error) and returns informative `InsertionError` variants.
  - [x] Add tests asserting statistics monotonicity and transactional behavior
        (beyond existing robustness tests in `tests/proptest_robust_bowyer_watson.rs`),
        for example:
    - `test_last_triangulation_statistics_records_global_validation_runs` in
      `triangulation_data_structure.rs` and
    - transactional rollback tests in `tests/test_insertion_algorithm_trait.rs`
      and `triangulation_data_structure.rs`.

- [x] **Phase 6 – Regression tests, docs, quality tools, and acceptance criteria**
  - [x] Promote important failing seeds to deterministic regression tests (`tests/regression_delaunay_*.rs`) that assert both structural and Delaunay validity.
- [x] Update crate-level docs, this document, and `tests/README.md` to describe:
- [x] The two-stage (fast + robust) insertion pipeline.
- [x] Zero-cell triangulation states and recovery.
- [x] Delaunay validation cadence and `DelaunayCheckPolicy` usage.
- [x] Unsalvageable vertex reporting.
  - [x] Run quality and configuration checks (`just fmt`, `just clippy`, `just markdown-lint`, `just spell-check`, `just validate-json`, `just validate-toml`).
  - [x] Re-run the main test suites (`just test`, `just test-release`,
        Delaunay-heavy tests, examples, and allocation/benchmark sanity checks) and
        treat any remaining `DelaunayViolation` as a bug to fix before completion.
