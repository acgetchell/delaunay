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
  - [x] `EveryN(NonZeroUsize)` variant placeholder (defined in the type but not yet used by insertion algorithms; treated equivalently to `EndOnly` for now).
- [ ] Integrate policy into the insertion pipeline:
  - [x] Maintain a count of successful insertions (tracked per `UnifiedInsertionPipeline`).
  - [x] For `EveryN(k)`, invoke global validation/repair every k insertions (wired via `run_global_delaunay_validation_with_policy` in the unified Stage 2 loop).
  - [ ] Always run at least one final validation/repair at the end.
  - [ ] No-op if there are no cells (zero-cell TDS).
- [ ] For robust configurations, global repair/validation should reuse `repair_global_delaunay_violations` and/or `validate_no_delaunay_violations`.

---

## Statistics and Error Semantics

Goal: keep strong observability and clear behavior on failure.

- [ ] Extend `InsertionStatistics` as needed to track:
  - [ ] Number of vertices processed.
  - [ ] Cells created/removed.
  - [ ] Fast vs robust attempts and successes.
  - [ ] Number of skipped unsalvageable vertices.
  - [ ] Number of global validation/repair runs.
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
  - [x] Zero-cell TDS behavior and later recovery via incremental insertion (current behavior: zero-cell fallback retains vertices; incremental `add` preserves validity and vertex count, without forcing an automatic rebuild).
- [ ] Integration tests:
  - [x] Fast 9 robust 9 skip pipeline on carefully chosen point sets (including existing regression cases).
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

- Stage 1 ("Robust Initial Simplex") and Stage 2 ("Unified Per-Vertex Insertion") are implemented and wired through `InsertionAlgorithm::triangulate` and the internal `UnifiedInsertionPipeline`.
- `RobustBowyerWatson` and `IncrementalBowyerWatson` both participate in the unified per-vertex pipeline, and unsalvageable vertices are tracked via `UnsalvageableVertexReport`.
- `DelaunayCheckPolicy` exists and is threaded into the unified pipeline; a final global Delaunay validation is wired, but finer-grained `EveryN(k)` cadence and any repair hooks remain partially implemented.
- Global Delaunay validation is available via `core::util::{is_delaunay, find_delaunay_violations}` and `Tds::validate_delaunay`, and is used at the end of triangulation.
- Structural invariant validation (`Tds::is_valid` / `Tds::validation_report`) and
  property-test suites are in place (`tests/proptest_triangulation.rs`,
  `tests/proptest_delaunay_condition.rs`, `tests/proptest_robust_bowyer_watson.rs`,
  `tests/integration_robust_bowyer_watson.rs`), but some high-dimensional Delaunay
  properties (especially 4D–5D) still exhibit violations or test drift.

## Focused Plan to Debug Remaining Delaunay Violations (2D–5D)

### Phase 1 – Establish a small, reproducible failure set

- [ ] Run `just test` once to confirm the current global state.
- [ ] Run the Delaunay-heavy suites with verbose output:
  - `cargo test --test proptest_delaunay_condition -- --nocapture`
  - `cargo test --test proptest_robust_bowyer_watson -- --nocapture`
  - `cargo test --test integration_robust_bowyer_watson -- --nocapture`
- [ ] For each failing proptest, capture:
  - test binary + test name,
  - dimension (2D–5D),
  - exact reproduction hints (`PROPTEST_SEED` / `--exact-seed`).
- [ ] Re-run those failures with `PROPTEST_CASES=1` and exact seeds to obtain a small set of canonical failing configurations per dimension.

### Phase 2 – Strengthen validators and add diagnostics

- [ ] Review and, if necessary, tighten `core::util::is_delaunay` and `find_delaunay_violations` so that:
  - Delaunay violations are defined as **strictly inside** the circumsphere (on-sphere is allowed).
  - Robust predicates (`robust_insphere`) are used consistently for classification.
  - Structural problems (missing vertices, invalid cells) are reported via `TriangulationValidationError`, not conflated with Delaunay violations.
- [ ] Add a debug-only helper in `core::util` (behind `cfg(any(test, debug_assertions))`) that:
  - Calls `find_delaunay_violations` (optionally on a subset of cells),
  - Dumps violating cells, their vertices, and offending external vertices via `eprintln!`,
  - Prints relevant neighbor information to help localize problems.

### Phase 3 – Debug `Tds::new` and incremental insertion paths

- [ ] For each canonical failing seed, build the vertex set and call `Tds::<f64, Option<()>, Option<()>, D>::new(&vertices)`, then:
  - Validate with `tds.is_valid()` and `tds.validate_delaunay()`,
  - On failure, use the new debug helper to inspect violating cells and points.
- [ ] For incremental failures:
  - Start from a small base triangulation (possibly constructed via `Tds::new`).
  - Insert points one-by-one using the existing insertion algorithms
    (`add`, `RobustBowyerWatson::insert_vertex`, or the unified pipeline).
  - After each successful insertion, ensure neighbors/incident cells are assigned and
    re-run `validate_delaunay`.
  - Use the debug helper to pinpoint the first insertion that introduces a Delaunay
    violation.

### Phase 4 – Integrate and test `DelaunayCheckPolicy` semantics

- [ ] Ensure `DelaunayCheckPolicy` is fully wired through the unified insertion pipeline:
  - `EndOnly`: run global Delaunay validation once at the end (current default).
  - `EveryN(k)`: run `validate_no_delaunay_violations` every `k` successful insertions **and** once at the end.
  - Skip validation when there are no cells (zero-cell triangulations).
- [ ] Add targeted tests (e.g., `tests/insertion_policy_validation.rs`) that:
  - Use small 2D–5D inputs to verify `EveryN(1)` catches violations at or near the insertion that introduces them,
  - Confirm `EndOnly` still catches violations in a final pass.
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
