# Debug Environment Variables

Comprehensive reference for all `DELAUNAY_*` environment variables used for
runtime diagnostics and debugging. All variables are **debug-only** unless
noted — they are gated behind `#[cfg(debug_assertions)]` and have no effect
in release builds.

**Activation**: most variables are presence-activated (any value works, e.g.
`DELAUNAY_DEBUG_CAVITY=1`). Variables that read a **value** are marked below.

**Output**: all diagnostic output uses the `tracing` crate. Enable with a
`tracing-subscriber` (e.g. `RUST_LOG=debug`).

---

## Contents

- [Construction & Insertion](#construction--insertion)
- [Point Location](#point-location)
- [Conflict Region](#conflict-region)
- [Cavity & Hull](#cavity--hull)
- [Orientation](#orientation)
- [Neighbor Wiring](#neighbor-wiring)
- [Flip Repair](#flip-repair)
- [Predicates & Validation](#predicates--validation)
- [Point Generation](#point-generation)
- [Large-Scale Debug Test Harness](#large-scale-debug-test-harness)
- [Proptest Configuration](#proptest-configuration)
- [Benchmarks](#benchmarks)
- [Miscellaneous](#miscellaneous)

---

## Construction & Insertion

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_INSERT_TRACE` | presence | `triangulation.rs` | Per-insertion summary (vertex index, location, conflict size, suspicion flags) |
| `DELAUNAY_DEBUG_SHUFFLE` | presence | `triangulation.rs` | Logs vertex shuffle order during batch construction |
| `DELAUNAY_DUPLICATE_METRICS` | presence | `triangulation/delaunay.rs` | Duplicate-detection metrics (spatial hash grid stats) |

## Point Location

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_LOCATE` | presence | `triangulation.rs` | Locate stats (walk steps, hint usage, fallback) |
| `DELAUNAY_DEBUG_VALIDATE_LOCATE` | presence | `locate.rs` | Cross-validates locate result against brute-force scan |

## Conflict Region

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_CONFLICT` | presence | `locate.rs` | Per-cell insphere classification during BFS, including BFS boundary cells |
| `DELAUNAY_DEBUG_CONFLICT_PROGRESS` | presence | `locate.rs` | Periodic progress during large BFS traversals |
| `DELAUNAY_DEBUG_CONFLICT_PROGRESS_EVERY` | **value** (integer) | `locate.rs` | Interval for progress logging (default: dimension-dependent) |
| `DELAUNAY_DEBUG_CONFLICT_VERIFY` | presence | `triangulation.rs` | Brute-force verification of BFS conflict-region completeness with reachability analysis |

## Cavity & Hull

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_CAVITY` | presence | `incremental_insertion.rs`, `locate.rs` | Cavity boundary diagnostics, cell creation provenance with orientation |
| `DELAUNAY_DEBUG_HULL` | presence | `incremental_insertion.rs` | Hull extension visibility, locate stats, neighbor summary |
| `DELAUNAY_DEBUG_HULL_DETAIL` | presence | `incremental_insertion.rs` | Per-facet orientation details during visibility computation |

## Orientation

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_ORIENTATION` | presence | `triangulation.rs` | Negative-orientation cell canonicalization and post-insertion audit |

## Neighbor Wiring

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_NEIGHBORS` | presence | `incremental_insertion.rs`, `triangulation.rs` | Neighbor symmetry checks after cavity wiring and hull extension |
| `DELAUNAY_DEBUG_RIDGE_LINK` | presence | `incremental_insertion.rs`, `triangulation.rs` | Ridge-link validation after wiring; skipped external facet matches |

## Flip Repair

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_REPAIR_TRACE` | presence | `flips.rs` | Per-flip trace: enqueue, skip, apply, context details |
| `DELAUNAY_REPAIR_DEBUG_FACETS` | presence | `flips.rs` | Facet-level flip skip reasons (degenerate, duplicate, non-manifold, existing simplex) |
| `DELAUNAY_REPAIR_DEBUG_PREDICATES` | presence | `flips.rs` | Insphere classification details for k=2 and k=3 violation checks |
| `DELAUNAY_REPAIR_DEBUG_RIDGE` | presence | `flips.rs` | Ridge context snapshots during k=3 repair |
| `DELAUNAY_REPAIR_DEBUG_RIDGE_LIMIT` | **value** (integer) | `flips.rs` | Maximum ridge debug snapshots (default: 64) |
| `DELAUNAY_REPAIR_DEBUG_SUMMARY` | presence | `flips.rs` | Per-attempt repair summary (flips, checks, cycles, ambiguous, skips) |

## Predicates & Validation

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_STRICT_INSPHERE_CONSISTENCY` | presence | `triangulation/delaunay.rs` | Cross-validates insphere results between fast and exact predicates |
| `DELAUNAY_DEBUG_LU_FALLBACK` | presence | `circumsphere.rs` | Logs when circumsphere computation falls back to LU decomposition |

## Point Generation

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_RANDOM_BUILDER` | presence | `triangulation_generation.rs` | Random triangulation builder diagnostics (point generation, retry logic) |
| `DELAUNAY_DEBUG_RANDOM_POINTSET_RETRIES` | presence | `point_generation.rs` | Retry attempts during random point set generation |

## Large-Scale Debug Test Harness

These variables configure `tests/large_scale_debug.rs` and run in both debug
and release builds.

| Variable | Activation | Description |
|---|---|---|
| `DELAUNAY_LARGE_DEBUG_N` | **value** | Number of vertices (global default) |
| `DELAUNAY_LARGE_DEBUG_N_{D}D` | **value** | Per-dimension vertex count override (e.g. `_N_3D=35`) |
| `DELAUNAY_LARGE_DEBUG_SEED` | **value** (hex) | Master RNG seed |
| `DELAUNAY_LARGE_DEBUG_CASE_SEED` | **value** (hex) | Per-case RNG seed (global default) |
| `DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D` | **value** (hex) | Per-dimension case seed override |
| `DELAUNAY_LARGE_DEBUG_DISTRIBUTION` | **value** | `ball` or `box` |
| `DELAUNAY_LARGE_DEBUG_BALL_RADIUS` | **value** | Radius for ball distribution |
| `DELAUNAY_LARGE_DEBUG_BOX_HALF_WIDTH` | **value** | Half-width for box distribution |
| `DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE` | **value** | `new` (batch) or `incremental` |
| `DELAUNAY_LARGE_DEBUG_DEBUG_MODE` | **value** | `cadenced` or `strict` |
| `DELAUNAY_LARGE_DEBUG_SHUFFLE_SEED` | **value** | Vertex shuffle seed |
| `DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY` | **value** | Progress logging interval |
| `DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY` | **value** | Validation interval |
| `DELAUNAY_LARGE_DEBUG_REPAIR_EVERY` | **value** | Repair interval |
| `DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS` | **value** | Flip budget override |
| `DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS` | **value** | Timeout (0 = no cap) |
| `DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS` | presence | Allow vertex insertion skips |
| `DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR` | presence | Skip final global repair pass |
| `DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL` | **value** | Total prefix probes for bisect mode |
| `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_PROBES` | **value** | Max probes per bisect run |
| `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_RUNTIME_SECS` | **value** | Bisect probe timeout |

## Proptest Configuration

These variables configure property tests in `tests/proptest_*.rs`.

| Variable | Activation | Description |
|---|---|---|
| `DELAUNAY_PROPTEST_CONSTRUCTION_ERRORS` | presence | Log construction errors during proptests |
| `DELAUNAY_PROPTEST_INSERT_ERRORS` | presence | Log insertion errors during proptests |
| `DELAUNAY_PROPTEST_REJECT_STATS` | presence | Log rejection statistics |
| `DELAUNAY_PROPTEST_COVERAGE_LOGS` | presence | Log coverage statistics |
| `DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT` | **value** | Minimum acceptance percentage |
| `DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT_{D}D` | **value** | Per-dimension acceptance override |
| `DELAUNAY_ALLOW_SLOW_COSPHERICAL_FILTER` | presence | Allow slow cospherical point filtering |

## Benchmarks

These variables configure `benches/ci_performance_suite.rs` and run in
release builds only.

| Variable | Activation | Description |
|---|---|---|
| `DELAUNAY_BENCH_LOG` | presence | Enable error logging inside Criterion measurement loops |
| `DELAUNAY_BENCH_DISCOVER_SEEDS` | presence | Seed-discovery mode: find and print stable seeds instead of running benchmarks |
| `DELAUNAY_BENCH_DISCOVER_SEEDS_LIMIT` | **value** (integer) | Maximum seeds to try per (dim, count) pair during discovery or fallback (default: 2000) |

## Miscellaneous

| Variable | Activation | Module | Description |
|---|---|---|---|
| `DELAUNAY_DEBUG_UNUSED_IMPORTS` | presence | `triangulation.rs` | Internal: suppress unused-import warnings in conditional compilation |
