# Test Coverage Improvement Plan

**Project:** delaunay  
**Current Coverage:** 56.75% (2,441/4,301 lines)  
**Target Coverage:** ≥70%  
**Timeline:** 3-4 weeks  
**Status:** 🟡 In Progress

---

## 📊 Current Coverage by Module

### ✅ Excellent Coverage (>80%)

- `collections.rs`: **100%** (10/10)
- `matrix.rs`: **100%** (6/6)
- `coordinate.rs`: **100%** (20/20)
- `lib.rs`: **100%** (2/2)
- `vertex.rs`: **82%** (79/96)
- `point.rs`: **81%** (79/97)

### ⚠️ Good Coverage (60-80%)

- `boundary.rs`: **73%** (11/15)
- `bowyer_watson.rs`: **73%** (44/60)
- `cell.rs`: **67%** (118/175)
- `facet.rs`: **66%** (87/132)
- `predicates.rs`: **63%** (90/142)

### 🔴 Priority Targets (<60%)

1. `facet_cache.rs`: **44%** (31/70) - 39 uncovered lines
2. `quality.rs`: **45%** (38/85) - 47 uncovered lines
3. `convex_hull.rs`: **46%** (126/271) - 145 uncovered lines
4. `robust_bowyer_watson.rs`: **48%** (223/463) - 240 uncovered lines
5. `insertion_algorithm.rs`: **51%** (459/893) - 434 uncovered lines
6. `triangulation_data_structure.rs`: **55%** (511/927) - 416 uncovered lines
7. `geometry/util.rs`: **57%** (295/518) - 223 uncovered lines

---

## 🎯 Coverage Goals

| Phase | Target | Modules | Timeline |
|-------|--------|---------|----------|
| **Phase 1** | 62-65% | Priority 1 modules (<50%) | Week 1-2 |
| **Phase 2** | 70%+ | Priority 2 modules (50-60%) | Week 3-4 |
| **Long-term** | 85% | Full codebase | 3-6 months |

---

## 📋 Execution Plan

### ✅ Phase 0: Baseline and Guardrails (1 hour)

**Goal:** Capture current state and add safety rails before changes.

**Actions:**

- [ ] Run baseline coverage and record module-level numbers

  ```bash
  just coverage
  # Open: target/tarpaulin/tarpaulin-report.html
  ```

- [ ] Record baseline metrics:
  - Overall: **_____%**
  - `src/core/traits/facet_cache.rs`: **_____%**
  - `src/geometry/quality.rs`: **_____%**
  - `src/geometry/algorithms/convex_hull.rs`: **_____%**
  - `src/core/algorithms/robust_bowyer_watson.rs`: **_____%**
  - `src/core/traits/insertion_algorithm.rs`: **_____%**
  - `src/core/triangulation_data_structure.rs`: **_____%**
  - `src/geometry/util.rs`: **_____%**
- [ ] Validate docs build: `just doc-check`
- [ ] Quality precheck (no code changes yet): `just quality`
- [ ] Decide test runtime budget:
  - Local: `PROPTEST_CASES=128`
  - CI: `PROPTEST_CASES=64` (or existing default)
- [ ] Create working branch (user-only):

  ```bash
  git checkout -b tests/priority1-2-coverage
  ```

**Completion Date:** ___________

---

### 📦 Phase 1: Priority 1 Modules (<50% Coverage)

#### Task 1: `tests/test_facet_cache.rs` (44% → ≥75%)

**Goal:** Exercise error paths, invalidation, generation tracking, and hit/miss behavior.

**Checklist:**

- [ ] Error paths
  - [ ] Build minimal test double implementing cache provider trait
  - [ ] Test `try_get_or_build_facet_cache()` surfaces exact error variants
  - [ ] Match on error type/message
- [ ] Cache invalidation
  - [ ] Provider with internal generation counter
  - [ ] Call `invalidate()` then request again
  - [ ] Assert: second call rebuilds cache and increments generation
- [ ] Generation tracking
  - [ ] Multiple `get()` calls without invalidation: same generation, no rebuild
  - [ ] After invalidation: generation increments exactly once
- [ ] Hit/miss pattern
  - [ ] Use `AtomicUsize` to count builder invocations
  - [ ] Pattern: get, get, invalidate, get, get → expect 2 builder calls
- [ ] Concurrency (optional if trait is `Sync + Send`)
  - [ ] Arc the provider and spawn N threads calling `get()`
  - [ ] Assert builder called once

**Suggested test names:**

- `test_error_path_is_propagated()`
- `test_invalidation_triggers_rebuild_and_generation_increments()`
- `test_cache_hits_do_not_rebuild()`
- `test_cache_hit_miss_pattern_counts()`
- `test_concurrent_get_builds_once()`

**Commands:**

```bash
just test-release
just coverage  # Target: facet_cache.rs ≥75%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### Task 2: `tests/test_quality_metrics.rs` (45% → ≥75%)

**Goal:** Cover edge cases, multi-dimension behavior, property invariants, and error handling.

**Checklist:**

- [ ] Edge cases
  - [ ] Degenerate simplices: collinear (2D), coplanar (3D), repeated points
  - [ ] Assert `Result::Err` or sentinel value
  - [ ] Extreme aspect ratios: skinny triangles/tetrahedra
  - [ ] Assert metric worse than near-equilateral
  - [ ] Equilateral/regular cases: assert metric is optimal within tolerance
- [ ] Dimension spanning (2D–6D)
  - [ ] Generate canonical simplices per dimension
  - [ ] Verify monotonic behavior and scale/translation invariance
- [ ] Property-based tests (proptest)
  - [ ] Scale invariance: `metric(P) == metric(s·P)`
  - [ ] Translation invariance: `metric(P) == metric(P + t)`
  - [ ] Permutation invariance: metric invariant under vertex order re-shuffle
  - [ ] Boundedness: metric stays within documented range or is non-negative
- [ ] Error handling
  - [ ] NaN/Inf coordinates → error
  - [ ] Duplicate vertices or zero-volume → error

**Suggested test names:**

- `test_degenerate_simplices_error()`
- `test_extreme_aspect_ratio_vs_equilateral()`
- `test_quality_invariance_across_dimensions()`
- `proptest_quality_scale_translation_permutation_invariance()`

**Commands:**

```bash
PROPTEST_CASES=128 just test-release
just coverage  # Target: quality.rs ≥75%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### Task 3: Audit `convex_hull.rs` + `tests/test_convex_hull_error_paths.rs` (46% → ≥70%)

**Goal:** Reduce redundancy and specifically cover uncovered paths and error handling.

#### Part A: Audit (no code removal yet)

- [ ] Run `just coverage` and open HTML report
- [ ] List uncovered functions/branches in `convex_hull.rs`
- [ ] Map existing 3,790 lines of tests to covered/uncovered paths
- [ ] Note redundant scenarios
- [ ] Record pre-change module coverage: _____%

#### Part B: New file `tests/test_convex_hull_error_paths.rs`

- [ ] StaleHull error detection
  - [ ] Induce stale/invalidated hull state using public APIs
  - [ ] Assert `Err(StaleHull)` if exposed by API
- [ ] Cache rebuild failure
  - [ ] Force underlying hull/facet cache rebuild to return error
  - [ ] Assert convex hull operation surfaces error variant
- [ ] Coordinate conversion errors
  - [ ] Use extreme f64 magnitudes, subnormals
  - [ ] Assert correct error variant
- [ ] Invalid facet index bounds
  - [ ] Call APIs with out-of-range indices
  - [ ] Expect `Err(IndexOutOfBounds/ValidationError)`
- [ ] Concurrent operations (if API supports)
  - [ ] Run same hull computation across clones in parallel threads
  - [ ] Validate thread-safety or assert non-Send/Sync at compile-time
- [ ] Extreme coordinate tests
  - [ ] Mix values spanning ±1e308 and 1e-308
  - [ ] Assert hull validity or error correctness

#### Part C: Consolidation plan (separate PR after coverage confirmed)

- [ ] Tag redundant tests that don't contribute unique coverage
- [ ] Propose merging overlapping scenarios into table-driven tests
- [ ] Reduce test lines without reducing coverage

#### Acceptance

- [ ] Re-run `just coverage` and capture new coverage
- [ ] Document before/after coverage delta: _____ % → _____ %
- [ ] Document any untestable paths (impossible branches)

**Commands:**

```bash
just coverage  # Before
# ... implement tests ...
just test-release
just coverage  # After - target: convex_hull.rs ≥70%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### Task 4: `tests/test_robust_fallbacks.rs` (48% → ≥70%)

**Goal:** Exercise robust predicate fallbacks, degenerate handling, all InsertionStrategy variants.

**Checklist:**

- [ ] Robust predicate fallback
  - [ ] Construct near-degenerate inputs (orientation/insphere numerically ambiguous)
  - [ ] Assert robust path triggers (check returned status or metrics)
- [ ] Degenerate insertion handling
  - [ ] Co-located points; collinear/coplanar sets
  - [ ] Ensure algorithm selects fallback handling path
  - [ ] Maintain TDS invariants
- [ ] InsertionStrategy variants
  - [ ] Iterate over all enum variants
  - [ ] Assert identical final hull/triangulation where applicable
  - [ ] Document expected differences
- [ ] Error propagation
  - [ ] Misconfigure predicate thresholds
  - [ ] Inject failing predicate provider
  - [ ] Assert errors bubble with correct context
- [ ] Predicate configuration
  - [ ] Toggle robust vs standard predicates (if configurable)
  - [ ] Assert chosen code paths differ
  - [ ] Results remain valid on stable inputs
- [ ] Robust-specific methods
  - [ ] Exercise any `robust_*` entry points directly with edge inputs

**Commands:**

```bash
PROPTEST_CASES=128 just test-release
just coverage  # Target: robust_bowyer_watson.rs ≥70%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### ✅ Phase 1 Checkpoint

**Goal:** Verify Phase 1 impact (target overall ≥62–65%)

- [ ] Run: `just coverage`
- [ ] Verify module targets met:
  - [ ] `facet_cache.rs` ≥75%: Achieved _____%
  - [ ] `quality.rs` ≥75%: Achieved _____%
  - [ ] `convex_hull.rs` ≥70%: Achieved _____%
  - [ ] `robust_bowyer_watson.rs` ≥70%: Achieved _____%
- [ ] Overall coverage: _____%
- [ ] If any targets missed, iterate on focused tests before proceeding

**Checkpoint Date:** ___________

---

### 📦 Phase 2: Priority 2 Modules (50-60% Coverage)

#### Task 5: `tests/test_insertion_algorithm_trait.rs` (51% → ≥70%)

**Goal:** Cover default trait implementations, error recovery, buffers, statistics, strategy paths.

**Checklist:**

- [ ] Default implementations
  - [ ] Create minimal struct implementing trait relying on defaults
  - [ ] Assert behavior matches docs for each default
- [ ] Error recovery
  - [ ] Simulate transient insertion failures
  - [ ] Assert recovery/retry logic restores algorithm progress and consistent TDS
- [ ] Buffer management
  - [ ] Fill, reuse, and clear internal buffers
  - [ ] Assert no leaks, correct reuse (capacity stays, length resets)
- [ ] Statistics
  - [ ] Exercise counters on edge paths: 0 insertions, all rejected, mixed
  - [ ] Assert accurate tallies
- [ ] Strategy path combinations
  - [ ] Cross all InsertionStrategy variants with optional flags
  - [ ] Assert valid outcomes

**Commands:**

```bash
just test-release
just coverage  # Target: insertion_algorithm.rs ≥70%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### Task 6: `tests/test_tds_edge_cases.rs` (55% → ≥70%)

**Goal:** Validate TDS integrity under removals, neighbor edge cases, corruption detection, scalability.

**Checklist:**

- [ ] Corruption detection
  - [ ] Construct illegal operations via public APIs
  - [ ] Assert detect/Err on `validate()`
- [ ] Removal error paths
  - [ ] Remove non-existent vertex/cell
  - [ ] Double-remove
  - [ ] Remove with dangling neighbors
  - [ ] Expect explicit errors
- [ ] Neighbor edge cases
  - [ ] After random insertions/removals
  - [ ] Assert neighbor symmetry and manifoldness invariants
- [ ] Stress tests (mark `#[ignore]` to keep CI fast)
  - [ ] 1,000 vertices: randomized insertion order
  - [ ] 5,000 vertices: assert `validate()` and no panics
  - [ ] 10,000 vertices: stress test
  - [ ] Run locally: `cargo test --release -- --ignored`
- [ ] Serialization/deserialization round-trip (if feature available)
  - [ ] Feature-gated: `cfg(feature = "serde")`
  - [ ] Serialize to JSON/bincode, deserialize
  - [ ] Assert equal invariants and hash/equality

**Commands:**

```bash
just test-release
cargo test --release -- --ignored  # Stress tests locally
just coverage  # Target: triangulation_data_structure.rs ≥70%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### Task 7: `tests/test_geometry_util.rs` (57% → ≥75%)

**Goal:** Cover all geometric utilities across dimensions, extremes, numerical stability.

**Checklist:**

- [ ] Dimensional sweep
  - [ ] For D in 2..=6 test all public utility functions
  - [ ] Canonical inputs and randomized sets
- [ ] Extreme values
  - [ ] Large magnitude (±1e308)
  - [ ] Tiny (±1e-308)
  - [ ] Mixed scales
  - [ ] Assert robustness or appropriate error
  - [ ] Include NaN/Inf inputs to assert error branches
- [ ] Stability/invariants (proptest)
  - [ ] Translation/rotation/scale invariance where applicable
  - [ ] Symmetry and antisymmetry for vector operations
  - [ ] Consistency across equivalent formulations

**Commands:**

```bash
PROPTEST_CASES=128 just test-release
just coverage  # Target: geometry/util.rs ≥75%
```

**Completion Date:** ___________  
**Coverage Achieved:** _____%

---

#### ✅ Phase 2 Checkpoint

**Goal:** Overall coverage target ≥70%

- [ ] Run: `just coverage`
- [ ] Confirm overall coverage ≥70%: Achieved _____%
- [ ] If below 70%, identify top remaining uncovered functions (tarpaulin HTML)
- [ ] Add 1–2 focused tests where payoff is highest

**Checkpoint Date:** ___________

---

### 🎯 Final Verification

**Goal:** Ensure all quality gates pass before merge/publish.

**Checklist:**

- [ ] Run: `just test-all` (Rust + Python)
- [ ] Validate docs again: `just doc-check`
- [ ] Run quality suite: `just quality`
- [ ] Spell-check modified files: `just spell-check`
  - [ ] Add legitimate terms to `cspell.json` words array
  - [ ] Validate JSON: `just validate-json`
  - [ ] Validate TOML: `just validate-toml`
- [ ] Examples sanity: `just examples`
- [ ] Allocation tests (optional): `just test-allocation`
- [ ] Document untestable code paths:
  - [ ] Add section to `docs/code_organization.md`
  - [ ] Add section to `tests/README.md`
  - [ ] Describe impossible branches with rationale

**Completion Date:** ___________

---

### 📝 Convex Hull Coverage Analysis

**Goal:** Answer: "Have we improved convex_hull.rs coverage? Are tests consolidated appropriately?"

**Checklist:**

- [ ] Before/After comparison:
  - Before Task 3: _____%
  - After Task 3: _____%
  - Delta: +_____%
- [ ] Consolidation without coverage loss:
  - [ ] Identify redundant tests (list in comment/file)
  - [ ] Prepare list for separate PR removing/merging them
  - [ ] Verify coverage remains ≥ post-Task-3 level after consolidation trial
  - [ ] Keep high-signal tests
  - [ ] Convert repetitive scenarios to parameterized/table-driven patterns

**Analysis Date:** ___________  
**Answer:** _________________________________________________________

---

### 🚀 Post-Merge Follow-ups (Optional but Recommended)

**Checklist:**

- [ ] Add mutation testing locally:

  ```bash
  cargo mutants -v --timeout 120 --tests
  ```

  - [ ] Use results to add missing assertions where mutants survive
- [ ] Add fuzzing harness (separate branch/PR):

  ```bash
  cargo fuzz init
  ```

  - [ ] Create fuzz targets for parsers/coordinate conversion
  - [ ] Create fuzz targets for hull building
- [ ] CI optimization:
  - [ ] Gate long-running tests with `#[ignore]`
  - [ ] Document how to run them
  - [ ] Consider nightly job to run ignored stress tests

---

### 📚 Documentation Updates

**Goal:** Update docs before any publish to crates.io.

**Checklist:**

- [ ] Update `tests/README.md`:
  - [ ] Document new test files
  - [ ] How to run ignored tests
  - [ ] Environment knobs (`PROPTEST_CASES`)
- [ ] Update `docs/code_organization.md`:
  - [ ] Reflect new test structure
  - [ ] Document test patterns
- [ ] Note project rule:
  - [ ] Update documentation before publishing
  - [ ] Do not publish docs-only changes without version bump

**Completion Date:** ___________

---

## 📊 Progress Tracking

### Weekly Status

**Week 1:**

- Status: ___________
- Tasks Completed: ___________
- Coverage: ___________

**Week 2:**

- Status: ___________
- Tasks Completed: ___________
- Coverage: ___________

**Week 3:**

- Status: ___________
- Tasks Completed: ___________
- Coverage: ___________

**Week 4:**

- Status: ___________
- Tasks Completed: ___________
- Coverage: ___________

### Final Results

- **Start Date:** ___________
- **Completion Date:** ___________
- **Starting Coverage:** 56.75%
- **Final Coverage:** _____%
- **Lines Added:** _____%
- **Tests Added:** _____
- **Modules Improved:** _____

---

## 🎓 Lessons Learned

### What Worked Well

_Document successful strategies, patterns, tools..._

### What Could Be Improved

_Document challenges, bottlenecks, improvements for next time..._

### Recommendations for Future Testing

_Document insights for maintaining >70% coverage going forward..._

---

## 📞 References

- **Coverage Reports:** `target/tarpaulin/tarpaulin-report.html`
- **Test Documentation:** `tests/README.md`
- **Code Organization:** `docs/code_organization.md`
- **Just Commands:** `justfile` or run `just --list`

---

**Last Updated:** 2025-11-04  
**Maintained By:** Project Contributors
