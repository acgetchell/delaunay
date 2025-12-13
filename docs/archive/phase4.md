# Phase 4 Benchmark Consolidation Plan

## Executive Summary

**Problem:** Currently using `triangulation_creation.rs` for Phase 4 benchmarking, but it's
the wrong tool - it only measures basic construction and is redundant with CI benchmarks.

**Solution:** Migrate to `large_scale_performance.rs`, which was **specifically designed for
Phase 4 SlotMap evaluation** and measures:

- ✅ Iteration performance (vertex/cell/neighbor traversals)
- ✅ Memory usage patterns (RSS tracking)
- ✅ Query performance (lookups, contains checks)
- ✅ Validation stress-testing (topology checks)
- ✅ 1K-10K scale appropriate for SlotMap comparisons

**Phase 4 Goal:** Enable swapping SlotMap implementations via Cargo feature flags
(SlotMap ↔ DenseSlotMap ↔ HopSlotMap) for benchmarking, targeting 10-15% iteration
performance improvement. The DenseSlotMap backend is gated by the `dense-slotmap` feature.

**Current state (2025-12-13):**

- Cargo feature `dense-slotmap` (DenseSlotMap backend) is enabled by default (`default = ["dense-slotmap"]`)
- SlotMap remains supported via `--no-default-features`
- Local comparison tooling: `uv run compare-storage-backends` (or `just compare-storage`)
- Cluster comparison tooling: `scripts/slurm_storage_comparison.sh` (saves Criterion baselines for `critcmp`)

## Current Benchmark Suite Issues

### Overlaps & Redundancies

1. **Triangulation Creation Overlap:**
   - `ci_performance_suite.rs`, `triangulation_creation.rs`, `microbenchmarks.rs`, and `profiling_suite.rs` all benchmark basic triangulation creation
   - **Problem:** `triangulation_creation.rs` is redundant with `ci_performance_suite.rs` (already used by CI/scripts)

2. **Assign Neighbors Duplication:**
   - Both `assign_neighbors_performance.rs` and `microbenchmarks.rs` test the same operation
   - `assign_neighbors_performance.rs` is more comprehensive (grid, spherical, scaling tests)

3. **Memory Measurement Overlap:**
   - `large_scale_performance.rs` and `profiling_suite.rs` both measure memory
   - Redundant memory bench files were consolidated into `profiling_suite.rs`

### Scripts Integration

- `scripts/benchmark_utils.py` is hardcoded to use **`ci_performance_suite.rs`** for baseline generation and regression testing
- Phase 4 backend comparison tooling exists via:
  - `scripts/compare_storage_backends.py` (`uv run compare-storage-backends`)
  - `scripts/slurm_storage_comparison.sh` (cluster runs; saves Criterion baselines for `critcmp`)

## Implementation Plan

### ✅ 1. Kickoff and Scope Alignment

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] Confirm Phase 4's primary goal: evaluate SlotMap-backed TDS performance and memory behavior at large scale
- [x] Make `large_scale_performance.rs` the primary Phase 4 benchmark (replacing `triangulation_creation.rs`)
- [x] Adopt one-cycle deprecation policy for redundant benches
- [x] Keep work isolated in feature branches when convenient (process note)

**Notes:**

- Backend selection remains compile-time via Cargo features (no runtime abstraction)
- `dense-slotmap` (DenseSlotMap backend) is now enabled by default (2025-12-13)

---

### ✅ 2. Inventory All Benchmark Files

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] List current benches: `ls benches/*.rs`
- [x] Record per file: purpose, operations covered, dataset sizes, CLI parameters/env vars, Criterion groups/IDs, output format
- [x] Validate expected bench set (current):
  - `ci_performance_suite.rs`
  - `large_scale_performance.rs` (Phase 4 primary)
  - `profiling_suite.rs`
  - `microbenchmarks.rs`
  - `assign_neighbors_performance.rs`
  - `circumsphere_containment.rs`
  - `triangulation_creation.rs` (deprecated harness)

**Deliverable:** `benches/INVENTORY.md` (temporary, to be folded into `benches/README.md`)

---

### ✅ 3. Map Benchmark Usage in GitHub Actions and Scripts

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] Scan workflows: `rg -n "bench" .github/workflows/benchmarks.yml .github/workflows/profiling-benchmarks.yml .github/workflows/generate-baseline.yml`
- [x] Scan Python scripts: `rg -n "(cargo bench|criterion|bench.*filter|--bench)" scripts/`
- [x] Scan justfile: `rg -n "(cargo bench|criterion|bench.*filter|--bench)" justfile`
- [x] Review `scripts/benchmark_utils.py` for hardcoded bench names, filters, baselines

**Deliverable:** `benches/USAGE_MAP.md` (temporary) - matrix of: benchmark file ↔ workflow job(s) ↔ script command(s) ↔ just targets

**Known Usage:**

- `ci_performance_suite.rs`: Used by `benchmark_utils.py` for baseline generation (line 1338, 1345, 1441, 1448)
- `circumsphere_containment.rs`: Used by `PerformanceSummaryGenerator` (line 296)
- `profiling_suite.rs`: Used by `.github/workflows/profiling-benchmarks.yml`

---

### ✅ 4. Document Purpose and Overlap of All Benches

**Status:** ✅ Completed (2025-10-20)

**Current Known Purposes:**

| Benchmark | Purpose | Scale | Operations | Phase 4 Relevant? |
|-----------|---------|-------|------------|-------------------|
| `ci_performance_suite.rs` | CI regression detection | 10-50 pts, 2D-5D | Basic construction | No |
| `triangulation_creation.rs` | Deprecated harness | N/A | Prints deprecation notice | No |
| `large_scale_performance.rs` | **Phase 4 backend eval** | 1K-10K vertices | Construction, memory, iteration, queries, validation | **YES - PRIMARY** |
| `profiling_suite.rs` | Comprehensive profiling | 10³-10⁶ pts | Scaling, memory profiling, query latency, etc. | Partial - too heavy |
| `microbenchmarks.rs` | Core operations | Various | Bowyer-Watson + validation microbenches | Keep |
| `assign_neighbors_performance.rs` | Neighbor assignment | 10-50 pts, 2D-5D | Distributions + scaling | Keep |
| `circumsphere_containment.rs` | Algorithm comparison | Random queries | Circumsphere predicates | Keep |

**Deliverable:** Consolidated section in `benches/README.md` with detailed table

---

### ✅ 5. Design the Consolidation Plan

**Status:** ✅ Completed (2025-10-20)

**Decisions (implemented):**

- [x] Deprecate `triangulation_creation.rs` (keep as a one-cycle deprecation harness)
- [x] Consolidate memory benchmarks into `profiling_suite.rs`
- [x] Deduplicate `microbenchmarks.rs` around `assign_neighbors`
- [x] Codify `large_scale_performance.rs` as the Phase 4 primary benchmark
- [ ] Standardize CLI/env controls across benches (optional; partially addressed via env vars)

**Deliverable:**

- This document (`docs/archive/phase4.md`) serves as the consolidation plan and progress log.

---

### ✅ 6. Deprecate triangulation_creation.rs

**Status:** ✅ Completed (2025-10-20)

**Implementation: Option A (One-Cycle Deprecation)** ✅

- [x] Replace contents with minimal harness that:
  - Prints clear deprecation message pointing to `large_scale_performance.rs` (Phase 4) and `ci_performance_suite.rs` (CI)
  - Exits early to avoid wasting CI time
- [x] Keep file for one release cycle

**Option B (Immediate Removal):**

- [ ] Delete file entirely
- [ ] Update all references in workflows, scripts, and justfile

**Tasks:**

- [x] Update `scripts/benchmark_utils.py` to stop referencing `triangulation_creation`
- [x] Update CI workflows if any reference it
- [x] Update `justfile` if any targets reference it

---

### ✅ 7. Consolidate Memory Benchmarks into profiling_suite.rs

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] Consolidate memory profiling into `profiling_suite.rs` (memory_profiling group)
- [x] Remove redundant memory benchmark files (`memory_scaling.rs`, `triangulation_vs_hull_memory.rs`)
- [x] Update docs and Cargo configuration to reflect the consolidation
- [ ] (Optional) Add richer metadata export (dataset/dim/seed/units) for automated analysis

**Memory Measurement Strategy:**

```rust
// Current approach in large_scale_performance.rs
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

fn get_memory_usage() -> u64 {
    // Returns RSS in KiB
}

// Keep this for Phase 4 SlotMap evaluation
// Optional: Add allocation-counter for detailed tracking
```

---

### ✅ 8. Deduplicate microbenchmarks Around assign_neighbors

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] Remove `assign_neighbors` duplicates from `microbenchmarks.rs`
- [x] Centralize neighbor-assignment benchmarking in `assign_neighbors_performance.rs`
- [x] Preserve baseline history by keeping benchmark names stable where possible
- [x] Add module docs pointing contributors to the consolidated benchmark

**What Remains in microbenchmarks.rs:**

- Bowyer-Watson triangulation benchmarks (unique)
- `remove_duplicate_cells` benchmarks (unique)
- Validation method benchmarks (unique)
- Incremental construction benchmarks (unique)

---

### ✅ 9. Elevate large_scale_performance.rs for Phase 4

**Status:** ✅ Completed (2025-10-20)

**Current State:**

- ✅ Already designed for Phase 4 SlotMap evaluation (see file comments)
- ✅ Tests iteration: `bench_vertex_iteration`, `bench_cell_iteration`
- ✅ Tests memory: `bench_memory_usage` with RSS tracking
- ✅ Tests queries: `bench_neighbor_queries`
- ✅ Tests validation: `bench_validation` (topology stress-test)
- ✅ Supports 1K-10K scale

**Enhancements Needed:**

- [ ] **Add standardized CLI/env control for datasets:**
  - `BENCH_N`: point count
  - `BENCH_DIM`: dimension (2-5)
  - `BENCH_SEED`: RNG seed
  - `BENCH_DISTRIBUTION`: grid, random, clustered
  - `BENCH_OP_MIX`: operation ratio
  - `BENCH_JSON_OUT`: structured results path

  Already supported:
  - `BENCH_LARGE_SCALE` (toggles larger 4D point counts)
  - `BENCH_SAMPLE_SIZE`, `BENCH_WARMUP_SECS`, `BENCH_MEASUREMENT_TIME` (Criterion tuning)

- [x] **Criterion IDs are stable and baseline-friendly**

  Current scheme: `<category>/<dim>/<n>v` (e.g. `construction/3D/1000v`).

- [ ] **Add cache locality measurement (optional):**

  ```rust
  // Behind feature flag: perf_guard or similar
  // Use perf/callgrind hooks for cache miss analysis
  ```

- [x] **Support SlotMap vs alternative comparison:**

  ```rust
  // Use feature flags + type aliases for zero-cost abstraction:
  #[cfg(feature = "dense-slotmap")]
  type StorageBackend<K, V> = DenseSlotMap<K, V>;
  
  #[cfg(not(feature = "dense-slotmap"))]
  type StorageBackend<K, V> = SlotMap<K, V>;
  
  // Benchmark with:
  // cargo bench --bench large_scale_performance  # default (feature: dense-slotmap)
  // cargo bench --no-default-features --bench large_scale_performance  # SlotMap
  ```

**Key Metrics for Phase 4:**

1. **Iteration speed**: Full vertex/cell traversals, neighbor walks
2. **Memory usage**: Peak RSS, per-element footprint estimates  
3. **Cache locality**: Traversal patterns (BFS vs random access)
4. **Query performance**: Lookups, contains checks, incident-entity queries

---

### ✅ 10. Phase 4 Storage Backend Comparison Tooling

**Status:** ✅ Completed (2025-12-13)

Instead of adding Phase 4 baseline JSON subcommands to `benchmark_utils.py`, Phase 4 backend
comparison is handled via Criterion baselines and dedicated scripts.

**Tooling:**

- `scripts/compare_storage_backends.py` (`uv run compare-storage-backends`)
  - Runs `cargo bench` twice:
    - DenseSlotMap (feature: `dense-slotmap`; default)
    - SlotMap (`--no-default-features`)
  - Generates a markdown report (default: `artifacts/storage_comparison.md`)

- `scripts/slurm_storage_comparison.sh`
  - Runs both backends on a Slurm cluster
  - Saves Criterion baselines (`slotmap`, `denseslotmap`) and supports `critcmp`

**Commands:**

```bash
# Local comparison report
just compare-storage

# Large scale comparison (sets BENCH_LARGE_SCALE=1)
just compare-storage-large

# Direct invocation
uv run compare-storage-backends --bench large_scale_performance
```

**Tasks:**

- [x] Local backend comparison + markdown report (`compare_storage_backends.py`)
- [x] Cluster backend comparison script saving baselines (`slurm_storage_comparison.sh`)
- [x] Documentation updated for `dense-slotmap` default (DenseSlotMap) + SlotMap via `--no-default-features`
- [ ] (Optional) Add Phase 4 baseline JSON generation to `benchmark_utils.py` for CI-style regression testing

---

### ✅ 11. Update GitHub Actions Workflows

**Status:** ✅ Completed (2025-10-20)

**benchmarks.yml (Performance Regression Testing):**

- [x] Replace any `triangulation_creation` references with `ci_performance_suite.rs` or `large_scale_performance.rs`
- [ ] Add optional Phase 4 job that runs reduced-size `large_scale_performance.rs` smoke test
- [x] Keep runtime reasonable for CI runs (use dev settings/timeouts where appropriate)

**profiling-benchmarks.yml (Comprehensive Profiling):**

- [x] Point memory jobs to `profiling_suite.rs` memory groups (after consolidation)
- [x] Gate heavy runs by workflow_dispatch labels or schedules
- [ ] Add Phase 4-specific profiling job (optional, manual trigger)

**generate-baseline.yml (Baseline Generation):**

- [ ] Add job to generate and upload `phase4_baseline.json` via new script command
- [ ] Store artifact with 90-day retention (same as release baselines)
- [ ] Trigger on: manual, monthly schedule, release tags

**Example Phase 4 Job:**

```yaml
phase4-smoke-test:
  name: Phase 4 SlotMap Smoke Test
  runs-on: macos-15
  timeout-minutes: 30
  
  steps:
    - uses: actions/checkout@v5
    
    - name: Install Rust toolchain
      uses: actions-rust-lang/setup-rust-toolchain@v1
    
    - name: Run Phase 4 benchmarks (reduced scale)
      run: |
        cargo bench --bench large_scale_performance -- \
          --sample-size 10 \
          "construction/3D/1000" \
          "queries/neighbors/3D/1000" \
          "iteration/vertices/3D/1000"
```

---

### ✅ 12. Documentation and Changelog Updates

**Status:** ✅ Completed (2025-10-20)

**benches/README.md:**

- [x] Add categorization section:
  - **CI Benchmarks**: `ci_performance_suite.rs` (fast, regression detection)
  - **Profiling Benchmarks**: `profiling_suite.rs` (comprehensive, 1-2 hours)
  - **Phase 4 Benchmarks**: `large_scale_performance.rs` (SlotMap evaluation)
  - **Algorithm Comparison**: `circumsphere_containment.rs`
  - **Specialized**: `assign_neighbors_performance.rs`
  - **Deprecated**: `triangulation_creation.rs` (use `ci_performance_suite.rs` or `large_scale_performance.rs`)

- [x] Add "When to use which" guidance:

  ```markdown
  ## Benchmark Selection Guide
  
  | Use Case | Benchmark | Command |
  |----------|-----------|---------|
  | Quick CI regression check | `ci_performance_suite.rs` | `just bench` or `cargo bench --bench ci_performance_suite` |
  | Phase 4 SlotMap evaluation | `large_scale_performance.rs` | `cargo bench --bench large_scale_performance` |
  | Deep profiling (1-2 hours) | `profiling_suite.rs` | `cargo bench --bench profiling_suite` |
  | Memory analysis | `profiling_suite.rs` (memory groups) | `cargo bench --bench profiling_suite -- memory` |
  | Algorithm comparison | `circumsphere_containment.rs` | `cargo bench --bench circumsphere_containment` |
  ```

- [x] Explicitly document `large_scale_performance.rs` as Phase 4 primary
- [x] Add deprecation notice for `triangulation_creation.rs`

**docs/code_organization.md:**

- [x] Update benchmark section to reflect new layout
- [x] Add Phase 4 benchmark responsibilities
- [x] Document memory benchmark consolidation

**CHANGELOG.md:**

- Note: `CHANGELOG.md` is auto-generated from git history in this repo.
- Do not edit it manually; ensure the relevant commits exist and run `just changelog` before release.

---

### ☐ 13. Add Missing Coverage (Time Permitting)

**Status:** Not Started

**Priority 1 (High Value):**

- [ ] **Convex hull timing benchmarks**
  - Add to `profiling_suite.rs` or separate file
  - Cover varied distributions (random, grid, clustered) and dimensions (2D-5D)
  - Currently only memory benchmarks exist in `profiling_suite.rs` (`memory_profiling` group)

**Priority 2 (Moderate Value):**

- [ ] **Serialization/deserialization benchmarks**
  - Add Criterion benches for Serde (bincode, JSON)
  - Vary triangulation sizes (1K, 10K, 100K vertices)
  - Measure throughput (MB/s) and time per operation
  
- [ ] **f32 vs f64 coordinate type comparison**
  - Matrix: dimensions (2D-5D) × sizes (1K, 10K) × distributions (random, grid)
  - Report relative speed and memory deltas
  - Use const generic benchmarks or feature flags

**Priority 3 (Nice to Have):**

- [ ] **Point location strategies** (if multiple implementations exist)
- [ ] **Incremental vs batch insertion** behavior analysis
- [ ] **Parallel construction benchmarks** (if parallelization exists/planned)

---

### ✅ 14. Quality Gates, Validation, and CI Safety

**Status:** ✅ Completed (2025-12-13)

**Pre-commit Checks (for all changes):**

```bash
# Format and lint
just fmt
just clippy
just doc-check

# Python quality
just python-lint

# Documentation quality
just markdown-lint
just spell-check

# Configuration validation
just validate-json
just validate-toml
```

**Benchmark-Specific Checks:**

```bash
# Verify benchmarks compile after Rust edits
just bench-compile

# Run Python script tests
uv run pytest

# Quick smoke test of benchmarks (reduced iterations)
cargo bench --bench ci_performance_suite -- --test
cargo bench --bench large_scale_performance -- --test
```

**PR Strategy:**

- [ ] **Split PRs**: docs/Python-only vs Rust changes
  - Docs-only PRs skip expensive CI benchmark runs
  - Rust PRs trigger full benchmark suite
- [ ] Use `[skip ci]` in commit messages for documentation-only changes
- [ ] Create feature branch for each major change
- [ ] Run quality gates before pushing

**Tasks:**

- [x] Run all quality gates on changed files (`just ci`)
- [x] Verify benchmark compilation with `just bench-compile`
- [x] Run Python tests with `uv run pytest`
- [x] Verify SlotMap builds/tests with `cargo test --no-default-features`
- [ ] Test smoke runs of modified benchmarks

---

### ◔ 15. Acceptance Criteria and Sign-Off

**Status:** ◔ In Progress (updated 2025-12-13)

**Benchmark Files:**

- [x] No broken references to removed/deprecated benches in workflows, scripts, or justfile
- [x] `large_scale_performance.rs` covers construction, memory, iteration, queries, and validation (2D–5D)
- [ ] Cache locality measurement (optional; not implemented)
- [x] Stable Criterion IDs (scheme: `<category>/<dim>/<n>v`)
- [ ] JSON output schema beyond Criterion (optional; not implemented)

**Backend Comparison Tooling:**

- [x] Local comparison report generator: `uv run compare-storage-backends`
- [x] Cluster comparison script: `scripts/slurm_storage_comparison.sh` (saves baselines for `critcmp`)
- [ ] Optional CI-style Phase 4 baseline JSON generation (deferred)

**Build System:**

- [x] Compare storage backends: `just compare-storage`, `just compare-storage-large`
- [x] Run large-scale benchmark directly: `cargo bench --bench large_scale_performance`

**Documentation:**

- [x] `benches/README.md` guides contributors and documents Phase 4 benchmarks
- [x] `docs/code_organization.md` reflects benchmark layout and memory consolidation
- [ ] Changelog entry is generated (do not edit `CHANGELOG.md` directly)

**Quality Gates:**

- [x] `just ci` passes locally
- [x] `cargo test --no-default-features` passes
- [x] `uv run pytest` passes

---

## Phase 4 SlotMap Evaluation Metrics

### Key Performance Indicators

Once the benchmark consolidation is complete, Phase 4 will evaluate these metrics:

1. **Iteration Performance** (10-15% improvement target with DenseSlotMap; feature: `dense-slotmap`)
   - Full vertex traversal time
   - Full cell traversal time
   - Neighbor-following traversal patterns
   - Filtered iteration (predicates)

2. **Memory Efficiency** (50% reduction target from Phase 3)
   - Peak RSS during construction
   - Per-vertex memory footprint
   - Per-cell memory footprint
   - Memory fragmentation analysis

3. **Cache Locality** (5-10% cache miss reduction target)
   - Sequential access patterns
   - Random access patterns
   - BFS traversal over adjacency graph
   - Optional: perf/cachegrind integration

4. **Query Performance** (maintain or improve)
   - Key lookup time
   - Contains-key checks
   - Neighbor queries
   - Incident-entity queries

### Collection Backend Comparison

| Backend | Iteration | Memory | Insertion | Removal | Best For |
|---------|-----------|--------|-----------|---------|----------|
| DenseSlotMap (`dense-slotmap`, default) | **Excellent** | Dense/contiguous | O(1) amortized | O(1) with moves | Stable/iteration |
| SlotMap (optional) | Good | Sparse | O(1) amortized | O(1) | Dynamic changes |
| HopSlotMap (future) | Good | Hop-optimized | O(1) | O(1) | Large scale |

### Success Criteria

- [ ] `dense-slotmap` (DenseSlotMap) implementation shows 10-15% iteration improvement
- [ ] No regression in other operations (insertion, removal, lookup)
- [ ] Memory usage comparable or better than current SlotMap
- [ ] 100% API compatibility maintained via trait abstraction
- [ ] Easy benchmarking via type parameter swap

---

## References

- **Phase 4 Roadmap:** `docs/OPTIMIZATION_ROADMAP.md` (lines 493-685)
- **Large Scale Benchmark:** `benches/large_scale_performance.rs` (Phase 4 evaluation comments on lines 16-23, 57-58)
- **Current Benchmark Suite:** `benches/README.md`
- **Benchmark Tooling:** `scripts/benchmark_utils.py`
- **CI Workflows:** `.github/workflows/benchmarks.yml`, `.github/workflows/profiling-benchmarks.yml`

---

## Progress Tracking

**Last Updated:** 2025-12-13

**Overall Status:** ✅ Benchmark consolidation complete; ✅ `dense-slotmap` (DenseSlotMap) is default

**Completed Steps:**

- ✅ Step 1: Kickoff and scope alignment
- ✅ Step 2: Inventory all benchmark files
- ✅ Step 3: Map benchmark usage in workflows/scripts
- ✅ Step 4: Document purpose and overlap
- ✅ Step 5: Consolidation plan (this document)
- ✅ Step 6: Deprecate `triangulation_creation.rs`
- ✅ Step 7: Consolidate memory benchmarks
- ✅ Step 8: Deduplicate `assign_neighbors` benchmarks
- ✅ Step 9: Elevate `large_scale_performance.rs` for Phase 4
- ✅ Step 10: Backend comparison tooling (local + Slurm)
- ✅ Step 12: Documentation updates
- ✅ Step 14: Quality gates and validation

**Next Steps (optional):**

1. ☐ Add Phase 4 smoke test job in CI for `large_scale_performance.rs` (reduced scale)
2. ☐ Add dataset CLI/env controls (`BENCH_N`, `BENCH_DIM`, `BENCH_SEED`, distributions)
3. ☐ Add cache locality measurement (optional)
4. ✅ Archived under `docs/archive/phase4.md`

---

## Notes

### Session 2025-10-20

**Completed:**

- ✅ Steps 2-4: Full inventory, usage mapping, and documentation
- ✅ Updated `benches/README.md` with:
  - Comprehensive "Benchmark Suite Overview" table
  - "Benchmark Selection Guide" with use cases
  - Phase 4 section explicitly documenting `large_scale_performance.rs` as primary
  - Deprecation notice for `triangulation_creation.rs`
- ✅ Fixed markdown linting issues (line length)
- ✅ Added profiling tool terms to spell check: `dhat`, `callgrind`, `cachegrind`

**Key Findings:**

- **Confirmed:** `triangulation_creation.rs` has ZERO usage (no workflows, scripts, justfile)
- **Confirmed:** `large_scale_performance.rs` already designed for Phase 4 (iteration, memory, queries)
- **Documented overlaps:**
  1. `triangulation_creation.rs` - 100% redundant with `ci_performance_suite.rs`
  2. `microbenchmarks.rs` - Has duplicate `assign_neighbors` tests
  3. Memory benchmarks overlap: `memory_scaling.rs`, `triangulation_vs_hull_memory.rs`, `profiling_suite.rs`

**Decisions Made:**

- Use `large_scale_performance.rs` as Phase 4 primary (not `triangulation_creation.rs`)
- Deprecate `triangulation_creation.rs` using Option A (one-cycle deprecation with notice)
- Consolidate memory benchmarks into `profiling_suite.rs`

**Implementation Details:**

- **Step 6 Deprecation:**
  - Replaced `triangulation_creation.rs` with minimal deprecation harness
  - Prints clear deprecation notice and migration guidance
  - Directs users to `ci_performance_suite.rs` (CI) and `large_scale_performance.rs` (Phase 4)
  - File compiles, passes clippy and fmt checks
  - Ready for removal in next major release

- **Step 7 Consolidation:**
  - Deleted `memory_scaling.rs` and `triangulation_vs_hull_memory.rs` (zero external usage)
  - Removed benchmark entries from `Cargo.toml`
  - Updated `benches/README.md` to remove deleted benchmarks from table
  - Memory profiling consolidated in `profiling_suite.rs` (already comprehensive)
  - Phase 4 memory evaluation uses `large_scale_performance.rs`
  - Benchmarks compile successfully, all quality checks pass

- **Step 8 Deduplication:**
  - Removed duplicate `assign_neighbors` benchmarks from `microbenchmarks.rs` (2D-5D)
  - Removed functions: `benchmark_assign_neighbors_2d/3d/4d/5d` and legacy wrapper
  - Removed from criterion_group targets (4 dimensional + 1 legacy = 5 functions)
  - Updated module doc to direct users to `assign_neighbors_performance.rs`
  - Comprehensive `assign_neighbors` testing now centralized with distributions (random, grid, spherical) and scaling
  - File compiles, passes fmt, clippy, and spell check

- **Step 9 Phase 4 Elevation:**
  - Added 5D benchmark suite with small point counts [500, 1K] (~30-60 min)
  - Added configurable scaling for 4D via `BENCH_LARGE_SCALE` env var
  - Default runtime: ~2-3 hours (2D/3D/4D/5D, suitable for local development)
  - Large scale: ~4-6 hours (4D@10K, requires compute cluster)
  - Point count strategy: 1K-10K (2D/3D), 1K-3K (4D default), 500-1K (5D)
  - Complete dimensional coverage: 2D, 3D, 4D, 5D
  - All Phase 4 metrics covered: construction, memory, iteration, queries, validation

- **Step 10 Justfile Targets (historical):**
  - The `bench-phase4*` targets were later removed (2025-12-13).
  - Use `cargo bench --bench large_scale_performance` and `just compare-storage*` instead.

- **Step 11 GitHub Actions Workflows:**
  - Updated profiling-benchmarks.yml: memory_scaling → profiling_suite (memory_profiling)
  - Set development mode for tag pushes to keep runtime reasonable (~1-2 hours vs 4-6 hours)
  - Full production profiling only for manual dispatch or scheduled monthly runs
  - Verified benchmarks.yml and generate-baseline.yml use benchmark-utils (no changes needed)
  - All workflows now reference correct benchmark files after consolidation

- **Step 12 Documentation Updates:**
  - Updated docs/code_organization.md: Removed deleted benchmarks, added large_scale_performance.rs, marked triangulation_creation.rs as deprecated
  - Added phase4.md to documentation tree
  - benches/README.md already updated in earlier steps
  - CHANGELOG.md update deferred to next release

**Blockers:**

- None

**Next Session:**

- Continue with step 10: Add Phase 4 tooling to scripts/justfile
- Then step 11: Update GitHub Actions workflows
- Then step 12: Final documentation and changelog

---

### General Guidelines

- Keep this document updated as work progresses
- Check off items in the TODO list as they're completed
- Add any blockers or issues encountered in session notes
- Reference this document when picking up work after breaks
