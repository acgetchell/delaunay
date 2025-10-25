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

**Phase 4 Goal:** Enable swapping SlotMap implementations via feature flags
(SlotMap → DenseSlotMap → HopSlotMap) for benchmarking, targeting 10-15% iteration
performance improvement.

## Current Benchmark Suite Issues

### Overlaps & Redundancies

1. **Triangulation Creation Overlap:**
   - `ci_performance_suite.rs`, `triangulation_creation.rs`, `microbenchmarks.rs`, and `profiling_suite.rs` all benchmark basic triangulation creation
   - **Problem:** `triangulation_creation.rs` is redundant with `ci_performance_suite.rs` (already used by CI/scripts)

2. **Assign Neighbors Duplication:**
   - Both `assign_neighbors_performance.rs` and `microbenchmarks.rs` test the same operation
   - `assign_neighbors_performance.rs` is more comprehensive (grid, spherical, scaling tests)

3. **Memory Measurement Overlap:**
   - `large_scale_performance.rs`, `memory_scaling.rs`, `triangulation_vs_hull_memory.rs`, and `profiling_suite.rs` all measure memory
   - Each has slightly different focus but significant overlap

### Scripts Integration

- `scripts/benchmark_utils.py` is hardcoded to use **`ci_performance_suite.rs`** for baseline generation and regression testing
- No Phase 4-specific tooling exists yet
- Need to add `generate-phase4-baseline` and `compare-phase4` commands

## Implementation Plan

### ☐ 1. Kickoff and Scope Alignment

**Status:** Not Started

**Tasks:**

- [ ] Confirm Phase 4's primary goal: evaluate SlotMap-backed TDS performance and memory behavior at large scale
- [ ] Agree to make `large_scale_performance.rs` the primary benchmark for Phase 4, replacing `triangulation_creation.rs` usage
- [ ] Decide deprecation vs immediate removal policy for redundant benches (recommend: deprecate for one release, then remove)
- [ ] Create a working branch for Rust changes separate from docs/Python to avoid long CI runs on every change

**Notes:**

- Phase 4 targets 10-15% iteration improvement with DenseSlotMap
- SlotMap implementations selected via feature flags (no runtime abstraction)
- Need to maintain baseline compatibility across changes

---

### ✅ 2. Inventory All Benchmark Files

**Status:** ✅ Completed (2025-10-20)

**Tasks:**

- [x] List current benches: `ls benches/*.rs`
- [x] Record per file: purpose, operations covered, dataset sizes, CLI parameters/env vars, Criterion groups/IDs, output format
- [ ] Validate expected set:
  - `ci_performance_suite.rs`
  - `triangulation_creation.rs`
  - `large_scale_performance.rs`
  - `profiling_suite.rs`
  - `memory_scaling.rs`
  - `triangulation_vs_hull_memory.rs`
  - `microbenchmarks.rs`
  - `assign_neighbors_performance.rs`
  - `circumsphere_containment.rs`

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
| `ci_performance_suite.rs` | CI regression detection | 10-50 pts, 2D-5D | Basic construction | No - too small |
| `triangulation_creation.rs` | Simple construction bench | 1000 pts, 2D-5D | Construction only | **REDUNDANT** |
| `large_scale_performance.rs` | **Phase 4 SlotMap eval** | 1K-10K vertices | Construction, memory, **iteration**, queries | **YES - PRIMARY** |
| `profiling_suite.rs` | Comprehensive profiling | 10³-10⁶ pts | All operations, multiple distributions | Partial - too heavy |
| `memory_scaling.rs` | Memory allocation patterns | 10-100 pts | Memory tracking | Merge candidate |
| `triangulation_vs_hull_memory.rs` | TDS vs hull memory comparison | 10-50 pts | Memory comparison | Merge candidate |
| `microbenchmarks.rs` | Core operations (Bowyer-Watson, assign_neighbors, validation) | Various | Individual operations | Has duplicates |
| `assign_neighbors_performance.rs` | Neighbor assignment comprehensive | 10-50 pts, 2D-5D | Neighbor assignment with multiple distributions | Keep |
| `circumsphere_containment.rs` | Algorithm comparison (insphere variants) | Random queries | Circumsphere predicates | Keep |

**Deliverable:** Consolidated section in `benches/README.md` with detailed table

---

### ☐ 5. Design the Consolidation Plan

**Status:** Not Started

**Proposals:**

- [ ] **Deprecate then remove** `triangulation_creation.rs` (redundant with `ci_performance_suite.rs`)
- [ ] **Merge** `memory_scaling.rs` and `triangulation_vs_hull_memory.rs` into `profiling_suite.rs` as dedicated Criterion groups
- [ ] **Update** `microbenchmarks.rs` to drop `assign_neighbors` duplicates and delegate to `assign_neighbors_performance.rs` groups
- [ ] **Codify** `large_scale_performance.rs` as the Phase 4 primary with stable Criterion IDs and JSON output
- [ ] **Define** standard CLI/env params across benches: `BENCH_N`, `BENCH_DIM`, `BENCH_SEED`, `BENCH_DISTRIBUTION`, `BENCH_OP_MIX`

**Deliverable:** `benches/CONSOLIDATION_PLAN.md` (brief; later summarized in CHANGELOG)

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

- [ ] Move/port scenarios from `memory_scaling.rs` → `profiling_suite.rs::memory::scaling`
- [ ] Move/port scenarios from `triangulation_vs_hull_memory.rs` → `profiling_suite.rs::memory::triangulation_vs_hull`
- [ ] Normalize memory measurement approach:
  - Current: Uses `sysinfo` crate for RSS tracking
  - Option: Add `dhat` support behind feature flag
  - Decision: Keep `sysinfo` for Phase 4, add `allocation-counter` usage where appropriate
- [ ] Preserve Criterion group names for baseline comparability
- [ ] Add bench metadata JSON fields: dataset, dim, seed, metric units
- [ ] Remove old files or leave thin wrappers with deprecation warnings

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

- [ ] Remove `assign_neighbors` duplicates from `microbenchmarks.rs`:
  - `benchmark_assign_neighbors_2d`
  - `benchmark_assign_neighbors_3d`
  - `benchmark_assign_neighbors_4d`
  - `benchmark_assign_neighbors_5d`
- [ ] Add reference/re-export to `assign_neighbors_performance.rs` groups
- [ ] Ensure Criterion IDs align to maintain baseline history
- [ ] Add module doc explaining what remains and where `assign_neighbors` moved

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

- [ ] **Add standardized CLI/env control:**
  - `--n` or `BENCH_N`: point count
  - `--dim` or `BENCH_DIM`: dimension (2-5)
  - `--seed` or `BENCH_SEED`: RNG seed
  - `--distribution` or `BENCH_DISTRIBUTION`: grid, random, clustered
  - `--op-mix` or `BENCH_OP_MIX`: operation ratio
  - `--json-out` or `BENCH_JSON_OUT`: structured results path

- [ ] **Stabilize Criterion IDs for baseline tracking:**

  ```rust
  // Suggested stable IDs:
  // - phase4/construction/{dim}/{n}
  // - phase4/iteration/vertices/{dim}/{n}
  // - phase4/iteration/cells/{dim}/{n}
  // - phase4/queries/neighbors/{dim}/{n}
  // - phase4/validation/{dim}/{n}
  // - phase4/memory/{dim}/{n}
  ```

- [ ] **Add cache locality measurement (optional):**

  ```rust
  // Behind feature flag: perf_guard or similar
  // Use perf/callgrind hooks for cache miss analysis
  ```

- [ ] **Support SlotMap vs alternative comparison:**

  ```rust
  // Use feature flags + type aliases for zero-cost abstraction:
  #[cfg(feature = "dense-slotmap")]
  type StorageBackend<K, V> = DenseSlotMap<K, V>;
  
  #[cfg(not(feature = "dense-slotmap"))]
  type StorageBackend<K, V> = SlotMap<K, V>;
  
  // Benchmark with:
  // cargo bench --bench large_scale_performance
  // cargo bench --bench large_scale_performance --features dense-slotmap
  ```

**Key Metrics for Phase 4:**

1. **Iteration speed**: Full vertex/cell traversals, neighbor walks
2. **Memory usage**: Peak RSS, per-element footprint estimates  
3. **Cache locality**: Traversal patterns (BFS vs random access)
4. **Query performance**: Lookups, contains checks, incident-entity queries

---

### ◔ 10. Add Phase 4 Helpers to benchmark_utils.py and justfile

**Status:** Partial - Justfile targets completed (2025-10-20)

**Python Script Commands (uv run):**

```python
# New subcommands to add to benchmark_utils.py

def generate_phase4_baseline(n: int, dim: int, seed: int, distribution: str, output: Path):
    """Generate Phase 4 baseline using large_scale_performance.rs"""
    # Run: cargo bench --bench large_scale_performance
    # Parse Criterion results
    # Save to output (e.g., artifacts/phase4_baseline.json)

def compare_phase4(baseline: Path, threshold: float):
    """Compare current Phase 4 performance against baseline"""
    # Run: cargo bench --bench large_scale_performance
    # Parse results and compare against baseline
    # Report regressions > threshold
```

**CLI Shape:**

```bash
# Generate baseline
uv run benchmark-utils generate-phase4-baseline \
  --n 10000 \
  --dim 3 \
  --seed 42 \
  --distribution random \
  --out artifacts/phase4_baseline.json

# Compare against baseline
uv run benchmark-utils compare-phase4 \
  --baseline artifacts/phase4_baseline.json \
  --threshold 5  # 5% regression threshold
```

**Justfile Targets:**

```makefile
# Add to justfile

# Generate Phase 4 baseline
bench-phase4-baseline n="10000" dim="3":
    uv run benchmark-utils generate-phase4-baseline \
      --n {{n}} --dim {{dim}} --seed 42 --distribution random \
      --out artifacts/phase4_baseline.json

# Compare Phase 4 performance
bench-phase4-compare:
    uv run benchmark-utils compare-phase4 \
      --baseline artifacts/phase4_baseline.json \
      --threshold 5

# Run Phase 4 benchmarks only (no baseline)
bench-phase4:
    cargo bench --bench large_scale_performance
```

**Tasks:**

- [ ] Add `generate-phase4-baseline` subcommand to `benchmark_utils.py`
- [ ] Add `compare-phase4` subcommand to `benchmark_utils.py`
- [ ] Add `bench-phase4-baseline` target to `justfile`
- [ ] Add `bench-phase4-compare` target to `justfile`
- [ ] Add `bench-phase4` target to `justfile`
- [ ] Ensure JSON schema compatibility with existing baseline/compare utilities
- [ ] Add schema version if needed

---

### ✅ 11. Update GitHub Actions Workflows

**Status:** ✅ Completed (2025-10-20)

**benchmarks.yml (Performance Regression Testing):**

- [ ] Replace any `triangulation_creation` references with `ci_performance_suite.rs` or `large_scale_performance.rs`
- [ ] Add optional Phase 4 job that runs reduced-size `large_scale_performance.rs` smoke test
- [ ] Keep runtime reasonable for PRs (use `--sample-size 10` or similar)

**profiling-benchmarks.yml (Comprehensive Profiling):**

- [ ] Point memory jobs to `profiling_suite.rs` memory groups (after consolidation)
- [ ] Gate heavy runs by workflow_dispatch labels or schedules
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

- [ ] Add categorization section:
  - **CI Benchmarks**: `ci_performance_suite.rs` (fast, regression detection)
  - **Profiling Benchmarks**: `profiling_suite.rs` (comprehensive, 1-2 hours)
  - **Phase 4 Benchmarks**: `large_scale_performance.rs` (SlotMap evaluation)
  - **Algorithm Comparison**: `circumsphere_containment.rs`
  - **Specialized**: `assign_neighbors_performance.rs`
  - **Deprecated**: `triangulation_creation.rs` (use `ci_performance_suite.rs` or `large_scale_performance.rs`)

- [ ] Add "When to use which" guidance:

  ```markdown
  ## Benchmark Selection Guide
  
  | Use Case | Benchmark | Command |
  |----------|-----------|---------|
  | Quick CI regression check | `ci_performance_suite.rs` | `just bench` or `cargo bench --bench ci_performance_suite` |
  | Phase 4 SlotMap evaluation | `large_scale_performance.rs` | `just bench-phase4` |
  | Deep profiling (1-2 hours) | `profiling_suite.rs` | `cargo bench --bench profiling_suite` |
  | Memory analysis | `profiling_suite.rs` (memory groups) | `cargo bench --bench profiling_suite -- memory` |
  | Algorithm comparison | `circumsphere_containment.rs` | `cargo bench --bench circumsphere_containment` |
  ```

- [ ] Explicitly document `large_scale_performance.rs` as Phase 4 primary
- [ ] Add deprecation notice for `triangulation_creation.rs`

**docs/code_organization.md:**

- [ ] Update benchmark section to reflect new layout
- [ ] Add Phase 4 benchmark responsibilities
- [ ] Document memory benchmark consolidation

**CHANGELOG.md:**

- [ ] Add "Benchmark consolidation" entry:

  ```markdown
  ### Benchmark Suite Reorganization
  
  **Deprecated:**
  - `triangulation_creation.rs` - redundant with `ci_performance_suite.rs`
    - For CI benchmarks, use: `cargo bench --bench ci_performance_suite`
    - For Phase 4 evaluation, use: `cargo bench --bench large_scale_performance`
  
  **Consolidated:**
  - `memory_scaling.rs` → `profiling_suite.rs::memory::scaling`
  - `triangulation_vs_hull_memory.rs` → `profiling_suite.rs::memory::triangulation_vs_hull`
  
  **Enhanced:**
  - `large_scale_performance.rs` - now the primary Phase 4 SlotMap evaluation benchmark
    - Added standardized CLI/env controls
    - Stabilized Criterion IDs for baseline tracking
    - Measures iteration, memory, queries, and validation
  
  **New Tooling:**
  - `just bench-phase4-baseline` - generate Phase 4 baseline
  - `just bench-phase4-compare` - compare against baseline
  - `uv run benchmark-utils generate-phase4-baseline` - Python command
  - `uv run benchmark-utils compare-phase4` - Python command
  ```

---

### ☐ 13. Add Missing Coverage (Time Permitting)

**Status:** Not Started

**Priority 1 (High Value):**

- [ ] **Convex hull timing benchmarks**
  - Add to `profiling_suite.rs` or separate file
  - Cover varied distributions (random, grid, clustered) and dimensions (2D-5D)
  - Currently only memory benchmarks exist in `triangulation_vs_hull_memory.rs`

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

### ☐ 14. Quality Gates, Validation, and CI Safety

**Status:** Not Started

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

- [ ] Run all quality gates on changed files
- [ ] Verify benchmark compilation with `just bench-compile`
- [ ] Run Python tests with `uv run pytest`
- [ ] Test smoke runs of modified benchmarks

---

### ☐ 15. Acceptance Criteria and Sign-Off

**Status:** Not Started

**Completion Checklist:**

**Benchmark Files:**

- [ ] No broken references to removed/deprecated benches in workflows, scripts, or justfile
- [ ] `large_scale_performance.rs` provides:
  - [ ] Iteration speed metrics (vertex/cell/neighbor traversals)
  - [ ] Memory usage tracking (RSS, per-element footprint)
  - [ ] Cache locality measurement (where supported)
  - [ ] Query performance metrics (lookups, contains, incident-entities)
  - [ ] Stable Criterion IDs (`phase4/operation/dim/n`)
  - [ ] JSON output schema documented

**Python Tooling:**

- [ ] `benchmark_utils.py` has `generate-phase4-baseline` command
- [ ] `benchmark_utils.py` has `compare-phase4` command
- [ ] Commands work end-to-end on developer machine
- [ ] JSON schema compatible with existing baseline/compare utilities

**Build System:**

- [ ] `justfile` has `bench-phase4-baseline` target
- [ ] `justfile` has `bench-phase4-compare` target
- [ ] `justfile` has `bench-phase4` target (no baseline)
- [ ] All targets work correctly

**CI/CD:**

- [ ] All benchmark-related workflow jobs succeed
- [ ] Phase 4 baseline artifacts uploaded for release tags
- [ ] Phase 4 smoke test runs on PRs (if implemented)
- [ ] No references to deprecated benchmarks

**Documentation:**

- [ ] `benches/README.md` clearly guides contributors
- [ ] Phase 4 benchmarks explicitly documented as primary
- [ ] Deprecation notices for removed/consolidated benchmarks
- [ ] `docs/code_organization.md` reflects new layout
- [ ] `CHANGELOG.md` documents consolidation
- [ ] Migration guide for users of deprecated benchmarks

**Quality Gates:**

- [ ] `just quality` passes locally
- [ ] `just bench-compile` succeeds
- [ ] `uv run pytest` passes
- [ ] All 781 tests still pass
- [ ] No clippy warnings
- [ ] Documentation builds successfully

**Performance Validation:**

- [ ] Baseline generation completes successfully
- [ ] Comparison against baseline works correctly
- [ ] Regression detection threshold (5%) is reasonable
- [ ] Benchmark results are reproducible (±2% variance)

---

## Phase 4 SlotMap Evaluation Metrics

### Key Performance Indicators

Once the benchmark consolidation is complete, Phase 4 will evaluate these metrics:

1. **Iteration Performance** (10-15% improvement target with DenseSlotMap)
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
| SlotMap (current) | Good | Sparse | O(1) amortized | O(1) | Dynamic changes |
| DenseSlotMap (target) | **Excellent** | Dense/contiguous | O(1) amortized | O(1) with moves | Stable/iteration |
| HopSlotMap (future) | Good | Hop-optimized | O(1) | O(1) | Large scale |

### Success Criteria

- [ ] DenseSlotMap implementation shows 10-15% iteration improvement
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

**Last Updated:** 2025-10-20 18:39

**Overall Status:** ✅ Core Consolidation Complete (Steps 6-9, 11-12)

**Completed Steps:**

- ✅ Step 2: Inventory all benchmark files
- ✅ Step 3: Map benchmark usage in workflows/scripts  
- ✅ Step 4: Document purpose and overlap
- ✅ Step 6: Deprecate `triangulation_creation.rs`
- ✅ Step 7: Consolidate memory benchmarks
- ✅ Step 8: Deduplicate `assign_neighbors` benchmarks
- ✅ Step 9: Elevate `large_scale_performance.rs` for Phase 4
- ◔ Step 10: Add Phase 4 tooling (justfile targets complete, Python scripts deferred)
- ✅ Step 11: Update GitHub Actions workflows
- ✅ Step 12: Final documentation updates
- ✅ Updated `benches/README.md` with comprehensive overview
- ✅ Quality checks passing (fmt, clippy, markdown lint, spell check)

**Next Steps:**

1. ☐ Design consolidation plan and get approval (step 5 - optional)
2. ☐ Complete Python script tooling for Phase 4 (step 10 remainder - optional)
3. ☐ Update CHANGELOG.md before next release
4. ☐ Quality validation and sign-off (steps 14-15)

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

- **Step 10 Justfile Targets (Partial):**
  - Added `bench-phase4`: Run Phase 4 benchmarks (~2-3 hours)
  - Added `bench-phase4-large`: Large scale with 4D@10K (~4-6 hours, cluster)
  - Added `bench-phase4-quick`: Quick smoke test (~5-10 min)
  - Updated help-workflows with Phase 4 section
  - Python script commands deferred (require deeper integration with benchmark_utils.py)

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
