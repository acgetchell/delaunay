# Performance Benchmarks

This directory contains performance benchmarks for the delaunay library, organized by purpose:

- **CI Regression Detection**: Fast benchmarks for every PR
- **Phase 4 SlotMap Evaluation**: Large-scale iteration and query performance
- **Comprehensive Profiling**: Deep analysis for optimization work
- **Algorithm Comparison**: Comparing different implementations
- **Specialized Benchmarks**: Focused testing of specific operations

## Benchmark Suite Overview

| Benchmark | Purpose | Scale | Runtime | Used By |
|-----------|---------|-------|---------|----------|
| `ci_performance_suite.rs` | **CI regression detection** | 10–50 vertices | ~5-10 min | CI workflows, baseline generation, performance summary |
| `circumsphere_containment.rs` | Predicate algorithm comparison | Random queries | ~5 min | Performance summary predicate subsection |
| `large_scale_performance.rs` | **Phase 4 SlotMap evaluation** | 1k–10k vertices | ~10-30 min (default); ~2-3 hours (BENCH_LARGE_SCALE=1) | Manual |
| `profiling_suite.rs` | Comprehensive profiling | 10³–10⁶ vertices | 1-2 hours | Monthly profiling, manual |
| `topology_guarantee_construction.rs` | Topology guarantee construction overhead | 2D–5D (small/medium point counts) | ~5–15 min | Manual |
| ~~`triangulation_creation.rs`~~ | ~~Simple construction~~ | ~~1000 vertices~~ | ~~N/A~~ | **DEPRECATED / REMOVED** |

### Benchmark Selection Guide

| Use Case | Benchmark | Command |
|----------|-----------|----------|
| CI regression check | `ci_performance_suite.rs` | `just bench-ci` or `cargo bench --profile perf --bench ci_performance_suite` |
| Release performance summary | `ci_performance_suite.rs` + `circumsphere_containment.rs` | `just bench-perf-summary` |
| Smoke-test benchmark harnesses | Workspace benches | `just bench-smoke` |
| Phase 4 SlotMap evaluation | `large_scale_performance.rs` | `cargo bench --profile perf --bench large_scale_performance` |
| Deep profiling (1-2 hours) | `profiling_suite.rs` | `cargo bench --profile perf --bench profiling_suite` |
| Memory analysis | `profiling_suite.rs` (memory groups) | `cargo bench --profile perf --bench profiling_suite -- memory_profiling` |
| Validation layer diagnostics | `profiling_suite.rs` (validation components) | `cargo bench --profile perf --bench profiling_suite -- validation_components` |
| Algorithm comparison | `circumsphere_containment.rs` | `cargo bench --profile perf --bench circumsphere_containment` |
| Topology guarantee overhead | `topology_guarantee_construction.rs` | See section below |

## Running Benchmarks

Benchmark discussions below are sorted lexicographically by benchmark name.

### Cargo Profiles

Benchmark commands that produce performance data use the `perf` profile:

```bash
just bench
just bench-ci
just bench-baseline
just bench-compare
just bench-perf-summary
cargo bench --profile perf --bench ci_performance_suite
```

Use smoke or compile-only recipes for fast validation when you do not need
performance data:

```bash
just bench-smoke
just bench-compile
just bench-test-compile
```

The `perf` profile inherits from release and restores ThinLTO with
`codegen-units = 1`. It is intentionally separate from the default release
profile used by `just ci` for comprehensive compile/test validation. `just ci`
does not need the `perf` profile because it is catching correctness, lint,
documentation, example, and build errors rather than publishing benchmark
numbers. Do not treat `bench-smoke` output as performance data.

### CI Performance Suite (`ci_performance_suite.rs`) (primary)

```bash
# Run CI performance suite benchmarks for 2D, 3D, 4D, and 5D
cargo bench --profile perf --bench ci_performance_suite
```

The CI Performance Suite is the primary benchmarking suite used for automated performance-regression testing and generated performance summaries:

- **Purpose**: Fast performance regression detection for regular CI/CD
- **Dimensions**: 2D–5D triangulations
- **Vertex counts**: 10, 25, 50 per dimension
- **Runtime**: ~5–10 minutes (hardware dependent)
- **Frequency**: Every PR and push to main
- **Integration**: `.github/workflows/benchmarks.yml` and `.github/workflows/generate-baseline.yml` (GitHub Actions)

### Circumsphere Containment (`circumsphere_containment.rs`)

```bash
# Run all circumsphere benchmarks
cargo bench --profile perf --bench circumsphere_containment

# Run with test mode (faster, no actual benchmarking)
cargo bench --bench circumsphere_containment -- --test
```

#### Methods Compared

1. **insphere**: Standard determinant-based method (most numerically stable)
2. **insphere_distance**: Distance-based method using explicit circumcenter calculation
3. **insphere_lifted**: Matrix determinant method with lifted paraboloid approach

#### Performance Results

📊 **[View Detailed Performance Results](PERFORMANCE_RESULTS.md)**

Comprehensive performance benchmarks, analysis, and recommendations have been moved to a dedicated file for easier
maintenance and automated updates. Circumsphere performance remains a dedicated subsection because these predicates
exercise `la-stack` code paths that are important to tune independently.

##### Quick Summary

- **Best Performance**: `insphere_lifted` method (fastest across all dimensions)
- **Best Stability**: `insphere` method (most numerically reliable)
- **Educational**: `insphere_distance` method (transparent algorithm)

##### Performance Data Maintenance

Performance results are automatically updated using the benchmark infrastructure:

```bash
# Prerequisite: install uv (e.g., `pipx install uv` or `brew install uv`)

# Generate updated performance summary
uv run benchmark-utils generate-summary

# Generate summary with fresh perf-profile benchmark data
uv run benchmark-utils generate-summary --run-benchmarks --profile perf

# Compare current performance against baseline
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt

# Compare two tags without re-running benchmarks (downloads baseline artifacts via gh)
uv run benchmark-utils compare-tags --old-tag vX.Y.Z --new-tag vA.B.C
```

#### Benchmark Structure

The `circumsphere_containment.rs` benchmark includes:

- **Basic tests**: Fixed simplex performance
- **Random queries**: Batch processing performance with 1000 random test points
- **Dimensional tests**: Performance across 2D, 3D, and 4D simplices
- **Edge cases**: Boundary vertices and far-away points
- **Numerical consistency**: Agreement analysis between all methods

### Large-scale Performance (`large_scale_performance.rs`) (Phase 4 SlotMap evaluation)

**Status:** Active (consolidation plan archived)  
**Documentation:** See [docs/archive/phase4.md](../docs/archive/phase4.md) for the archived consolidation plan and progress log

#### Purpose

Phase 4 aims to abstract SlotMap usage behind traits to enable swapping implementations
(SlotMap → DenseSlotMap → HopSlotMap) without code changes, targeting **10-15% iteration
performance improvement**.

#### Key Metrics

1. **Iteration Performance** (10-15% improvement target)
   - Full vertex/cell traversals
   - Neighbor-following patterns
   - Filtered iteration

2. **Memory Efficiency**
   - Peak RSS during construction
   - Per-element footprint

3. **Cache Locality** (5-10% cache miss reduction target)
   - Sequential vs random access
   - BFS traversal patterns

4. **Query Performance** (maintain or improve)
   - Key lookups
   - Neighbor queries

#### Running Phase 4 Benchmarks

```bash
# Run large-scale performance benchmarks (Phase 4 primary)
cargo bench --profile perf --bench large_scale_performance

# Run specific test categories
cargo bench --profile perf --bench large_scale_performance -- "construction/3D"
cargo bench --profile perf --bench large_scale_performance -- "queries/neighbors"
cargo bench --profile perf --bench large_scale_performance -- "iteration/vertices"

# Storage backend comparison
just compare-storage       # Compare SlotMap vs DenseSlotMap (~4-6 hours)
just compare-storage-large # Large scale comparison (~8-12 hours, compute cluster)
```

**Note:** `large_scale_performance.rs` was specifically designed for Phase 4 evaluation.
It measures iteration speed, memory usage, query performance, and validation - all critical
for SlotMap comparison.

### Profiling Suite (`profiling_suite.rs`) (comprehensive)

```bash
# Run comprehensive profiling suite (1-2 hours, 10³-10⁶ vertices)
cargo bench --profile perf --bench profiling_suite --features count-allocations

# Development mode (faster, reduced scale)
PROFILING_DEV_MODE=1 cargo bench --profile perf --bench profiling_suite --features count-allocations

# Override measurement times for faster iteration
BENCH_MEASUREMENT_TIME=10 cargo bench --profile perf --bench profiling_suite --features count-allocations

# Run specific profiling categories
cargo bench --profile perf --bench profiling_suite --features count-allocations -- triangulation_scaling
cargo bench --profile perf --bench profiling_suite --features count-allocations -- memory_profiling
cargo bench --profile perf --bench profiling_suite --features count-allocations -- query_latency
cargo bench --profile perf --bench profiling_suite --features count-allocations -- algorithmic_bottlenecks
cargo bench --profile perf --bench profiling_suite --features count-allocations -- validation_components

# Run only memory profiling group (useful for focused analysis)
cargo bench --profile perf --bench profiling_suite --features count-allocations -- "memory_profiling"
```

The **Profiling Suite** provides comprehensive performance analysis for optimization work:

- **Purpose**: Comprehensive performance analysis for optimization work
- **Runtime**: 1-2 hours (full production mode)
- **Scale**: Large point counts (10³ to 10⁶ points)
- **Frequency**: Monthly scheduled + manual triggers + release tags
- **Integration**: `.github/workflows/profiling-benchmarks.yml`

**📊 Key Features & Improvements**:

- **Optimized Query Benchmarks**: Precomputes simplex vertices outside inner loops to reduce per-iteration allocations
- **Enhanced Memory Profiling**: Reports mean, max, and 95th percentile allocation statistics for better spike detection
- **Complete Dimensional Coverage**: Memory profiling and triangulation scaling for all dimensions 2D-5D
- **Environment Variable Control**: Use `BENCH_MEASUREMENT_TIME` to override default measurement times for CI tuning
- **Eliminated Double Point Generation**: Sample points generated once per benchmark to reduce setup noise

**⚠️ Performance & Hardware Considerations**:

- **Feature Overhead**: The `count-allocations` feature can materially slow benchmark runs (20-50% overhead) and increase memory usage for allocation tracking
- **Hardware Requirements**: Recommend ≥16 GB RAM and ≥4 CPU cores to prevent timeouts during large-scale runs (10⁶ vertices)
- **CI/Local Timeouts**: Without adequate resources, runs may exceed typical CI timeouts (30-60 minutes).
  Note that 10⁶ vertex benchmarks can take several hours and are not suitable for standard CI environments
- **Development Mode**: Use `PROFILING_DEV_MODE=1` for faster iteration during optimization work
- **Flexible Timing**: Use `BENCH_MEASUREMENT_TIME=N` to set measurement time in seconds for all benchmark groups

#### Automated Profiling Triggers

```bash
# Manual trigger via GitHub Actions UI:
# 1. Go to Actions tab → "Comprehensive Profiling Benchmarks"
# 2. Click "Run workflow" 
# 3. Select mode: development (faster) or production (full scale)
# 4. Optional: Filter specific benchmarks

# Automatic triggers:
# - Monthly: First Sunday of each month at 2 AM UTC  
# - Release tags: Every version tag (v*.*.*) for baseline generation
```

#### Profiling Results and Artifacts

- **Profiling Results**: Available as GitHub Actions artifacts for 30 days
- **Profiling Baselines**: Release-tagged baselines kept for 90 days
- **Memory Analysis**: Detailed allocation tracking with `count-allocations` feature
- **HTML Reports**: Criterion-generated performance reports with statistical analysis

#### Development Workflow

1. **Regular Development**: Use CI Performance Suite for quick feedback
2. **Optimization Work**: Trigger Profiling Suite manually in development mode
3. **Release Preparation**: Full profiling suite runs automatically on version tags
4. **Performance Monitoring**: Monthly automated runs track long-term trends

### Topology Guarantee Construction (`topology_guarantee_construction.rs`) (manual)

This benchmark compares construction cost under `TopologyGuarantee::Pseudomanifold`,
`TopologyGuarantee::PLManifold` (incremental), and `TopologyGuarantee::PLManifoldStrict`
across 2D–5D.
It is intended for **manual** runs (not used by CI).

```bash
# Run the full suite (recommended: disable plot generation)
cargo bench --profile perf --bench topology_guarantee_construction -- --noplot

# Filter by dimension
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/2d"
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/3d"
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/4d"
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/5d"
```

### All Benchmarks

```bash
# Run all available benchmarks (includes CI + profiling suites + manual comparisons)
just bench

# Run all benchmarks with memory tracking (direct cargo command)
cargo bench --profile perf --features count-allocations

# Compile-only check (useful for CI validation without running benchmarks)
just bench-compile
```

**💡 Targeted Benchmark Runs**: Use `cargo bench --profile perf --bench profiling_suite -- --help` to see available filters and Criterion flags for scoping
long-running benchmarks. This helps target specific test categories or adjust measurement parameters for faster iteration.
