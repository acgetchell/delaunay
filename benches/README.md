# Performance Benchmarks

This directory contains performance benchmarks for the delaunay library, including circumsphere containment algorithms,
triangulation data structure operations, and small-scale triangulation performance testing.

## Running Benchmarks

### Circumsphere Containment Benchmarks

```bash
# Run all benchmarks
cargo bench --bench circumsphere_containment

# Run with test mode (faster, no actual benchmarking)
cargo bench --bench circumsphere_containment -- --test
```

### CI Performance Suite (primary)

```bash
# Run CI performance suite benchmarks for 2D, 3D, 4D, and 5D (optimized for CI)
cargo bench --bench ci_performance_suite
```

The CI Performance Suite is the primary benchmarking suite used for automated performance-regression testing:

- **Dimensions**: 2D–5D triangulations
- **Point counts**: 10, 25, 50 per dimension
- **Runtime**: ~5–10 minutes (hardware dependent)
- **Integration**: Used by GitHub Actions for automated baseline generation and comparison

### Profiling Suite (comprehensive)

```bash
# Run comprehensive profiling suite (1-2 hours, 10³-10⁶ points)
cargo bench --bench profiling_suite --features count-allocations

# Development mode (faster, reduced scale)
PROFILING_DEV_MODE=1 cargo bench --bench profiling_suite --features count-allocations

# Override measurement times for faster iteration
BENCH_MEASUREMENT_TIME=10 cargo bench --bench profiling_suite --features count-allocations

# Run specific profiling categories
cargo bench --bench profiling_suite --features count-allocations -- triangulation_scaling
cargo bench --bench profiling_suite --features count-allocations -- memory_profiling
cargo bench --bench profiling_suite --features count-allocations -- query_latency
cargo bench --bench profiling_suite --features count-allocations -- algorithmic_bottlenecks
```

The **Profiling Suite** provides comprehensive performance analysis for optimization work:

- **Large-scale triangulation performance** (10³ to 10⁶ points across multiple decades)
- **Complete dimensional coverage** (2D through 5D triangulation scaling)
- **Multiple point distributions** (random, grid, Poisson disk)
- **Memory allocation tracking** with 95th percentile statistics (requires `--features count-allocations`)
- **Query latency analysis** (circumsphere tests, optimized with precomputed simplices)
- **Algorithmic bottleneck identification** (boundary facets, convex hull operations)

**📊 Key Features & Improvements**:

- **Optimized Query Benchmarks**: Precomputes simplex vertices outside inner loops to reduce per-iteration allocations
- **Enhanced Memory Profiling**: Reports mean, max, and 95th percentile allocation statistics for better spike detection
- **Complete Dimensional Coverage**: Memory profiling and triangulation scaling for all dimensions 2D-5D
- **Environment Variable Control**: Use `BENCH_MEASUREMENT_TIME` to override default measurement times for CI tuning
- **Eliminated Double Point Generation**: Sample points generated once per benchmark to reduce setup noise

**⚠️ Performance & Hardware Considerations**:

- **Feature Overhead**: The `count-allocations` feature can materially slow benchmark runs (20-50% overhead) and increase memory usage for allocation tracking
- **Hardware Requirements**: Recommend ≥16GB RAM and ≥4 CPU cores to prevent timeouts during large-scale runs (10⁶ points)
- **CI/Local Timeouts**: Without adequate resources, runs may exceed typical CI timeouts (30-60 minutes) or cause local system slowdowns
- **Development Mode**: Use `PROFILING_DEV_MODE=1` for faster iteration during optimization work
- **Flexible Timing**: Use `BENCH_MEASUREMENT_TIME=N` to set measurement time in seconds for all benchmark groups

**⚠️ Note**: This suite is designed for optimization work and takes significantly longer than CI benchmarks.

### All Benchmarks

```bash
# Run all available benchmarks (includes CI + profiling suites)
cargo bench

# Run all benchmarks with memory tracking
cargo bench --features count-allocations

# Compile-only check (useful for CI validation without running benchmarks)
cargo bench --no-run
```

**💡 Targeted Benchmark Runs**: Use `cargo bench --bench profiling_suite -- --help` to see available filters and Criterion flags for scoping
long-running benchmarks. This helps target specific test categories or adjust measurement parameters for faster iteration.

## Methods Compared

1. **insphere**: Standard determinant-based method (most numerically stable)
2. **insphere_distance**: Distance-based method using explicit circumcenter calculation
3. **insphere_lifted**: Matrix determinant method with lifted paraboloid approach

## Performance Results

📊 **[View Detailed Performance Results](PERFORMANCE_RESULTS.md)**

Comprehensive performance benchmarks, analysis, and recommendations have been moved to a dedicated file for easier maintenance and automated updates.

### Quick Summary

- **Best Performance**: `insphere_lifted` method (fastest across all dimensions)
- **Best Stability**: `insphere` method (most numerically reliable)
- **Educational**: `insphere_distance` method (transparent algorithm)

### Performance Data Maintenance

Performance results are automatically updated using the benchmark infrastructure:

```bash
# Prerequisite: install uv (e.g., `pipx install uv` or `brew install uv`)

# Generate updated performance summary
uv run performance-summary-utils generate

# Generate summary with fresh benchmark data
uv run performance-summary-utils generate --run-benchmarks

# Compare current performance against baseline (separate utility)
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
```

**Note**: The performance results file contains detailed benchmark data, version comparisons,
optimization analysis, and recommendations that were previously embedded in this README.

## Benchmark Structure

The `circumsphere_containment.rs` benchmark includes:

- **Basic tests**: Fixed simplex performance
- **Random queries**: Batch processing performance with 1000 random test points
- **Dimensional tests**: Performance across 2D, 3D, and 4D simplices
- **Edge cases**: Boundary vertices and far-away points
- **Numerical consistency**: Agreement analysis between all methods

## GitHub Actions Integration

### Workflow Separation Strategy

The project uses a two-tier benchmarking approach to balance comprehensive analysis with CI efficiency:

#### 1. **CI Performance Suite** (`ci_performance_suite.rs`)

- **Purpose**: Fast performance regression detection for regular CI/CD
- **Runtime**: ~5-10 minutes
- **Scale**: Small point counts (10, 25, 50 points)
- **Frequency**: Every PR and push to main
- **Integration**: `.github/workflows/benchmarks.yml` and `.github/workflows/generate-baseline.yml`

#### 2. **Profiling Suite** (`profiling_suite.rs`)

- **Purpose**: Comprehensive performance analysis for optimization work
- **Runtime**: 1-2 hours (full production mode)
- **Scale**: Large point counts (10³ to 10⁶ points)
- **Frequency**: Monthly scheduled + manual triggers + release tags
- **Integration**: `.github/workflows/profiling-benchmarks.yml`

### Automated Profiling Triggers

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

### Profiling Results and Artifacts

- **Profiling Results**: Available as GitHub Actions artifacts for 30 days
- **Profiling Baselines**: Release-tagged baselines kept for 90 days
- **Memory Analysis**: Detailed allocation tracking with `count-allocations` feature
- **HTML Reports**: Criterion-generated performance reports with statistical analysis

### Development Workflow

1. **Regular Development**: Use CI Performance Suite for quick feedback
2. **Optimization Work**: Trigger Profiling Suite manually in development mode
3. **Release Preparation**: Full profiling suite runs automatically on version tags
4. **Performance Monitoring**: Monthly automated runs track long-term trends
