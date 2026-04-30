# Code Organization Guide

This document provides a comprehensive guide to the delaunay project's code organization, from the overall project architecture to detailed individual module patterns.

## Table of Contents

- [Project Structure](#project-structure)
  - [Complete Directory Tree](#complete-directory-tree)
  - [Architecture Overview](#architecture-overview)
  - [Architectural Principles](#architectural-principles)
- [Module Organization Patterns](#module-organization-patterns)
  - [Canonical Section Sequence](#canonical-section-sequence)
  - [Comment Separators](#comment-separators)
  - [Section-by-Section Analysis](#section-by-section-analysis)
  - [Module-Specific Variations](#module-specific-variations)
  - [Key Conventions](#key-conventions)

---

## Project Structure

The delaunay project follows a standard Rust library structure with additional tooling for computational geometry research.

### Complete Directory Tree

> **Tip**: Generate this tree in CI
>
> ```bash
> # Requires tree command (install with: brew install tree or apt-get install tree)
> git --no-pager ls-files | LC_ALL=C sort | \
>   LC_ALL=C tree -a --charset UTF-8 --dirsfirst --noreport \
>     -I 'target|.git|**/*.png|**/*.svg' -F --fromfile
>
> # Alternative using find (when tree is not available):
> find . -type f \( -name "*.rs" -o -name "*.md" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) | LC_ALL=C sort
> ```
>
> This keeps the directory tree automatically synchronized with the actual project structure.

```text
delaunay/
├── .cargo/
│   └── config.toml
├── .config/
│   └── nextest.toml
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── config.yml
│   ├── workflows/
│   │   ├── audit.yml
│   │   ├── benchmarks.yml
│   │   ├── ci.yml
│   │   ├── codacy.yml
│   │   ├── codecov.yml
│   │   ├── codeql.yml
│   │   ├── generate-baseline.yml
│   │   ├── profiling-benchmarks.yml
│   │   └── rust-clippy.yml
│   ├── CODEOWNERS
│   └── dependabot.yml
├── benches/
│   ├── PERFORMANCE_RESULTS.md
│   ├── README.md
│   ├── ci_performance_suite.rs
│   ├── circumsphere_containment.rs
│   ├── cold_path_predicates.rs
│   ├── large_scale_performance.rs
│   ├── profiling_suite.rs
│   └── topology_guarantee_construction.rs
├── docs/
│   ├── archive/
│   │   ├── changelog/
│   │   │   ├── 0.2.md
│   │   │   ├── 0.3.md
│   │   │   ├── 0.4.md
│   │   │   ├── 0.5.md
│   │   │   └── 0.6.md
│   │   ├── OPTIMIZATION_ROADMAP.md
│   │   ├── fix-delaunay.md
│   │   ├── invariant_validation_plan.md
│   │   ├── issue_120_investigation.md
│   │   ├── issue_204_investigation.md
│   │   ├── jaccard.md
│   │   ├── known_issues_4d_2026-04-23.md
│   │   ├── optimization_recommendations_historical.md
│   │   ├── phase2_bowyer_watson_optimization.md
│   │   ├── phase2_uuid_iter_optimization.md
│   │   ├── phase4.md
│   │   ├── phase_3a_implementation_guide.md
│   │   ├── phase_3c_action_plan.md
│   │   ├── testing.md
│   │   ├── todo_2026-04-23.md
│   │   └── topology_integration_design_historical.md
│   ├── dev/
│   │   ├── commands.md
│   │   ├── debug_env_vars.md
│   │   ├── python.md
│   │   ├── rust.md
│   │   └── testing.md
│   ├── templates/
│   │   └── README.md
│   ├── ORIENTATION_SPEC.md
│   ├── README.md
│   ├── RELEASING.md
│   ├── api_design.md
│   ├── code_organization.md
│   ├── invariants.md
│   ├── limitations.md
│   ├── numerical_robustness_guide.md
│   ├── property_testing_summary.md
│   ├── roadmap.md
│   ├── topology.md
│   ├── validation.md
│   └── workflows.md
├── examples/
│   ├── README.md
│   ├── convex_hull_3d_100_points.rs
│   ├── delaunayize_repair.rs
│   ├── into_from_conversions.rs
│   ├── memory_analysis.rs
│   ├── pachner_roundtrip_4d.rs
│   ├── point_comparison_and_hashing.rs
│   ├── topology_editing_2d_3d.rs
│   ├── triangulation_3d_100_points.rs
│   └── zero_allocation_iterator_demo.rs
├── scripts/
│   ├── ci/
│   │   └── capture_profiling_metadata.sh
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_archive_changelog.py
│   │   ├── test_benchmark_models.py
│   │   ├── test_benchmark_utils.py
│   │   ├── test_compare_storage_backends.py
│   │   ├── test_hardware_utils.py
│   │   ├── test_postprocess_changelog.py
│   │   ├── test_subprocess_utils.py
│   │   └── test_tag_release.py
│   ├── README.md
│   ├── archive_changelog.py
│   ├── benchmark_models.py
│   ├── benchmark_utils.py
│   ├── compare_storage_backends.py
│   ├── hardware_utils.py
│   ├── postprocess_changelog.py
│   ├── run_all_examples.sh
│   ├── slurm_storage_comparison.sh
│   ├── subprocess_utils.py
│   └── tag_release.py
├── src/
│   ├── core/
│   │   ├── algorithms/
│   │   │   ├── flips.rs
│   │   │   ├── incremental_insertion.rs
│   │   │   ├── locate.rs
│   │   │   └── pl_manifold_repair.rs
│   │   ├── collections/
│   │   │   ├── aliases.rs
│   │   │   ├── buffers.rs
│   │   │   ├── helpers.rs
│   │   │   ├── key_maps.rs
│   │   │   ├── secondary_maps.rs
│   │   │   ├── spatial_hash_grid.rs
│   │   │   └── triangulation_maps.rs
│   │   ├── traits/
│   │   │   ├── boundary_analysis.rs
│   │   │   ├── data_type.rs
│   │   │   └── facet_cache.rs
│   │   ├── util/
│   │   │   ├── canonical_points.rs
│   │   │   ├── deduplication.rs
│   │   │   ├── delaunay_validation.rs
│   │   │   ├── facet_keys.rs
│   │   │   ├── facet_utils.rs
│   │   │   ├── hashing.rs
│   │   │   ├── hilbert.rs
│   │   │   ├── jaccard.rs
│   │   │   ├── measurement.rs
│   │   │   └── uuid.rs
│   │   ├── adjacency.rs
│   │   ├── boundary.rs
│   │   ├── cell.rs
│   │   ├── edge.rs
│   │   ├── facet.rs
│   │   ├── operations.rs
│   │   ├── tds.rs
│   │   ├── triangulation.rs
│   │   └── vertex.rs
│   ├── geometry/
│   │   ├── algorithms/
│   │   │   └── convex_hull.rs
│   │   ├── traits/
│   │   │   └── coordinate.rs
│   │   ├── util/
│   │   │   ├── circumsphere.rs
│   │   │   ├── conversions.rs
│   │   │   ├── measures.rs
│   │   │   ├── norms.rs
│   │   │   ├── point_generation.rs
│   │   │   └── triangulation_generation.rs
│   │   ├── kernel.rs
│   │   ├── matrix.rs
│   │   ├── point.rs
│   │   ├── predicates.rs
│   │   ├── quality.rs
│   │   ├── robust_predicates.rs
│   │   └── sos.rs
│   ├── topology/
│   │   ├── characteristics/
│   │   │   ├── euler.rs
│   │   │   └── validation.rs
│   │   ├── spaces/
│   │   │   ├── euclidean.rs
│   │   │   ├── spherical.rs
│   │   │   └── toroidal.rs
│   │   ├── traits/
│   │   │   ├── global_topology_model.rs
│   │   │   └── topological_space.rs
│   │   └── manifold.rs
│   ├── triangulation/
│   │   ├── builder.rs
│   │   ├── delaunay.rs
│   │   ├── delaunayize.rs
│   │   └── flips.rs
│   └── lib.rs
├── tests/
│   ├── semgrep/
│   │   └── src/
│   │       └── core/
│   │           └── algorithms/
│   │               └── no_std_hash_collections.rs
│   ├── COVERAGE.md
│   ├── README.md
│   ├── allocation_api.rs
│   ├── check_perturbation_stats.rs
│   ├── circumsphere_debug_tools.rs
│   ├── conflict_region_verification.rs
│   ├── coordinate_conversion_errors.rs
│   ├── dedup_batch_construction.rs
│   ├── delaunay_edge_cases.rs
│   ├── delaunay_incremental_insertion.rs
│   ├── delaunay_public_api_coverage.rs
│   ├── delaunay_repair_fallback.rs
│   ├── delaunayize_workflow.rs
│   ├── euler_characteristic.rs
│   ├── insert_with_statistics.rs
│   ├── k3_cycle_predicate.rs
│   ├── large_scale_debug.rs
│   ├── prelude_exports.rs
│   ├── proptest_cell.rs
│   ├── proptest_convex_hull.rs
│   ├── proptest_delaunay_triangulation.proptest-regressions
│   ├── proptest_delaunay_triangulation.rs
│   ├── proptest_euler_characteristic.rs
│   ├── proptest_facet.rs
│   ├── proptest_geometry.rs
│   ├── proptest_orientation.rs
│   ├── proptest_point.rs
│   ├── proptest_predicates.rs
│   ├── proptest_safe_conversions.rs
│   ├── proptest_serialization.rs
│   ├── proptest_sos.rs
│   ├── proptest_tds.rs
│   ├── proptest_toroidal.rs
│   ├── proptest_triangulation.rs
│   ├── proptest_vertex.rs
│   ├── public_topology_api.rs
│   ├── regressions.rs
│   ├── serialization_vertex_preservation.rs
│   ├── storage_backend_compatibility.rs
│   ├── tds_orientation.rs
│   └── triangulation_builder.rs
├── .codacy.yml
├── .codecov.yml
├── .coderabbit.yml
├── .gitignore
├── .gitleaks.toml
├── .markdownlint.json
├── .python-version
├── .taplo.toml
├── .yamllint
├── AGENTS.md
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Cargo.lock
├── Cargo.toml
├── LICENSE
├── README.md
├── REFERENCES.md
├── SECURITY.md
├── cliff.toml
├── clippy.toml
├── justfile
├── proptest.toml
├── pyproject.toml
├── rust-toolchain.toml
├── rustfmt.toml
├── semgrep.yaml
├── ty.toml
├── typos.toml
└── uv.lock

```

**Note**: `tests/circumsphere_debug_tools.rs` contains interactive debugging test functions that can be run with:

```bash
# Run debug tests with interactive output (just command)
just test-diagnostics

# Or run specific test functions with verbose output (direct cargo)
cargo test --test circumsphere_debug_tools --features diagnostics test_2d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools --features diagnostics test_3d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools --features diagnostics test_all_debug -- --nocapture
# Or run all debug tests at once
cargo test --test circumsphere_debug_tools --features diagnostics -- --nocapture
```

**Note**: Memory allocation profiling is available through the `count-allocations` feature:

```bash
# Run allocation profiling tests (just command)
just test-allocation

# Run benchmarks with allocation counting (direct cargo for specific bench)
cargo bench --profile perf --bench profiling_suite --features count-allocations
```

> **Allocator Requirements**: Results depend on the system allocator (typically the default allocator on stable Rust).
> For consistent results across environments, ensure the same allocator is used. The `allocation-counter` crate works
> with the global allocator interface.

**Note**: Benchmark-style tests are available through the `bench` feature for performance analysis and demonstrations:

```bash
# Run regular tests (just command)
just test

# Run all tests including benchmark-style performance analysis
cargo test --lib --features bench
```

> **CI Stability**: The `bench` feature gates timing-based tests that may be flaky in CI environments.
> These tests are designed for local performance analysis and ergonomics validation rather than
> deterministic unit testing. Use `--features bench` when conducting performance investigations.

**Note**: Python tests in `scripts/tests/` are executed via pytest (recommended: `uv run pytest`) and discovered via `pyproject.toml`. Run with:

```bash
# Run all Python utility tests
just test-python

# Or run a specific test file
uv run pytest scripts/tests/test_benchmark_utils.py
uv run pytest scripts/tests/test_postprocess_changelog.py
```

**Note**: The changelog is generated by [git-cliff](https://git-cliff.org/) (`just changelog-update`), post-processed by
`postprocess_changelog.py` for markdown hygiene, and then archived by `archive_changelog.py` which splits completed minor
series into `docs/archive/changelog/X.Y.md` files, keeping only Unreleased + the active minor in the root `CHANGELOG.md`.
Tag creation (`just changelog-tag`) handles GitHub's tag annotation size limits and automatically falls back to archived
files when the requested version is no longer in the root changelog.

**Note**: Benchmarks, baselines, and performance summaries are generated via the benchmark utilities CLI:

```bash
# Generate a baseline artifact (used for comparisons)
just bench-baseline

# Generate benches/PERFORMANCE_RESULTS.md (runs benchmarks; longer)
just bench-perf-summary

# Or use the CLI directly
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
```

The `benchmark-utils` CLI provides integrated benchmark workflow functionality, with convenient `just` shortcuts for common workflows.

### Architecture Overview

#### Core Library (`src/`)

**`src/core/`** - Triangulation data structures and algorithms:

- `tds.rs` - Main `Tds` struct
- `triangulation.rs` - Generic Triangulation layer with kernel
- `vertex.rs`, `cell.rs`, `facet.rs` - Core geometric primitives
- `edge.rs` - Canonical `EdgeKey` for topology traversal
- `adjacency.rs` - Optional `AdjacencyIndex` builder outputs (opt-in)
- `collections/` - Optimized collection types and spatial acceleration structures
  - `spatial_hash_grid.rs` - Hash-grid spatial index for duplicate detection and locate-hint selection
- `boundary.rs` - Boundary detection and analysis
- `algorithms/` - Core algorithms (incremental insertion, flips, point location, PL-manifold repair)
- `traits/` - Core trait definitions including FacetCacheProvider for performance optimization
- `util/` - General utility functions organized by functionality (replaced single `util.rs` file)
  - `uuid.rs` - UUID generation and validation
  - `hashing.rs` - Stable, deterministic hash primitives
  - `deduplication.rs` - Vertex deduplication utilities
  - `measurement.rs` - Allocation measurement helper (feature-gated)
  - `facet_utils.rs` - Facet helpers (adjacency, vertex extraction, combination generation)
  - `facet_keys.rs` - Facet key derivation and consistency helpers
  - `jaccard.rs` - Set similarity utilities and diagnostics macro
  - `delaunay_validation.rs` - Delaunay property validation helpers (expensive; debug-oriented)
  - `hilbert.rs` - Hilbert ordering utilities (pure; triangulation-agnostic)
  - `canonical_points.rs` - Canonical vertex-ordering helpers for geometric predicate call sites (SoS consistency)
- `operations.rs` - Semantic classification and telemetry for topological operations

**`src/geometry/`** - Geometric algorithms and predicates:

- `kernel.rs` - Kernel abstraction (`AdaptiveKernel` default, `RobustKernel`, `FastKernel`) and `ExactPredicates` marker trait
- `point.rs` - NaN-aware Point operations
- `predicates.rs`, `robust_predicates.rs` - Geometric tests (see [Numerical Robustness Guide](numerical_robustness_guide.md))
- `sos.rs` - Simulation of Simplicity (SoS) for deterministic degeneracy resolution (orientation and insphere)
- `quality.rs` - Cell quality metrics (radius ratio, normalized volume) for d-dimensional simplices; provides mesh quality analysis to identify
  poorly-shaped cells (supports 2D-6D)
- `matrix.rs` - Linear algebra support
- `algorithms/convex_hull.rs` - Hull extraction
- `traits/coordinate.rs` - Coordinate abstractions
- `util/` - Geometric utility functions organized by functionality
  - `conversions.rs` - Safe coordinate type conversions with finite-value checking
  - `norms.rs` - Vector norms and distance computations (squared_norm, hypot)
  - `circumsphere.rs` - Circumcenter and circumradius calculations for simplices
  - `measures.rs` - Simplex volume, inradius, facet measure, surface measure computations
  - `point_generation.rs` - Random point generation (uniform, grid, Poisson disk sampling)
  - `triangulation_generation.rs` - Random triangulation generation with topology guarantees

**`src/triangulation/`** - Triangulation-facing public APIs:

- `builder.rs` - Fluent builder API for Euclidean and toroidal/periodic construction
- `delaunay.rs` - `DelaunayTriangulation` implementation (top layer) with incremental insertion
- `delaunayize.rs` - End-to-end "repair then delaunayize" workflow (`delaunayize_by_flips`);
  bounded topology repair + flip-based Delaunay repair + optional fallback rebuild
- `flips.rs` - High-level bistellar flip (Pachner move) trait and supporting public types; delegates to `core::algorithms::flips`

**`src/topology/`** - Topology analysis and validation:

- `characteristics/euler.rs` - Euler characteristic computation for full complexes and boundaries
- `characteristics/validation.rs` - Topological validation functions
- `manifold.rs` - Topology-only manifold invariants (e.g., closed boundary checks; see
  [`invariants.md`](invariants.md))
- `spaces/euclidean.rs` - Euclidean space topology helper implementation (f64-oriented)
- `spaces/spherical.rs` - Spherical space topology helper implementation (f64-oriented)
- `spaces/toroidal.rs` - Toroidal space topology helper implementation (f64-oriented)
- `traits/topological_space.rs` - Public `GlobalTopology<D>` metadata enum and `TopologyKind`
- `traits/global_topology_model.rs` - Internal scalar-generic `GlobalTopologyModel<D>` trait with
  concrete implementations (`EuclideanModel`, `ToroidalModel`, `SphericalModel`, `HyperbolicModel`);
  provides topology-specific behavior (canonicalization, lifting, periodic domain) used by core
  triangulation and builder code

#### Development Infrastructure

- **`examples/`** - Usage demos and trait examples, including memory profiling
  (see: [examples/memory_analysis.rs](../examples/README.md#5-memory-analysis-across-dimensions-memory_analysisrs)), Pachner move roundtrips
  (see: [examples/pachner_roundtrip_4d.rs](../examples/pachner_roundtrip_4d.rs)), and zero-allocation iterator demonstrations
- **`benches/`** - Performance benchmarks with automated baseline management (2D-5D coverage) and memory allocation tracking
  (see: [benches/profiling_suite.rs](../benches/README.md#profiling-suite-comprehensive))
- **`tests/`** - Integration tests including basic TDS validation (creation, neighbor assignment, boundary analysis),
  debugging utilities, regression testing, allocation profiling tools
  (see: [tests/allocation_api.rs](../tests/README.md#allocation_apirs)), and robust predicates validation
- **`docs/`** - User and contributor documentation, including architecture/reference guides,
  `docs/dev/` workflow rules for agents, archived design notes, and templates
- **`scripts/`** - Python utilities for automation and CI integration
  - **`archive_changelog.py`** - Archive completed minor series from root CHANGELOG.md into per-minor files
  - **`postprocess_changelog.py`** - Markdown hygiene for git-cliff output (typo correction, reflow, list normalization, summary injection)
  - **`tag_release.py`** - Extract changelog section for git tag annotations (with 125KB limit handling and archive fallback)
  - **`benchmark_utils.py`** - Performance benchmarking, regression testing, and baseline management
  - **`hardware_utils.py`** - Cross-platform hardware detection for performance tracking
  - **`tests/`** - Test suite for the Python utilities (regressions and tooling behavior)

#### Configuration

- **Quality Control**: `.codacy.yml`, `rustfmt.toml`, `pyproject.toml`, linting configurations
- **Environment**: `rust-toolchain.toml`, `.python-version`, `.cargo/config.toml`, GitHub Actions workflows
- **Development Workflow**: `justfile` with automated commands for common development tasks (see [Development Workflow](#development-workflow) below)
- **Memory Profiling**: `count-allocations` feature flag, allocation-counter dependency, profiling benchmarks
- **Performance Analysis**: `bench` feature flag for timing-based tests and performance demos (see "Benchmark-style tests" note above)
- **Project Metadata**: `CITATION.cff`, `REFERENCES.md`, `AGENTS.md`

### Architectural Principles

The project structure reflects several key architectural decisions:

1. **Separation of Concerns**: Clear boundaries between data structures (`core/`) and algorithms (`geometry/`)
2. **Generic Design**: Extensive use of generics for coordinate types, data associations, and dimensionality
3. **Trait-Based Architecture**: Heavy use of traits for extensibility and code reuse
4. **Performance Focus**: Dedicated benchmarking infrastructure, performance regression detection, and memory allocation profiling
5. **Memory Profiling**: Comprehensive allocation tracking with `count-allocations` feature for detailed memory analysis
6. **Performance Analysis (opt-in)**: `bench` feature for timing-based tests and ergonomics checks; distinct from CI-driven regression detection in item 4
7. **Academic Integration**: Strong support for research use with comprehensive citations and references
8. **Performance-Oriented Design**: Optimized collections, key-based APIs, and optional spatial indexing to reduce hot-path overhead
9. **Enhanced Robustness**: Rollback mechanisms, atomic operations, and comprehensive error handling
10. **Cross-Platform Development**: Modern Python tooling alongside traditional Rust development
11. **Quality Assurance**: Multiple layers of automated quality control and testing

This structure supports both library users (through examples and documentation) and contributors (through comprehensive
development tooling and clear architectural guidance).

#### Memory Profiling System

The project includes optional memory profiling capabilities:

- **Allocation Tracking**: Optional `count-allocations` feature using the `allocation-counter` crate
- **Memory Benchmarks**: Dedicated benchmarks for memory scaling analysis (`profiling_suite.rs`) - comprehensive profiling suite
  with typical runtime of 1-2 hours (10³-10⁶ points). **Recommended for manual profiling runs** rather than CI due to
  long execution time. Use `PROFILING_DEV_MODE=1` for faster iteration (10x speedup).
- **Profiling Examples**: `memory_analysis.rs` demonstrates allocation counting across different operations
- **Integration Testing**: `allocation_api.rs` provides utilities for testing memory usage in various scenarios
- **CI Integration**: Automated profiling benchmarks with detailed allocation reports

#### Performance-oriented infrastructure

The codebase includes several performance-focused components that are relevant when working on hot paths:

- **Fast collections**: `FastHashMap`, `FastHashSet`, and small-buffer helpers in `src/core/collections/`
- **Key-based internal APIs**: core types use key handles (`VertexKey`, `CellKey`) for fast lookups
- **Spatial hash-grid index**: `src/core/collections/spatial_hash_grid.rs` (duplicate detection and locate-hint selection)
- **Zero-allocation iterators / helpers**: used in performance-sensitive traversal paths
- **Benchmark suite + utilities**: `benches/` and the `benchmark-utils` tooling used by `just bench-*`

Exact performance characteristics depend on dimension, input distribution, and kernel choice; use the benchmarks to measure changes.

#### Development Workflow

The project uses [`just`](https://github.com/casey/just) as a command runner to simplify common development tasks. Key workflows include:

**Recommended Workflow:**

```bash
just fix           # Apply formatters/auto-fixes (mutating)
just check         # All non-mutating lints/validators
just test          # Tests + benchmark/release compile smoke
```

**Full CI / Pre-Push Validation:**

```bash
just ci            # Comprehensive checks + tests + examples
just ci-slow       # CI + slow tests (100+ vertices)
```

**Testing Workflows:**

```bash
just test-unit     # Lib and doc tests only
just test-integration # All integration tests (includes proptests)
just test-all      # All tests (lib + doc + integration + Python)
just test-python   # Python tests only (pytest)
just test-release  # All tests in release mode
just test-slow     # Run slow/stress tests with --features slow-tests
just test-slow-release # Slow tests in release mode (faster)
just test-diagnostics # Run diagnostics tools with output
just test-allocation  # Run allocation profiling tests
```

**Quality and Linting:**

```bash
just lint          # All linting (code + docs + config)
just lint-code     # Code linting (Rust, Python, Shell)
just lint-docs     # Documentation linting (Markdown, Spelling)
just lint-config   # Configuration validation (JSON, TOML, Actions)
just fmt           # Format Rust code
just clippy        # Run Clippy with strict settings
just doc-check     # Validate documentation builds
just python-lint   # Format and lint Python scripts
just spell-check   # Check spelling across project files
```

**Benchmarks and Performance:**

```bash
just bench         # Run all benchmarks with perf profile (ThinLTO)
just bench-baseline # Generate perf-profile performance baseline
just bench-ci      # CI regression benchmarks with perf profile (~5-10 min)
just bench-compare # Compare against baseline with perf profile
just bench-dev     # Reduced-sample perf-profile comparison (~1-2 min)
just bench-smoke   # Smoke-test benchmark harnesses (minimal samples)
just bench-perf-summary # Generate perf-profile release summary (~30-45 min)
```

**Storage Backend Comparison (large-scale):**

```bash
# DenseSlotMap (default)
cargo bench --profile perf --bench large_scale_performance

# SlotMap (disable default DenseSlotMap)
cargo bench --profile perf --no-default-features --bench large_scale_performance

# Enable larger 4D point counts (use on a compute cluster)
BENCH_LARGE_SCALE=1 cargo bench --profile perf --bench large_scale_performance

# Compare SlotMap (--no-default-features) vs DenseSlotMap (default)
just compare-storage    # SlotMap (--no-default-features) vs DenseSlotMap (default) (~4-6 hours)
```

**Performance Analysis:**

```bash
just perf-help     # Show performance analysis commands
just perf-baseline # Save current performance as baseline
just perf-check    # Check for performance regressions
just perf-compare  # Compare with specific baseline file
just profile       # Profile full triangulation_scaling benchmark
just profile-dev   # Profile 3D dev mode (faster iteration)
just profile-mem   # Profile memory allocations
```

**CI Simulation:**

```bash
just ci            # Comprehensive local CI run
just ci-baseline   # CI + save performance baseline
```

**Utilities:**

```bash
just setup         # Set up development environment
just clean         # Clean build artifacts
just build         # Build the project
just build-release # Build in release mode
just changelog     # Generate enhanced changelog
just changelog-tag <version> # Create git tag with changelog content
just examples      # Run all examples
```

**Complete Command Reference:**

```bash
just --list        # Show all available commands
just help-workflows # Show common workflow patterns
```

This `justfile`-based workflow provides consistent, cross-platform development commands and integrates seamlessly with the existing tooling ecosystem.

---

## Module Organization Patterns

The canonical organizational patterns found across key modules in the codebase:
`cell.rs`, `vertex.rs`, `facet.rs`, `boundary.rs`, and the `util/` submodules
under `src/core/util/`.

### Canonical Section Sequence

Based on analysis of the modules, the standard ordering follows this sequence:

1. **Module Documentation** (`//!` doc comments)
2. **Imports** (with section separator)
3. **Error Types** (with section separator)
4. **Convenience Macros and Helpers** (with section separator)
5. **Struct Definitions** (with section separator)
6. **Deserialization Implementation** (with section separator)
7. **Core Implementation Blocks** (with section separator)
8. **Advanced Implementation Blocks** (specialized trait bounds)
9. **Standard Trait Implementations** (with section separator)
10. **Specialized Trait Implementations** (e.g., Hashing, Equality)
11. **Tests** (with section separator)

### Comment Separators

#### Primary Section Separators

All modules consistently use this pattern for major sections:

```rust
// =============================================================================
// SECTION NAME
// =============================================================================
```

#### Subsection Separators

Within test modules, subsections use consistent formatting:

```rust
    // =============================================================================
    // SUBSECTION NAME TESTS
    // =============================================================================
```

### Section-by-Section Analysis

#### 1. Module Documentation (`//!` comments)

**Pattern**: Comprehensive module-level documentation with:

- Brief description of the module's purpose
- Key features (bulleted list with `**bold**` headings)
- Usage examples with code blocks
- References to external concepts (linked where appropriate)

**Example Structure**:

```rust
//! Brief description of the module
//!
//! Detailed explanation of what the module provides
//!
//! # Key Features
//!
//! - **Feature 1**: Description
//! - **Feature 2**: Description
//!
//! # Examples
//!
//! ```rust
//! // Code example
//! ```
```

#### 2. Imports Section

**Pattern**: Organized into logical groups with clear hierarchy:

1. `super::` imports (internal crate modules)
2. `crate::` imports (other crate modules)
3. External crate imports (alphabetically ordered)
4. Standard library imports

**Consistent Elements**:

- Section header: `// IMPORTS`
- Clear grouping with spacing
- Trait imports explicitly named

#### 3. Error Types Section

**Pattern**: Custom error enums using `thiserror::Error`:

```rust
/// Errors that can occur during [operation] validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum [Module]ValidationError {
    /// Description of error variant
    #[error("Error message: {source}")]
    VariantName {
        /// Description of source field
        #[from]
        source: SourceErrorType,
    },
}
```

**Consistent Elements**:

- Descriptive enum names ending in `ValidationError` or `Error`
- Full derive macro set: `Clone, Debug, Error, PartialEq, Eq`
- Detailed documentation for each variant
- `#[from]` attribute for error chaining

#### 4. Convenience Macros and Helpers Section

**Pattern**: Procedural macros with comprehensive documentation:

```rust
/// Convenience macro for creating [items] with less boilerplate.
///
/// Detailed description of macro functionality
///
/// # Returns
/// Description of return type
///
/// # Panics
/// Description of panic conditions
///
/// # Usage
/// ```rust
/// // Usage examples
/// ```
#[macro_export]
macro_rules! item_name {
    // Pattern definitions
}

// Re-export at crate level
pub use crate::macro_name;
```

**Helper Function Pattern**:

```rust
/// Helper function description
fn helper_function<generics>(parameters) -> ReturnType
where
    // trait bounds
{
    // implementation
}
```

#### 5. Struct Definitions Section

**Pattern**: Builder pattern with comprehensive documentation:

```rust
#[derive(Builder, Clone, Debug, Default, Serialize)]
#[builder(build_fn(validate = "Self::validate"))]
/// Comprehensive struct documentation
///
/// # Generic Parameters
/// * `T` - Description
/// * `U` - Description
/// * `const D` - Description
///
/// # Properties
/// - **field**: Description
///
/// # Usage
/// ```rust
/// // Usage example
/// ```
pub struct StructName<generics>
where
    // trait bounds
{
    /// Field documentation
    field: Type,
    
    #[builder(setter(skip), default = "default_value()")]
    auto_field: Type,
}
```

#### 6. Deserialization Implementation Section

**Pattern**: Manual `Deserialize` implementation with visitor pattern:

```rust
/// Manual implementation of Deserialize for [Type]
impl<'de, G> serde::Deserialize<'de> for Type<G>
where
    G: /* trait bounds as needed */,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Visitor pattern implementation
        // Ok(Self { /* ... */ })
    }
}
```

#### 7. Core Implementation Blocks

**Pattern**: Primary functionality with clear method groupings:

```rust
impl<generics> StructName<generics>
where
    // basic trait bounds
{
    /// Method documentation with examples
    ///
    /// # Arguments
    /// * `param` - Description
    ///
    /// # Returns
    /// Description
    ///
    /// # Example
    /// ```rust
    /// // Example code
    /// ```
    pub fn method_name(self) -> ReturnType {
        // implementation
    }
}
```

#### 8. Advanced Implementation Blocks

**Pattern**: Specialized implementations with additional trait bounds:

```rust
// Advanced implementation block for methods requiring ComplexField
impl<generics> StructName<generics>
where
    T: CoordinateScalar + Clone + ComplexField<generics> + PartialEq + PartialOrd + Sum,
    // additional specialized bounds
{
    /// Advanced method requiring specialized traits
    pub fn advanced_method(self) -> ReturnType {
        // implementation
    }
}
```

#### 9. Standard Trait Implementations Section

**Pattern**: Standard Rust traits with clear documentation:

```rust
/// Description of trait implementation behavior
impl<generics> TraitName for StructName<generics>
where
    // trait bounds
{
    /// Implementation documentation
    #[inline]
    fn trait_method(self, other: Self) -> ReturnType {
        // implementation
    }
}
```

**Common Standard Traits**:

- `PartialEq` - based on core data, excluding metadata
- `PartialOrd` - lexicographic ordering
- `Eq` - marker trait
- `From`/`Into` conversions

#### 10. Specialized Trait Implementations

**Pattern**: Complex traits like `Hash` with detailed contract documentation:

```rust
/// Custom Hash implementation using only [criteria] for consistency with `PartialEq`.
///
/// This ensures that items with the same [criteria] have the same hash,
/// maintaining the Eq/Hash contract: if a == b, then hash(a) == hash(b).
///
/// Note: [excluded fields] are excluded from hashing to match
/// the `PartialEq` implementation.
impl<G> core::hash::Hash for StructName<G>
where
    G: /* trait bounds as needed */,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // implementation with explanation comments
    }
}
```

#### 11. Tests Section

**Pattern**: Comprehensive test organization with multiple subsections:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // additional test imports
    
    // Type aliases for commonly used types to reduce repetition
    type TestType = StructName<generics>;
    
    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================
    
    /// Helper function for common test setup
    fn helper_function() -> TestType {
        // setup code
    }
    
    // =============================================================================
    // CATEGORY TESTS
    // =============================================================================
    // Tests covering [specific functionality]
    
    #[test]
    fn test_function_name() {
        // test implementation
    }
}
```

**Test Categories** (in order of appearance):

1. **Helper Functions** - Common test utilities
2. **Convenience Macro Tests** - Macro functionality
3. **Trait Implementation Tests** - Core Rust traits
4. **Core Methods Tests** - Primary functionality
5. **Dimensional Tests** - Multi-dimensional support
6. **Serialization Tests** - Serde functionality
7. **Geometric Properties Tests** - Domain-specific logic
8. **Error Handling Tests** - Validation and error cases
9. **Edge Case Tests** - Boundary conditions

### Module-Specific Variations

#### `cell.rs` (large module)

- Most comprehensive implementation
- Multiple specialized implementation blocks
- Extensive geometric predicates integration
- Detailed Hash/Eq contract documentation

#### `vertex.rs` (large module)

- Strong focus on coordinate validation
- Comprehensive equality testing
- Multiple numeric type support
- Detailed serialization testing

#### `facet.rs` (medium module)

- Geometric relationship focus
- Key generation utilities
- Adjacency testing
- Error handling for geometric constraints

#### `boundary.rs` (small module)

- Trait implementation focused
- Algorithm-specific testing
- Performance benchmarking
- Integration with TDS

#### `util/` (utility modules)

- Function-focused (not struct-focused)
- Split into dedicated modules under `src/core/util/` and wired explicitly in `src/lib.rs`.
- Major submodules:
  - `uuid.rs`: UUID generation and validation
  - `hashing.rs`: stable, deterministic hash primitives
  - `deduplication.rs`: vertex deduplication utilities
  - `measurement.rs`: allocation measurement helper (feature-gated)
  - `facet_utils.rs`: facet helpers (adjacency, vertex extraction, combination generation)
  - `facet_keys.rs`: facet key derivation + facet index consistency helpers
  - `jaccard.rs`: set similarity utilities + diagnostics macro `assert_jaccard_gte!`
  - `delaunay_validation.rs`: Delaunay property validation helpers (expensive; debug-oriented)
  - `hilbert.rs`: Hilbert ordering utilities (pure; triangulation-agnostic)
  - `canonical_points.rs`: Canonical vertex-ordering helpers for geometric predicate call sites (SoS consistency)
- Unit tests live alongside each submodule for cohesion (instead of a single giant util test module).

### Key Conventions

#### Documentation Standards

- Always include examples in public API documentation
- Use `///` for item documentation, `//!` for module documentation
- Include `# Arguments`, `# Returns`, `# Errors`, `# Panics` sections where applicable
- Reference other types using `[Type]` notation

#### Naming Conventions

- Error types: `[Module]ValidationError` or `[Module]Error`
- Test functions: `test_[functionality]_[specific_case]`
- Helper functions: `create_[item]` or `assert_[property]`
- Type aliases in tests: `Test[Type][Dimension]` (e.g., `TestCell3D`)

#### Code Organization

- Group related functionality within implementation blocks
- Separate basic and advanced functionality into different impl blocks
- Use consistent indentation and spacing
- Include inline comments for complex logic

#### Testing Patterns

- Comprehensive edge case coverage
- Both positive and negative test cases
- Type parameter variation testing (f32, f64, different dimensions)
- Serialization round-trip testing
- Error message validation

This organizational pattern provides a consistent, maintainable structure that scales well across different module
complexities while maintaining readability and discoverability.
