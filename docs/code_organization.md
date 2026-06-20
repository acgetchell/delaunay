# Code Organization Guide

This document provides a comprehensive guide to the delaunay project's code
organization, from the overall project architecture to detailed individual
module patterns.

## Table of Contents

- [Project Structure](#project-structure)
- [Directory Tree Snapshot](#directory-tree-snapshot)
  - [Architecture Overview](#architecture-overview)
  - [Focused Prelude Reference](#focused-prelude-reference)
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

### Directory Tree Snapshot

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
> Use this command to refresh the snapshot when files move. The tree below is a
> human-maintained orientation aid, not generated during every documentation
> build.

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
│   ├── instructions/
│   │   └── codacy.instructions.md
│   ├── workflows/
│   │   ├── audit.yml
│   │   ├── benchmarks.yml
│   │   ├── ci.yml
│   │   ├── codacy.yml
│   │   ├── codecov.yml
│   │   ├── codeql.yml
│   │   ├── generate-baseline.yml
│   │   ├── profiling-benchmarks.yml
│   │   ├── release-benchmarks.yml
│   │   ├── rust-clippy.yml
│   │   ├── semgrep-sarif.yml
│   │   └── zizmor.yml
│   ├── CODEOWNERS
│   └── dependabot.yml
├── benches/
│   ├── common/
│   │   ├── bench_utils.rs
│   │   ├── flip_fixtures.rs
│   │   └── flip_workflows.rs
│   ├── PERFORMANCE_RESULTS.md
│   ├── README.md
│   ├── allocation_hot_paths.rs
│   ├── boundary_uuid_iter.rs
│   ├── ci_performance_suite.rs
│   ├── circumsphere_containment.rs
│   ├── cold_path_predicates.rs
│   ├── profiling_suite.rs
│   ├── remove_vertex.rs
│   ├── tds_clone.rs
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
│   │   ├── issue_341_n1_repair_plan.md
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
│   │   ├── testing.md
│   │   └── tooling-alignment.md
│   ├── templates/
│   │   └── README.md
│   ├── ORIENTATION_SPEC.md
│   ├── README.md
│   ├── RELEASING.md
│   ├── api_design.md
│   ├── code_organization.md
│   ├── diagnostics.md
│   ├── invariants.md
│   ├── limitations.md
│   ├── numerical_robustness_guide.md
│   ├── production_review_remediation_checklist.md
│   ├── property_testing_summary.md
│   ├── roadmap.md
│   ├── topology.md
│   ├── validation.md
│   └── workflows.md
├── examples/
│   ├── README.md
│   ├── delaunayize_repair.rs
│   ├── diagnostics.rs
│   ├── into_from_conversions.rs
│   ├── numerical_robustness.rs
│   ├── point_comparison_and_hashing.rs
│   ├── topology_editing.rs
│   └── triangulation_and_hull.rs
├── scripts/
│   ├── ci/
│   │   ├── capture_profiling_metadata.sh
│   │   └── filter_codacy_sarif.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_archive_changelog.py
│   │   ├── test_benchmark_models.py
│   │   ├── test_benchmark_utils.py
│   │   ├── test_filter_codacy_sarif.py
│   │   ├── test_hardware_utils.py
│   │   ├── test_postprocess_changelog.py
│   │   ├── test_subprocess_utils.py
│   │   └── test_tag_release.py
│   ├── README.md
│   ├── archive_changelog.py
│   ├── benchmark_models.py
│   ├── benchmark_utils.py
│   ├── hardware_utils.py
│   ├── postprocess_changelog.py
│   ├── run_all_examples.sh
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
│   │   ├── construction.rs
│   │   ├── edge.rs
│   │   ├── facet.rs
│   │   ├── insertion.rs
│   │   ├── operations.rs
│   │   ├── orientation.rs
│   │   ├── query.rs
│   │   ├── repair.rs
│   │   ├── simplex.rs
│   │   ├── tds/
│   │   │   ├── equality.rs
│   │   │   ├── errors.rs
│   │   │   ├── incidence.rs
│   │   │   ├── keys.rs
│   │   │   ├── mutation.rs
│   │   │   ├── snapshot.rs
│   │   │   ├── storage.rs
│   │   │   └── validation.rs
│   │   ├── triangulation.rs
│   │   ├── validation.rs
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
│   │   ├── coordinate_range.rs
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
│   ├── delaunay/
│   │   ├── builder.rs
│   │   ├── construction.rs
│   │   ├── delaunayize.rs
│   │   ├── diagnostics.rs
│   │   ├── flips.rs
│   │   ├── insertion.rs
│   │   ├── locality.rs
│   │   ├── query.rs
│   │   ├── repair.rs
│   │   ├── serialization.rs
│   │   ├── triangulation.rs
│   │   └── validation.rs
│   ├── lib.rs
│   └── ...
├── tests/
│   ├── semgrep/
│   │   ├── .github/
│   │   │   └── workflows/
│   │   │       └── action_policy.yml
│   │   ├── docs/
│   │   │   └── command_order.sh
│   │   ├── doctests/
│   │   │   └── unwrap_expect.txt
│   │   ├── scripts/
│   │   │   └── tests/
│   │   │       ├── python_exceptions.py
│   │   │       └── python_parse_boundaries.py
│   │   └── src/
│   │       ├── core/
│   │       │   └── algorithms/
│   │       │       └── no_std_hash_collections.rs
│   │       └── project_rules/
│   │           └── rust_style.rs
│   ├── COVERAGE.md
│   ├── README.md
│   ├── allocation_api.rs
│   ├── benchmark_flip_fixtures.rs
│   ├── circumsphere_debug_tools.rs
│   ├── coordinate_conversion_errors.rs
│   ├── dedup_batch_construction.rs
│   ├── delaunay_edge_cases.rs
│   ├── delaunay_incremental_insertion.rs
│   ├── delaunay_repair_fallback.rs
│   ├── delaunayize_workflow.rs
│   ├── euler_characteristic.rs
│   ├── example_workflows.rs
│   ├── insert_with_statistics.rs
│   ├── large_scale_debug.rs
│   ├── pachner_roundtrip.rs
│   ├── prelude_exports.rs
│   ├── proptest_convex_hull.rs
│   ├── proptest_delaunay_triangulation.proptest-regressions
│   ├── proptest_delaunay_triangulation.rs
│   ├── proptest_euler_characteristic.rs
│   ├── proptest_facet.rs
│   ├── proptest_flips.rs
│   ├── proptest_geometry.rs
│   ├── proptest_orientation.rs
│   ├── proptest_point.rs
│   ├── proptest_predicates.rs
│   ├── proptest_safe_conversions.rs
│   ├── proptest_serialization.rs
│   ├── proptest_simplex.rs
│   ├── proptest_sos.rs
│   ├── proptest_tds.rs
│   ├── proptest_toroidal.rs
│   ├── proptest_triangulation.rs
│   ├── proptest_vertex.rs
│   ├── public_topology_api.rs
│   ├── regressions.rs
│   ├── serialization_vertex_preservation.rs
│   ├── trait_bound_ergonomics.rs
│   └── triangulation_builder.rs
├── .codacy.yml
├── .codecov.yml
├── .coderabbit.yml
├── .gitignore
├── .gitleaks.toml
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
├── dprint.json
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
cargo test --test circumsphere_debug_tools --features diagnostics test_all_debug -- --exact --nocapture
# Or run all debug tests at once
cargo test --test circumsphere_debug_tools --features diagnostics -- --nocapture
```

**Note**: Memory allocation profiling is available through the `count-allocations` feature:

```bash
# Verify allocation measurement wiring (just command)
just test-allocation

# Run hot-path allocation contracts (just command)
just bench-allocations

# Run broader profiling benchmarks with allocation counting (direct cargo)
cargo bench --profile perf --bench profiling_suite --features count-allocations
```

> **Allocator Requirements**: Results depend on the system allocator (typically the default allocator on stable Rust).
> For consistent results across environments, ensure the same allocator is used. The `allocation-counter` crate works
> with the global allocator interface.

**Note**: Benchmark-style measurements live in Criterion harnesses under `benches/`:

```bash
# Run regular tests
just test

# Run focused benchmark harnesses
cargo bench --profile perf --bench boundary_uuid_iter -- --noplot
```

> **CI Stability**: Timing-based measurements are not unit tests. Keep
> performance analysis in `benches/` so correctness tests remain deterministic.

**Note**: Python tests in `scripts/tests/` are executed via pytest and discovered via `pyproject.toml`. Run the usual suite through the `just` recipe:

```bash
# Run all Python utility tests
just test-python

# Or run a specific test file directly when narrowing failures
uv run pytest scripts/tests/test_benchmark_utils.py
uv run pytest scripts/tests/test_postprocess_changelog.py
```

**Note**: The changelog is generated by [git-cliff](https://git-cliff.org/) (`just changelog`), post-processed by
`postprocess_changelog.py` for markdown hygiene, and then archived by `archive_changelog.py` which splits completed minor
series into `docs/archive/changelog/X.Y.md` files, keeping only Unreleased + the active minor in the root `CHANGELOG.md`.
Tag creation (`just tag`, with `just changelog-tag` kept as a compatibility alias)
handles GitHub's tag annotation size limits and automatically falls back to
archived files when the requested version is no longer in the root changelog.

**Note**: Benchmarks, local baselines, release baseline packaging, and performance summaries are generated via the benchmark
utilities CLI:

```bash
# Prepare the default local main baseline artifact (manual same-machine comparisons)
just perf-baseline

# Generate benches/PERFORMANCE_RESULTS.md (runs benchmarks; longer)
just bench-perf-summary

# Or use the CLI directly
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
uv run benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
```

The `benchmark-utils` CLI provides integrated benchmark workflow functionality,
with convenient `just` shortcuts for common workflows. GitHub Actions compares
PRs and pushes on `ubuntu-latest` against the latest stable GitHub Release
benchmark asset produced by `.github/workflows/release-benchmarks.yml`; local
developer-machine runs should use the ignored same-machine baseline artifact
paths instead.

### Architecture Overview

#### Core Library (`src/`)

**`src/core/`** - Triangulation data structures and algorithms:

- `tds/storage.rs` - Main `Tds` struct, storage accessors, identity helpers, and construction tests
- `tds/errors.rs` - TDS error/report vocabulary re-export boundary
- `tds/equality.rs` - TDS equality implementation and stable simplex identity helpers
- `tds/incidence.rs` - Invariant-bearing vertex-to-simplices incidence index
- `tds/keys.rs` - Slotmap-backed `VertexKey` and `SimplexKey` handle types
- `tds/mutation.rs` - TDS topology mutation, orientation repair, and neighbor maintenance
- `tds/snapshot.rs` - TDS persistence boundary: raw codec records parse into a validated UUID snapshot before hydration
  allocates fresh slotmap keys, preserving vertex and simplex payload data without serializing storage-local handles
- `tds/validation.rs` - TDS Level 2 structural validation and adjacency consistency checks
- `triangulation.rs` - Generic Triangulation layer with kernel
- `construction.rs` - Generic triangulation construction helpers and initial-simplex setup
- `insertion.rs` - Generic transactional insertion, duplicate detection, and insertion telemetry
- `orientation.rs` - Generic simplex orientation validation, lifted-coordinate
  handling, and positive-orientation canonicalization
- `query.rs` - Read-only generic triangulation accessors, adjacency indices,
  and topology traversal helpers
- `repair.rs` - Generic local topology repair, stale incident-simplex repair,
  and vertex-removal cavity retriangulation
- `validation.rs` - Generic validation vocabulary and Level 3 orchestration;
  Level 1 remains with `vertex.rs`/`simplex.rs`, Level 2 with `tds/validation.rs`, and
  Delaunay Level 4 with `src/delaunay/validation.rs`
- `vertex.rs`, `simplex.rs`, `facet.rs` - Core geometric primitives
- `edge.rs` - Canonical `EdgeKey` for topology traversal
- `adjacency.rs` - Optional `AdjacencyIndex` builder outputs and
  lifetime-bound `TriangulationAdjacency` views (opt-in)
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
  - `facet_utils.rs` - Facet helpers (adjacency extraction, combination generation)
  - `facet_keys.rs` - Facet key derivation and consistency helpers
  - `jaccard.rs` - Set similarity utilities and diagnostics macro
  - `delaunay_validation.rs` - Delaunay property validation helpers (expensive; debug-oriented)
  - `hilbert.rs` - Hilbert ordering utilities (pure; triangulation-agnostic)
  - `canonical_points.rs` - Canonical vertex-ordering helpers for geometric predicate call sites (SoS consistency)
- `operations.rs` - Semantic classification and telemetry for topological operations

Public namespace policy: `crate::core` is the internal implementation namespace
for the low-level TDS and algorithm layer. The public low-level surface is
exposed through curated modules and focused preludes (`delaunay::tds`,
`delaunay::collections`, `delaunay::algorithms`, `delaunay::query`, and their
`delaunay::prelude::*` counterparts) rather than a broad public
`delaunay::core` module.

### Focused Prelude Reference

Use the narrowest prelude that matches the workflow. This keeps examples,
benchmarks, and tests clear about which part of the API they exercise.

| Task | Import |
|---|---|
| Bistellar flips / Edit API | `use delaunay::prelude::flips::*` |
| Collection aliases and small buffers | `use delaunay::prelude::collections::*` |
| Construct/configure a Delaunay triangulation | `use delaunay::prelude::construction::*` |
| Construction telemetry diagnostics | `use delaunay::prelude::diagnostics::*` |
| Construction validation cadence/policy | `use delaunay::prelude::validation::*` |
| Delaunay repair diagnostics and policies | `use delaunay::prelude::repair::*` |
| Delaunayize workflow | `use delaunay::prelude::delaunayize::*` |
| Hilbert ordering and quantization utilities | `use delaunay::prelude::ordering::*` |
| Low-level incremental insertion building blocks | `use delaunay::prelude::insertion::*` |
| Low-level TDS simplices, facets, keys, and validation reports | `use delaunay::prelude::tds::*` |
| Points, coordinate ranges, kernels, predicates, and geometric measures | `use delaunay::prelude::geometry::*` |
| Random points or triangulations for examples, tests, and benchmarks | `use delaunay::prelude::generators::*` |
| Read-only traversal, adjacency, convex hulls, and comparison helpers | `use delaunay::prelude::query::*` |
| Topological spaces and topology traits | `use delaunay::prelude::topology::spaces::*` |
| Topology validation and Euler characteristic helpers | `use delaunay::prelude::topology::validation::*` |

`use delaunay::prelude::*` remains available for quick experiments and broad
interactive use, but repository examples and benchmarks prefer focused preludes.

**`src/geometry/`** - Geometric algorithms and predicates:

- `coordinate_range.rs` - Validated coordinate-range value type used by random
  point and triangulation generator APIs; external tuple inputs are parsed at
  constructor boundaries and internal generation code consumes `CoordinateRange`
- `kernel.rs` - Kernel abstraction (`AdaptiveKernel` default, `RobustKernel`, `FastKernel`) and `ExactPredicates` marker trait
- `point.rs` - NaN-aware Point operations
- `predicates.rs`, `robust_predicates.rs` - Geometric tests (see [Numerical Robustness Guide](numerical_robustness_guide.md))
- `sos.rs` - Simulation of Simplicity (SoS) for deterministic degeneracy resolution (orientation and insphere)
- `quality.rs` - Simplex quality metrics (radius ratio, normalized volume) for d-dimensional simplices; provides mesh quality analysis to identify
  poorly-shaped simplices (supports 2D-6D)
- `matrix.rs` - Linear algebra support
- `algorithms/convex_hull.rs` - Hull extraction
- `traits/coordinate.rs` - Coordinate abstractions and typed coordinate
  diagnostic payloads (`FiniteCoordinateValue`, `CoordinateConversionValue`,
  `CoordinateValues`)
- `util/` - Geometric utility functions organized by functionality
  - `conversions.rs` - Safe coordinate type conversions with finite-value
    checking and `ValueConversionError`
  - `norms.rs` - Vector norms and distance computations (squared_norm, hypot)
  - `circumsphere.rs` - Circumcenter and circumradius calculations for
    simplices, plus `CircumcenterError`
  - `measures.rs` - Simplex volume, inradius, facet measure, surface measure
    computations, plus `SurfaceMeasureError`
  - `point_generation.rs` - Random point generation (uniform, grid, Poisson
    disk sampling), generator-specific range/count errors, and
    `InvalidPositiveScalar`
  - `triangulation_generation.rs` - Random triangulation generation with topology guarantees

**`src/delaunay/`** - Delaunay-facing implementation modules:

- `builder.rs` - Fluent builder API for Euclidean and toroidal/periodic construction
- `construction.rs` - Batch construction options, errors, statistics, and
  high-level constructors
- `insertion.rs` - Post-construction vertex insertion/removal and repair policy orchestration
- `query.rs` - Read-only `DelaunayTriangulation` accessors and traversal helpers
- `triangulation.rs` - `DelaunayTriangulation` storage type and insertion-state cache
- `delaunayize.rs` - End-to-end "repair then delaunayize" workflow (`delaunayize_by_flips`);
  bounded topology repair + flip-based Delaunay repair + optional fallback rebuild
- `flips.rs` - High-level bistellar flip (Pachner move) trait and supporting public types; delegates to `core::algorithms::flips`
- `locality.rs` - Local seed/frontier helpers for Hilbert-local construction and repair
- `repair.rs` - Delaunay repair policies, heuristic rebuild config, and repair outcomes
- `serialization.rs` - Conversion to/from `Tds` with topology metadata reset rules
- `validation.rs` - Level 4 validation errors plus construction validation cadence helpers

**`src/lib.rs`** - Crate root, public module declarations, root re-exports, and
focused preludes. Delaunay-facing modules are exposed directly as
`delaunay::builder`, `delaunay::construction`, `delaunay::flips`,
`delaunay::repair`, `delaunay::validation`, and related focused preludes rather
than through a `delaunay::delaunay` or `delaunay::triangulation` facade.

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

- **`examples/`** - User-facing API demos and workflow examples, including
  3D/4D construction plus hull queries, topology editing, diagnostics, conversion
  ergonomics, numerical robustness, and Delaunay repair
- **`benches/`** - Performance benchmarks with automated baseline management (2D-5D coverage) and memory allocation tracking
  (see: [benches/profiling_suite.rs](../benches/README.md#profiling-suite) and
  [benches/allocation_hot_paths.rs](../benches/README.md))
- **`tests/`** - Integration tests including basic TDS validation (creation, neighbor assignment, boundary analysis),
  debugging utilities, regression testing, allocation-measurement smoke coverage
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
- **Performance Analysis**: Criterion benchmark targets under `benches/` for timing-based measurements and performance demos
- **Project Metadata**: `CITATION.cff`, `REFERENCES.md`, `AGENTS.md`

### Architectural Principles

The project structure reflects several key architectural decisions:

1. **Separation of Concerns**: Clear boundaries between data structures (`core/`) and algorithms (`geometry/`)
2. **Generic Design**: Extensive use of generics for data associations and dimensionality, with `f64` as the only currently supported coordinate scalar
3. **Trait-Based Architecture**: Heavy use of traits for extensibility and code reuse
4. **Performance Focus**: Dedicated benchmarking infrastructure, performance regression detection, and memory allocation profiling
5. **Memory Profiling**: Comprehensive allocation tracking with `count-allocations` feature for detailed memory analysis
6. **Performance Analysis (opt-in)**: Criterion benchmark targets for timing-based
   measurements and ergonomics checks, distinct from CI-driven regression detection in item 4
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
- **Memory Benchmarks**: Dedicated benchmarks for RSS and allocation scaling analysis (`profiling_suite.rs`) - comprehensive
  profiling suite with calibrated large-scale 2D-5D point counts. **Recommended for manual profiling runs** rather than CI due
  to long execution time. Use `PROFILING_DEV_MODE=1` for faster auxiliary diagnostics.
- **Allocation Contracts**: `allocation_hot_paths.rs` keeps zero-allocation and bounded-allocation hot-path checks over
  calibrated 2D-5D triangulations in Criterion benchmarks, while `allocation_api.rs` only smoke-tests that allocation
  measurement is wired correctly.
- **CI Integration**: Automated profiling benchmarks with detailed allocation reports

#### Performance-oriented infrastructure

The codebase includes several performance-focused components that are relevant when working on hot paths:

- **Fast collections**: `FastHashMap`, `FastHashSet`, and small-buffer helpers in `src/core/collections/`
- **Key-based internal APIs**: core types use key handles (`VertexKey`, `SimplexKey`) for fast lookups
- **Spatial hash-grid index**: `src/core/collections/spatial_hash_grid.rs` (duplicate detection and locate-hint selection)
- **Zero-allocation iterators / helpers**: used in performance-sensitive traversal paths
- **Benchmark suite + utilities**: `benches/` and the `benchmark-utils` tooling used by `just bench-*`

Exact performance characteristics depend on dimension, input distribution, and kernel choice; use the benchmarks to measure changes.

#### Development Workflow

The project uses [`just`](https://github.com/casey/just) as a command runner to simplify common development tasks. Key workflows include:

**Recommended Workflow:**

```bash
just check         # All non-mutating lints/validators
just fix           # Apply formatters/auto-fixes (mutating)
just test          # Tests + benchmark/release compile smoke
```

**Full CI / Pre-Push Validation:**

```bash
just ci            # Comprehensive checks + tests + examples
just ci-slow       # CI + slow correctness tests
```

**Testing Workflows:**

```bash
just test-unit     # Lib and doc tests only
just test-integration # All integration tests (includes proptests)
just test-all      # All tests (lib + doc + integration + Python)
just test-python   # Python tests only (pytest)
just test-release  # All tests in release mode
just test-slow     # Run correctness tests over the 10s default-suite budget
just test-slow-release # Compatibility alias for just test-slow
just test-diagnostics # Run diagnostics tools with output
just test-allocation  # Verify allocation measurement wiring
just bench-allocations # Run allocation-contract microbenchmarks
```

**Quality and Linting:**

```bash
just lint          # All linting (code + docs + config)
just lint-code     # Code linting (Rust, Python, Shell)
just lint-docs     # Documentation linting (Markdown, Spelling)
just lint-config   # Configuration checks (JSON, TOML, YAML/CFF, Actions)
just clippy        # Run Clippy with strict settings
just doc-check     # Validate documentation builds
just python-check  # Python format/lint/typecheck checks
just spell-check   # Check spelling across project files
just fix           # Apply formatters/auto-fixes after reviewing checks
```

**Benchmarks and Performance:**

```bash
just bench         # Run all benchmarks with perf profile (ThinLTO)
just bench-ci      # CI regression benchmarks with perf profile (~5-10 min)
just perf-baseline # Prepare dev-mode local main baseline
just perf-compare  # Compare against a specific local baseline file
just perf-vs-ref   # Compare current tree against a cached same-machine ref baseline
just perf-no-regressions # Fast pre-PR 2D-5D regression guard
just bench-smoke   # Smoke-test benchmark harnesses (minimal samples)
just bench-perf-summary # Generate perf-profile release summary (~30-45 min)
```

**Large-scale Profiling:**

```bash
# Large-scale performance benchmarks
cargo bench --profile perf --bench profiling_suite

# Enable larger 4D point counts (use on a compute cluster)
BENCH_LARGE_SCALE=1 cargo bench --profile perf --bench profiling_suite
```

**Performance Analysis:**

```bash
just perf-help     # Show performance analysis commands
just perf-baseline # Prepare dev-mode local main baseline
just perf-compare  # Compare with a specific local baseline file
just perf-vs-ref   # Compare current tree with a specific release/ref
just perf-no-regressions # Check for performance regressions
just profile       # Run ci_performance_suite for a compiler/code pair
just profile-dev   # Profile 3D construction in profiling_suite
just profile-mem   # Profile memory allocations
```

**CI Simulation:**

```bash
just ci            # Comprehensive local CI run
just ci-baseline   # CI + persist default performance baseline
```

**Utilities:**

```bash
just setup         # Set up development environment
just clean         # Clean build artifacts
just build         # Build the project
just build-release # Build in release mode
just changelog     # Generate enhanced changelog
just tag <version> # Create git tag with changelog content
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
`simplex.rs`, `vertex.rs`, `facet.rs`, `boundary.rs`, and the `util/` submodules
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
    T: Clone + ComplexField<generics> + PartialEq + PartialOrd + Sum,
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

#### `simplex.rs` (large module)

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
  - `facet_utils.rs`: facet helpers (adjacency extraction, combination generation)
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
- Type aliases in tests: `Test[Type][Dimension]` (e.g., `TestSimplex3D`)

#### Code Organization

- Group related functionality within implementation blocks
- Separate basic and advanced functionality into different impl blocks
- Use consistent indentation and spacing
- Include inline comments for complex logic

#### Coordinate Scalar Policy

The currently supported caller-visible coordinate scalar is `f64`. Core types may
remain generic over `T` so combinatorial topology, payload data, and geometric
operations stay orthogonal, but the supported construction, predicate,
validation, and generator APIs use `f64` coordinates. This is intentional: the
crate prioritizes topology, manifoldness, and strict geometric correctness over
surface-level numeric generality.

Exact arithmetic is already used internally by robust predicate fallbacks. If
exact coordinates become caller-visible input in a future release, they should be
introduced as an explicit documented coordinate model/API rather than by
loosening ordinary `f64` APIs to arbitrary numeric types.

#### Testing Patterns

- Comprehensive edge case coverage
- Both positive and negative test cases
- Dimension variation testing for f64 coordinates
- Serialization round-trip testing
- Error message validation

This organizational pattern provides a consistent, maintainable structure that scales well across different module
complexities while maintaining readability and discoverability.
