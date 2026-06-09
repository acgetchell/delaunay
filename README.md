# delaunay

[![DOI](https://zenodo.org/badge/729897852.svg)](https://doi.org/10.5281/zenodo.16931097)
[![Crates.io](https://img.shields.io/crates/v/delaunay.svg)](https://crates.io/crates/delaunay)
[![Downloads](https://img.shields.io/crates/d/delaunay.svg)](https://crates.io/crates/delaunay)
[![License](https://img.shields.io/crates/l/delaunay.svg)](https://github.com/acgetchell/delaunay/blob/main/LICENSE)
[![Docs.rs](https://docs.rs/delaunay/badge.svg)](https://docs.rs/delaunay)
[![CI][ci-badge]][ci-workflow]
[![CodeQL][codeql-badge]][codeql-workflow]
[![rust-clippy analyze][clippy-badge]][clippy-workflow]
[![codecov](https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/delaunay)
[![Audit dependencies][audit-badge]][audit-workflow]
[![Codacy Badge][codacy-badge]][codacy-dashboard]

[Rust] crate providing D-dimensional [Delaunay triangulations] and
[convex hulls][Convex hulls] (2D through 5D explicitly tested) constructed with
a [PL-manifold] (default) or [pseudomanifold][Pseudomanifold] guarantee on
finite point sets with Euclidean and toroidal global topologies. Uses
[exact predicates] and [Simulation of Simplicity] for robustness and degeneracy
handling, and [Hilbert curve]s for deterministic insertion ordering and
efficient spatial indexing. Provides an explicit
[4-level validation hierarchy][Validation Guide] on individual elements,
triangulation data structure validity, manifold topology, and Delaunay property
adherence. Allows for the complete set of [Pachner moves] up to D=5 using
bistellar flips, vertex insertion and deletion, and the conversion of
non-Delaunay triangulations into Delaunay triangulations via bounded
flip/rebuilds. Auxiliary data may be stored directly in vertices and simplices
with external [secondary maps][Secondary maps] provided for vertex- and
simplex-keyed algorithm use, and the entire data structure is
serializable/deserializable. Written in safe Rust with no unsafe code.

## 📐 Introduction

This library implements dimension-generic [Delaunay triangulations] in [Rust]
for computational geometry and scientific workflows. It is inspired by [CGAL],
which is a [C++] library for computational geometry, and [Spade], a [Rust]
library focused on 2D triangulations. The goal is to provide a lightweight but
rigorous Rust-native option for workflows that need explicit topology settings,
validation levels, deterministic construction controls, and repair behavior.

The core idea is an implemented and validated algorithmic framework for robust
Delaunay construction, PL-manifold-aware local moves, and bistellar repair. The
software artifact is part of the contribution: public APIs, examples,
property-based tests, diagnostics, and generated performance summaries are kept
together so the library can be inspected, reused, and benchmarked rather than
treated as a one-off implementation.

Use this crate when you want:

- Delaunay triangulations or convex hulls in 2D through 5D.
- A Rust-native bistellar flip / Pachner move API for topological editing and
  Delaunay repair.
- Exact predicates and deterministic Simulation of Simplicity (SoS) handling
  for degenerate inputs.
- PL-manifold checks and explicit topology guarantees for triangulations that
  need more than a bag of simplices.
- Typed construction, insertion, validation, topology, and repair diagnostics
  instead of stringly error handling.
- Validation reports that separate element, structure, topology, and Delaunay
  failures.
- Batch construction controls for insertion order, deduplication, repair
  cadence, and retry behavior.
- Release-mode 3D construction through 10,000 vertices with final Levels 1–4
  validation in the current large-scale characterization harness.
- PL-manifold-aware editing via bistellar flips.

This is not a replacement for full meshing packages such as CGAL, TetGen, or
Gmsh when you need constrained Delaunay triangulations, out-of-core meshing,
GPU/parallel meshing, or production-scale dynamic remeshing.

## 🧪 Scientific Basis

This crate models triangulations of finite point sets in `R^d` as oriented
simplicial complexes with explicit combinatorial and geometric checks. The
framework couples robust predicates, topology-aware local moves, repair
policies, validation reports, and benchmarked workflows instead of treating
these as separate utilities.

- Core operational invariant for editing/repair: coherent orientation +
  PL-manifold validity.
- Local move system: exposed bistellar flips (`k = 1, 2, 3` and inverses),
  providing the supported [Pachner moves] set in dimensions up to 5D.
- Local reversibility basis: benchmark-owned flip fixtures include n=1
  ergodicity checks, where one admissible Pachner move followed by its inverse
  must recover the same triangulation, not merely another valid triangulation;
  [Jaccard similarity] is used only as failure-path diagnostics for near misses.
- Geometric convergence basis in finite-point workflows: Herbert Edelsbrunner
  and Nimish R. Shah, *Incremental Topological Flipping Works for Regular
  Triangulations*, **Algorithmica (1996)**,
  <https://doi.org/10.1007/BF01975867>.
- Robust predicate basis: Shewchuk-style floating-point filters with exact
  fallback, plus deterministic Simulation of Simplicity for exact degeneracies.
- Topology basis: Level 3 validation checks PL-manifold structure, links, and
  Euler/topological consistency before Level 4 Delaunay predicates are trusted.
- Artifact basis: unit tests, integration tests, property tests, examples,
  release benchmark summaries, and docs.rs documentation exercise the same
  public APIs users copy.
- Scope of claims: guarantees apply to supported library workflows
  (construction, flip-based repair, and validation APIs), not arbitrary external
  mutation of internal structures.

See [REFERENCES.md](REFERENCES.md), [Invariants](docs/invariants.md), and the
[Numerical Robustness Guide](docs/numerical_robustness_guide.md) for the
complete technical background.

## ✨ Features

- [x]  Copyable data types associated with vertices and simplices (integers,
  floats, chars, custom enums), plus `SimplexSecondaryMap` and
  `VertexSecondaryMap` aliases for caller-owned key-indexed algorithm state
- [x]  D-dimensional [Delaunay triangulations]
- [x]  D-dimensional [Convex hulls]
- [x]  Bistellar flip / [Pachner moves] Edit API up to 5D: k-flips for
  k = 1, 2, 3 plus inverse moves
- [x]  [Delaunay repair] using bistellar flips for k=2/k=3 with inverse
  edge/triangle queues in 4D/5D
- [x]  Simulation of Simplicity (SoS) for deterministic handling of degenerate
  orientation and in-sphere configurations
- [x]  [PL-manifold] topology validation by default, with [Pseudomanifold]
  available as an explicit opt-out
- [x]  Toroidal triangulations via [`DelaunayTriangulationBuilder`]:
  `.toroidal(...)` builds a validated periodic image-point quotient in 2D and
  compact 3D, while `.canonicalized_toroidal(...)` canonicalizes points into the
  fundamental domain without quotient rewiring
- [x]  Geometry quality metrics for simplices: radius ratio and normalized
  volume (dimension-agnostic)
- [x]  Serialization/deserialization of all data structures to/from [JSON]
- [x]  Tested for 2-, 3-, 4-, and 5-dimensional triangulations
- [x]  Focused public preludes for construction, insertion, repair, validation,
  topology, query, geometry, generators, ordering, collections, TDS, and
  diagnostics workflows
- [x]  Configurable predicate kernels: `AdaptiveKernel` (default; exact
  arithmetic + SoS), `RobustKernel` (exact, preserves degeneracy signals),
  `FastKernel` (raw floating-point predicates for well-conditioned exploratory
  work; not accepted by explicit repair APIs)
- [x]  Bulk insertion ordering (`InsertionOrderStrategy`): [Hilbert curve]
  (default) or input order
- [x]  Batch construction options (`ConstructionOptions`): optional
  deduplication, configurable local Delaunay repair cadence, and deterministic
  retries
- [x]  Incremental construction APIs: insertion, insertion statistics, and
  transactional vertex removal (`remove_vertex`)
- [x]  4-level validation hierarchy (element validity → TDS structural
  validity → manifold topology → Delaunay property), including full
  diagnostics via `validation_report`
- [x]  Coherent combinatorial orientation validation/normalization for simplices,
  maintaining oriented simplicial complexes
- [x]  Typed construction, insertion, TDS, topology, validation, and repair
  errors that preserve source context for callers and diagnostics
- [x]  Reusable research software artifact: public examples, property tests,
  diagnostics, docs.rs landing-page documentation, and benchmark summaries
  generated from checked public workflows
- [x]  Release performance summaries generated from the public API benchmark
  contract, current Criterion run metadata, and generated simplex counts
- [x]  10,000-vertex 3D large-scale characterization run: zero skipped
  vertices, final flip repair clean, and `validation_report` OK for Levels 1–4
  in roughly 100 seconds on maintainer Apple M4 Max hardware
- [x]  Safe Rust: `#![forbid(unsafe_code)]`

See [CHANGELOG.md](CHANGELOG.md) for release details. Older releases are
archived under [docs/archive/changelog/](docs/archive/changelog/).

## 🟢 Minimal Construction Example

The construction API has two entry points:

- [`DelaunayTriangulationBuilder`] - primary construction interface for the common case and advanced configuration (custom options,
  toroidal topology, custom kernels)
- `DelaunayTriangulation::new(&vertices)` - legacy convenience constructor

Add the library to your crate:

```bash
cargo add delaunay
```

Choose the smallest prelude that matches the task:

| Task | Import |
|---|---|
| Construct/configure a Delaunay triangulation | `use delaunay::prelude::construction::*` |
| Read-only traversal, adjacency, convex hulls, and comparison helpers | `use delaunay::prelude::query::*` |
| Points, kernels, predicates, and geometric measures | `use delaunay::prelude::geometry::*` |
| Random points or triangulations for examples, tests, and benchmarks | `use delaunay::prelude::generators::*` |
| Low-level incremental insertion building blocks | `use delaunay::prelude::insertion::*` |
| Bistellar flips / Edit API | `use delaunay::prelude::flips::*` |
| Delaunay repair diagnostics and policies | `use delaunay::prelude::repair::*` |
| Delaunayize workflow | `use delaunay::prelude::delaunayize::*` |
| Construction telemetry diagnostics | `use delaunay::prelude::diagnostics::*` |
| Construction validation cadence/policy | `use delaunay::prelude::validation::*` |
| Hilbert ordering and quantization utilities | `use delaunay::prelude::ordering::*` |
| Low-level TDS simplices, facets, keys, and validation reports | `use delaunay::prelude::tds::*` |
| Collection aliases and small buffers | `use delaunay::prelude::collections::*` |
| Topology validation and Euler characteristic helpers | `use delaunay::prelude::topology::validation::*` |
| Topological spaces and topology traits | `use delaunay::prelude::topology::spaces::*` |

`use delaunay::prelude::*` remains available for quick experiments, but examples
and benchmarks in this repository prefer focused preludes so imports document
intent. The broad `delaunay::prelude::*` import is retained for
compatibility, but new docs and tests should prefer the narrow workflow preludes
above.

### Low-level imports

`delaunay::core` is an internal implementation namespace. Public low-level APIs
are exposed through `delaunay::tds`, `delaunay::collections`,
`delaunay::algorithms`, and `delaunay::query`, plus the matching focused
preludes. Contributors should follow the namespace policy in
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/code_organization.md](docs/code_organization.md).

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2]),
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    assert_eq!(dt.dim(), 4);
    assert_eq!(dt.number_of_vertices(), 6);

    // Optional verification:
    // - `dt.is_valid()` checks Level 4 only (Delaunay property).
    // - `dt.validate()` checks Levels 1-4 (elements + structure + topology + Delaunay).
    assert!(dt.validate().is_ok());
    Ok(())
}
```

### Toroidal (Periodic) Triangulations

For periodic boundary conditions, use `DelaunayTriangulationBuilder`:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyKind, vertex,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        vertex!([0.1, 0.2]),
        vertex!([0.8, 0.3]),
        vertex!([0.5, 0.7]),
        vertex!([1.2, 0.4]), // Wraps to [0.2, 0.4]
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .canonicalized_toroidal([1.0, 1.0])
        .build::<()>()?;

    assert_eq!(dt.topology_kind(), TopologyKind::Toroidal);
    Ok(())
}
```

For boundary-facet identification and periodic neighbor pointers, use
`.toroidal([..])` in 2D or compact 3D; see the
[toroidal construction workflow] for the full recipe and current 4D/5D
guardrails.

### Need more control?

- **Editing with flips (Edit API)**:
  see [`docs/workflows.md`](docs/workflows.md) for a minimal example and
  [`docs/api_design.md`](docs/api_design.md) for details.
- **Flip-based Delaunay repair**, including the heuristic rebuild fallback
  (`repair_delaunay_with_flips*`):
  see [`docs/workflows.md`](docs/workflows.md).
- **Repair diagnostics and mutating-operation rollback**:
  `remove_vertex` rolls back if post-removal repair or orientation
  canonicalization fails, and repair failures preserve typed source errors for
  debugging.
- **Insertion outcomes and statistics** (`insert_with_statistics`,
  `insert_best_effort_with_statistics`, `InsertionOutcome`,
  `InsertionStatistics`):
  see [`docs/workflows.md`](docs/workflows.md) and
  [`docs/numerical_robustness_guide.md`](docs/numerical_robustness_guide.md).
- **Topology guarantees** (`TopologyGuarantee`) and **automatic topology
  validation** (`ValidationPolicy`):
  see [`docs/validation.md`](docs/validation.md) and [`docs/topology.md`](docs/topology.md).
- **Release benchmark summaries**:
  see [`benches/README.md`](benches/README.md) and
  [`benches/PERFORMANCE_RESULTS.md`](benches/PERFORMANCE_RESULTS.md).

## ✅ Validation and Guarantees

| Level | What is validated | Primary API |
|---|---|---|
| 1 | Element validity (vertex/simplex primitives) | `dt.validate()` / `dt.validation_report()` |
| 2 | TDS structural validity (keys, incidences, neighbors) | `dt.tds().is_valid()` |
| 3 | Manifold topology (link checks, Euler/topological consistency) | `dt.as_triangulation().is_valid()` |
| 4 | Delaunay property (empty-circumsphere via local predicates) | `dt.is_valid()` |
| 1-4 | Cumulative checks with diagnostics | `dt.validate()` or `dt.validation_report()` |

`TopologyGuarantee` controls which Level 3 manifold constraints are enforced,
and `ValidationPolicy` controls when Level 3 checks run automatically during
incremental insertion. Incompatible combinations are rejected by the fallible
`try_set_*` policy setters; use `ValidationPolicy::ExplicitOnly` for
caller-owned full-validation checkpoints with the default PL-manifold guarantee.

## 🔬 Reproducibility

The construction pipeline exposes deterministic controls for experiments and
regression testing:

- Deterministic insertion ordering via `InsertionOrderStrategy`:
  `Hilbert` (default) or `Input`
  (use `Input` to preserve caller-provided order exactly)
- Deterministic preprocessing via `DedupPolicy`
- Deterministic retry behavior via `RetryPolicy` (including seeded shuffled
  retries) or `RetryPolicy::Disabled`
- Cadenced batch repair behavior via `ConstructionOptions` when large batch
  construction should repair local Delaunay fronts during insertion
- Explicit topology/validation configuration via `TopologyGuarantee` and
  `ValidationPolicy`

```rust
use delaunay::prelude::construction::{
    ConstructionOptions, DedupPolicy, DelaunayTriangulationBuilder, InsertionOrderStrategy,
    RetryPolicy, TopologyGuarantee, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    let options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_dedup_policy(DedupPolicy::Exact)
        .with_retry_policy(RetryPolicy::Disabled);

    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build::<()>()?;

    dt.set_validation_policy(ValidationPolicy::Always);
    assert!(dt.validate().is_ok());
    Ok(())
}
```

For reproducible checks in CI/local runs, use `just check`, `just test`,
`just doc-check`, or `just ci`.

## ⚠️ Limitations

- **Dimension coverage:** CI and property-test coverage target 2D–5D.
- **Exact predicate limits:** exact orientation is available through D=6; exact
  in-sphere is available through D=5. For D≥6, in-sphere classification relies
  on symbolic perturbation and deterministic tie-breaking because the
  `(D+2)×(D+2)` determinant exceeds the stack matrix limit.
- **Periodic domains:** `.toroidal()` uses the periodic image-point method and
  is release-validated in 2D and compact 3D. `.canonicalized_toroidal()`
  canonicalizes coordinates into the fundamental domain without quotient
  rewiring. 4D/5D periodic quotients fail fast pending scalable construction
  work in issue #416.
- **Large 4D+ batches:** thousands of 4D points can be expensive to
  investigate. Use release mode and the large-scale debug harness for
  characterization.
- **3D scale:** the default `just debug-large-scale-3d` helper uses 7,500
  vertices for the near-one-minute acceptance path. The 10,000-vertex run has
  also passed full Levels 1–4 validation as a heavier characterization probe;
  use `just debug-large-scale-3d 10000 1` for local numbers.
- **Feature gaps:** [Constrained Delaunay triangulations], [Voronoi diagrams],
  built-in visualization, GPU/parallel meshing, and out-of-core construction
  are out of scope today.
- **Validation/repair guarantees** assume the library-managed
  construction/editing pipeline.

See [Limitations and Scope](docs/limitations.md) for details and
[Roadmap](docs/roadmap.md) for active follow-up work.

## 🚧 Project History

This crate was originally maintained at
[https://github.com/oovm/shape-rs](https://github.com/oovm/shape-rs) through
version `0.1.0`. The original implementation provided basic Delaunay
triangulation functionality.

Starting with version `0.3.4`, maintenance transferred to
[this repository](https://github.com/acgetchell/delaunay), which hosts a
rewritten d-dimensional implementation focused on computational geometry
research applications.

- 📚 Docs for the original implementation (`0.1.0`): <https://docs.rs/delaunay/0.1.0/delaunay/>
- 📚 Docs for the rewritten implementation (`0.3.4+`): <https://docs.rs/delaunay>

## 🤝 How to Contribute

We welcome contributions! Here's a quickstart:

```bash
# Clone and setup
git clone https://github.com/acgetchell/delaunay.git
cd delaunay

# Setup development environment (installs tools, builds project)
cargo install just
just setup            # Installs all development tools and dependencies

# Development workflow
just check            # All non-mutating lints/validators
just fix              # Apply formatters/auto-fixes (mutating)
just test             # Tests + benchmark/release compile smoke
just ci               # Comprehensive checks + tests + examples
just --list           # See all available commands
just help-workflows   # Show common workflow patterns
```

Benchmark commands that produce performance data use the `perf` Cargo profile
for consistent ThinLTO settings. `just ci` remains the comprehensive validation
path: it runs checks, the test workflow, and examples, but it does not pay the
`perf` profile cost unless measuring performance.

For release performance documentation, run `just bench-perf-summary` from the
release PR branch after version and documentation updates. It refreshes
`benches/PERFORMANCE_RESULTS.md` from the public API Criterion suite,
circumsphere predicate benchmarks, current run metadata, and generated simplex
counts.

**Try the examples:**

```bash
just examples         # Run all examples
# Or run specific examples:
cargo run --release --example triangulation_and_hull
cargo run --release --example delaunayize_repair
```

## 📋 Examples

The `examples/` directory contains several demonstrations:

- **`delaunayize_repair`**: Demonstrates the `delaunayize_by_flips` workflow
  (bounded topology repair + flip-based Delaunay repair + optional fallback)
- **`diagnostics`**: Opt-in structured diagnostics for validation and
  deliberately non-Delaunay TDS examples
- **`into_from_conversions`**: Demonstrates Into/From trait conversions and
  utilities
- **`numerical_robustness`**: Compares `FastKernel`, `RobustKernel`, and
  `AdaptiveKernel` on degenerate predicate inputs
- **`point_comparison_and_hashing`**: Demonstrates point comparison and hashing
  behavior
- **`topology_editing`**: Builder API vs Edit API in 2D/3D (bistellar
  flips and Delaunay preservation)
- **`triangulation_and_hull`**: Seeded 3D and 4D triangulations, boundary
  traversal, convex hull extraction, and hull containment/visibility queries

For detailed documentation, sample output, and usage instructions for each
example, see [examples/README.md](examples/README.md).

For comprehensive guidelines on development environment setup, testing,
benchmarking, performance analysis, and development workflow, please see
[CONTRIBUTING.md](CONTRIBUTING.md).

This includes information about:

- Building and testing the library
- Running benchmarks and performance analysis
- Code style and standards
- Submitting changes and pull requests
- Project structure and development tools

## 📖 Documentation

- **[API Design](docs/api_design.md)** - Builder vs Edit API design (explicit bistellar flips)
- **[Benchmarks](benches/README.md)** - Benchmark suites, perf-profile workflow, generated summaries, and release canaries
- **[Code Organization](docs/code_organization.md)** - Project structure and module patterns
- **[Diagnostics](docs/diagnostics.md)** - Diagnostic helpers, structured reports, telemetry, and debug switches
- **[Invariants](docs/invariants.md)** - Theoretical background and rationale for the topological and geometric invariants
- **[Numerical Robustness Guide](docs/numerical_robustness_guide.md)** - Robustness strategies, kernels, and retry/repair behavior
- **[Orientation Spec](docs/ORIENTATION_SPEC.md)** - Coherent combinatorial and geometric orientation rules
- **[Property Testing Summary](docs/property_testing_summary.md)** - Property-based testing with proptest (where tests live, how to run)
- **[Limitations and Scope](docs/limitations.md)** - Supported dimensions, predicate limits, and feature gaps
- **[Releasing](docs/RELEASING.md)** - Release workflow (changelog + benchmarks + publish)
- **[Roadmap](docs/roadmap.md)** - Current follow-up work and deferred features
- **[Topology](docs/topology.md)** - Level 3 topology validation (manifoldness + Euler characteristic) and module overview
- **[Validation Guide](docs/validation.md)** - Comprehensive 4-level validation hierarchy guide (element → structural → manifold → Delaunay)
- **[Workflows](docs/workflows.md)** - Happy-path construction plus practical Builder/Edit recipes (stats, repairs, and minimal flips)

## 📎 How to Cite

If you use this software in academic work, cite the Zenodo DOI and include the
software metadata from [`CITATION.cff`](CITATION.cff).

- DOI: <https://doi.org/10.5281/zenodo.16931097>
- Citation metadata: [`CITATION.cff`](CITATION.cff)

```bibtex
@software{getchell_delaunay,
  author = {Adam Getchell},
  title = {delaunay: A d-dimensional Delaunay triangulation library},
  doi = {10.5281/zenodo.16931097},
  url = {https://github.com/acgetchell/delaunay}
}
```

For release-specific fields (version, release date, ORCID), prefer
`CITATION.cff`.

## 📚 References

For a comprehensive list of academic references and bibliographic citations used
throughout the library, see [REFERENCES.md](REFERENCES.md).

## 🤖 AI Agents

AI coding assistants should read [AGENTS.md](AGENTS.md) before proposing or
applying changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository's
AI-assisted development note.

[Rust]: https://rust-lang.org
[audit-badge]: https://github.com/acgetchell/delaunay/actions/workflows/audit.yml/badge.svg
[audit-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/audit.yml
[ci-badge]: https://github.com/acgetchell/delaunay/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/ci.yml
[clippy-badge]: https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml/badge.svg
[clippy-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml
[codacy-badge]: https://app.codacy.com/project/badge/Grade/3cad94f994f5434d877ae77f0daee692
[codacy-dashboard]: https://app.codacy.com/gh/acgetchell/delaunay/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
[codeql-badge]: https://github.com/acgetchell/delaunay/actions/workflows/codeql.yml/badge.svg
[codeql-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/codeql.yml
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://crates.io/crates/spade
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://en.wikipedia.org/wiki/Voronoi_diagram
[Convex hulls]: https://en.wikipedia.org/wiki/Convex_hull
[Hilbert curve]: https://en.wikipedia.org/wiki/Hilbert_curve
[exact predicates]: docs/numerical_robustness_guide.md
[Simulation of Simplicity]: docs/numerical_robustness_guide.md#simulation-of-simplicity-sos
[Secondary maps]: docs/workflows.md#builder-api-auxiliary-vertex-and-simplex-data
[Validation Guide]: docs/validation.md
[Pseudomanifold]: https://en.wikipedia.org/wiki/Pseudomanifold
[PL-manifold]: https://en.wikipedia.org/wiki/Piecewise_linear_manifold
[Delaunay repair]: https://link.springer.com/article/10.1007/BF01975867
[Pachner moves]: https://en.wikipedia.org/wiki/Pachner_move
[Jaccard similarity]: https://en.wikipedia.org/wiki/Jaccard_index
[`DelaunayTriangulationBuilder`]: src/delaunay/builder.rs
[toroidal construction workflow]: docs/workflows.md#builder-api-toroidal-periodic-triangulations
