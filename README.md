# delaunay

[![DOI](https://zenodo.org/badge/729897852.svg)](https://doi.org/10.5281/zenodo.16931097)
[![Crates.io](https://img.shields.io/crates/v/delaunay.svg)](https://crates.io/crates/delaunay)
[![Downloads](https://img.shields.io/crates/d/delaunay.svg)](https://crates.io/crates/delaunay)
[![License](https://img.shields.io/crates/l/delaunay.svg)](https://github.com/acgetchell/delaunay/blob/main/LICENSE)
[![Docs.rs](https://docs.rs/delaunay/badge.svg)](https://docs.rs/delaunay)
[![CI](https://github.com/acgetchell/delaunay/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/ci.yml)
[![rust-clippy analyze](https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/delaunay)
[![Audit dependencies](https://github.com/acgetchell/delaunay/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/audit.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3cad94f994f5434d877ae77f0daee692)](https://app.codacy.com/gh/acgetchell/delaunay/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

D-dimensional Delaunay triangulations and convex hulls in [Rust], with explicit invariants,
multi-level validation, and theory-backed flip workflows inspired by [CGAL].

## 📐 Introduction

This library implements d-dimensional Delaunay triangulations in [Rust] for computational geometry and scientific workflows.
It is inspired by [CGAL], which is a [C++] library for computational geometry, and [Spade], a [Rust] library that
implements 2D [Delaunay triangulations], [Constrained Delaunay triangulations], and [Voronoi diagrams].
The goal is to provide a lightweight but rigorous alternative to [CGAL] in the [Rust] ecosystem, with explicit
topology settings, validation levels, and repair behavior.

## 🧪 Scientific Basis

This crate models triangulations of finite point sets in `R^d` as oriented simplicial complexes with explicit
combinatorial and geometric checks.

- Core operational invariant for editing/repair: coherent orientation + PL-manifold validity
- Local move system: bistellar flips (`k = 1, 2, 3` and inverses), providing the supported [Pachner moves] set in dimensions ≤ 5
- Geometric convergence basis in finite-point workflows: Herbert Edelsbrunner and Nina R. Shah,
  *Incremental Topological Flipping Works for Regular Triangulations*, **Discrete & Computational Geometry (1996)**,
  <https://doi.org/10.1007/BF01975867>
- Scope of claims: guarantees apply to supported library workflows (construction, flip-based repair, and validation APIs),
  not arbitrary external mutation of internal structures

## ✨ Features

- [x]  Copyable data types associated with vertices and cells (integers, floats, chars, custom enums)
- [x]  d-dimensional [Delaunay triangulations]
- [x]  d-dimensional [Convex hulls]
- [x]  Toroidal (periodic) triangulations via [`DelaunayTriangulationBuilder`] with Phase 1 (canonicalization) and Phase 2 (image-point method) support
- [x]  Geometry quality metrics for simplices: radius ratio and normalized volume (dimension-agnostic)
- [x]  Serialization/deserialization of all data structures to/from [JSON]
- [x]  Tested for 2-, 3-, 4-, and 5-dimensional triangulations
- [x]  Configurable predicate kernels: `FastKernel` (speed) vs `RobustKernel` (degenerate / near-degenerate robustness)
- [x]  Bulk insertion ordering (`InsertionOrderStrategy`): [Hilbert curve] (default), [Z-order curve] / Morton, lexicographic, or input order
- [x]  Batch construction options (`ConstructionOptions`): optional deduplication and deterministic retries
- [x]  Incremental construction APIs: insertion plus vertex removal (`remove_vertex`)
- [x]  4-level validation hierarchy (element validity → TDS structural validity → manifold topology → Delaunay property), including full diagnostics via `validation_report`
- [x]  Local topology validation ([PL-manifold] default, [Pseudomanifold] opt-out)
- [x]  Coherent combinatorial orientation validation/normalization for cells, maintaining oriented simplicial complexes
- [x]  The complete set of [Pachner moves] up to 5D implemented as bistellar k-flips for k = 1, 2, 3 plus inverse moves
- [x]  [Delaunay repair] using bistellar flips for k=2/k=3 with inverse edge/triangle queues in 4D/5D
- [x]  Safe Rust: `#![forbid(unsafe_code)]`

See [CHANGELOG.md](CHANGELOG.md) for details.

## 🟢 Minimal Construction Example

The construction API has two entry points:

- `DelaunayTriangulation::new(&vertices)` - simple constructor for the common case
- [`DelaunayTriangulationBuilder`] - Advanced configuration (custom options, toroidal topology)

Add the library to your crate:

```bash
cargo add delaunay
```

```rust
use delaunay::prelude::triangulation::*;

// Create a 4D Delaunay triangulation from a set of vertices (with the default fast kernel).
let vertices = vec![
    vertex!([0.0, 0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0, 0.0]),
    vertex!([0.0, 0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 0.0, 1.0]), // 5 vertices (D+1) creates first 4-simplex
    vertex!([0.2, 0.2, 0.2, 0.2]), // Adding this vertex creates new simplices via flips
];

let dt = DelaunayTriangulation::new(&vertices).unwrap();

assert_eq!(dt.dim(), 4);
assert_eq!(dt.number_of_vertices(), 6);
assert_eq!(dt.number_of_cells(), 5); // 1 simplex from first 5 vertices + 4 new simplices from last vertex

// Optional verification:
// - `dt.is_valid()` checks Level 4 only (Delaunay property).
// - `dt.validate()` checks Levels 1–4 (elements + structure + topology + Delaunay).
assert!(dt.is_valid().is_ok());
```

### Toroidal (Periodic) Triangulations

For periodic boundary conditions, use `DelaunayTriangulationBuilder`:

```rust
use delaunay::prelude::triangulation::*;

// Phase 1: Canonicalization (wraps coordinates into [0, 1)²)
let vertices = vec![
    vertex!([0.1, 0.2]),
    vertex!([0.8, 0.3]),
    vertex!([0.5, 0.7]),
    vertex!([1.2, 0.4]),  // Wraps to [0.2, 0.4]
];

let dt = DelaunayTriangulationBuilder::new(&vertices)
    .toroidal([1.0, 1.0])  // Fundamental domain periods
    .build::<()>()
    .unwrap();

assert_eq!(dt.topology_kind(), TopologyKind::Toroidal);
```

For the full periodic image-point method (Phase 2), see the [`DelaunayTriangulationBuilder`] documentation.

### Need more control?

- **Editing with flips (Edit API)**:
  see [`docs/workflows.md`](docs/workflows.md) for a minimal example and [`docs/api_design.md`](docs/api_design.md) for details.
- **Flip-based Delaunay repair**, including the heuristic rebuild fallback (`repair_delaunay_with_flips*`):
  see [`docs/workflows.md`](docs/workflows.md).
- **Insertion outcomes and statistics** (`insert_with_statistics`, `InsertionOutcome`, `InsertionStatistics`):
  see [`docs/workflows.md`](docs/workflows.md) and [`docs/numerical_robustness_guide.md`](docs/numerical_robustness_guide.md).
- **Topology guarantees** (`TopologyGuarantee`) and **automatic topology validation** (`ValidationPolicy`):
  see [`docs/validation.md`](docs/validation.md) and [`docs/topology.md`](docs/topology.md).

## ✅ Validation and Guarantees

| Level | What is validated | Primary API |
|---|---|---|
| 1 | Element validity (vertex/cell primitives) | `dt.validate()` / `dt.validation_report()` |
| 2 | TDS structural validity (keys, incidences, neighbors) | `dt.tds().is_valid()` |
| 3 | Manifold topology (link checks, Euler/topological consistency) | `dt.as_triangulation().is_valid()` |
| 4 | Delaunay property (empty-circumsphere via local predicates) | `dt.is_valid()` |
| 1-4 | Cumulative checks with diagnostics | `dt.validate()` or `dt.validation_report()` |

`TopologyGuarantee` controls which Level 3 manifold constraints are enforced, and `ValidationPolicy`
controls when Level 3 checks run automatically during incremental insertion.

## 🔬 Reproducibility

The construction pipeline exposes deterministic controls for experiments and regression testing:

- Deterministic insertion ordering via `InsertionOrderStrategy`:
  `Hilbert` (default), `Morton` (Z-order), `Lexicographic`, or `Input`
  (use `Input` to preserve caller-provided order exactly)
- Deterministic preprocessing via `DedupPolicy`
- Deterministic retry behavior via `RetryPolicy` (including seeded shuffled retries) or `RetryPolicy::Disabled`
- Explicit topology/validation configuration via `TopologyGuarantee` and `ValidationPolicy`

```rust
use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, DedupPolicy, InsertionOrderStrategy, RetryPolicy,
};
use delaunay::core::triangulation::{TopologyGuarantee, ValidationPolicy};
use delaunay::prelude::triangulation::*;

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
    .build::<()>()
    .unwrap();

dt.set_validation_policy(ValidationPolicy::Always);
assert!(dt.validate().is_ok());
```

For reproducible checks in CI/local runs, use `just check`, `just test`, `just doc-check`, or `just ci`.

## ⚠️ Limitations

- CI and property-test coverage currently targets 2D-5D.
- Large-scale and 4D bulk-construction caveats are documented in [Known Issues](docs/KNOWN_ISSUES_4D.md).
- Validation/repair guarantees assume the library-managed construction/editing pipeline.

## 🚧 Project History

This crate was originally maintained at [https://github.com/oovm/shape-rs](https://github.com/oovm/shape-rs) through version `0.1.0`.
The original implementation provided basic Delaunay triangulation functionality.

Starting with version `0.3.4`, maintenance transferred to [this repository](https://github.com/acgetchell/delaunay), which hosts a completely
rewritten d-dimensional implementation focused on computational geometry research applications.

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
just fix              # Apply formatters/auto-fixes (mutating)
just check            # Lint/validators (non-mutating)
just ci               # Full CI run (checks + all tests + examples + bench compile)
just ci-slow          # CI + slow tests (100+ vertices)
just --list           # See all available commands
just help-workflows   # Show common workflow patterns
```

**Try the examples:**

```bash
just examples         # Run all examples
# Or run specific examples:
cargo run --release --example triangulation_3d_100_points
cargo run --release --example convex_hull_3d_100_points
```

## 📋 Examples

The `examples/` directory contains several demonstrations:

- **`convex_hull_3d_100_points`**: 3D convex hull extraction and analysis on the same 100-point configuration
- **`into_from_conversions`**: Demonstrates Into/From trait conversions and utilities
- **`memory_analysis`**: Memory usage analysis for triangulations across dimensions with allocation tracking
- **`pachner_roundtrip_4d`**: 4D Pachner move (k=1,2,3) roundtrip checks (flip + inverse preserves the triangulation)
- **`point_comparison_and_hashing`**: Demonstrates point comparison and hashing behavior
- **`topology_editing_2d_3d`**: Builder API vs Edit API in 2D/3D (bistellar flips and Delaunay preservation)
- **`triangulation_3d_100_points`**: 3D Delaunay triangulation with a stable 100-point random configuration
- **`zero_allocation_iterator_demo`**: Performance comparison between allocation and zero-allocation iterators

For detailed documentation, sample output, and usage instructions for each example, see [examples/README.md](examples/README.md).

For comprehensive guidelines on development environment setup,
testing, benchmarking, performance analysis, and development workflow, please see [CONTRIBUTING.md](CONTRIBUTING.md).

This includes information about:

- Building and testing the library
- Running benchmarks and performance analysis
- Code style and standards
- Submitting changes and pull requests
- Project structure and development tools

## 📖 Documentation

- **[API Design](docs/api_design.md)** - Builder vs Edit API design (explicit bistellar flips)
- **[Code Organization](docs/code_organization.md)** - Project structure and module patterns
- **[Invariants](docs/invariants.md)** - Theoretical background and rationale for the topological and geometric invariants
- **[Numerical Robustness Guide](docs/numerical_robustness_guide.md)** - Robustness strategies, kernels, and retry/repair behavior
- **[Property Testing Summary](docs/property_testing_summary.md)** - Property-based testing with proptest (where tests live, how to run)
- **[Known Issues](docs/KNOWN_ISSUES_4D.md)** - Known limitations for large-scale and 4D bulk construction
- **[Releasing](docs/RELEASING.md)** - Release workflow (changelog + benchmarks + publish)
- **[Topology](docs/topology.md)** - Level 3 topology validation (manifoldness + Euler characteristic) and module overview
- **[Validation Guide](docs/validation.md)** - Comprehensive 4-level validation hierarchy guide (element → structural → manifold → Delaunay)
- **[Workflows](docs/workflows.md)** - Happy-path construction plus practical Builder/Edit recipes (stats, repairs, and minimal flips)

## 📎 How to Cite

If you use this software in academic work, cite the Zenodo DOI and include the software
metadata from [`CITATION.cff`](CITATION.cff).

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

For release-specific fields (version, release date, ORCID), prefer `CITATION.cff`.

## 📚 References

For a comprehensive list of academic references and bibliographic citations used throughout the library, see [REFERENCES.md](REFERENCES.md).

## 🤖 AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants
for all AI coding assistants and autonomous agents working on this codebase.

AI tools (including ChatGPT, Claude, GitHub Copilot, Cursor, Warp, and CI repair agents) are
expected to read and follow `AGENTS.md` when proposing or applying changes.

Portions of this library were developed with the assistance of these AI tools:

- [ChatGPT]
- [Claude]
- [CodeRabbit]
- [GitHub Copilot]
- [KiloCode]
- [WARP]
>
> All code was written and/or reviewed and validated by the author.

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://crates.io/crates/spade
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://grokipedia.com/page/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://grokipedia.com/page/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://grokipedia.com/page/Voronoi_diagram
[Convex hulls]: https://grokipedia.com/page/Convex_hull
[Hilbert curve]: https://grokipedia.com/page/Hilbert_curve
[Z-order curve]: https://grokipedia.com/page/Z-order_curve
[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[CodeRabbit]: https://coderabbit.ai/
[GitHub Copilot]: https://github.com/features/copilot
[KiloCode]: https://kilocode.ai/
[WARP]: https://www.warp.dev
[Pseudomanifold]: https://grokipedia.com/page/Pseudomanifold
[PL-manifold]: https://grokipedia.com/page/Piecewise_linear_manifold
[Delaunay repair]: https://link.springer.com/article/10.1007/BF01975867
[Pachner moves]: https://grokipedia.com/page/pachner_moves
[`DelaunayTriangulationBuilder`]: https://docs.rs/delaunay/latest/delaunay/core/builder/struct.DelaunayTriangulationBuilder.html
