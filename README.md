# delaunay

[![DOI](https://zenodo.org/badge/729897852.svg)](https://doi.org/10.5281/zenodo.16931097)
[![Crates.io](https://badgen.net/crates/v/delaunay)](https://crates.io/crates/delaunay)
[![Downloads](https://badgen.net/crates/d/delaunay)](https://crates.io/crates/delaunay)
[![License](https://badgen.net/github/license/acgetchell/delaunay)](https://github.com/acgetchell/delaunay/blob/main/LICENSE)
[![Docs.rs](https://docs.rs/delaunay/badge.svg)](https://docs.rs/delaunay)
[![CI][ci-badge]][ci-workflow]
[![CodeQL][codeql-badge]][codeql-workflow]
[![rust-clippy analyze][clippy-badge]][clippy-workflow]
[![codecov](https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/delaunay)
[![Audit dependencies][audit-badge]][audit-workflow]
[![Codacy Badge][codacy-badge]][codacy-dashboard]

D-dimensional [Delaunay triangulations] and [convex hulls][Convex hulls] in [Rust], with exact predicates,
deterministic degeneracy handling, explicit topology validation, and bistellar flips for finite point sets.

## Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Quickstart](#-quickstart)
- [Scientific Basis](#-scientific-basis)
- [Validation Model](#-validation-model)
- [Documentation Map](#-documentation-map)
- [Ecosystem](#-ecosystem)
- [Benchmarking](#-benchmarking)
- [Limitations and Roadmap](#-limitations-and-roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [References](#-references)
- [AI-assisted Development](#-ai-assisted-development)
- [License](#-license)

## 📐 Introduction

Rust crate providing D-dimensional [Delaunay triangulations] and [convex hulls][Convex hulls]
(2D through 5D explicitly tested) constructed with a [PL-manifold] (default) or
[pseudomanifold][Pseudomanifold] guarantee on finite point sets with Euclidean and toroidal global
topologies. Uses [exact predicates] and [Simulation of Simplicity] for robustness and degeneracy
handling, and [Hilbert curve]s for deterministic insertion ordering and efficient spatial indexing.
Provides an explicit [4-level validation hierarchy][Validation Guide] on individual elements,
triangulation data structure validity, manifold topology, and Delaunay property adherence. Allows for
the complete set of [Pachner moves] up to D=5 using bistellar flips, vertex insertion and deletion,
and the conversion of non-Delaunay triangulations into Delaunay triangulations via bounded
flip/rebuilds. Auxiliary data may be stored directly in vertices and simplices with external
[secondary maps][Secondary maps] provided for vertex- and simplex-keyed algorithm use, and the entire
data structure is serializable/deserializable. Written in safe Rust with no unsafe code.

Use this crate when you want:

- Delaunay triangulations or convex hulls in 2D through 5D.
- Exact predicates and deterministic SoS handling for degenerate inputs.
- PL-manifold checks and explicit topology guarantees.
- PL-manifold-aware editing via bistellar flips and bounded Delaunay repair.
- Typed construction, insertion, validation, topology, and repair diagnostics.
- Validation reports that separate element, structure, topology, and Delaunay failures.

This is not a replacement for full meshing packages such as [CGAL], TetGen, or Gmsh when you need
constrained Delaunay triangulations, direct Voronoi extraction, out-of-core meshing, GPU/parallel
meshing, or production-scale dynamic remeshing.

## ✨ Features

- [x] Batch construction controls for insertion order, deduplication, repair cadence, and deterministic
  retries.
- [x] Complete set of bistellar flip / [Pachner moves] through D=5 via the Edit API, plus bounded
  Delaunay repair.
- [x] Configurable predicate kernels: `AdaptiveKernel` by default, `RobustKernel` for exact
  degeneracy-preserving predicates, and `FastKernel` for well-conditioned exploratory work.
- [x] D-dimensional [Convex hulls] and [Delaunay triangulations].
- [x] Euclidean and toroidal construction through `DelaunayTriangulationBuilder`:
  `.try_toroidal(...)` builds the periodic image-point quotient in validated dimensions, while
  `.try_canonicalized_toroidal(...)` wraps coordinates without quotient rewiring.
- [x] Exact predicates, stack-allocated linear algebra through [la-stack], and deterministic SoS
  degeneracy handling.
- [x] Focused public preludes for common construction, query, geometry, repair, topology, and diagnostic
  workflows.
- [x] Geometry measures and simplex quality metrics such as simplex volume, inradius, radius ratio, and
  normalized volume, plus Jaccard set-similarity diagnostics.
- [x] Incremental insertion, insertion statistics, and transactional `delete_vertex` rollback on failed
  repair/canonicalization.
- [x] JSON-exportable simplicial-complex primitives with stable vertex/simplex UUIDs for notebooks and
  downstream analysis tools.
- [x] Optional Cargo feature gates for allocation counting, diagnostics, benchmark logging, and slow
  correctness tests.
- [x] PL-manifold validation by default, with pseudomanifold checks available as an explicit opt-out.
- [x] Safe Rust: `#![forbid(unsafe_code)]`.
- [x] Serialization/deserialization through [JSON].
- [x] Vertex/simplex payloads plus secondary maps for caller-owned algorithm state.

See [CHANGELOG.md](CHANGELOG.md) for release history and [`docs/roadmap.md`](docs/roadmap.md) for
current direction, near-term candidates, and non-goals.

## 🚀 Quickstart

Add the crate to your project:

```bash
cargo add delaunay@0.7.8
```

Use `cargo add delaunay` instead if you want Cargo to select the newest published release.

- Rust 1.96.0 or newer, pinned by `Cargo.toml` and `rust-toolchain.toml`.
- `f64` coordinates for caller-facing construction, predicate, validation, and generator APIs.

```rust
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    assert_eq!(dt.dim(), 3);
    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.validate().is_ok());
    Ok(())
}
```

For toroidal domains, auxiliary vertex/simplex data, insertion statistics, vertex deletion, and
explicit flips, see [`docs/workflows.md`](docs/workflows.md).

## 🧪 Scientific Basis

The crate models triangulations as oriented simplicial complexes with separate combinatorial and
geometric checks. Robustness comes from the same layered predicate strategy used throughout the code:
fast f64 filters when the sign is provable, exact arithmetic fallback when it is not, and deterministic
SoS resolution for degenerate configurations.

The validation contract is computational and finite-dimensional. The crate checks that constructed or
edited triangulations satisfy implemented element, topology, and Delaunay invariants; it does not claim
to solve meshing constraints or certify unsupported geometric models.

For the detailed contract, see [REFERENCES.md](REFERENCES.md), [`docs/invariants.md`](docs/invariants.md),
and [`docs/numerical_robustness_guide.md`](docs/numerical_robustness_guide.md).

## ✅ Validation Model

| Level | Validates | Primary API |
|---|---|---|
| 1 | Vertex, simplex, and facet element invariants | `vertex.is_valid()` / `simplex.is_valid()` |
| 2 | TDS keys, incidences, and neighbor links | `dt.tds().is_valid()` |
| 3 | Manifold topology, ridge links, and Euler consistency | `dt.as_triangulation().is_valid()` |
| 4 | Delaunay property via local predicates | `dt.is_valid()` |
| 1-4 | Cumulative diagnostics | `dt.validate()` / `dt.validation_report()` |

`TopologyGuarantee` controls which Level 3 topology invariants are enforced. `ValidationPolicy`
controls when Level 3 checks run during incremental insertion. The default is PL-manifold topology with
explicit full-validation checkpoints.

## 🗺️ Documentation Map

- [API Design](docs/api_design.md) - construction, vertex lifecycle, and explicit Pachner moves.
- [Benchmarks](benches/README.md) - Criterion suites, perf-profile workflow, release summaries, and canary sizes.
- [Code Organization](docs/code_organization.md) - Architecture hub with links to module maps, focused preludes, and file layout.
- [Diagnostics](docs/diagnostics.md) - Structured reports, telemetry, and debug switches.
- [Examples](examples/README.md) - Runnable examples for construction, hulls, topology editing, diagnostics, and repair.
- [Invariants](docs/invariants.md) - Topological and geometric invariants enforced by the crate.
- [Limitations](docs/limitations.md) - Supported dimensions, predicate limits, toroidal modes, and feature gaps.
- [Mesh Export](docs/mesh_export.md) - Stable UUID-based simplicial-complex export for notebooks and downstream tools.
- [Numerical Robustness Guide](docs/numerical_robustness_guide.md) - Predicate kernels, SoS, retry, and repair behavior.
- [Orientation Spec](docs/ORIENTATION_SPEC.md) - Coherent combinatorial and geometric orientation rules.
- [Property Testing Summary](docs/property_testing_summary.md) - Property-test layout and coverage summary.
- [Releasing](docs/RELEASING.md) - Changelog, benchmark, and publish workflow.
- [Roadmap](docs/roadmap.md) - Current release sequence and deferred feature tracks.
- [Topology](docs/topology.md) - Level 3 topology validation and global topology models.
- [Validation Guide](docs/validation.md) - Validation hierarchy and policy configuration.
- [Workflows](docs/workflows.md) - Practical recipes for construction, repair, toroidal domains, payloads, and flips.

## 🧩 Ecosystem

`delaunay` sits in a small Rust research stack:

- [`la-stack`](https://crates.io/crates/la-stack) - stack-allocated linear algebra and exact determinant support.
- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) - downstream CDT research crate built on
  Delaunay-backed geometry primitives.
- [`markov-chain-monte-carlo`](https://crates.io/crates/markov-chain-monte-carlo) - composable MCMC traits used by the
  broader simulation ecosystem.

Within this crate, `src/core/` owns the topology data structures, `src/geometry/` owns predicates and
geometric helpers, `src/delaunay/` owns user-facing construction/query/repair APIs, and `src/topology/`
owns topology spaces and validation.

## 📈 Benchmarking

Correctness validation and performance measurement are separate workflows. For ordinary local validation:

```bash
just check
just test
just examples
```

For full CI parity:

```bash
just ci
```

Performance-sensitive work uses Criterion suites and same-machine baselines:

```bash
just perf-no-regressions
just bench-ci
just bench-perf-summary
```

See [`benches/README.md`](benches/README.md) for benchmark selection, fixture sizes, release baselines,
and large-scale profiling guidance.

## 🛣️ Limitations and Roadmap

Current routine coverage targets 2D through 5D. Exact orientation is available through D=6; exact
in-sphere is available through D=5. For D≥6, in-sphere classification relies on symbolic perturbation
and deterministic tie-breaking because the determinant exceeds the current stack-matrix limit.

Toroidal support has two modes:

- `.try_canonicalized_toroidal([..])` wraps coordinates into the fundamental domain before Euclidean
  construction.
- `.try_toroidal([..])` builds a true periodic quotient through the image-point method; it is validated
  in 2D and compact 3D, while 4D/5D fail fast pending scalable quotient work.

Not implemented today: constrained Delaunay triangulations, Voronoi diagram extraction, built-in
visualization, massively parallel/GPU construction, out-of-core meshing, and full spherical or
hyperbolic triangulation semantics.

See [`docs/limitations.md`](docs/limitations.md) for operational limits and [`docs/roadmap.md`](docs/roadmap.md)
for the v0.8.0 paper-facing API/topology push and later feature tracks.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide: project layout, development
workflow, code style, testing, documentation, benchmarking, and release support. Community expectations
live in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). AI assistants should follow [AGENTS.md](AGENTS.md).

Quick local workflow:

```bash
git clone https://github.com/acgetchell/delaunay.git
cd delaunay
cargo install --locked just
just setup
just check
just test
```

For the full command list, run `just --list`.

## 📚 Citation

If you use this software in academic work or downstream research software, cite the Zenodo DOI and
include the software metadata from [CITATION.cff](CITATION.cff).

- DOI: <https://doi.org/10.5281/zenodo.16931097>
- Citation metadata: [CITATION.cff](CITATION.cff)

```bibtex
@software{getchell_delaunay,
  author = {Adam Getchell},
  title = {delaunay: A d-dimensional Delaunay triangulation library},
  doi = {10.5281/zenodo.16931097},
  url = {https://github.com/acgetchell/delaunay}
}
```

For release-specific fields such as version, release date, and ORCID, prefer [CITATION.cff](CITATION.cff).

## 🔎 References

For academic references and bibliographic citations used throughout the library, see [REFERENCES.md](REFERENCES.md).

This includes foundational work on:

- Delaunay triangulations and convex hulls.
- Robust geometric predicates and exact arithmetic.
- Simulation of Simplicity.
- PL-manifold topology and Pachner moves.

## 🤖 AI-assisted Development

This repository contains [AGENTS.md](AGENTS.md), which defines the rules and invariants for AI coding
assistants and autonomous agents working on this codebase.

Portions of this library were developed with the assistance of AI tools including [ChatGPT], [Claude],
[Codex], and [CodeRabbit]. All accepted code and documentation changes are reviewed, edited, and
validated by the author.

For tool citation metadata, see the [AI-assisted development tools](REFERENCES.md#ai-assisted-development-tools)
section of [REFERENCES.md](REFERENCES.md).

## 📜 License

This project is licensed under the [BSD 3-Clause License](https://github.com/acgetchell/delaunay/blob/main/LICENSE).

---

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
[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[CodeRabbit]: https://coderabbit.ai/
[Codex]: https://openai.com/codex
[Convex hulls]: https://en.wikipedia.org/wiki/Convex_hull
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[exact predicates]: docs/numerical_robustness_guide.md
[Hilbert curve]: https://en.wikipedia.org/wiki/Hilbert_curve
[JSON]: https://www.json.org/json-en.html
[la-stack]: https://crates.io/crates/la-stack
[Pachner moves]: https://en.wikipedia.org/wiki/Pachner_move
[PL-manifold]: https://en.wikipedia.org/wiki/Piecewise_linear_manifold
[Pseudomanifold]: https://en.wikipedia.org/wiki/Pseudomanifold
[Secondary maps]: docs/workflows.md#builder-api-auxiliary-vertex-and-simplex-data
[Simulation of Simplicity]: docs/numerical_robustness_guide.md#simulation-of-simplicity-sos
[Validation Guide]: docs/validation.md
