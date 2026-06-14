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
bistellar flips insertion and deletion, and the conversion of
non-Delaunay triangulations into Delaunay triangulations via bounded
flip/rebuilds. Auxiliary data may be stored directly in vertices and simplices
with external [secondary maps][Secondary maps] provided for vertex- and
simplex-keyed algorithm use, and the entire data structure is
serializable/deserializable. Written in safe Rust with no unsafe code.

## 📐 Introduction

This library is for computational geometry and scientific workflows that need
more than a set of triangles: explicit topology settings, deterministic
construction controls, typed diagnostics, validation levels, and repair
behavior. It is inspired by [CGAL] and [Spade], but focuses on a safe,
inspectable, d-dimensional Rust API.

Use this crate when you want:

- Delaunay triangulations or convex hulls in 2D through 5D.
- Exact predicates and deterministic SoS handling for degenerate inputs.
- PL-manifold checks and explicit topology guarantees.
- PL-manifold-aware editing via bistellar flips and bounded Delaunay repair.
- Typed construction, insertion, validation, topology, and repair diagnostics.
- Validation reports that separate element, structure, topology, and Delaunay
  failures.

This is not a replacement for full meshing packages such as CGAL, TetGen, or
Gmsh when you need constrained Delaunay triangulations, out-of-core meshing,
GPU/parallel meshing, or production-scale dynamic remeshing.

## 🧪 Scientific Basis

The crate models finite point-set triangulations as oriented simplicial
complexes with separate combinatorial and geometric checks. Its robustness story
comes from Shewchuk-style floating-point filters, exact arithmetic fallback,
deterministic Simulation of Simplicity, and topology validation before Delaunay
predicates are trusted.

See [REFERENCES.md](REFERENCES.md), [Invariants](docs/invariants.md), and the
[Numerical Robustness Guide](docs/numerical_robustness_guide.md) for the
complete technical background.

## ✨ Features

- [x] Batch construction controls for insertion order, deduplication, repair
  cadence, and deterministic retries.
- [x] Complete set of bistellar flip / [Pachner moves] through D=5 via the Edit
  API, plus bounded Delaunay repair.
- [x] Configurable predicate kernels: `AdaptiveKernel` by default,
  `RobustKernel` for exact degeneracy-preserving predicates, and `FastKernel`
  for well-conditioned exploratory work.
- [x] D-dimensional [Convex hulls] and [Delaunay triangulations].
- [x] Euclidean and toroidal construction through
  [`DelaunayTriangulationBuilder`]: `.try_toroidal(...)` builds the periodic
  image-point quotient in validated dimensions, while
  `.try_canonicalized_toroidal(...)` wraps coordinates without quotient rewiring.
- [x] Exact predicates, stack-allocated linear algebra through [la-stack], and
  deterministic SoS degeneracy handling.
- [x] Focused public preludes for common construction, query, geometry, repair,
  topology, and diagnostic workflows.
- [x] Geometry measures and simplex quality metrics such as simplex volume,
  inradius, radius ratio, and normalized volume, plus Jaccard set-similarity
  diagnostics.
- [x] Incremental insertion, insertion statistics, and transactional
  `remove_vertex` rollback on failed repair/canonicalization.
- [x] Optional Cargo feature gates for allocation counting, diagnostics,
  benchmark logging, and slow correctness tests.
- [x] PL-manifold validation by default, with pseudomanifold checks available
  as an explicit opt-out.
- [x] Safe Rust: `#![forbid(unsafe_code)]`.
- [x] Serialization/deserialization through [JSON].
- [x] Vertex/simplex payloads plus secondary maps for caller-owned algorithm
  state.

See [CHANGELOG.md](CHANGELOG.md) for release details. Older releases are
archived under [docs/archive/changelog/](docs/archive/changelog/).

## 🟢 Minimal Construction Example

For ordinary point-cloud construction, start with one of two common entry
points:

- [`DelaunayTriangulationBuilder`] - primary construction interface for common
  and advanced configuration.
- `DelaunayTriangulation::new(&vertices)` - convenience constructor.

Add the library to your crate:

```bash
cargo add delaunay
```

Examples prefer focused preludes so imports document intent. For the full
prelude map and namespace policy, see the
[Focused Prelude Reference](docs/code_organization.md#focused-prelude-reference).

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2])
            .expect("finite vertex coordinates"),
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

### Toroidal Triangulations

For coordinate wrapping on a toroidal domain, use
`DelaunayTriangulationBuilder::try_canonicalized_toroidal`:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyKind,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.1, 0.2])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.8, 0.3])
            .expect("finite vertex coordinates"),
        delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.7])
            .expect("finite vertex coordinates"),
        // Wraps to [0.2, 0.4].
        delaunay::prelude::Vertex::<(), _>::try_new([1.2, 0.4])
            .expect("finite vertex coordinates"),
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .expect("unit toroidal domain is valid")
        .build::<()>()?;

    assert_eq!(dt.topology_kind(), TopologyKind::Toroidal);
    Ok(())
}
```

For boundary-facet identification and periodic neighbor pointers, use
`.try_toroidal([..])` in 2D or compact 3D; see the
[toroidal construction workflow] for the full recipe and current 4D/5D
guardrails.

### Need more control?

- **Editing with flips (Edit API)**:
  see [`docs/workflows.md`](docs/workflows.md) for a minimal example and
  [`docs/api_design.md`](docs/api_design.md) for details.
- **Flip-based Delaunay repair**, including the heuristic rebuild fallback
  (`repair_delaunay_with_flips*`):
  see [`docs/workflows.md`](docs/workflows.md).
- **Insertion outcomes and statistics** (`insert_with_statistics`,
  `insert_best_effort_with_statistics`, `InsertionOutcome`,
  `InsertionStatistics`):
  see [`docs/workflows.md`](docs/workflows.md) and
  [`docs/numerical_robustness_guide.md`](docs/numerical_robustness_guide.md).
- **Release benchmark summaries**:
  see [`benches/README.md`](benches/README.md) and
  [`benches/PERFORMANCE_RESULTS.md`](benches/PERFORMANCE_RESULTS.md).
- **Repair diagnostics and mutating-operation rollback**:
  `remove_vertex` rolls back if post-removal repair or orientation
  canonicalization fails, and repair failures preserve typed source errors for
  debugging.
- **Topology guarantees** (`TopologyGuarantee`) and **automatic topology
  validation** (`ValidationPolicy`):
  see [`docs/validation.md`](docs/validation.md) and [`docs/topology.md`](docs/topology.md).

## ✅ Validation and Guarantees

| Level | What is validated | Primary API |
|---|---|---|
| 1 | Element validity (vertex/simplex primitives) | `vertex.is_valid()` / `simplex.is_valid()` |
| 2 | TDS structural validity (keys, incidences, neighbors) | `dt.tds().is_valid()` |
| 3 | Manifold topology (link checks, Euler/topological consistency) | `dt.as_triangulation().is_valid()` |
| 4 | Delaunay property (empty-circumsphere via local predicates) | `dt.is_valid()` |
| 1-4 | Cumulative checks with diagnostics | `dt.validate()` or `dt.validation_report()` |

`TopologyGuarantee` controls which Level 3 manifold constraints are enforced,
and `ValidationPolicy` controls when Level 3 checks run automatically during
incremental insertion. Incompatible combinations are rejected by the fallible
`try_set_*` policy setters; use `ValidationPolicy::ExplicitOnly` for
caller-owned full-validation checkpoints with the default PL-manifold guarantee.

### Coordinate Input Type

The default supported coordinate input type is `f64`, matching the crate's
current linear algebra backend and geometric-primitive correctness guarantees.
Exact arithmetic is already used internally for robust predicate fallbacks, and
exact coordinate input may be supported explicitly in the future.

## 🔬 Reproducibility

The construction pipeline exposes deterministic controls for experiments and
regression testing:

- `ConstructionOptions` for repair cadence and batch-construction behavior.
- `DedupPolicy` for preprocessing duplicate coordinates.
- `InsertionOrderStrategy` for Hilbert ordering or caller-provided input order.
- `RetryPolicy` for deterministic retry behavior.
- `TopologyGuarantee` and `ValidationPolicy` for topology/validation policy.

For reproducible checks in CI/local runs, use `just check`, `just test`,
`just doc-check`, or `just ci`.

## ⚠️ Limitations

- **3D scale:** the default `just debug-large-scale-3d` helper uses 7,500
  vertices for the near-one-minute acceptance path. The 10,000-vertex run has
  also passed full Levels 1–4 validation as a heavier characterization probe;
  use `just debug-large-scale-3d 10000 1` for local numbers.
- **Dimension coverage:** CI and property-test coverage target 2D–5D.
- **Exact predicate limits:** exact orientation is available through D=6; exact
  in-sphere is available through D=5. For D≥6, in-sphere classification relies
  on symbolic perturbation and deterministic tie-breaking because the
  `(D+2)×(D+2)` determinant exceeds the stack matrix limit.
- **Feature gaps:** [Constrained Delaunay triangulations], [Voronoi diagrams],
  built-in visualization, GPU/parallel meshing, and out-of-core construction
  are out of scope today.
- **Large 4D+ batches:** `just debug-large-scale-4d 800 1` is the current
  release-mode acceptance harness; a recent local run inserted all 800 vertices,
  skipped none, ran final repair, and passed `validation_report` in about
  52 seconds. The 3,000-point 4D harness remains a manual characterization probe
  for issue #340 rather than routine CI.
- **Periodic domains:** `.try_toroidal()` uses the periodic image-point method and
  is release-validated in 2D and compact 3D. `.try_canonicalized_toroidal()`
  canonicalizes coordinates into the fundamental domain without quotient
  rewiring. 4D/5D periodic quotients fail fast pending scalable construction
  work in issue #416.
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

We welcome contributions. Start with the contributor guide, then use `just` for
the common local workflows:

```bash
git clone https://github.com/acgetchell/delaunay.git
cd delaunay
cargo install --locked just
just setup
just help-workflows
just check
just test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing,
benchmarks, style, and pull-request guidance. Run `just ci` before opening or
updating a pull request.

```bash
just examples
cargo run --release --example delaunayize_repair
cargo run --release --example triangulation_and_hull
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

For sample output and usage notes for each example, see
[examples/README.md](examples/README.md).

## 📖 Documentation

- **[API Design](docs/api_design.md)** - Builder vs Edit API design (explicit bistellar flips)
- **[Benchmarks](benches/README.md)** - Benchmark suites, perf-profile workflow, generated summaries, and release canaries
- **[Code Organization](docs/code_organization.md)** - Project structure and module patterns
- **[Diagnostics](docs/diagnostics.md)** - Diagnostic helpers, structured reports, telemetry, and debug switches
- **[Invariants](docs/invariants.md)** - Theoretical background and rationale for the topological and geometric invariants
- **[Limitations and Scope](docs/limitations.md)** - Supported dimensions, predicate limits, and feature gaps
- **[Numerical Robustness Guide](docs/numerical_robustness_guide.md)** - Robustness strategies, kernels, and retry/repair behavior
- **[Orientation Spec](docs/ORIENTATION_SPEC.md)** - Coherent combinatorial and geometric orientation rules
- **[Property Testing Summary](docs/property_testing_summary.md)** - Property-based testing with proptest (where tests live, how to run)
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
[Spade]: https://crates.io/crates/spade
[JSON]: https://www.json.org/json-en.html
[la-stack]: https://crates.io/crates/la-stack
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
[Pachner moves]: https://en.wikipedia.org/wiki/Pachner_move
[`DelaunayTriangulationBuilder`]: src/delaunay/builder.rs
[toroidal construction workflow]: docs/workflows.md#builder-api-toroidal-periodic-triangulations
