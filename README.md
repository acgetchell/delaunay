# delaunay

[![DOI](https://badgen.net/badge/DOI/10.5281%2Fzenodo.16931097/blue)](https://doi.org/10.5281/zenodo.16931097)
[![Crates.io](https://badgen.net/crates/v/delaunay)](https://crates.io/crates/delaunay)
[![Downloads](https://badgen.net/crates/d/delaunay)](https://crates.io/crates/delaunay)
[![License](https://badgen.net/github/license/acgetchell/delaunay)](https://github.com/acgetchell/delaunay/blob/v0.8.1/LICENSE)
[![Docs.rs](https://docs.rs/delaunay/badge.svg)](https://docs.rs/delaunay)
[![CI][ci-badge]][ci-workflow]
[![CodeQL][codeql-badge]][codeql-workflow]
[![rust-clippy analyze][clippy-badge]][clippy-workflow]
[![codecov](https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/delaunay)
[![Audit dependencies][audit-badge]][audit-workflow]
[![Codacy Badge][codacy-badge]][codacy-dashboard]

D-dimensional [Delaunay triangulations] and [convex hulls][Convex hulls] in [Rust], with exact predicates,
deterministic degeneracy handling, explicit topology validation, and bistellar flips for finite point sets.

![S² Delaunay triangulation with 160 points][readme-hero]

## Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [API at a glance](#api-at-a-glance)
- [Quickstart](#-quickstart)
- [Scientific Basis](#-scientific-basis)
- [Validation Model](#-validation-model)
- [Documentation Map](#readme-documentation-map)
- [Ecosystem](#-ecosystem)
- [Benchmarking](#-benchmarking)
- [Limitations and Roadmap](#readme-limitations-and-roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [References](#-references)
- [AI Agents](#-ai-agents)
- [License](#-license)

## 📐 Introduction

Rust crate providing D-dimensional [Delaunay triangulations] and [convex hulls][Convex hulls]
constructed with a [PL-manifold] (default) or [pseudomanifold][Pseudomanifold] guarantee on finite
point sets. Euclidean construction is explicitly tested in 2D through 5D, periodic toroidal
construction is validated on T² and for compact T³ inputs, and bounded spherical S²/S³ construction
is available as a prototype. Uses [exact predicates] and [Simulation of Simplicity] for robustness and
degeneracy handling, and [Hilbert curve]s for deterministic insertion ordering and efficient spatial indexing.
Provides an explicit [5-level validation hierarchy][Construction and Validation Guide] on individual elements,
combinatorial consistency, intrinsic PL topology, valid realization in the active
model, and geometric predicates such as Delaunay. Allows for the complete set of [Pachner moves] up to D=5
using bistellar flips, vertex insertion and deletion, and the conversion of non-Delaunay
triangulations into Delaunay triangulations via bounded flip/rebuilds. Auxiliary data may be stored
directly in vertices and simplices with external [secondary maps][Secondary maps] provided for
vertex- and simplex-keyed algorithm use, and the entire data structure is
serializable/deserializable. Written in safe Rust with no unsafe code.

Use this crate when you want:

- Delaunay triangulations or convex hulls in 2D through 5D.
- Exact predicates and deterministic SoS handling for degenerate inputs.
- PL-manifold checks and explicit topology guarantees.
- PL-manifold-aware editing via bistellar flips and bounded Delaunay repair.
- Typed construction, insertion, validation, topology, and repair diagnostics.
- Valid-realization checks for Euclidean, toroidal, and spherical models independent of Delaunay predicates.
- Validation reports that separate element, combinatorial, intrinsic topology, realization, and
  geometric-predicate failures.

This is not a replacement for full meshing packages such as [CGAL], TetGen, or Gmsh when you need
constrained Delaunay triangulations, direct Voronoi extraction, out-of-core meshing, GPU/parallel
meshing, or production-scale dynamic remeshing.

## ✨ Features

- [x] Batch construction controls for insertion order, deduplication, repair cadence, and deterministic
  retries.
- [x] Complete set of bistellar flip / [Pachner moves] through D=5 via the Edit API, plus bounded
  Delaunay repair.
- [x] Configurable predicate kernels: `AdaptiveKernel` by default for deterministic SoS tie-breaking,
  `RobustKernel` for exact diagnostics and higher-dimensional fallbacks, and `FastKernel` for a lean
  filtered-exact path that preserves degeneracy signals through D ≤ 6.
- [x] D-dimensional [Convex hulls] and [Delaunay triangulations].
- [x] Euclidean construction and periodic `T^2`/`T^3` image-point quotients through
  `DelaunayTriangulationBuilder`.
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
- [x] [Jupyter] notebook interface for quickstart visualization, generated JSON artifacts, and README
  hero image reproduction.
- [x] Optional Cargo feature gates for allocation counting, diagnostics, benchmark logging, and slow
  correctness tests.
- [x] PL-manifold validation by default, with pseudomanifold checks available as an explicit opt-out.
- [x] Prototype spherical `S^2`/`S^3` construction through `SphericalDelaunayBuilder`, with
  Level 3 Intrinsic PL Topology, spherical Level 4 realization checks, and spherical Level 5
  empty-cap predicate checks.
- [x] Safe Rust: `#![forbid(unsafe_code)]`.
- [x] Serialization/deserialization through [JSON].
- [x] Topology-aware simplex barycenters for local-editing workflows, including periodic image-point
  lifting and canonicalization.
- [x] Vertex/simplex payloads plus secondary maps for caller-owned algorithm state.

See [CHANGELOG.md][changelog] for release history and [`docs/roadmap.md`][roadmap] for
current direction, near-term candidates, and non-goals.

## API at a glance

Use these public entry points with the linked recipes and runnable examples.
The full [import selector][import-selector] stays in the API documentation.
Here, `dt` is a `DelaunayTriangulation` and `tri` is a `Triangulation`.

| Capability | Public entry points | Learn by doing |
|---|---|---|
| Construction and configuration | `DelaunayTriangulationBuilder`, `ConstructionOptions` | [Recipe][construction-recipe] |
| Insertion and vertex deletion | `dt.insert_with_statistics()`, `dt.delete_vertex()` | [Lifecycle example][lifecycle-example] |
| Queries, quality, and hulls | `dt.locate()`, `radius_ratio()`, `ConvexHull` | [Example][hull-example] |
| Validation and diagnostics | `dt.validate()`, `dt.validation_report()`, `tri.realization_report()` | [Guide][Construction and Validation Guide] |
| Repair and Pachner editing | `DelaunayRefinementBuilder`, `PachnerMoves` | [Repair][repair-example]; [editing][pachner-example] |
| Toroidal/spherical construction | `.try_toroidal()`, `SphericalDelaunayBuilder` | [Torus][toroidal-example]; [sphere][spherical-example] |
| Payloads, checkpoints, JSON, export | `Vertex`, `Simplex`, `prelude::checkpoint` | [JSON][data-example]; [export][mesh-export-guide] |

Toroidal construction covers `T^2` and compact `T^3`; spherical construction is a
bounded `S^2`/`S^3` prototype. These library workflows use default features. The
[diagnostics example][diagnostics-example] requires `diagnostics`; the notebook/binary
workflow below requires `cli`. [Checkpoint contracts][checkpoint-guide] describe
versioned persistence; `dt.to_visualization_data()` creates a detached export.
Pachner moves preserve
their promised topology/realization scope; use repair and Level 5 validation when
you need the Delaunay property. See [scope and limitations][limitations-guide]
before choosing a dimension or geometric model.

## 🚀 Quickstart

Choose the path that matches your use case:

- Consume the crate as a Rust library when your application needs D-dimensional Delaunay
  triangulations, convex hulls, validation, or local-editing APIs.
- Use the repository notebook and binary workflow when you want to generate JSON or PNG artifacts,
  reproduce the README image, or explore larger point clouds interactively.

### Rust library

Add the crate to your project:

```bash
cargo add delaunay@0.8.1
```

Use `cargo add delaunay` instead if you want Cargo to select the newest published release.

- Rust 1.98.1 or newer. The minimum supported version is declared in
  `Cargo.toml`, while `rust-toolchain.toml` pins the exact repository toolchain.
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

    let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    assert_eq!(dt.dim(), 3);
    assert_eq!(dt.number_of_vertices(), 4);
    dt.validate()?;
    Ok(())
}
```

For runnable Rust workflows spanning toroidal and spherical construction,
auxiliary data, serialization, insertion statistics, deletion, queries,
quality metrics, and explicit flips, see the
[`examples/` coverage index][examples-guide].

### Notebook and binary

From a repository checkout, start with the notebook-first workflow:

```bash
just notebook-setup
just notebook
```

`just notebook-setup` installs the uv-managed notebook dependency group, and `just notebook`
launches JupyterLab with [`notebooks/00_quickstart.ipynb`][quickstart-notebook]. The
notebook uses the opt-in `delaunay` binary as the engine, loads generic simplicial-complex
visualization and convex-hull JSON, and writes a transparent preview under
`target/notebooks/00_quickstart/`.
The notebook and `just run` recipes enable the Cargo `cli` feature, which pulls in the binary and
notebook-support dependencies; ordinary library builds do not need them.
The [reviewer artifact guide][artifact-guide] and paper-claim mapping consume
this visual-inspection workflow without duplicating its implementation.
For validation-layer failure visuals, open
[`notebooks/01_validation.ipynb`][validation-notebook];
it runs `delaunay validation-demo` and renders generated validation figures for docs and papers.
The tracked spherical hero is generated from the real `S²` prototype by
[`notebooks/02_spherical_hero.ipynb`][spherical-notebook].
Refresh it deliberately with `just spherical-readme-hero`; routine notebook checks only lint this
computational artifact.

For headless CI or batch execution, use:

```bash
just notebook-execute
```

Use the binary directly when you want a scriptable artifact run:

```bash
just run generate visualization \
  --dimension 3 --vertices 1000 --distribution ball --seed 873 \
  --output target/notebooks/00_quickstart/visualization_3d.json
```

Before committing edited notebooks, clear generated outputs and execution counts:

```bash
just notebook-clear-outputs-all
```

## 🧪 Scientific Basis

The crate separates the abstract simplicial complex, its coordinate realization,
and the Delaunay predicate contract through the five validation levels below.
The [scientific basis overview][scientific-basis-guide] connects API selection,
supported geometric models, numerical assumptions, and methods to their sources.
It also distinguishes correctness evidence from performance measurements.

For the detailed contracts, see [`docs/construction_and_validation.md`][Construction and Validation Guide],
[`docs/invariants.md`][invariants-guide], [`docs/topology.md`][topology-guide],
[`docs/numerical_robustness_guide.md`][exact predicates],
[`docs/limitations.md`][limitations-guide], and [`benches/README.md`][benchmarks-guide].

## ✅ Validation Model

| Level | Validates | Primary API |
|---|---|---|
| 1 | Element Validity: vertex, simplex, facet, coordinate, and local-object invariants | `is_valid()` / element reports |
| 2 | Combinatorial Consistency: TDS incidence, adjacency, indexes, and stored orientation | `is_valid_structure()` / `structure_report()` |
| 3 | Intrinsic PL Topology: manifold links, components, Euler consistency, and orientability | `is_valid_topology()` / `topology_report()` |
| 4 | Valid Realization: affine-chart validity or bounded spherical simplex nondegeneracy, by backend | `is_valid_realization()` / `realization_report()` |
| 5 | Geometric Predicates: Delaunay and future geometry-specific optimality predicates | `is_valid_delaunay()` / `delaunay_report()` |
| 1-5 | Cumulative diagnostics | `dt.validate()` / `dt.validation_report()` |

`TopologyGuarantee` controls which Level 3 Intrinsic PL Topology invariants are enforced. `ValidationPolicy`
controls when cumulative Levels 1–4 audits run during incremental insertion. Level 4 realization validation is
backend-specific: Euclidean and toroidal paths validate affine-chart realizations, with toroidal
checks lifted to periodic covering-space charts, while the spherical prototype validates simplices on
`S^D \subset R^(D+1)`. Level 5 geometric predicates are likewise
backend-specific: Euclidean/toroidal Delaunay paths use empty-circumsphere predicates, while the
spherical prototype uses the empty-cap / ambient-hull-facet predicate. Use
`dt.as_triangulation().validate_realization()` when you want
cumulative Levels 1-4 validation for ordinary triangulations. `dt.as_triangulation().realization_report()`
returns simplex keys, simplex UUIDs, and offending vertex keys/UUIDs for Level 4 repair planning. The default is
PL-manifold topology with explicit full-validation
checkpoints. Layer-local APIs use `is_valid()` for unambiguous element/TDS owners, `is_valid_*`
for higher-level fast-fail checks, and `*_diagnostic` / `*_report` for diagnostics; cumulative
APIs use `validate()` / `validation_report()`.
`orientation_witness()` exposes the supported 2D/3D Level 3 orientability certificate directly.

For generated failure pictures, public test anchors, and diagnostics for each layer, run
[`notebooks/01_validation.ipynb`][validation-notebook]. For the paper-facing mathematical
exposition, see [`papers/validation.tex`][validation-paper-source] and the compiled reviewer copy at
[`papers/validation.pdf`][validation-paper].

<!-- Keep this explicit anchor stable across GitHub and rustdoc rendering. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="readme-documentation-map"></a>

## 🗺️ Documentation Map

- [API Design][api-design-guide] - construction, vertex lifecycle, and explicit Pachner moves.
- [Artifact Guide][artifact-guide] - v0.8.0 reviewer reproduction paths, claim map, evidence, and limits.
- [Benchmarks][benchmarks-guide] - Criterion suites, perf-profile workflow, release summaries, and canary sizes.
- [Code Organization][code-organization-guide] - Architecture hub with links to module maps, focused preludes, and file layout.
- [Construction and Validation Guide] - Proof-bearing construction, validation hierarchy, and policy configuration.
- [Construction and Validation Paper][validation-paper] - Reviewer-facing architecture paper.
- [Diagnostics][diagnostics-guide] - Structured reports, telemetry, and debug switches.
- [Examples and Notebooks][examples-guide] - Coverage map for runnable Rust workflows and visual computational artifacts.
- [Invariants][invariants-guide] - Topological and geometric invariants enforced by the crate.
- [Limitations][limitations-guide] - Supported dimensions, predicate limits, toroidal modes, and feature gaps.
- [Mesh Export][mesh-export-guide] - Stable UUID-based simplicial-complex export for notebooks and downstream tools.
- [Numerical Robustness Guide][exact predicates] - Predicate kernels, SoS, retry, and repair behavior.
- [Orientation Spec][orientation-spec] - Coherent combinatorial and geometric orientation rules.
- [Performance Report][performance-report] - Legacy release-to-release benchmark evidence without a retained artifact bundle.
- [Property Testing Summary][property-testing-guide] - Property-test layout and coverage summary.
- [Releasing][releasing-guide] - Changelog, benchmark, and publish workflow.
- [Roadmap][roadmap] - Current release sequence and deferred feature tracks.
- [Scientific Basis][scientific-basis-guide] - API selection, scientific scope, assumptions, methods, and supporting sources.
- [Topology][topology-guide] - Level 3 Intrinsic PL Topology validation, orientability, and global topology models.
- [Workflows][workflows-guide] - Practical recipes for construction, repair, toroidal domains, payloads, and flips.

## 🧩 Ecosystem

`delaunay` sits in a small Rust research stack:

- [`la-stack`](https://crates.io/crates/la-stack) - stack-allocated linear algebra, exact rational determinants, and exact linear solves.
- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) - downstream CDT research crate built on
  Delaunay-backed geometry primitives.

Within this crate, `src/core/` owns the topology data structures, `src/geometry/` owns predicates and
geometric helpers, `src/delaunay/` owns user-facing construction/query/repair APIs, and `src/topology/`
owns topology spaces and validation.

## 📈 Benchmarking

Benchmarking follows the same invariant-first model as the rest of the crate:
first confirm the measured workflow maintains the scientific invariants, then
compare same-machine performance, then publish curated release evidence. A fast
run that violates triangulation, predicate, topology, or diagnostic invariants
is a failed run, not a performance improvement.

For ordinary local validation:

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

See the legacy [`docs/performance.md`][performance-report] report for historical,
provenance-limited release evidence and [`benches/README.md`][benchmarks-guide]
for benchmark selection, fixture sizes, baseline workflows, and large-scale
profiling guidance.

Release PRs publish a compact group-level snapshot from the validated retained
CSV/provenance bundle. The geometric-mean ratios are descriptive summaries;
the generated full report retains every benchmark result represented in that
bundle.

<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->

No retained release-comparison bundle has been published to the README yet.

<!-- PERFORMANCE_RELEASE_TABLE:END -->

<!-- Keep this explicit anchor stable across GitHub and rustdoc rendering. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="readme-limitations-and-roadmap"></a>

## 🛣️ Limitations and Roadmap

Current routine construction coverage targets 2D through 5D. Exact orientation, relative in-sphere,
and complete symbolic tie-breaking are available through D=6. The in-sphere fast path bounds the
determinant with outward-rounded intervals; its cold path forms the relative matrix exactly from the
original binary64 coordinates before asking `la-stack`'s `RationalMatrix` for the sign. D≥7
falls back to the floating-point circumcenter/radius predicate and therefore lacks exact-sign
protection near degeneracy.

`.try_toroidal([..])` builds a periodic quotient through the image-point method. It is validated on
`T^2` and compact `T^3`, while `T^4`/`T^5` fail fast pending scalable quotient work.

Not implemented today: constrained Delaunay triangulations, Voronoi diagram extraction, built-in
visualization, massively parallel/GPU construction, out-of-core meshing, full spherical integration
beyond the bounded `S^2`/`S^3` prototype, and hyperbolic triangulation semantics.

See [`docs/limitations.md`][limitations-guide] for operational limits and [`docs/roadmap.md`][roadmap]
for v0.8.1 follow-up work and later feature tracks.

## 🤝 Contributing

See [CONTRIBUTING.md][contributing-guide] for the full contributor guide: project layout, development
workflow, code style, testing, documentation, benchmarking, and release support. Community expectations
live in [CODE_OF_CONDUCT.md][code-of-conduct]. AI assistants should follow [AGENTS.md][agent-guide].

Quick local workflow:

```bash
git clone https://github.com/acgetchell/delaunay.git
cd delaunay
bash scripts/bootstrap_just.sh
just setup
just check
just test
```

For the full command list, run `just --list`.

## 📚 Citation

If you use this software in academic work or downstream research software, cite the Zenodo DOI and
include the software metadata from [CITATION.cff][citation-metadata].

- DOI: <https://doi.org/10.5281/zenodo.16931097>
- Citation metadata: [CITATION.cff][citation-metadata]

```bibtex
@software{getchell_delaunay,
  author = {Adam Getchell},
  title = {delaunay: A d-dimensional Delaunay triangulation library},
  doi = {10.5281/zenodo.16931097},
  url = {https://github.com/acgetchell/delaunay}
}
```

For release-specific fields such as version, release date, and ORCID, prefer [CITATION.cff][citation-metadata].

## 🔎 References

For academic references and bibliographic citations used throughout the library, see [REFERENCES.md][references-guide].

This includes foundational work on:

- Delaunay triangulations and convex hulls.
- Robust geometric predicates and exact arithmetic.
- Simulation of Simplicity.
- PL-manifold topology and Pachner moves.

<!-- Preserve links to the former AI-assisted Development heading. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="-ai-assisted-development"></a>

## 🤖 AI Agents

AI coding assistants should read [AGENTS.md][agent-guide]
before proposing or applying changes. See [CONTRIBUTING.md][ai-development-guide]
for the repository's AI-assisted development note.

## 📜 License

This project is licensed under the [BSD 3-Clause License](https://github.com/acgetchell/delaunay/blob/v0.8.1/LICENSE).

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
[Convex hulls]: https://en.wikipedia.org/wiki/Convex_hull
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[exact predicates]: https://github.com/acgetchell/delaunay/blob/main/docs/numerical_robustness_guide.md
[Hilbert curve]: https://en.wikipedia.org/wiki/Hilbert_curve
[Jupyter]: https://jupyter.org/
[JSON]: https://www.json.org/json-en.html
[la-stack]: https://crates.io/crates/la-stack
[Pachner moves]: https://en.wikipedia.org/wiki/Pachner_move
[PL-manifold]: https://en.wikipedia.org/wiki/Piecewise_linear_manifold
[Pseudomanifold]: https://en.wikipedia.org/wiki/Pseudomanifold
[readme-hero]: https://raw.githubusercontent.com/acgetchell/delaunay/main/docs/assets/readme/delaunay_spherical_readme.png
[Secondary maps]: https://github.com/acgetchell/delaunay/blob/main/docs/USING_TRIANGULATIONS.md#builder-api-auxiliary-vertex-and-simplex-data
[Simulation of Simplicity]:
  <https://github.com/acgetchell/delaunay/blob/main/docs/numerical_robustness_guide.md#identity-based-sos-perturbation-via-canonical-vertex-ordering>
[Construction and Validation Guide]: https://github.com/acgetchell/delaunay/blob/main/docs/construction_and_validation.md

<!-- Repository guides follow main so links work in both GitHub and included rustdoc. -->
[agent-guide]: https://github.com/acgetchell/delaunay/blob/main/AGENTS.md
[ai-development-guide]: https://github.com/acgetchell/delaunay/blob/main/CONTRIBUTING.md#ai-assisted-development
[api-design-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/api_design.md
[artifact-guide]: https://github.com/acgetchell/delaunay/blob/main/papers/ARTIFACT.md
[benchmarks-guide]: https://github.com/acgetchell/delaunay/blob/main/benches/README.md
[changelog]: https://github.com/acgetchell/delaunay/blob/main/CHANGELOG.md
[checkpoint-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/construction_and_validation.md#owner-checkpoint-manifests
[citation-metadata]: https://github.com/acgetchell/delaunay/blob/main/CITATION.cff
[code-of-conduct]: https://github.com/acgetchell/delaunay/blob/main/CODE_OF_CONDUCT.md
[code-organization-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/code_organization.md
[contributing-guide]: https://github.com/acgetchell/delaunay/blob/main/CONTRIBUTING.md
[construction-recipe]: https://github.com/acgetchell/delaunay/blob/main/docs/USING_TRIANGULATIONS.md#builder-api-the-happy-path
[data-example]: https://github.com/acgetchell/delaunay/blob/main/examples/data_and_serialization.rs
[diagnostics-example]: https://github.com/acgetchell/delaunay/blob/main/examples/diagnostics.rs
[diagnostics-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/diagnostics.md
[examples-guide]: https://github.com/acgetchell/delaunay/blob/main/examples/README.md
[hull-example]: https://github.com/acgetchell/delaunay/blob/main/examples/triangulation_and_hull.rs
[import-selector]: https://docs.rs/delaunay/latest/delaunay/#which-import-do-i-need
[invariants-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/invariants.md
[limitations-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/limitations.md
[lifecycle-example]: https://github.com/acgetchell/delaunay/blob/main/examples/dynamic_lifecycle.rs
[mesh-export-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/mesh_export.md
[orientation-spec]: https://github.com/acgetchell/delaunay/blob/main/docs/orientation_spec.md
[pachner-example]: https://github.com/acgetchell/delaunay/blob/main/examples/topology_editing.rs
[performance-report]: https://github.com/acgetchell/delaunay/blob/main/docs/performance.md
[property-testing-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/property_testing_summary.md
[quickstart-notebook]: https://github.com/acgetchell/delaunay/blob/main/notebooks/00_quickstart.ipynb
[references-guide]: https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md
[releasing-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/RELEASING.md
[repair-example]: https://github.com/acgetchell/delaunay/blob/main/examples/delaunayize_repair.rs
[roadmap]: https://github.com/acgetchell/delaunay/blob/main/docs/roadmap.md
[scientific-basis-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/scientific_basis.md
[spherical-example]: https://github.com/acgetchell/delaunay/blob/main/examples/spherical_construction.rs
[spherical-notebook]: https://github.com/acgetchell/delaunay/blob/main/notebooks/02_spherical_hero.ipynb
[topology-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/topology.md
[toroidal-example]: https://github.com/acgetchell/delaunay/blob/main/examples/toroidal_construction.rs
[validation-notebook]: https://github.com/acgetchell/delaunay/blob/main/notebooks/01_validation.ipynb
[validation-paper]: https://github.com/acgetchell/delaunay/blob/main/papers/validation.pdf
[validation-paper-source]: https://github.com/acgetchell/delaunay/blob/main/papers/validation.tex
[workflows-guide]: https://github.com/acgetchell/delaunay/blob/main/docs/USING_TRIANGULATIONS.md
