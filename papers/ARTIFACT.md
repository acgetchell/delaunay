# v0.8.0 Reviewer Artifact Guide

This file is the reviewer entry point for the `delaunay` v0.8.0 software
artifact. It indexes the immutable release, reproduction commands, validation
evidence, generated figures, benchmark evidence, and known limits. The linked
documents remain the source of truth for their respective contracts.

> **Release status:** v0.8.0 was published on 2026-07-28 as an annotated tag, a
> crates.io package, a [GitHub release][github-release], and a
> [versioned Zenodo record][zenodo-version]. The GitHub release does not contain
> the planned Criterion baseline asset because its release-benchmark workflow
> failed; see [Known benchmark exception](#known-benchmark-exception).

## Artifact identity

| Field | v0.8.0 identity | Check |
|---|---|---|
| Source tag | [`v0.8.0`][source-tag] | The annotated release tag, not `main` |
| Commit | `ee0b1992071552ecf8669418ea74444a699343a9` | `git rev-list -n 1 v0.8.0` |
| GitHub archive | [`delaunay` at `v0.8.0`][source-tag] | Tag and commit must agree |
| GitHub release | [`v0.8.0`][github-release] | Published 2026-07-28 from the annotated tag |
| Crate | [`delaunay` 0.8.0 on crates.io][crate-release] | Package version must be `0.8.0` |
| Rust | 1.97.1 | `rustc --version` and [`rust-toolchain.toml`](../rust-toolchain.toml) |
| Zenodo collection | [`10.5281/zenodo.16931097`][zenodo-concept] | Stable software concept DOI |
| Zenodo v0.8.0 record | [`10.5281/zenodo.21635965`][zenodo-version] | SHA-256 `7ec456da83d105f807ac40737d38e102b30f5963467d410354925cf197cb45a7` |

The published commit is recorded explicitly above and must also resolve from
the annotated tag. In a Git checkout, verify both identities with:

```bash
TAG=v0.8.0
RELEASE_COMMIT=ee0b1992071552ecf8669418ea74444a699343a9
test "$(git rev-list -n 1 "$TAG")" = "$RELEASE_COMMIT"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git show -s --format='tag=%D%ncommit=%H%ncommit-date=%cI' "$TAG^{commit}"
rustc --version
```

GitHub and Zenodo source archives do not contain `.git` metadata. For an
archive, verify its published checksum and confirm that the release metadata
records the same tag and peeled commit before running the reproduction paths.
The Zenodo file is `acgetchell/delaunay-v0.8.0.zip`; the record also reports
MD5 `6274e49bb1ef91e1e3023225131199d7`. Verify the stronger checksum with:

```bash
printf '%s\n' '7ec456da83d105f807ac40737d38e102b30f5963467d410354925cf197cb45a7  acgetchell/delaunay-v0.8.0.zip' | sha256sum --check
```

The release-specific Zenodo DOI is
[`10.5281/zenodo.21635965`][zenodo-version].
[`CITATION.cff`](../CITATION.cff) intentionally uses the stable concept DOI
[`10.5281/zenodo.16931097`][zenodo-concept] to identify the software collection
across releases; cite the version DOI when referring specifically to v0.8.0.

### Which download to use

Use the GitHub release source archive or the Zenodo v0.8.0 archive for this
reproduction workflow. Those snapshots contain the `justfile`, support scripts,
notebooks, paper files, tests, benchmarks, and this guide.

The crates.io package is the library distribution, not the reviewer
reproduction bundle. Its explicit `Cargo.toml` allowlist includes this guide as
a reviewer signpost, along with public source, active docs, examples, tests,
notebooks, the paper snapshot, and benchmark sources. It deliberately excludes
the `justfile`, repository support scripts, and CI workflows, so the canonical
commands cannot be run from the crate package alone. Use the GitHub or Zenodo
snapshot for executable reproduction.

## Environment and setup

The repository pins Rust 1.97.1 in both [`Cargo.toml`](../Cargo.toml) and
[`rust-toolchain.toml`](../rust-toolchain.toml). From the exact release
checkout, install the repository-owned tools and dependencies with:

```bash
just setup
```

The quick and standard paths require Rust/Cargo, `just`, the uv-managed Python
environment, and the test tools installed by repository setup. The paper path
also needs the native TeX/Tectonic prerequisites listed in
[`docs/dev/commands.md`](../docs/dev/commands.md#paper-build). Exact command
ownership, platform notes, and validator selection remain in that command
guide.

## Reproduction paths

Run one path from the release root. Runtime classes are qualitative because
cold compilation, exact predicates, notebook rendering, and benchmarks depend
on hardware and local caches.

- **Quick correctness**
  - Commands: `just test`
  - Coverage: default Rust unit, doctest, integration, CLI, and Python
    correctness buckets
  - Runtime class: routine; no slow tests, figures, or benchmarks
- **Standard artifact**
  - Commands: `just ci`
  - Coverage: GitHub-equivalent formatting, config, docs, version sync,
    Rust/Python/notebook checks, default tests, examples, and benchmark
    compilation
  - Runtime class: standard CI workload
- **Full correctness and paper**
  - Commands: `just ci-slow`, then `just papers`, then `just publish-check`
  - Coverage: standard CI, explicit slow correctness tests, regenerated
    validation figures, paper lint/build/check, and crates.io metadata rules
  - Runtime class: extended; paper tools and slow tests required
- **Manual evidence**
  - Commands: run only the focused commands below
  - Coverage: visual, large-scale, or measured performance evidence
  - Runtime class: hardware-sensitive and potentially long-running

Successful correctness commands exit with status 0 and report no failed tests
or validators. `just papers` refreshes tracked outputs; on the immutable release
checkout they should reproduce without a content change:

```bash
git --no-pager diff --exit-code -- \
  docs/assets/validation papers/validation.pdf
```

### Visual inspection and validation figures

The generic visualization/export artifact is implemented by
[`delaunay::io::visualization`](../src/io/visualization.rs), specified in the
[`Mesh Export Schema`](../docs/mesh_export.md), and covered by
[`tests/mesh_export.rs`](../tests/mesh_export.rs). It exports stable UUID-based
simplicial-complex primitives rather than exposing private TDS handles.

Run the quickstart visual-inspection notebook:

```bash
just notebook-execute notebooks/00_quickstart.ipynb
```

The executed notebook, JSON exports, and PNG previews are written below
`target/notebooks/00_quickstart/`. The source notebook remains output-free.

Regenerate the hierarchy overview and the five standalone validation figures:

```bash
just validation-doc-figures
```

The expected tracked outputs are:

- [`validation_hierarchy.png`](../docs/assets/validation/validation_hierarchy.png)
- [`validation_level_1_element_validity.png`](../docs/assets/validation/validation_level_1_element_validity.png)
- [`validation_level_2_combinatorial_consistency.png`](../docs/assets/validation/validation_level_2_combinatorial_consistency.png)
- [`validation_level_3_intrinsic_pl_topology.png`](../docs/assets/validation/validation_level_3_intrinsic_pl_topology.png)
- [`validation_level_4_valid_realization.png`](../docs/assets/validation/validation_level_4_valid_realization.png)
- [`validation_level_5_geometric_predicates.png`](../docs/assets/validation/validation_level_5_geometric_predicates.png)

The canonical generator is
[`notebooks/01_validation.ipynb`](../notebooks/01_validation.ipynb). The same
figures are consumed by [`docs/validation.md`](../docs/validation.md) and the
paper snapshot; there is no duplicate artifact implementation here. The
spherical README image has a separate deliberate refresh path:

```bash
just spherical-readme-hero
```

### Benchmark evidence

Correctness evidence comes from validators and tests. Benchmarks characterize
cost and observability only after those invariants pass.

| Purpose | Command | Output |
|---|---|---|
| Curated release-signal measurements | `just bench-latest` | Fresh Criterion results under `target/criterion/` |
| Rebuild the checked-in public summary | `just bench-perf-summary` | [`benches/PERFORMANCE_RESULTS.md`](../benches/PERFORMANCE_RESULTS.md) |
| Same-machine comparison with the latest published release | `just perf-local` | `target/bench-reports/performance.md` |
| Compare two durable GitHub release assets | `just perf-github-assets "$TAG" "$PREVIOUS_TAG"` | `target/bench-reports/github-assets-performance.md` |
| Coarse 2D–5D large-scale guard | `just perf-large-scale-smoke` | Per-dimension pass/fail wall-clock diagnostics |

Absolute timings are not portable across machines. Use same-machine comparisons
for regression claims and stored GitHub release assets for runner-to-runner
release comparisons when both releases provide them. See
[`benches/README.md`](../benches/README.md) for the benchmark contract and
[`docs/dev/perf-tuning.md`](../docs/dev/perf-tuning.md) for evidence rules.

#### Known benchmark exception

The v0.8.0 GitHub release has no
`delaunay-v0.8.0-criterion-baseline.tar.gz` asset. The
[release-benchmark run][release-benchmark-run] failed when
`ci_performance_suite` exceeded its 900-second command timeout, before packaging
or upload. The checked-in v0.8.0 performance summary also identifies commit
`1f15b9352456aca9eade44d116cd8d15d3d1ac20`, not the peeled release commit.
Consequently, do not use that summary or `perf-github-assets` as durable v0.8.0
release-comparison evidence. Run `just perf-local` for a same-machine comparison,
or regenerate measurements from the exact release checkout and record the
environment metadata alongside the results.

## Five-level claim map

The canonical definitions and API semantics are in the
[`Validation Guide`](../docs/validation.md) and
[`Invariant Guide`](../docs/invariants.md). This table only maps those
contracts to concrete artifact evidence.

### Level 1 — Element Validity

- Implementation: [`vertex.rs`](../src/core/vertex.rs),
  [`simplex.rs`](../src/core/simplex.rs), and
  [`tds/validation.rs`](../src/core/tds/validation.rs)
- Representative tests: [`proptest_vertex.rs`](../tests/proptest_vertex.rs),
  [`proptest_simplex.rs`](../tests/proptest_simplex.rs), and
  [`proptest_tds.rs`](../tests/proptest_tds.rs)
- Documentation: [Level 1 contract](../docs/validation.md#level-1-element-validity)
  and [figure](../docs/assets/validation/validation_level_1_element_validity.png)
- Performance evidence: `structure` component in
  [`profiling_suite.rs`](../benches/profiling_suite.rs)

### Level 2 — Combinatorial Consistency

- Implementation: [`tds/validation.rs`](../src/core/tds/validation.rs) and
  [`adjacency.rs`](../src/core/adjacency.rs)
- Representative tests: [`proptest_tds.rs`](../tests/proptest_tds.rs),
  [`proptest_triangulation.rs`](../tests/proptest_triangulation.rs), and
  [`public_topology_api.rs`](../tests/public_topology_api.rs)
- Documentation:
  [Level 2 contract](../docs/validation.md#level-2-combinatorial-consistency)
  and
  [figure](../docs/assets/validation/validation_level_2_combinatorial_consistency.png)
- Performance evidence: `structure` component in
  [`profiling_suite.rs`](../benches/profiling_suite.rs)

### Level 3 — Intrinsic PL Topology

- Implementation: [`triangulation/validation.rs`](../src/triangulation/validation.rs),
  [`manifold.rs`](../src/topology/manifold.rs), and
  [`characteristics/validation.rs`](../src/topology/characteristics/validation.rs)
- Representative tests:
  [`proptest_euler_characteristic.rs`](../tests/proptest_euler_characteristic.rs),
  [`public_topology_api.rs`](../tests/public_topology_api.rs), and
  [`proptest_toroidal.rs`](../tests/proptest_toroidal.rs)
- Documentation:
  [Level 3 contract](../docs/validation.md#level-3-intrinsic-pl-topology)
  and
  [figure](../docs/assets/validation/validation_level_3_intrinsic_pl_topology.png)
- Performance evidence: `triangulation` component in
  [`profiling_suite.rs`](../benches/profiling_suite.rs)

### Level 4 — Valid Realization

- Implementation: [`triangulation/realization.rs`](../src/triangulation/realization.rs),
  [`geometry/realization.rs`](../src/geometry/realization.rs), and the spherical
  backend in [`spherical.rs`](../src/delaunay/spherical.rs)
- Representative tests: unit tests in
  [`triangulation/realization.rs`](../src/triangulation/realization.rs),
  [`triangulation_builder.rs`](../tests/triangulation_builder.rs),
  [`proptest_toroidal.rs`](../tests/proptest_toroidal.rs), and
  [`spherical_delaunay.rs`](../tests/spherical_delaunay.rs)
- Documentation: [Level 4 contract](../docs/validation.md#level-4-valid-realization)
  and [figure](../docs/assets/validation/validation_level_4_valid_realization.png)
- Performance evidence: `realization` component in
  [`profiling_suite.rs`](../benches/profiling_suite.rs)

### Level 5 — Geometric Predicates

- Implementation: [`delaunay/validation.rs`](../src/delaunay/validation.rs),
  [`property_validation.rs`](../src/delaunay/property_validation.rs), and the
  spherical backend in [`spherical.rs`](../src/delaunay/spherical.rs)
- Representative tests:
  [`proptest_delaunay_triangulation.rs`](../tests/proptest_delaunay_triangulation.rs),
  [`proptest_toroidal.rs`](../tests/proptest_toroidal.rs), and
  [`spherical_delaunay.rs`](../tests/spherical_delaunay.rs)
- Documentation: [Level 5 contract](../docs/validation.md#level-5-geometric-predicates)
  and [figure](../docs/assets/validation/validation_level_5_geometric_predicates.png)
- Performance evidence: `delaunay` and cumulative `full` components in
  [`profiling_suite.rs`](../benches/profiling_suite.rs)

[`ci_performance_suite.rs`](../benches/ci_performance_suite.rs) supplies the
stable cumulative Levels 1–5 release-signal benchmark. Component benchmarks
make cost visible; they do not replace the layer-specific correctness tests.

### Cross-cutting evidence

- **Adaptive exact predicate signs and deterministic degeneracy handling**
  - Implementation: [`predicates.rs`](../src/geometry/predicates.rs),
    [`robust_predicates.rs`](../src/geometry/robust_predicates.rs), and
    [`sos.rs`](../src/geometry/sos.rs)
  - Tests and documentation:
    [`proptest_predicates.rs`](../tests/proptest_predicates.rs),
    [`proptest_sos.rs`](../tests/proptest_sos.rs), and the
    [Numerical Robustness Guide](../docs/numerical_robustness_guide.md)
- **Generic stable-ID visualization/export data**
  - Implementation: [`io/visualization.rs`](../src/io/visualization.rs)
  - Tests and documentation: [`mesh_export.rs`](../tests/mesh_export.rs), the
    [schema guide](../docs/mesh_export.md), and
    [`00_quickstart.ipynb`](../notebooks/00_quickstart.ipynb)
- **Dimension-generic construction and invariant coverage**
  - Implementation: [`delaunay/construction.rs`](../src/delaunay/construction.rs)
    and [`triangulation/construction.rs`](../src/triangulation/construction.rs)
  - Tests and documentation: the [test index](../tests/README.md),
    [property-testing summary](../docs/property_testing_summary.md), and 2D–5D
    release-signal cases in
    [`ci_performance_suite.rs`](../benches/ci_performance_suite.rs)
- **Safe Rust implementation**
  - Implementation: [`lib.rs`](../src/lib.rs) and the manifest lint policy in
    [`Cargo.toml`](../Cargo.toml)
  - Tests and documentation: `just ci` compiles all supported targets with
    unsafe code forbidden

## Known limitations

[`docs/limitations.md`](../docs/limitations.md) is the canonical and more
detailed scope statement. Reviewers should interpret results within these
boundaries:

- **Euclidean:** this is the default and best-covered construction path.
  Routine release coverage is strongest in 2D through 5D. Exact orientation is
  available through D=6, while exact insphere determinants are limited to D≤5.
- **Toroidal:** periodic quotient construction is release-covered on `T^2` and
  compact `T^3`; `T^4`/`T^5` quotient construction fails explicitly pending
  scalable selection and diagnostics.
- **Spherical:** bounded `S^2`/`S^3` construction and model-specific Level 4/5
  validation are prototypes. Full `S^2`–`S^5` integration and the ordinary mutable
  editing surface remain out of scope for v0.8.0.
- **Performance:** exact predicates dominate some near-degenerate and
  high-dimensional inputs. Large-scale and Criterion timings depend on the
  machine, compiler, profile, load, and fixture.
- **Feature scope:** constrained Delaunay triangulation, direct Voronoi
  extraction, native GUI visualization, parallel/GPU construction, and
  out-of-core meshing are not implemented.

## Prior art and novelty boundary

This artifact does not claim novelty for established Delaunay theory,
triangulation data structures, or standard computational-geometry algorithms.
The repository's [`REFERENCES.md`](../REFERENCES.md) records the algorithmic
literature. Relevant software prior art includes:

- [CGAL](https://www.cgal.org/) for broad, production computational geometry
  data structures and algorithms, including triangulation and meshing.
- [Spade](https://docs.rs/spade/latest/spade/) for robust two-dimensional
  Delaunay and constrained Delaunay triangulations in Rust.

The artifact evidence is limited to the documented implementation, validation,
reproducibility, and performance contracts of this release.

## Manuscript snapshot

The full GitHub/Zenodo artifact includes [`papers/validation.tex`](../papers/validation.tex)
and the compiled reviewer copy [`papers/validation.pdf`](../papers/validation.pdf)
because they reuse the same validation tables and generated figures. They are a
source snapshot, not a claim that the manuscript is submission-complete.
Substantive author-owned manuscript completion and submission packaging are
tracked separately in [issue #522][manuscript].

## Published release record

The annotated tag, crates.io package, GitHub release, and Zenodo record were
published on 2026-07-28. The source identity is the peeled commit recorded above.
The Zenodo archive supplies the immutable version DOI and checksum for reviewer
verification. The missing GitHub Criterion asset is the sole incomplete release
artifact identified by this guide and is documented as a benchmark limitation,
not as evidence that the source release is unpublished.

[crate-release]: https://crates.io/crates/delaunay/0.8.0
[github-release]: https://github.com/acgetchell/delaunay/releases/tag/v0.8.0
[manuscript]: https://github.com/acgetchell/delaunay/issues/522
[release-benchmark-run]: https://github.com/acgetchell/delaunay/actions/runs/30325806574
[source-tag]: https://github.com/acgetchell/delaunay/tree/v0.8.0
[zenodo-concept]: https://doi.org/10.5281/zenodo.16931097
[zenodo-version]: https://doi.org/10.5281/zenodo.21635965
