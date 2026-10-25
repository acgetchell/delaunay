# v0.8.0 Reviewer Artifact Guide

This file is the reviewer entry point for the `delaunay` v0.8.0 software
artifact. It indexes the immutable release, reproduction commands, validation
evidence, generated figures, benchmark evidence, and known limits. The linked
documents at commit `ee0b1992071552ecf8669418ea74444a699343a9` remain the source
of truth for the release's contracts, including their historical paths and commands.

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
| Rust | 1.97.1 | `rustc --version` and [`rust-toolchain.toml`][release-toolchain] |
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
[`CITATION.cff`][release-citation] intentionally uses the stable concept DOI
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

The release pins Rust 1.97.1 in both [`Cargo.toml`][release-cargo] and
[`rust-toolchain.toml`][release-toolchain]. From the exact release
checkout, install the repository-owned tools and dependencies with:

```bash
just setup
```

The quick and standard paths require Rust/Cargo, `just`, the uv-managed Python
environment, and the test tools installed by repository setup. The paper path
also needs the native TeX/Tectonic prerequisites listed in
[`docs/dev/commands.md`][release-paper-build]. Exact command
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
[`delaunay::io::visualization`][release-visualization], specified in the
[`Mesh Export Schema`][release-mesh-export-guide], and covered by
[`tests/mesh_export.rs`][release-mesh-export-test]. It exports stable UUID-based
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

- [`validation_hierarchy.png`][release-hierarchy-figure]
- [`validation_level_1_element_validity.png`][release-level-1-figure]
- [`validation_level_2_combinatorial_consistency.png`][release-level-2-figure]
- [`validation_level_3_intrinsic_pl_topology.png`][release-level-3-figure]
- [`validation_level_4_valid_realization.png`][release-level-4-figure]
- [`validation_level_5_geometric_predicates.png`][release-level-5-figure]

The canonical generator is
[`notebooks/01_validation.ipynb`][release-validation-notebook]. The same
figures are consumed by [`docs/validation.md`][release-validation-guide] and the
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
| Rebuild the checked-in public summary | `just bench-perf-summary` | [`benches/PERFORMANCE_RESULTS.md`][release-performance-results] |
| Same-machine comparison with the latest published release | `just perf-local` | `target/bench-reports/performance.md` |
| Compare two durable GitHub release assets | `just perf-github-assets "$TAG" "$PREVIOUS_TAG"` | `target/bench-reports/github-assets-performance.md` |
| Coarse 2D–5D large-scale guard | `just perf-large-scale-smoke` | Per-dimension pass/fail wall-clock diagnostics |

Absolute timings are not portable across machines. Use same-machine comparisons
for regression claims and stored GitHub release assets for runner-to-runner
release comparisons when both releases provide them. See
[`benches/README.md`][release-benchmark-guide] for the benchmark contract and
[`docs/dev/perf-tuning.md`][release-performance-guide] for evidence rules.

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
[`Validation Guide`][release-validation-guide] and
[`Invariant Guide`][release-invariants]. This table only maps those
contracts to concrete artifact evidence.

### Level 1 — Element Validity

- Implementation: [`vertex.rs`][release-vertex],
  [`simplex.rs`][release-simplex], and
  [`tds/validation.rs`][release-tds-validation]
- Representative tests: [`proptest_vertex.rs`][release-proptest-vertex],
  [`proptest_simplex.rs`][release-proptest-simplex], and
  [`proptest_tds.rs`][release-proptest-tds]
- Documentation: [Level 1 contract][release-level-1]
  and [figure][release-level-1-figure]
- Performance evidence: `structure` component in
  [`profiling_suite.rs`][release-profiling]

### Level 2 — Combinatorial Consistency

- Implementation: [`tds/validation.rs`][release-tds-validation] and
  [`adjacency.rs`][release-adjacency]
- Representative tests: [`proptest_tds.rs`][release-proptest-tds],
  [`proptest_triangulation.rs`][release-proptest-triangulation], and
  [`public_topology_api.rs`][release-public-topology]
- Documentation:
  [Level 2 contract][release-level-2]
  and
  [figure][release-level-2-figure]
- Performance evidence: `structure` component in
  [`profiling_suite.rs`][release-profiling]

### Level 3 — Intrinsic PL Topology

- Implementation: [`core/validation.rs`][release-topology-validation],
  [`manifold.rs`][release-manifold], and
  [`characteristics/validation.rs`][release-characteristics]
- Representative tests:
  [`proptest_euler_characteristic.rs`][release-proptest-euler],
  [`public_topology_api.rs`][release-public-topology], and
  [`proptest_toroidal.rs`][release-proptest-toroidal]
- Documentation:
  [Level 3 contract][release-level-3]
  and
  [figure][release-level-3-figure]
- Performance evidence: `triangulation` component in
  [`profiling_suite.rs`][release-profiling]

### Level 4 — Valid Realization

- Implementation: [`core/realization.rs`][release-owner-realization],
  [`geometry/realization.rs`][release-geometry-realization], and the spherical
  backend in [`spherical.rs`][release-spherical]
- Representative tests: unit tests in
  [`core/realization.rs`][release-owner-realization],
  [`triangulation_builder.rs`][release-triangulation-builder],
  [`proptest_toroidal.rs`][release-proptest-toroidal], and
  [`spherical_delaunay.rs`][release-spherical-test]
- Documentation: [Level 4 contract][release-level-4]
  and [figure][release-level-4-figure]
- Performance evidence: `realization` component in
  [`profiling_suite.rs`][release-profiling]

### Level 5 — Geometric Predicates

- Implementation: [`delaunay/validation.rs`][release-delaunay-validation],
  [`property_validation.rs`][release-property-validation], and the
  spherical backend in [`spherical.rs`][release-spherical]
- Representative tests:
  [`proptest_delaunay_triangulation.rs`][release-proptest-delaunay],
  [`proptest_toroidal.rs`][release-proptest-toroidal], and
  [`spherical_delaunay.rs`][release-spherical-test]
- Documentation: [Level 5 contract][release-level-5]
  and [figure][release-level-5-figure]
- Performance evidence: `delaunay` and cumulative `full` components in
  [`profiling_suite.rs`][release-profiling]

[`ci_performance_suite.rs`][release-ci-performance] supplies the
stable cumulative Levels 1–5 release-signal benchmark. Component benchmarks
make cost visible; they do not replace the layer-specific correctness tests.

### Cross-cutting evidence

- **Adaptive exact predicate signs and deterministic degeneracy handling**
  - Implementation: [`predicates.rs`][release-predicates],
    [`robust_predicates.rs`][release-robust-predicates], and
    [`sos.rs`][release-sos]
  - Tests and documentation:
    [`proptest_predicates.rs`][release-proptest-predicates],
    [`proptest_sos.rs`][release-proptest-sos], and the
    [Numerical Robustness Guide][release-robustness-guide]
- **Generic stable-ID visualization/export data**
  - Implementation: [`io/visualization.rs`][release-visualization]
  - Tests and documentation: [`mesh_export.rs`][release-mesh-export-test], the
    [schema guide][release-mesh-export-guide], and
    [`00_quickstart.ipynb`][release-quickstart]
- **Dimension-generic construction and invariant coverage**
  - Implementation: [`delaunay/construction.rs`][release-delaunay-construction]
    and [`core/construction.rs`][release-triangulation-construction]
  - Tests and documentation: the [test index][release-test-index],
    [property-testing summary][release-property-testing], and 2D–5D
    release-signal cases in
    [`ci_performance_suite.rs`][release-ci-performance]
- **Safe Rust implementation**
  - Implementation: [`lib.rs`][release-lib] and the manifest lint policy in
    [`Cargo.toml`][release-cargo]
  - Tests and documentation: `just ci` compiles all supported targets with
    unsafe code forbidden

## Known limitations

[`docs/limitations.md`][release-limitations] is the canonical and more
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
The release's [`REFERENCES.md`][release-references] records the algorithmic
literature. Relevant software prior art includes:

- [CGAL](https://www.cgal.org/) for broad, production computational geometry
  data structures and algorithms, including triangulation and meshing.
- [Spade](https://docs.rs/spade/latest/spade/) for robust two-dimensional
  Delaunay and constrained Delaunay triangulations in Rust.

The artifact evidence is limited to the documented implementation, validation,
reproducibility, and performance contracts of this release.

## Manuscript snapshot

The full GitHub/Zenodo artifact includes [`papers/validation.tex`][release-paper-source]
and the compiled reviewer copy [`papers/validation.pdf`][release-paper-pdf]
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
[release-adjacency]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/adjacency.rs
[release-benchmark-guide]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/benches/README.md
[release-cargo]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/Cargo.toml
[release-characteristics]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/topology/characteristics/validation.rs
[release-ci-performance]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/benches/ci_performance_suite.rs
[release-citation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/CITATION.cff
[release-delaunay-construction]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/delaunay/construction.rs
[release-delaunay-validation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/delaunay/validation.rs
[release-geometry-realization]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/geometry/realization.rs
[release-hierarchy-figure]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_hierarchy.png
[release-invariants]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/invariants.md
[release-level-1]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md#level-1-element-validity
[release-level-1-figure]:
<https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_level_1_element_validity.png>
[release-level-2]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md#level-2-combinatorial-consistency
[release-level-2-figure]:
<https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_level_2_combinatorial_consistency.png>
[release-level-3]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md#level-3-intrinsic-pl-topology
[release-level-3-figure]:
<https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_level_3_intrinsic_pl_topology.png>
[release-level-4]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md#level-4-valid-realization
[release-level-4-figure]:
<https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_level_4_valid_realization.png>
[release-level-5]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md#level-5-geometric-predicates
[release-level-5-figure]:
<https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/assets/validation/validation_level_5_geometric_predicates.png>
[release-lib]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/lib.rs
[release-limitations]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/limitations.md
[release-manifold]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/topology/manifold.rs
[release-mesh-export-guide]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/mesh_export.md
[release-mesh-export-test]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/mesh_export.rs
[release-owner-realization]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/realization.rs
[release-paper-build]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/dev/commands.md#paper-build
[release-paper-pdf]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/papers/validation.pdf
[release-paper-source]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/papers/validation.tex
[release-performance-guide]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/dev/perf-tuning.md
[release-performance-results]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/benches/PERFORMANCE_RESULTS.md
[release-predicates]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/geometry/predicates.rs
[release-profiling]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/benches/profiling_suite.rs
[release-property-testing]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/property_testing_summary.md
[release-property-validation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/delaunay/property_validation.rs
[release-proptest-delaunay]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_delaunay_triangulation.rs
[release-proptest-euler]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_euler_characteristic.rs
[release-proptest-predicates]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_predicates.rs
[release-proptest-simplex]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_simplex.rs
[release-proptest-sos]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_sos.rs
[release-proptest-tds]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_tds.rs
[release-proptest-toroidal]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_toroidal.rs
[release-proptest-triangulation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_triangulation.rs
[release-proptest-vertex]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/proptest_vertex.rs
[release-public-topology]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/public_topology_api.rs
[release-quickstart]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/notebooks/00_quickstart.ipynb
[release-references]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/REFERENCES.md
[release-robust-predicates]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/geometry/robust_predicates.rs
[release-robustness-guide]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/numerical_robustness_guide.md
[release-simplex]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/simplex.rs
[release-sos]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/geometry/sos.rs
[release-spherical]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/delaunay/spherical.rs
[release-spherical-test]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/spherical_delaunay.rs
[release-tds-validation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/tds/validation.rs
[release-test-index]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/README.md
[release-toolchain]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/rust-toolchain.toml
[release-topology-validation]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/validation.rs
[release-triangulation-builder]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/tests/triangulation_builder.rs
[release-triangulation-construction]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/construction.rs
[release-validation-guide]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/docs/validation.md
[release-validation-notebook]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/notebooks/01_validation.ipynb
[release-vertex]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/core/vertex.rs
[release-visualization]: https://github.com/acgetchell/delaunay/blob/ee0b1992071552ecf8669418ea74444a699343a9/src/io/visualization.rs
