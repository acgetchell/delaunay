# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes

- Replace custom changelog pipeline with git-cliff

### Changed

- 📝 Add docstrings to `chore/simplify-changelog-pipeline` [`99f5449`](https://github.com/acgetchell/delaunay/commit/99f5449a04cf0bf734dc7c9444dc8a278675dad3)

### Maintenance

- [**breaking**] Replace custom changelog pipeline with git-cliff
  [`17c5b50`](https://github.com/acgetchell/delaunay/commit/17c5b500ebaee07a6e630ab7ec388c8bab19f84f)

Replace ~4,000 lines of custom Python changelog generation
  (changelog_utils.py, enhance_commits.py) with git-cliff and two
  focused scripts (~640 lines total):

- postprocess_changelog.py: lightweight markdown hygiene
    (MD004, MD007, MD012, MD013, MD030, MD031, MD032, MD040)
    plus summary section injection (Merged PRs, Breaking Changes)

- tag_release.py: extract latest version section for git tag messages

  Pipeline changes:

- cliff.toml: full git-cliff config with conventional commit parsing,
    PR auto-linking, and Tera template for Keep a Changelog format

- justfile: new changelog-update, changelog-tag, changelog recipes
    replacing the old generate-changelog workflow

- Idempotent output: postprocessor matches markdownlint --fix exactly,
    so `just changelog-update && just fix` produces zero diff

  Tooling simplification:

- Remove mypy in favor of ty (Astral) — mypy's permissive config was
    catching nothing that ty doesn't already cover

- Disable markdownlint MD037 (false positives on cron expressions and
    glob patterns like saturating_*)

## [0.7.2] - 2026-03-10

### Merged Pull Requests

- Release v0.7.2 [#246](https://github.com/acgetchell/delaunay/pull/246)
- Exact insphere predicates with f64 fast filter [#245](https://github.com/acgetchell/delaunay/pull/245)
- Use exact-sign orientation in robust_orientation() [#244](https://github.com/acgetchell/delaunay/pull/244)
- Bump actions/setup-node from 6.2.0 to 6.3.0 [#239](https://github.com/acgetchell/delaunay/pull/239)
- Bump taiki-e/install-action from 2.68.16 to 2.68.25 [#237](https://github.com/acgetchell/delaunay/pull/237)
- Use exact arithmetic for orientation predicates [#235](https://github.com/acgetchell/delaunay/pull/235)
  [#236](https://github.com/acgetchell/delaunay/pull/236)
- Switch FastKernel to insphere_lifted and enable LTO [#234](https://github.com/acgetchell/delaunay/pull/234)
- Deduplicate D<4 repair fallback and improve diagnostics [#232](https://github.com/acgetchell/delaunay/pull/232)
- Resolve 3D seeded bulk construction orientation convergence failure [#228](https://github.com/acgetchell/delaunay/pull/228)
- Bump actions-rust-lang/setup-rust-toolchain [#226](https://github.com/acgetchell/delaunay/pull/226)
- Bump taiki-e/install-action from 2.68.8 to 2.68.16 [#225](https://github.com/acgetchell/delaunay/pull/225)
- Bump astral-sh/setup-uv from 7.3.0 to 7.3.1 [#224](https://github.com/acgetchell/delaunay/pull/224)
- Bump actions/upload-artifact from 6 to 7 [#223](https://github.com/acgetchell/delaunay/pull/223)
- Bump actions/download-artifact from 7.0.0 to 8.0.0 [#222](https://github.com/acgetchell/delaunay/pull/222)
- Introduce GlobalTopology behavior model adapter [#221](https://github.com/acgetchell/delaunay/pull/221)
- Enforce coherent orientation as a first-class invariant [#219](https://github.com/acgetchell/delaunay/pull/219)
- Use bulk Hilbert API in order_vertices_hilbert [#218](https://github.com/acgetchell/delaunay/pull/218)
- Improve Hilbert curve correctness and add bulk API [#207](https://github.com/acgetchell/delaunay/pull/207)
  [#216](https://github.com/acgetchell/delaunay/pull/216)
- Update docs for DelaunayTriangulationBuilder and toroidal topology [#215](https://github.com/acgetchell/delaunay/pull/215)
- Feat/210 toroidalspace periodic [#213](https://github.com/acgetchell/delaunay/pull/213)
- Bump taiki-e/install-action from 2.67.30 to 2.68.8 [#211](https://github.com/acgetchell/delaunay/pull/211)

### Added

- Enforce coherent orientation as a first-class invariant [#219](https://github.com/acgetchell/delaunay/pull/219)
  [`350f614`](https://github.com/acgetchell/delaunay/commit/350f614c3e18d148bfc88809c28fdc2de362dd9a)

- feat(tds): enforce coherent orientation as a first-class invariant

  - add Level 2 coherent-orientation validation via `is_coherently_oriented()` and `OrientationViolation` diagnostics
  - preserve/normalize orientation across flips, cavity/neighbor rebuild paths, periodic quotient reconstruction, and vertex-removal retriangulation
  - add orientation coverage in `tests/tds_orientation.rs`, document the invariant, and update related docs/doctest examples
- Use exact arithmetic for orientation predicates [#235](https://github.com/acgetchell/delaunay/pull/235)
  [#236](https://github.com/acgetchell/delaunay/pull/236) [`a62437f`](https://github.com/acgetchell/delaunay/commit/a62437f25c27259f145d3c193ce149ee14b421c7)

- feat: use exact arithmetic for orientation predicates [#235](https://github.com/acgetchell/delaunay/pull/235)

  Switch to la-stack v0.2.1 with the `exact` feature to obtain provably
  correct simplex orientation via `det_sign_exact()`.

  Orientation predicates:

  - Add `orientation_from_matrix()` using `det_sign_exact()` with a
    finite-entry guard and adaptive-tolerance fallback for non-finite cases

  - `simplex_orientation()` now delegates to `orientation_from_matrix()`,
    eliminating manual tolerance comparison

  insphere_lifted optimization:

  - Reuse the relative-coordinate block already in the lifted matrix for
    orientation instead of re-converting all D+1 simplex points

  - Combine the dimension-parity sign and (-1)^D orientation correction
    into a single simplified formula: det_norm = −det × rel_sign

  - Remove `dimension_is_even`, `parity_sign`, and `orient_sign` variables

  Stack-matrix dispatch cleanup:

  - Reduce MAX_STACK_MATRIX_DIM from 18 to 7 (matching tested D≤5 range)
  - Replace 19 hand-written match arms with a compact repetition macro
  - Add `matrix_zero_like()` helper for creating same-sized zero matrices
    within macro-dispatched blocks without nested dispatch

- Use exact-sign orientation in robust_orientation() [#244](https://github.com/acgetchell/delaunay/pull/244)
  [`2869cfe`](https://github.com/acgetchell/delaunay/commit/2869cfea111dbca3641e7f88119d67b93a0d4841)

- feat: use exact-sign orientation in robust_orientation()

  Replace f64 determinant + adaptive tolerance in `robust_orientation()`
  with `orientation_from_matrix()`, which uses `det_sign_exact()` for
  provably correct sign classification on finite inputs.

  - Make `orientation_from_matrix` pub(crate) so robust_predicates can call it
  - Remove adaptive_tolerance / manual threshold comparison from robust_orientation()
  - Add near-degenerate 2D and 3D tests that exercise the exact-sign path
- Exact insphere predicates with f64 fast filter [#245](https://github.com/acgetchell/delaunay/pull/245)
  [`fed429f`](https://github.com/acgetchell/delaunay/commit/fed429f281cb2bc2e4a97cd99ac1770ade76a202)

- feat: exact insphere predicates with f64 fast filter

  - Add `insphere_from_matrix` helper in predicates.rs using a 3-stage
    approach: f64 fast filter → exact Bareiss → f64 fallback

  - Update `insphere`, `insphere_lifted`, `adaptive_tolerance_insphere`,
    and `conditioned_insphere` to use exact-sign path

  - Remove dead `interpret_insphere_determinant` function
  - Add near-cocircular and near-cospherical exact-sign tests
  - Switch convex hull performance test to FastKernel to avoid 5×5
    exact Bareiss on cospherical inputs

  - Document lint suppression preference (`expect` over `allow`) in
    docs/dev/rust.md

### Changed

- Feat/210 toroidalspace periodic [#213](https://github.com/acgetchell/delaunay/pull/213)
  [`c172796`](https://github.com/acgetchell/delaunay/commit/c1727967ae2c92440c42413c10e1d5859d4cb561)
- Introduce GlobalTopology behavior model adapter [#221](https://github.com/acgetchell/delaunay/pull/221)
  [`e56265b`](https://github.com/acgetchell/delaunay/commit/e56265bafeb4a1e0e65c72b2037ab0e747af7ffa)

- refactor(topology): introduce GlobalTopology behavior model adapter

  - add internal `GlobalTopologyModel` abstraction and concrete models
    (euclidean, toroidal, spherical scaffold, hyperbolic scaffold)

  - add `GlobalTopologyModelAdapter` and `GlobalTopology::model()` bridge to keep
    `GlobalTopology<D>` as the stable public metadata/config surface

  - migrate triangulation orientation lifting to model-based behavior calls
  - migrate builder toroidal validation/canonicalization to model-based calls
  - update topology/code-organization docs for metadata-vs-behavior split

  - Changed: Improve global topology model validation and consistency

  Enhances periodic cell offset validation by leveraging `supports_periodic_facet_signatures`.
  Introduces robust checks for non-finite coordinates during point canonicalization
  in toroidal models, preventing invalid states.
  Refactors the `GlobalTopologyModelAdapter` to consistently delegate all trait
  method calls to specific underlying topology model implementations,
  improving maintainability.
  Updates error messages for clarity during topology validation.
  Optimizes `periodic_domain` to return a reference, avoiding data copies.
  Adjusts internal module visibility and re-exports `ToroidalConstructionMode` to prelude.

  - refactor: add comprehensive documentation and tests for GlobalTopologyModel

  Enhance the global_topology_model module with extensive documentation and unit test coverage:

  Documentation improvements:

  - Add module-level overview explaining trait abstraction and concrete implementations
  - Enhance trait method documentation for periodic_domain() and supports_periodic_facet_signatures()
  - Document public methods: ToroidalModel::new(), GlobalTopology::model(), GlobalTopologyModelAdapter::from_global_topology()

  Test coverage (41 tests added):

  - EuclideanModel: comprehensive trait method coverage
  - SphericalModel and HyperbolicModel: scaffolded model behavior
  - GlobalTopologyModelAdapter: delegation verification for all trait methods
  - Error handling: zero/negative/infinite/NaN periods, non-finite coordinates
  - Edge cases: large coordinates, exact periods, zero/large offsets, f32 scalars, 5D

  Code quality improvements:

  - Fix nitpick: delegate kind() and allows_boundary() in GlobalTopologyModelAdapter
  - Fix nitpick: add NaN coordinate test for canonicalize_point_in_place
  - Fix nitpick: add finiteness validation to lift_for_orientation with test
  - Apply cargo fmt formatting
  - Fix clippy warnings (float_cmp, suboptimal_flops)

  - Changed: Optimize facet vertex processing and improve periodic facet key determinism

  Moves the facet vertex buffer initialization to only execute on the non-periodic path,
  avoiding unnecessary work for periodic cells and improving efficiency.

  Enhances the `periodic_facet_key_from_lifted_vertices` function to ensure
  deterministic sorting by considering both the vertex key value and its periodic
  offset. This prevents inconsistencies when multiple lifted vertices share the same
  base key.

- Make Codacy analysis step continue on error [`24a1ad6`](https://github.com/acgetchell/delaunay/commit/24a1ad6a7e17025f40ea9d4f626f302670664c39)

Configures the Codacy analysis step in the CI workflow to continue on
  error. This prevents the entire workflow from failing due to intermittent
  issues with Codacy's opengrep/semgrep engine, ensuring subsequent steps,
  like SARIF report uploads, can still execute. This improves CI robustness.
  This is an internal CI workflow improvement.

- Enhance Codacy CI reliability and performance [`b980630`](https://github.com/acgetchell/delaunay/commit/b9806304bff5de8d6ee554e36ae5543929c6300f)

Disables the Semgrep engine within Codacy due to intermittent failures
  and excessively long runtimes observed in CI. Additionally, adds a
  timeout to the Codacy analysis step to prevent hung analyzers from
  consuming the full job timeout, improving overall workflow stability
  and resource utilization. This is an internal CI/CD change.

- Update MSRV to 1.93.1 and `sysinfo` dependency [`a2a42d5`](https://github.com/acgetchell/delaunay/commit/a2a42d58ed913b46bf81489658356dc4d09c3637)

Increment the Minimum Supported Rust Version (MSRV) to 1.93.1
  across all project configuration and documentation. This updates
  the pinned Rust toolchain, `Cargo.toml` settings, and `clippy.toml`
  MSRV, ensuring consistency and compatibility. Additionally, the
  `sysinfo` dependency is updated to 0.38.3. This is an internal
  maintenance change.

- Improve 3D incremental prefix debug harness [`515dade`](https://github.com/acgetchell/delaunay/commit/515dadeb975250217702a9ae1c0a705fc16f620f)

Refactors the `run_incremental_prefix_3d` function to use a batch construction
  method, aligning it with other large-scale debug tests and simplifying logic.
  This enhances initial triangulation robustness and error reporting by capturing
  detailed statistics from construction failures.

  Adds new environment variables, `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_PROBES` and
  `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_RUNTIME_SECS`, to control the bisection
  process, enabling more efficient and targeted debugging of failure points.

  Updates the `env_usize` helper to correctly parse environment variables
  provided in a `key=value` format, improving test configuration flexibility.

- Deduplicate D<4 repair fallback and improve diagnostics [#232](https://github.com/acgetchell/delaunay/pull/232)
  [`14ff1b3`](https://github.com/acgetchell/delaunay/commit/14ff1b3aad4ab4c1869018d6cbbb59d2d0456fd3)

- refactor: deduplicate D<4 repair fallback and improve diagnostics

  - Extract try_d_lt4_global_repair_fallback helper to eliminate
    duplicated repair-or-abort logic between the stats and non-stats
    branches of insert_remaining_vertices_seeded.

  - Enrich the soft post-condition diagnostic in
    normalize_and_promote_positive_orientation with the count of
    residual negative cells and a sample of up to 5 CellKeys.

  - Add test_construction_options_global_repair_fallback_toggle unit
    test verifying the without_global_repair_fallback builder toggle.

  - fix: switch default kernel from FastKernel to RobustKernel

  - Change all convenience constructors (new, new_with_topology_guarantee,
    new_with_options, empty, etc.) to use RobustKernel<f64>

  - Change builder build() default to RobustKernel<T>
  - Add RobustKernel to prelude::query exports
  - Update type annotations across tests, benches, and doc tests
  - Preserve FastKernel in tests that explicitly test it via with_kernel()

  - Changed: use RobustKernel for random generation and 3D examples

  Update examples and random triangulation utilities to use RobustKernel,
  aligning them with the core library's default. FastKernel is now
  explicitly documented as unreliable for 3D workloads due to floating-
  point precision issues in near-degenerate configurations. This change
  also adds topological admissibility checks for flip-based repairs and
  improves error diagnostics for per-insertion failures.

### Documentation

- Update docs for DelaunayTriangulationBuilder and toroidal topology [#215](https://github.com/acgetchell/delaunay/pull/215)
  [`a90526c`](https://github.com/acgetchell/delaunay/commit/a90526cd53be4cbe07c0add0b52ef04bd7243c3d)

- docs: update docs for DelaunayTriangulationBuilder and toroidal topology

  Update all documentation to reflect that toroidal topology is fully
  implemented and accessible via DelaunayTriangulationBuilder.

  Documentation updates:

  - docs/topology.md: Replace "future plumbing" language with current
    implementation status; add complete toroidal triangulation examples

  - docs/api_design.md: Split Builder API section into simple (::new())
    vs advanced (Builder) construction with toroidal examples

  - docs/workflows.md: Add new section for toroidal/periodic triangulations
    with practical examples and construction modes

  - docs/code_organization.md: Update file tree with missing files
    (invariants.md, workflows.md, tests, geometry/util/ subdirectory)

  - README.md: Add toroidal feature to Builder API section with example

  Code updates:

  - src/lib.rs: Export TopologyKind and GlobalTopology from
    prelude::triangulation for ergonomic imports

  - src/core/delaunay_triangulation.rs: Add "Advanced Configuration"
    section to ::new() documentation mentioning Builder for toroidal
    and custom options; fix redundant rustdoc link

  - examples/topology_editing_2d_3d.rs: Migrate to DelaunayTriangulationBuilder
  - benches/profiling_suite.rs: Migrate to DelaunayTriangulationBuilder

  - Changed: Adopt DelaunayTriangulationBuilder and update related documentation

  Migrates benchmark and example code to consistently use the
  DelaunayTriangulationBuilder for creating triangulations.
  This reflects the full implementation and accessibility of toroidal
  topology via the builder, and updates documentation across various
  sections (API design, workflows, topology) to guide users on its
  proper and advanced configuration. Includes internal exports for
  ergonomic usage.

### Fixed

- Ensure deterministic sorting and enforce coherent orientation
  [`c0e4d4f`](https://github.com/acgetchell/delaunay/commit/c0e4d4fffc9e3ae116062b8bd2d89baf58678517)

Resolves multiple issues to ensure deterministic behavior and strong
  invariants across the triangulation data structure.

  Stabilizes vertex ordering, particularly for Hilbert curve sorts, by
  refining tie-breaking, error handling, and fallback logic. This prevents
  non-deterministic results and corrects inverse permutation calculations,
  addressing previously identified breaking changes related to sorting.

  Enforces a consistent positive geometric orientation for all cells
  throughout the triangulation lifecycle, making coherent orientation a
  first-class invariant. This fixes 4D construction failures and improves
  periodic self-neighbor validation by handling lifted coordinates.

  Enhances the global topology model, especially for toroidal domains,
  by improving validation, canonicalization, and periodic facet key
  derivation. This addresses edge cases related to non-finite coordinates,
  zero/negative periods, and ensures consistent behavior.

- Resolve 3D seeded bulk construction orientation convergence failure [#228](https://github.com/acgetchell/delaunay/pull/228)
  [`c181f28`](https://github.com/acgetchell/delaunay/commit/c181f289c7d36a6668a8441b78aed596b9dae36c)

- Soften post-condition in normalize_and_promote_positive_orientation:
    add canonicalize_global_orientation_sign before the promote loop to
    prevent oscillation; demote the residual negative-orientation check
    from a hard error to a diagnostic log so near-degenerate but
    structurally valid simplices no longer abort insertion.

  - Replace enlarged local repair fallback with global
    repair_delaunay_with_flips_k2_k3 when D<4 per-insertion local repair
    cycles on co-spherical FP configurations. The multi-attempt global
    repair uses robust predicates and alternate queue orders to break
    cycling.

  - Gate global repair fallback on a new ConstructionOptions field
    (use_global_repair_fallback, default true) threaded through the
    build chain via DelaunayInsertionState. The periodic builder disables
    it (.without_global_repair_fallback()) so global repair cannot
    disrupt the image-point topology; the existing 24-attempt shuffle
    retry finds a working vertex ordering instead.

### Maintenance

- Bump taiki-e/install-action from 2.67.30 to 2.68.8 [#211](https://github.com/acgetchell/delaunay/pull/211)
  [`a133758`](https://github.com/acgetchell/delaunay/commit/a13375884047d06aab280ff5cb17f48910e496f0)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.30 to 2.68.8.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/288875dd3d64326724fa6d9593062d9f8ba0b131...cfdb446e391c69574ebc316dfb7d7849ec12b940)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.68.8
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Update docs, remove old files [`18679dc`](https://github.com/acgetchell/delaunay/commit/18679dc73e5764277906ee83d1465cb0134a9780)
- Bump actions-rust-lang/setup-rust-toolchain [#226](https://github.com/acgetchell/delaunay/pull/226)
  [`3e2cd26`](https://github.com/acgetchell/delaunay/commit/3e2cd26f67d4845445f1b876ec31695a20e745be)

Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.2 to 1.15.3.

- [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
- [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
- [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/1780873c7b576612439a134613cc4cc74ce5538c...a0b538fa0b742a6aa35d6e2c169b4bd06d225a98)

  ---
  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
    dependency-version: 1.15.3
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/download-artifact from 7.0.0 to 8.0.0 [#222](https://github.com/acgetchell/delaunay/pull/222)
  [`c172f09`](https://github.com/acgetchell/delaunay/commit/c172f09dd0a0f533f47e943bfbbbfc54240e40b0)

Bumps [actions/download-artifact](https://github.com/actions/download-artifact) from 7.0.0 to 8.0.0.

- [Release notes](https://github.com/actions/download-artifact/releases)
- [Commits](https://github.com/actions/download-artifact/compare/37930b1c2abaa49bbe596cd826c3c89aef350131...70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3)

  ---
  updated-dependencies:

- dependency-name: actions/download-artifact
    dependency-version: 8.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump taiki-e/install-action from 2.68.8 to 2.68.16 [#225](https://github.com/acgetchell/delaunay/pull/225)
  [`25a09a4`](https://github.com/acgetchell/delaunay/commit/25a09a46f1c0d46775fd085e6b34053888d69277)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.8 to 2.68.16.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/cfdb446e391c69574ebc316dfb7d7849ec12b940...d6e286fa45544157a02d45a43742857ebbc25d12)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.68.16
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/upload-artifact from 6 to 7 [#223](https://github.com/acgetchell/delaunay/pull/223)
  [`0a25bfa`](https://github.com/acgetchell/delaunay/commit/0a25bfa55c5b9d4579e7747e1e7c8661185c4542)

Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 6 to 7.

- [Release notes](https://github.com/actions/upload-artifact/releases)
- [Commits](https://github.com/actions/upload-artifact/compare/v6...v7)

  ---
  updated-dependencies:

- dependency-name: actions/upload-artifact
    dependency-version: '7'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump astral-sh/setup-uv from 7.3.0 to 7.3.1 [#224](https://github.com/acgetchell/delaunay/pull/224)
  [`1533402`](https://github.com/acgetchell/delaunay/commit/15334025d07c4dfeef43d01bbf3a190eaed54688)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.3.0 to 7.3.1.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/eac588ad8def6316056a12d4907a9d4d84ff7a3b...5a095e7a2014a4212f075830d4f7277575a9d098)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.3.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.68.16 to 2.68.25 [#237](https://github.com/acgetchell/delaunay/pull/237)
  [`abe7925`](https://github.com/acgetchell/delaunay/commit/abe79250b63aaf695a8a4447250402b6f1cbdcc8)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.16 to 2.68.25.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/d6e286fa45544157a02d45a43742857ebbc25d12...a37010ded18ff788be4440302bd6830b1ae50d8b)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.68.25
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/setup-node from 6.2.0 to 6.3.0 [#239](https://github.com/acgetchell/delaunay/pull/239)
  [`eb0000b`](https://github.com/acgetchell/delaunay/commit/eb0000bf27cb583a97370190861c01d4b8902bfe)

Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.2.0 to 6.3.0.

- [Release notes](https://github.com/actions/setup-node/releases)
- [Commits](https://github.com/actions/setup-node/compare/6044e13b5dc448c55e2357c09f80417699197238...53b83947a5a98c8d113130e565377fae1a50d02f)

  ---
  updated-dependencies:

- dependency-name: actions/setup-node
    dependency-version: 6.3.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Release v0.7.2 [#246](https://github.com/acgetchell/delaunay/pull/246)
  [`2c7d26d`](https://github.com/acgetchell/delaunay/commit/2c7d26d4a2bfb9ec6173de7b7480171cba04598f)

- chore(release): release v0.7.2

  - Bump version to v0.7.2
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.7.2

  - Changed: update robustness guide and fix release documentation

  Clarify evaluation stages for orientation and insphere predicates,
  detailing the use of Shewchuk-style error bounds for D ≤ 4. Correct a
  step reference in the release guide. This is an internal change.

### Performance

- Improve Hilbert curve correctness and add bulk API [#207](https://github.com/acgetchell/delaunay/pull/207)
  [#216](https://github.com/acgetchell/delaunay/pull/216) [`2d198e7`](https://github.com/acgetchell/delaunay/commit/2d198e7d2f1f41f1b2e47009a1cf7cc12079fe05)

- perf: improve Hilbert curve correctness and add bulk API [#207](https://github.com/acgetchell/delaunay/pull/207)

  Implements correctness fixes, API improvements, and comprehensive testing
  for the Hilbert space-filling curve ordering utilities.

  ## Correctness Fixes

  - Add debug_assert guards in hilbert_index_from_quantized for parameter
    validation (bits range and overflow checks)

  - Fix quantization truncation bias by changing from NumCast::from(scaled)
    to scaled.round().to_u32() for fairer spatial distribution across grid
    cells, improving point ordering quality

  ## API Design

  - Add HilbertError enum with InvalidBitsParameter, IndexOverflow, and
    DimensionTooLarge variants for proper error handling

  - Implement hilbert_indices_prequantized() returning Result<Vec<u128>,
    HilbertError> for bulk processing of pre-quantized coordinates

  - Bulk API avoids redundant quantization computation, significantly
    improving performance for large insertion batches

  ## Testing

  - Add 4D continuity test verifying Hilbert curve property on 256-point
    grid (bits=2)

  - Add quantization rounding distribution test validating fair cell
    assignment

  - Add 5 comprehensive tests for prequantized API covering success cases,
    empty input, and all error conditions

  - All 17 Hilbert-specific tests pass (11 existing + 6 new)

  ## Known Issue

  Temporarily ignore repair_fallback_produces_valid_triangulation test as
  the rounding change affects insertion order, exposing a latent geometric
  degeneracy issue in triangulation construction. This is properly
  documented and tracked under issue #204 for investigation.

  - Added: Explicitly handle zero-dimensional inputs in Hilbert index calculation

  Ensures correct behavior for `hilbert_indices_prequantized` when
  the dimensionality `D` is zero. In such a space, all points map to
  the single origin, and their Hilbert curve index is always 0.
  This change adds an early return for this specific edge case.

- Use bulk Hilbert API in order_vertices_hilbert [#218](https://github.com/acgetchell/delaunay/pull/218)
  [`4782905`](https://github.com/acgetchell/delaunay/commit/478290556e88f770e0fcda07fb6137a3404b70f5)

- perf: use bulk Hilbert API in order_vertices_hilbert

  Refactored `order_vertices_hilbert` to use the bulk `hilbert_indices_prequantized` API
  instead of calling `hilbert_index` individually for each vertex. This eliminates
  redundant parameter validation (N validations → 1 validation for N vertices).

- Switch FastKernel to insphere_lifted and enable LTO [#234](https://github.com/acgetchell/delaunay/pull/234)
  [`91e290f`](https://github.com/acgetchell/delaunay/commit/91e290fca1a633f5b084accc367767f235780a49)

- perf: switch FastKernel to insphere_lifted and enable LTO

  Switch FastKernel::in_sphere() to use insphere_lifted() for 5.3x speedup in 3D.
  Add release profile optimization with thin LTO and codegen-units=1.

  Benchmarks across dimensions (2D-5D):

  - insphere_lifted is 5.3x faster in 3D (15.5 ns vs 81.7 ns)
  - Random query test: 3.75x faster (20.0 µs vs 75.0 µs for 1000 queries)
  - insphere_lifted consistently fastest across all dimensions

  Performance gains attributed to la-stack v0.2.0's closed-form determinants for D=1-4.

## [0.7.1] - 2026-02-20

### Merged Pull Requests

- Release v0.7.1 [#205](https://github.com/acgetchell/delaunay/pull/205)
- Prevents timeout in 4D bulk construction [#203](https://github.com/acgetchell/delaunay/pull/203)
- Bump taiki-e/install-action from 2.67.26 to 2.67.30 [#202](https://github.com/acgetchell/delaunay/pull/202)
- Bump the dependencies group with 3 updates [#201](https://github.com/acgetchell/delaunay/pull/201)
- Feat/ball pointgen and debug harness [#200](https://github.com/acgetchell/delaunay/pull/200)
- Ci/perf baselines by tag [#199](https://github.com/acgetchell/delaunay/pull/199)
- Refactors changelog generation to use git-cliff [#198](https://github.com/acgetchell/delaunay/pull/198)
- Bump astral-sh/setup-uv from 7.2.1 to 7.3.0 [#196](https://github.com/acgetchell/delaunay/pull/196)
- Bump taiki-e/install-action from 2.67.18 to 2.67.26 [#195](https://github.com/acgetchell/delaunay/pull/195)
- Improves flip algorithm with topology index [#194](https://github.com/acgetchell/delaunay/pull/194)
- Removes `CoordinateScalar` bound from `Cell`, `Tds`, `Vertex` [#193](https://github.com/acgetchell/delaunay/pull/193)
- Moves `TopologyEdit` to `triangulation::flips` [#192](https://github.com/acgetchell/delaunay/pull/192)
- Correctly wires neighbors after K2 flips [#191](https://github.com/acgetchell/delaunay/pull/191)
- Use borrowed APIs in utility functions [#190](https://github.com/acgetchell/delaunay/pull/190)
- Add ScalarSummable/ScalarAccumulative supertraits [#189](https://github.com/acgetchell/delaunay/pull/189)
- Refactors point access for efficiency (internal) [#187](https://github.com/acgetchell/delaunay/pull/187)
- Corrects kernel parameter passing in triangulation [#186](https://github.com/acgetchell/delaunay/pull/186)
- Examples to error and struct definitions [#185](https://github.com/acgetchell/delaunay/pull/185)
- Validates random triangulations for Euler consistency [#184](https://github.com/acgetchell/delaunay/pull/184)
- Validates ridge links locally after Delaunay repair [#183](https://github.com/acgetchell/delaunay/pull/183)
- Bump astral-sh/setup-uv from 7.2.0 to 7.2.1 [#182](https://github.com/acgetchell/delaunay/pull/182)
- Bump taiki-e/install-action from 2.67.17 to 2.67.18 [#181](https://github.com/acgetchell/delaunay/pull/181)
- Stabilizes Delaunay property tests with bistellar flips [#180](https://github.com/acgetchell/delaunay/pull/180)
- Bump taiki-e/install-action from 2.66.1 to 2.67.11 [#177](https://github.com/acgetchell/delaunay/pull/177)
- Bump actions/checkout from 6.0.1 to 6.0.2 [#176](https://github.com/acgetchell/delaunay/pull/176)
- Bump actions/setup-node from 6.1.0 to 6.2.0 [#175](https://github.com/acgetchell/delaunay/pull/175)
- Feature/bistellar flips [#172](https://github.com/acgetchell/delaunay/pull/172)

### Added

- Examples to error and struct definitions [#185](https://github.com/acgetchell/delaunay/pull/185)
  [`a1bce55`](https://github.com/acgetchell/delaunay/commit/a1bce556cfd9a799f2a6aabb716443a14aaf6772)

- Added: Examples to error and struct definitions

  Adds code examples to various error enums and struct definitions
  to improve documentation and provide usage guidance.

  This change enhances the discoverability and understanding of
  various components, such as `AdjacencyIndexBuildError`,
  `BistellarFlipKind`, `FlipDirection`, `FlipError`, `FlipInfo`,
  `TriangleHandle`, `RidgeHandle`, `DelaunayRepairStats`,
  `RepairQueueOrder`, `DelaunayRepairDiagnostics`,
  `DelaunayRepairError`, `HullExtensionReason`, `InsertionError`,
  `LocateResult`, `LocateError`, `ConflictError`, `LocateFallbackReason`,
  `LocateFallback`, `LocateStats`, `CellValidationError`, `EdgeKey`,
  `FacetError`, `TopologicalOperation`, `RepairDecision`,
  `InsertionResult`, `InsertionStatistics`,
  `TriangulationConstructionState`, `TdsConstructionError`,
  `DelaunayValidationError`, `DelaunayRepairPolicy`,
  `DelaunayRepairHeuristicConfig`, `DelaunayRepairHeuristicSeeds`,
  `DelaunayRepairOutcome`, `DelaunayCheckPolicy`,
  `UuidValidationError`, `VertexValidationError`,
  `ConvexHullValidationError`, `ConvexHullConstructionError`,
  `MatrixError`, `InSphere`, `Orientation`, `QualityError`,
  `ConsistencyResult`, `CoordinateConversionError`,
  `CoordinateValidationError`, `CircumcenterError`,
  `SurfaceMeasureError`, `RandomPointGenerationError` and
  `ValueConversionError` to make the crate easier to use.

  - Changed: Improves examples and updates doc tests

  Updates doc tests to use clearer examples and more
  idiomatic syntax, enhancing code readability and
  maintainability. Modifies BistellarFlipKind to use a
  getter method. Addresses issues identified during
  documentation review. (internal)

- Add ScalarSummable/ScalarAccumulative supertraits [#189](https://github.com/acgetchell/delaunay/pull/189)
  [`abdeeb2`](https://github.com/acgetchell/delaunay/commit/abdeeb2f80ab03c998b9a29108c57ff9f0c54393)

- feat(geometry): add ScalarSummable/ScalarAccumulative supertraits

  - Add ScalarSummable (CoordinateScalar + Sum) and ScalarAccumulative (CoordinateScalar + AddAssign + SubAssign + Sum)
  - Refactor repeated scalar bounds across geometry/core modules to use the new supertraits
  - Allow “supertrait(s)” in cspell
- Document geometric and topological invariants [`0283bf0`](https://github.com/acgetchell/delaunay/commit/0283bf01cd1860c1a44ff9f645ac304fa44b7345)

Adds `invariants.md` to document the theoretical background and
  rationale for the topological and geometric invariants enforced by
  the `delaunay` crate. This includes simplicial complexes,
  PL-manifolds, link-based validation, insertion strategies, and
  convergence considerations. Updates `README.md` and `lib.rs` to
  reference the new document. Also adds a `examples/README.md` file.

### Changed

- Feature/bistellar flips [#172](https://github.com/acgetchell/delaunay/pull/172)
  [`66c7028`](https://github.com/acgetchell/delaunay/commit/66c7028d0c3d9dbc00f6b1a9cb791c41d39ab933)
- Refactors point access for efficiency (internal) [#187](https://github.com/acgetchell/delaunay/pull/187)
  [`8020065`](https://github.com/acgetchell/delaunay/commit/8020065afe66fc066ef51307a9de02621f087a54)

- Changed: Refactors point access for efficiency (internal)

  Simplifies vertex coordinate access using `.coords()` instead of `.into()`,
  improving code clarity and potentially performance. This change is
  internal, affecting predicate calculations and geometric algorithms.
  Also, moves the issue 120 investigation document to the archive.

- Use borrowed APIs in utility functions [#190](https://github.com/acgetchell/delaunay/pull/190)
  [`bee065b`](https://github.com/acgetchell/delaunay/commit/bee065bd13f9f7adb8a7767b5b907ea7886248d5)

- Changed: Use borrowed APIs in utility functions

  Updates `into_hashmap`, `dedup_vertices_exact`,
  `dedup_vertices_epsilon`, and `filter_vertices_excluding`
  functions to accept slices instead of vectors, improving
  performance by avoiding unnecessary cloning.

  This aligns with the Rust agent's preference for borrowed
  APIs, taking references as arguments and returning borrowed
  views when possible, and only taking ownership when required.

- Moves `TopologyEdit` to `triangulation::flips` [#192](https://github.com/acgetchell/delaunay/pull/192)
  [`c491bb9`](https://github.com/acgetchell/delaunay/commit/c491bb913e1b49ff38d4ce52c180d5220e9db9df)

- Changed: Moves `TopologyEdit` to `triangulation::flips`

  Moves the `TopologyEdit` trait to `triangulation::flips` and renames it to `BistellarFlips`.

  This change involves updating imports and references throughout the codebase and documentation to reflect the new location and name of the trait.

  - Changed: Refactors prelude modules for clarity (internal)

  Streamlines the prelude modules to provide clearer and more
  focused exports for common triangulation tasks. This change
  affects import statements in documentation and examples,
  requiring more specific paths for certain types.

  - Removed: Topology validation prelude module

  Removes the redundant topology validation prelude module.

  Moves its contents into the main topology prelude, simplifying
  module structure and reducing code duplication. This change
  internally refactors the prelude modules for better organization.

- Removes `CoordinateScalar` bound from `Cell` , `Tds` , `Vertex` [#193](https://github.com/acgetchell/delaunay/pull/193)
  [`e69f3d1`](https://github.com/acgetchell/delaunay/commit/e69f3d153961050e03a520e5b5457a165097c834)

- Changed: Removes `CoordinateScalar` bound from `Cell`, `Tds`, `Vertex`

  Relaxes trait bounds on `Cell`, `Tds`, and `Vertex` structs by
  removing the `CoordinateScalar` requirement.

  This change prepares the triangulation data structure for combinatorial
  operations independent of geometry. The `validate` method in `Tds`
  now requires `CoordinateScalar` to perform coordinate validation,
  where applicable. (Internal change).

  - Changed: Clarifies `Vertex` constraints and moves `point`

  Clarifies the `Vertex` struct's constraints, emphasizing
  `CoordinateScalar` requirement for geometric operations and
  serialization but allowing combinatorial use without it.

  Moves the `point` method definition to ensure consistent API
  presentation. (Internal refactoring, no functional change).

- Improves flip algorithm with topology index [#194](https://github.com/acgetchell/delaunay/pull/194)
  [`c4e37ed`](https://github.com/acgetchell/delaunay/commit/c4e37ed9e899170978196af4edad4e2c0a248141)

- Changed: Improves flip algorithm with topology index

  Improves flip algorithm by introducing a topology index to
  efficiently check for duplicate cells and non-manifold facets.
  This avoids redundant scans of the triangulation data
  structure, especially during repair operations, by pre-computing
  and storing facet and cell signatures. This change is internal.

- Updates typos-cli installation in CI workflow [`6168e83`](https://github.com/acgetchell/delaunay/commit/6168e83512509123f9c0443f9716752f88cc2aa3)

Updates the typos-cli installation in the CI workflow to use the
  `taiki-e/install-action` for simpler and more reliable installation.
  This aligns with the switch from cspell to typos in the codebase.

- Refactors changelog generation to use git-cliff [#198](https://github.com/acgetchell/delaunay/pull/198)
  [`63553fa`](https://github.com/acgetchell/delaunay/commit/63553fa170a5603c98c3c5eee87caca756e0bb89)
- Ci/perf baselines by tag [#199](https://github.com/acgetchell/delaunay/pull/199)
  [`0c94dec`](https://github.com/acgetchell/delaunay/commit/0c94dec96ac73004b9b9c924a977499e29dfaf19)
- Feat/ball pointgen and debug harness [#200](https://github.com/acgetchell/delaunay/pull/200)
  [`79bf0e9`](https://github.com/acgetchell/delaunay/commit/79bf0e96b0d9c4f8cbdda8539910adfa18f412a4)

### Documentation

- Refresh docs and add workflows guide [`695e9a0`](https://github.com/acgetchell/delaunay/commit/695e9a0f700b8cb6bf7cb5a3acb6e4510c4b3939)

- Add a new workflows guide covering Builder/Edit API usage, topology guarantees, and repair
  - Wire README into doctests and refresh crate-level docs/examples (simplify type annotations, link to workflows)
  - Update README Features + references (kernels, insertion ordering, construction options, validation/repair)
  - Reorganize docs index and archive historical/roadmap material; refresh topology/robustness/validation guides
  - Replace debug `eprintln!` with `tracing` in Hilbert utilities/tests
  - Tweak spell-check + release docs (typos-cli, rename handling, release steps) and update CHANGELOG

### Fixed

- Stabilizes Delaunay property tests with bistellar flips [#180](https://github.com/acgetchell/delaunay/pull/180)
  [`e3bd4bf`](https://github.com/acgetchell/delaunay/commit/e3bd4bfa77258484e6dab088a2980139efd0f182)

- Fixed: Stabilizes Delaunay property tests with bistellar flips

  Enables previously failing Delaunay property tests by
  implementing bistellar flips for robust Delaunay repair.
  Includes automatic repair and fast validation.

  Updates MSRV to 1.93.0.

- Validates ridge links locally after Delaunay repair [#183](https://github.com/acgetchell/delaunay/pull/183)
  [`7bc4792`](https://github.com/acgetchell/delaunay/commit/7bc4792b33ee1e6ecbeda729834a33fbf06cd0e6)

- Fixed: Validates ridge links locally after Delaunay repair

  Addresses potential topology violations (non-manifold configurations)
  introduced by flip-based Delaunay repair by validating ridge links
  for affected cells post-insertion. This prevents committing
  invalid triangulations and surfaces topology validation failures
  as insertion errors, enabling transactional rollback.

- Validates random triangulations for Euler consistency [#184](https://github.com/acgetchell/delaunay/pull/184)
  [`72eb272`](https://github.com/acgetchell/delaunay/commit/72eb2726f8c683339a23b17a63991ac87a35e412)

Ensures that random triangulations satisfy Euler characteristic
  validation to prevent construction errors or invalid classifications.

  Adds a validation function to check topology/Euler validity after
  triangulation construction or robust fallback attempts, catching
  potential issues that can lead to incorrect results. Removes
  redundant validation checks.

- Corrects kernel parameter passing in triangulation [#186](https://github.com/acgetchell/delaunay/pull/186)
  [`df3c490`](https://github.com/acgetchell/delaunay/commit/df3c49033dd96558ee5ee0de572913e82c55f210)

- Fixed: Corrects kernel parameter passing in triangulation

  Addresses an issue where the kernel was being passed by value
  instead of by reference in the Delaunay triangulation
  construction. This change ensures that the kernel is correctly
  accessed and used, preventing potential errors and improving
  reliability. The fix involves modifying the `with_kernel` method
  signatures and call sites to accept a kernel reference instead of
  a kernel value. This affects benchmark code, documentation,
  examples, and internal code.

- Correctly wires neighbors after K2 flips [#191](https://github.com/acgetchell/delaunay/pull/191)
  [`5ab686c`](https://github.com/acgetchell/delaunay/commit/5ab686c2177562bdeb852fe843c946181e03753a)

- Fixed: Correctly wires neighbors after K2 flips

  Fixes an issue where external neighbors across the cavity
  boundary were not being correctly rewired after a K2 flip.

  Introduces `external_facets_for_boundary` to collect the set
  of external facets that are shared with the flip cavity boundary,
  and then uses these to correctly wire up neighbors.

  Adds a test case to verify that external neighbors are correctly
  rewired after the flip, ensuring that the triangulation remains
  valid and consistent.
  Refs: refactor/wire-cavity-neighbors

  - Added: K=3 flip rewiring test

  Adds a test to verify correct rewiring of external neighbors
  after a k=3 flip. This validates the boundary handling and
  neighbor update logic in the bistellar flip implementation.

  This test constructs an explicit k=3 ridge-flip fixture and
  checks neighbor rewiring.

- Prevents timeout in 4D bulk construction [#203](https://github.com/acgetchell/delaunay/pull/203)
  [`b071fb7`](https://github.com/acgetchell/delaunay/commit/b071fb787bf06ff196fade35fd5dab180822985b)

- Fixed: Prevents timeout in 4D bulk construction

  Addresses a timeout issue in 4D bulk construction
  by implementing per-insertion local Delaunay repair
  (soft-fail) during bulk construction to prevent
  violation accumulation, which slows down the global
  repair process. Also adds a hard wall-clock time
  limit to the test harness.

### Maintenance

- Bump actions/setup-node from 6.1.0 to 6.2.0 [#175](https://github.com/acgetchell/delaunay/pull/175)
  [`8d15114`](https://github.com/acgetchell/delaunay/commit/8d15114553e05ecb9d46cfb9bd78eb9e27379796)

Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.1.0 to 6.2.0.

- [Release notes](https://github.com/actions/setup-node/releases)
- [Commits](https://github.com/actions/setup-node/compare/395ad3262231945c25e8478fd5baf05154b1d79f...6044e13b5dc448c55e2357c09f80417699197238)

  ---
  updated-dependencies:

- dependency-name: actions/setup-node
    dependency-version: 6.2.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump actions/checkout from 6.0.1 to 6.0.2 [#176](https://github.com/acgetchell/delaunay/pull/176)
  [`c703f94`](https://github.com/acgetchell/delaunay/commit/c703f944a9928651b639d5e5b2a06db3b1b75e4f)

Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.1 to 6.0.2.

- [Release notes](https://github.com/actions/checkout/releases)
- [Commits](https://github.com/actions/checkout/compare/v6.0.1...v6.0.2)

  ---
  updated-dependencies:

- dependency-name: actions/checkout
    dependency-version: 6.0.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.66.1 to 2.67.11 [#177](https://github.com/acgetchell/delaunay/pull/177)
  [`92292e0`](https://github.com/acgetchell/delaunay/commit/92292e01f5777fe4f13ab387a1f9ee4e39930ab3)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.66.1 to 2.67.11.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/3522286d40783523f9c7880e33f785905b4c20d0...887bc4e03483810873d617344dd5189cd82e7b8b)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.67.11
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump astral-sh/setup-uv from 7.2.0 to 7.2.1 [#182](https://github.com/acgetchell/delaunay/pull/182)
  [`cfaaf60`](https://github.com/acgetchell/delaunay/commit/cfaaf600fef814bfe563524fcdbcb0ab83fa9028)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.2.0 to 7.2.1.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/61cb8a9741eeb8a550a1b8544337180c0fc8476b...803947b9bd8e9f986429fa0c5a41c367cd732b41)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.2.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.67.17 to 2.67.18 [#181](https://github.com/acgetchell/delaunay/pull/181)
  [`068f314`](https://github.com/acgetchell/delaunay/commit/068f314ed00a837678c968734f75b4b69327d9e0)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.17 to 2.67.18.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/29feb09ac22f4fde4175fe7b5c3548952234f69a...650c5ca14212efbbf3e580844b04bdccf68dac31)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.67.18
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump rand to 0.10 and trim unused deps [`fe3e406`](https://github.com/acgetchell/delaunay/commit/fe3e406f19dcc6c9c72666d89b285626ea00465c)

- Update rand to v0.10 and fix RNG trait imports (use rand::RngExt) in tests/utilities
  - Move test-only crates to dev-dependencies (approx, serde_json)
  - Remove unused runtime dependencies (anyhow, clap, serde_test)
  - Drop clippy allow for multiple_crate_versions
  - Update Cargo.lock and regenerate CHANGELOG.md
- Bump astral-sh/setup-uv from 7.2.1 to 7.3.0 [#196](https://github.com/acgetchell/delaunay/pull/196)
  [`6c4662a`](https://github.com/acgetchell/delaunay/commit/6c4662a4008e5a43de71d3b83591350dbf4778b0)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.2.1 to 7.3.0.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/803947b9bd8e9f986429fa0c5a41c367cd732b41...eac588ad8def6316056a12d4907a9d4d84ff7a3b)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.3.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.67.18 to 2.67.26 [#195](https://github.com/acgetchell/delaunay/pull/195)
  [`1cd6008`](https://github.com/acgetchell/delaunay/commit/1cd6008503d0bc2694e934060f8ed986f2d9e05e)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.18 to 2.67.26.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/650c5ca14212efbbf3e580844b04bdccf68dac31...509565405a8a987e73cf742e26b26dcc72c4b01a)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.67.26
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.67.26 to 2.67.30 [#202](https://github.com/acgetchell/delaunay/pull/202)
  [`e99f0d3`](https://github.com/acgetchell/delaunay/commit/e99f0d3f17cf45e85c51274799b8baee3b28e8e5)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.26 to 2.67.30.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/509565405a8a987e73cf742e26b26dcc72c4b01a...288875dd3d64326724fa6d9593062d9f8ba0b131)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.67.30
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump the dependencies group with 3 updates [#201](https://github.com/acgetchell/delaunay/pull/201)
  [`fa69c80`](https://github.com/acgetchell/delaunay/commit/fa69c800e6b01fa2cb4f9b1468473e09ed9f0108)

Bumps the dependencies group with 3 updates: [arc-swap](https://github.com/vorner/arc-swap) , [uuid](https://github.com/uuid-rs/uuid) and
[sysinfo](https://github.com/GuillaumeGomez/sysinfo) .

  Updates `arc-swap` from 1.8.1 to 1.8.2

- [Changelog](https://github.com/vorner/arc-swap/blob/master/CHANGELOG.md)
- [Commits](https://github.com/vorner/arc-swap/compare/v1.8.1...v1.8.2)

  Updates `uuid` from 1.20.0 to 1.21.0

- [Release notes](https://github.com/uuid-rs/uuid/releases)
- [Commits](https://github.com/uuid-rs/uuid/compare/v1.20.0...v1.21.0)

  Updates `sysinfo` from 0.38.1 to 0.38.2

- [Changelog](https://github.com/GuillaumeGomez/sysinfo/blob/main/CHANGELOG.md)
- [Commits](https://github.com/GuillaumeGomez/sysinfo/compare/v0.38.1...v0.38.2)

  ---
  updated-dependencies:

- dependency-name: arc-swap
    dependency-version: 1.8.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: uuid
    dependency-version: 1.21.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies

- dependency-name: sysinfo
    dependency-version: 0.38.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Release v0.7.1 [#205](https://github.com/acgetchell/delaunay/pull/205)
  [`df890d3`](https://github.com/acgetchell/delaunay/commit/df890d377c4f52f3502e3339ebfdac17c522cd76)

- chore(release): release v0.7.1

  - Bump version to 0.7.1
  - Update CHANGELOG.md with latest changes
  - Update documentation for release
  - Add performance results for v0.7.1

  - Changed: Updates performance benchmark results (internal)

  Updates the performance benchmark results in PERFORMANCE_RESULTS.md
  based on the latest benchmark run. Also, modifies the benchmark
  utility script to correctly extract dimension information for
  performance summaries. This is an internal change to reflect
  performance improvements.

  - Fixed: Correctly parses dimension info in performance summaries

  Fixes an issue where parentheses within the dimension information
  were being incorrectly removed when parsing performance
  summaries.

  Uses `removeprefix` and `removesuffix` instead of `strip` to
  avoid accidentally removing internal parentheses in dimension
  descriptions.

### Removed

- Replace cspell with typos for spell checking [`4b5e1a1`](https://github.com/acgetchell/delaunay/commit/4b5e1a1641037f137663612c74bb39cd187f90e9)

Replaces the cspell tool with typos for spell checking across the
  project. This change involves removing cspell configurations and
  dependencies, and integrating typos, including its configuration file.

## [0.7.0] - 2026-01-13

### ⚠️ Breaking Changes

- Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)

### Merged Pull Requests

- Refactors topology guarantee and manifold validation [#171](https://github.com/acgetchell/delaunay/pull/171)
- Bump astral-sh/setup-uv from 7.1.6 to 7.2.0 [#170](https://github.com/acgetchell/delaunay/pull/170)
- Bump taiki-e/install-action from 2.65.13 to 2.66.1 [#169](https://github.com/acgetchell/delaunay/pull/169)
- Feature/manifolds [#168](https://github.com/acgetchell/delaunay/pull/168)
- Refactors Gram determinant calculation with LDLT [#167](https://github.com/acgetchell/delaunay/pull/167)
- Bump clap from 4.5.53 to 4.5.54 in the dependencies group [#166](https://github.com/acgetchell/delaunay/pull/166)
- Bump taiki-e/install-action from 2.65.7 to 2.65.13 [#165](https://github.com/acgetchell/delaunay/pull/165)
- Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)

### Added

- [**breaking**] Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)
  [`3748ebb`](https://github.com/acgetchell/delaunay/commit/3748ebb24ded08154875b2be371128c77d43eed3)

- feat: add public topology traversal API

  - Introduce canonical `EdgeKey` and read-only topology traversal helpers on `Triangulation`
  - Add opt-in `AdjacencyIndex` builder for faster repeated adjacency queries
  - Add integration tests for topology traversal and adjacency index invariants
  - Refresh repo tooling/CI configs and supporting scripts/tests

  - Changed: Exposes public topology traversal API

  Makes topology traversal APIs public for external use.

  Exposes `edges()`, `incident_edges()`, and `cell_neighbors()` on the
  `DelaunayTriangulation` struct as convenience wrappers. Updates
  documentation, examples, and benchmarks to use new API.

  This allows external users to traverse the triangulation's topology
  without needing to access internal implementation details.

  - Changed: Expose topology query APIs on DelaunayTriangulation

  Exposes cell and vertex query APIs on `DelaunayTriangulation` for zero-allocation topology traversal.

  Also includes internal refactoring to improve vertex incidence
  validation and ensure more robust handling of invalid key references.
  Now TDS validation detects isolated vertices.

### Changed

- Updates CHANGELOG.md for unreleased changes [`271353f`](https://github.com/acgetchell/delaunay/commit/271353f1b1af82fb45eb619520d62cd7474e4541)

Updates the changelog to reflect recent changes, including adding
  a new public topology traversal API, refreshing repository
  tooling/CI configurations, and clarifying TDS validation and API
  documentation.

- Refactors Gram determinant calculation with LDLT [#167](https://github.com/acgetchell/delaunay/pull/167)
  [`561a259`](https://github.com/acgetchell/delaunay/commit/561a259d58401d2baa61cd6313dd0ada01179f4a)

- Changed: Refactors Gram determinant calculation with LDLT

  Refactors the Gram determinant calculation to use LDLT factorization from the `la-stack` crate for improved efficiency and numerical stability by exploiting
  symmetry.

  Also, updates the `la-stack` dependency version.

  - Fixed: Improves robustness of incremental insertion

  Addresses rare topological invalidations during incremental
  insertion by:

  - Adding connectedness validation to conflict region checks.

  - Adding codimension-2 boundary manifoldness validation
      ("no boundary of boundary") to triangulation's `is_valid`
       method.

  - Ensuring that strict Level 3 validation is enabled during
      batch construction in debug builds.
  Refs: feat/la-stack-ldlt-factorization

  - Changed: Rename SimplexCounts to FVector for clarity

  Renames the `SimplexCounts` struct to `FVector` to better reflect
  its mathematical meaning as the f-vector in topology, representing
  the counts of simplices of different dimensions.

  This change improves code readability and aligns the naming
  convention with standard topological terminology.
  (Internal refactoring, no API change.)

  - Changed: Improves simplex generation algorithm

  Improves the algorithm for generating simplex combinations in
  the Euler characteristic calculation. This change enhances
  efficiency by using a lexicographic approach to generate
  combinations, reducing unnecessary computations.

- Feature/manifolds [#168](https://github.com/acgetchell/delaunay/pull/168)
  [`10abbe1`](https://github.com/acgetchell/delaunay/commit/10abbe1899a381d1b0d4855727dfef7797952549)
- Refactors topology guarantee and manifold validation [#171](https://github.com/acgetchell/delaunay/pull/171)
  [`dfdba5a`](https://github.com/acgetchell/delaunay/commit/dfdba5a745d6a41b4fa92d66b71c0c3d3dc87e54)

- Changed: Refactors topology guarantee and manifold validation

  Refactors manifold validation mode to topology guarantee for
  clarity. Updates Level 3 validation configuration, improves error
  reporting, and adds comprehensive manifold validation tests.
  Also improves robustness of incremental insertion.

  - Changed: Updates topology guarantee defaults and validation

  Updates the default topology guarantee to `Pseudomanifold` for new
  triangulations and deserialized triangulations. Also, clarifies
  validation policy and its relationship to topology guarantees in
  documentation. Introduces a test-only function to repair degenerate
  cells by removing them and clearing dangling references.

  - Fixed: Corrects triangulation perturbation logic

  Fixes a bug in the vertex insertion perturbation logic that
  caused non-equivalent results when translating the input
  point set by using a translation-invariant anchor for
  perturbation scaling.

  Also, preserves the caller-provided vertex UUID across
  perturbation retries to maintain vertex identity.

  Updates documentation on topology guarantees to clarify
  manifoldness invariants.

  - Changed: Improves PL-manifold validation with vertex-link check

  Replaces ridge-link validation with vertex-link validation for
  PL-manifold topology guarantee. This change provides a more
  robust and canonical check for PL-manifoldness, ensuring that
  the link of every vertex is a sphere or ball of the appropriate
  dimension.

### Maintenance

- Bump clap from 4.5.53 to 4.5.54 in the dependencies group [#166](https://github.com/acgetchell/delaunay/pull/166)
  [`ebd0d32`](https://github.com/acgetchell/delaunay/commit/ebd0d32e3552b2e94af021995df8c9c1431ccc1b)

Bumps the dependencies group with 1 update: [clap](https://github.com/clap-rs/clap).

  Updates `clap` from 4.5.53 to 4.5.54

- [Release notes](https://github.com/clap-rs/clap/releases)
- [Changelog](https://github.com/clap-rs/clap/blob/master/CHANGELOG.md)
- [Commits](https://github.com/clap-rs/clap/compare/clap_complete-v4.5.53...clap_complete-v4.5.54)

  ---
  updated-dependencies:

- dependency-name: clap
    dependency-version: 4.5.54
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.65.7 to 2.65.13 [#165](https://github.com/acgetchell/delaunay/pull/165)
  [`6b5f723`](https://github.com/acgetchell/delaunay/commit/6b5f723cf9f71599bedef098b3f226f4786ce539)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.65.7 to 2.65.13.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/4c6723ec9c638cccae824b8957c5085b695c8085...0e76c5c569f13f7eb21e8e5b26fe710062b57b62)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.65.13
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 7.1.6 to 7.2.0 [#170](https://github.com/acgetchell/delaunay/pull/170)
  [`b50bb8c`](https://github.com/acgetchell/delaunay/commit/b50bb8c4157496c99e9879b5c7214815ddbca633)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.6 to 7.2.0.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/681c641aba71e4a1c380be3ab5e12ad51f415867...61cb8a9741eeb8a550a1b8544337180c0fc8476b)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.2.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.65.13 to 2.66.1 [#169](https://github.com/acgetchell/delaunay/pull/169)
  [`44d3add`](https://github.com/acgetchell/delaunay/commit/44d3add45bdac8daff9cff6b686929959ccf84a3)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.65.13 to 2.66.1.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/0e76c5c569f13f7eb21e8e5b26fe710062b57b62...3522286d40783523f9c7880e33f785905b4c20d0)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.66.1
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

## [0.6.2] - 2026-01-01

### Merged Pull Requests

- Release v0.6.2 [#162](https://github.com/acgetchell/delaunay/pull/162)
- Bump taiki-e/install-action from 2.65.1 to 2.65.7 [#161](https://github.com/acgetchell/delaunay/pull/161)
- Bump serde_json in the dependencies group [#160](https://github.com/acgetchell/delaunay/pull/160)
- Bump taiki-e/install-action from 2.63.3 to 2.65.1 [#159](https://github.com/acgetchell/delaunay/pull/159)
- Refactor/validation hierarchy [#157](https://github.com/acgetchell/delaunay/pull/157)

### Changed

- Refactor/validation hierarchy [#157](https://github.com/acgetchell/delaunay/pull/157)
  [`c23cefb`](https://github.com/acgetchell/delaunay/commit/c23cefb223e17b16c6b1d60e8be385777da59f6f)

### Maintenance

- Bump taiki-e/install-action from 2.63.3 to 2.65.1 [#159](https://github.com/acgetchell/delaunay/pull/159)
  [`105b893`](https://github.com/acgetchell/delaunay/commit/105b8938903cab3bb8b94d68ea28bba535b94afc)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.63.3 to 2.65.1.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/d850aa816998e5cf15f67a78c7b933f2a5033f8a...b9c5db3aef04caffaf95a1d03931de10fb2a140f)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.65.1
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.65.1 to 2.65.7 [#161](https://github.com/acgetchell/delaunay/pull/161)
  [`b12b253`](https://github.com/acgetchell/delaunay/commit/b12b2537c3107a1d97c97d88d7a7285b26d6b9f1)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.65.1 to 2.65.7.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/b9c5db3aef04caffaf95a1d03931de10fb2a140f...4c6723ec9c638cccae824b8957c5085b695c8085)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.65.7
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump serde_json in the dependencies group [#160](https://github.com/acgetchell/delaunay/pull/160)
  [`cf3368c`](https://github.com/acgetchell/delaunay/commit/cf3368cf54c2ff68a7acfa34e3e51a2806a35f96)

Bumps the dependencies group with 1 update: [serde_json](https://github.com/serde-rs/json).

  Updates `serde_json` from 1.0.147 to 1.0.148

- [Release notes](https://github.com/serde-rs/json/releases)
- [Commits](https://github.com/serde-rs/json/compare/v1.0.147...v1.0.148)

  ---
  updated-dependencies:

- dependency-name: serde_json
    dependency-version: 1.0.148
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Release v0.6.2 [#162](https://github.com/acgetchell/delaunay/pull/162)
  [`66df9a6`](https://github.com/acgetchell/delaunay/commit/66df9a65fc87a527a80df40451ef984149e4e525)

- Bump version to v0.6.2
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.6.2

## [0.6.1] - 2025-12-17

### Merged Pull Requests

- Release v0.6.1 [#156](https://github.com/acgetchell/delaunay/pull/156)
- Migrate geometry predicates to la-stack [#155](https://github.com/acgetchell/delaunay/pull/155)
- Bump actions/cache from 4 to 5 [#153](https://github.com/acgetchell/delaunay/pull/153)
- Bump taiki-e/install-action from 2.62.63 to 2.63.3 [#152](https://github.com/acgetchell/delaunay/pull/152)
- Bump astral-sh/setup-uv from 7.1.5 to 7.1.6 [#151](https://github.com/acgetchell/delaunay/pull/151)
- Bump actions/upload-artifact from 5 to 6 [#150](https://github.com/acgetchell/delaunay/pull/150)
- Bump codecov/codecov-action from 5.5.1 to 5.5.2 [#149](https://github.com/acgetchell/delaunay/pull/149)
- Bump actions/download-artifact from 6.0.0 to 7.0.0 [#148](https://github.com/acgetchell/delaunay/pull/148)
- Add InsertionOutcome return type with insertion statistics [#147](https://github.com/acgetchell/delaunay/pull/147)
- Make DenseSlotMap default storage backend [#146](https://github.com/acgetchell/delaunay/pull/146)
- Add dimensional-generic topology module with Euler validation [#145](https://github.com/acgetchell/delaunay/pull/145)
- Bump astral-sh/setup-uv from 7.1.4 to 7.1.5 [#143](https://github.com/acgetchell/delaunay/pull/143)
- Bump actions/checkout from 6.0.0 to 6.0.1 [#142](https://github.com/acgetchell/delaunay/pull/142)
- Bump actions/setup-node from 6.0.0 to 6.1.0 [#141](https://github.com/acgetchell/delaunay/pull/141)

### Added

- Add --partition and --account options to slurm storage comparison script
  [`d249077`](https://github.com/acgetchell/delaunay/commit/d249077342a977f1b0d14302c9722fbd24593dac)

Add command-line options to override Slurm partition and account settings
  when submitting storage comparison benchmarks, with defaults of med2 and
  adamgrp respectively.

- Add --partition=NAME option to specify queue (default: med2)
- Add --account=NAME option to specify allocation (default: adamgrp)
- Update help text with new options and usage examples
- Pass options through to sbatch invocation
- Display account and partition in job submission confirmation
- Add dimensional-generic topology module with Euler validation [#145](https://github.com/acgetchell/delaunay/pull/145)
  [`e30dbab`](https://github.com/acgetchell/delaunay/commit/e30dbab900b41c86eda538be3d89c10eac94a573)

- feat: add dimensional-generic topology module with Euler validation

  Implement comprehensive topology analysis framework that works across all dimensions, replacing dimension-specific Euler characteristic validation with a
  unified, extensible design.

  Core Features:

  - Simplex counting and Euler characteristic (χ) calculation for any dimension
  - Topology classification (Empty, SingleSimplex, Ball, ClosedSphere, Unknown)
  - Complete validation with detailed diagnostics
  - Stub topology space types (Euclidean, Spherical, Toroidal) for future work
- Add InsertionOutcome return type with insertion statistics [#147](https://github.com/acgetchell/delaunay/pull/147)
  [`ba96912`](https://github.com/acgetchell/delaunay/commit/ba96912e47020f1233070d786175e69036ab655f)

- feat: add InsertionOutcome return type with insertion statistics

  Breaking Changes:

  - Bump MSRV from 1.91.0 to 1.92.0 across all configurations
  - `insert_with_statistics` now returns `Result<(InsertionOutcome, InsertionStatistics), InsertionError>`
  - `insert_transactional` signature changed to match new return type

  New Features:

  - Add `InsertionOutcome` enum with `Inserted` and `Skipped` variants
  - `Inserted` variant includes optional `hint` for next insertion optimization
  - `insert_with_statistics` provides detailed outcomes for both successful and skipped insertions
  - Skipped insertions return statistics without throwing error in statistics path

### Documentation

- Enhance Slurm script documentation with CLI options [`72f7f24`](https://github.com/acgetchell/delaunay/commit/72f7f24cd89e340260062006475ca07bf2d606ae)

- Add command-line option examples for partition and account configuration
  - Document all available CLI flags (--large, --time, --partition, --account, --help)
  - Add example combining multiple options
  - Clarify that CLI arguments override script header defaults
  - Add "myaccount" to cspell.json dictionary (placeholder account name)
- Canonicalize validation guide and refresh performance results
  [`7ab0bdf`](https://github.com/acgetchell/delaunay/commit/7ab0bdffdb04f7400646849febd71659b5fffdea)

- Make docs/validation.md the canonical reference for the 4-level validation hierarchy
    (including focused structural invariant helpers and Issue #120 caveats for Level 4).

  - Shorten crate-level docs in src/lib.rs to avoid duplicating validation details while
    keeping an overview + link to the guide.

  - Update benches/PERFORMANCE_RESULTS.md with refreshed benchmark data and metadata.

### Fixed

- Slurm_storage_comparison.sh [`1f6fb35`](https://github.com/acgetchell/delaunay/commit/1f6fb352dbd397b5e04ca3d415d9cc3b52c9c076)

Fixed an issue where running the script did not submit correctly via sbatch and instead ran interactively

- Increase BENCH_SAMPLE_SIZE to meet Criterion minimum requirement
  [`006cd97`](https://github.com/acgetchell/delaunay/commit/006cd97a3b38c2870850c4e734ef9c466493717d)

- Change BENCH_SAMPLE_SIZE from 5 to 10 in memory-stress-test job
  - Fixes assertion failure: Criterion requires at least 10 samples
  - Update both job-level and step-level env configurations

  The memory stress testing workflow was failing with "assertion failed: n >= 10" because Criterion's statistical analysis requires a minimum of 10 samples.
  This change ensures the CI configuration meets that requirement while keeping benchmarks fast for CI environments.

- Correctly generate and upload Codacy SARIF reports [`3645ab4`](https://github.com/acgetchell/delaunay/commit/3645ab493ffda3d56fb38a3232af3fb02705a5e3)

Fixes the Codacy workflow to correctly generate and upload SARIF reports.

  This involves creating a dedicated workspace, ensuring the SARIF file is correctly located and formatted, and handling cases where no issues are found.

  The changes also include setting a timeout, skipping uncommitted files, and improving error handling.

### Maintenance

- Fix memory stress testing timeout and schedule issues [`5676c92`](https://github.com/acgetchell/delaunay/commit/5676c92412272d0f5231e64a24231152a2b42435)

Root cause: Environment variables were not properly inherited by the Rust
  benchmark binary, causing tests to run at production scale (100k points)
  instead of CI-friendly development scale.

- Bump actions/setup-node from 6.0.0 to 6.1.0 [#141](https://github.com/acgetchell/delaunay/pull/141)
  [`828c850`](https://github.com/acgetchell/delaunay/commit/828c850915c183ca8e8352b8bdba8452890cf6c7)

Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.0.0 to 6.1.0.

- [Release notes](https://github.com/actions/setup-node/releases)
- [Commits](https://github.com/actions/setup-node/compare/2028fbc5c25fe9cf00d9f06a71cc4710d4507903...395ad3262231945c25e8478fd5baf05154b1d79f)

  ---
  updated-dependencies:

- dependency-name: actions/setup-node
    dependency-version: 6.1.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump actions/checkout from 6.0.0 to 6.0.1 [#142](https://github.com/acgetchell/delaunay/pull/142)
  [`e6775ca`](https://github.com/acgetchell/delaunay/commit/e6775caab00c5343bbeb72354b88567fd5e68daf)

Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.0 to 6.0.1.

- [Release notes](https://github.com/actions/checkout/releases)
- [Commits](https://github.com/actions/checkout/compare/v6...v6.0.1)

  ---
  updated-dependencies:

- dependency-name: actions/checkout
    dependency-version: 6.0.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 7.1.4 to 7.1.5 [#143](https://github.com/acgetchell/delaunay/pull/143)
  [`f82e673`](https://github.com/acgetchell/delaunay/commit/f82e673332b254898024a39d685b02630ffaa8e6)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.4 to 7.1.5.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/1e862dfacbd1d6d858c55d9b792c756523627244...ed21f2f24f8dd64503750218de024bcf64c7250a)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.5
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/download-artifact from 6.0.0 to 7.0.0 [#148](https://github.com/acgetchell/delaunay/pull/148)
  [`2b79dc4`](https://github.com/acgetchell/delaunay/commit/2b79dc482d8baa03d316235bf862e3b9761ad93d)

Bumps [actions/download-artifact](https://github.com/actions/download-artifact) from 6.0.0 to 7.0.0.

- [Release notes](https://github.com/actions/download-artifact/releases)
- [Commits](https://github.com/actions/download-artifact/compare/018cc2cf5baa6db3ef3c5f8a56943fffe632ef53...37930b1c2abaa49bbe596cd826c3c89aef350131)

  ---
  updated-dependencies:

- dependency-name: actions/download-artifact
    dependency-version: 7.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump actions/cache from 4 to 5 [#153](https://github.com/acgetchell/delaunay/pull/153)
  [`9ad6f99`](https://github.com/acgetchell/delaunay/commit/9ad6f99d8a6eab55f3413f8ad60ccd78e4cd628b)

Bumps [actions/cache](https://github.com/actions/cache) from 4 to 5.

- [Release notes](https://github.com/actions/cache/releases)
- [Commits](https://github.com/actions/cache/compare/v4...v5)

  ---
  updated-dependencies:

- dependency-name: actions/cache
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump codecov/codecov-action from 5.5.1 to 5.5.2 [#149](https://github.com/acgetchell/delaunay/pull/149)
  [`a72754b`](https://github.com/acgetchell/delaunay/commit/a72754b2ae0110448783fe50dfc69e687d09ccc7)

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.1 to 5.5.2.

- [Release notes](https://github.com/codecov/codecov-action/releases)
- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/codecov/codecov-action/compare/5a1091511ad55cbe89839c7260b706298ca349f7...671740ac38dd9b0130fbe1cec585b89eea48d3de)

  ---
  updated-dependencies:

- dependency-name: codecov/codecov-action
    dependency-version: 5.5.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/upload-artifact from 5 to 6 [#150](https://github.com/acgetchell/delaunay/pull/150)
  [`fd20d59`](https://github.com/acgetchell/delaunay/commit/fd20d59586efe5164edfa3f10e57144418c585b5)

Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 5 to 6.

- [Release notes](https://github.com/actions/upload-artifact/releases)
- [Commits](https://github.com/actions/upload-artifact/compare/v5...v6)

  ---
  updated-dependencies:

- dependency-name: actions/upload-artifact
    dependency-version: '6'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump astral-sh/setup-uv from 7.1.5 to 7.1.6 [#151](https://github.com/acgetchell/delaunay/pull/151)
  [`d8d3aac`](https://github.com/acgetchell/delaunay/commit/d8d3aacc0be6dfd4264eabea202a9125a465dfbe)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.5 to 7.1.6.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/ed21f2f24f8dd64503750218de024bcf64c7250a...681c641aba71e4a1c380be3ab5e12ad51f415867)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.6
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.62.63 to 2.63.3 [#152](https://github.com/acgetchell/delaunay/pull/152)
  [`f4754f5`](https://github.com/acgetchell/delaunay/commit/f4754f508aa75398bc136178d5f930f497bd3549)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.62.63 to 2.63.3.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/50708e9ba8d7b6587a2cb575ddaa9a62e927bc06...d850aa816998e5cf15f67a78c7b933f2a5033f8a)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.63.3
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Release v0.6.1 [#156](https://github.com/acgetchell/delaunay/pull/156)
  [`5ab8086`](https://github.com/acgetchell/delaunay/commit/5ab8086fab96c2c7b45405929ebd7d64a83c0937)

- chore(release): release v0.6.1

  Performance increases!

  - Uses DenseSlotMap for backend storage of Tds
  - Uses la-stack instead of nalgebra for fast, stack-based linear algebra

  Bump version to v0.6.1

  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.6.1

### Performance

- Make DenseSlotMap default storage backend [#146](https://github.com/acgetchell/delaunay/pull/146)
  [`7dfd42a`](https://github.com/acgetchell/delaunay/commit/7dfd42a4fa7aa08fce22efafabe31d96a873d291)

- perf: make DenseSlotMap default storage backend

  - Switch default Cargo feature set to `dense-slotmap` (SlotMap remains available via `--no-default-features`)
  - Update docs/tests to reflect the new default and the correct build/test commands for each backend
  - Align storage-backend comparison tooling (local + Slurm) with the new feature/default behavior
  - Archive Phase 4 doc under `docs/archive/` (no stub left in `docs/`) and update references
  - Remove Phase 4–specific `just` recipes while keeping storage-backend comparison workflows
  - Misc hygiene: ignore storage comparison outputs, fix markdownlint line wrap, add spellcheck word

  - ci: stabilize benchmark CI and keep SlotMap backend exercised

  Run same-runner A/B perf comparisons in benchmarks workflow and upload baseline/harness artifacts for debugging

  - Reduce CI benchmark variance by pre-generating inputs and timing only triangulation construction
  - Fix storage backend comparison attribution by clearing Criterion output between runs and making backend flags explicit
  - Add a SlotMap clippy pass (--no-default-features) to prevent bit-rot
  - Docs: mark Phase 4 complete, standardize `dense-slotmap` naming, document SlotMap selection, and label `topology-validation` as proposed
- Migrate geometry predicates to la-stack [#155](https://github.com/acgetchell/delaunay/pull/155)
  [`3ac8b11`](https://github.com/acgetchell/delaunay/commit/3ac8b111a6c5ae33f069a32764718bfe4707db22)

- perf: migrate geometry predicates to la-stack

  - Geometry:
    - Replace nalgebra-backed dynamic matrices with la-stack fixed-size matrices via macro dispatch (D+1 / D+2).
    - Refactor predicates + robust predicates to use la-stack determinant/LU and shared tolerance helpers.
    - Update circumcenter + Gram-matrix utilities to solve via la-stack LU (no inversion) and compute Gram dets with la-stack.
  - Tests:
    - Remove nalgebra from circumsphere_debug_tools and update remaining predicate/util tests accordingly.

  - fix(geometry): avoid panic in stack-matrix dispatch

  - Add StackMatrixDispatchError (UnsupportedDim/La) and try_with_la_stack_matrix! helper
  - Switch predicate/utility call sites to fallible dispatch and propagate via existing error types
  - Keep with_la_stack_matrix! as the panic-on-programmer-error variant for internal uses

  - refactor(geometry): reduce duplication and clarify tolerance/degeneracy docs

  - Compute adaptive tolerance before consuming matrices for determinant evaluation (predicates + robust predicates)
  - Factor robust insphere (D+2)x(D+2) matrix assembly into a shared helper to keep layouts in sync
  - Clarify CoordinateConversionError::ConversionFailed coordinate_index semantics for non-per-coordinate failures
  - Clarify clamp_gram_determinant docs: clamping is numerical hygiene; zero/non-positive determinants are treated as degenerate

## [0.6.0] - 2025-12-07

### Merged Pull Requests

- Release v0.6.0 [#140](https://github.com/acgetchell/delaunay/pull/140)
- Add Level 3 manifold validation with validate_manifold() [#139](https://github.com/acgetchell/delaunay/pull/139)
- Investigate Issue #120 and document bistellar flip requirement [#138](https://github.com/acgetchell/delaunay/pull/138)
- Bump slotmap in the dependencies group across 1 directory [#137](https://github.com/acgetchell/delaunay/pull/137)
- Bump taiki-e/install-action from 2.62.56 to 2.62.60 [#134](https://github.com/acgetchell/delaunay/pull/134)
- Feat/locate and insert [#132](https://github.com/acgetchell/delaunay/pull/132)

### Added

- Add Level 3 manifold validation with validate_manifold() [#139](https://github.com/acgetchell/delaunay/pull/139)
  [`37845f5`](https://github.com/acgetchell/delaunay/commit/37845f505aa665bfb5c5fa7d8d414898a3024de3)

- feat: add Level 3 manifold validation with validate_manifold()

  Implements a comprehensive 4-level validation hierarchy with new Level 3
  (Manifold Topology) validation between TDS structural checks and Delaunay
  property validation.

  Core Implementation:

  - Add Triangulation::validate_manifold() method with facet sharing and Euler characteristic validation
  - Implement validate_manifold_facets() to ensure each facet has exactly 1 or 2 incident cells
  - Implement validate_euler_characteristic() with dimension-specific expected values (0D-3D)
  - Add TriangulationValidationError with detailed error messages for manifold violations
  - Generate macro-based tests for 2D-5D dimensions plus edge cases (6 test functions)

### Changed

- Feat/locate and insert [#132](https://github.com/acgetchell/delaunay/pull/132)
  [`5c8d87a`](https://github.com/acgetchell/delaunay/commit/5c8d87a6fe6e1dcbdea5900a864a383cf242ccc0)

### Documentation

- Investigate Issue #120 and document bistellar flip requirement [#138](https://github.com/acgetchell/delaunay/pull/138)
  [`9e48381`](https://github.com/acgetchell/delaunay/commit/9e48381be33b907cec76b9199ee5e8bde881391f)

- docs: investigate Issue #120 and document bistellar flip requirement

  Investigation determined that property test failures correctly identify
  missing algorithmic capability rather than test configuration issues.

  The incremental Bowyer-Watson algorithm can produce locally non-Delaunay
  configurations that cannot be repaired without topology-changing operations
  (bistellar flips). Current global repair mechanism can detect and remove
  violated cells but cannot flip edges to restore Delaunay property.

### Fixed

- Iteration limit on Delaunay repairs [`14c81a0`](https://github.com/acgetchell/delaunay/commit/14c81a0149d0c3c5b74f5b7991e0fe391474a8e8)

Also add targeted test

### Maintenance

- Improve SLURM storage comparison script reliability and add spell check word
  [`c73dbe9`](https://github.com/acgetchell/delaunay/commit/c73dbe9aff32879ab456ea24188fcf35bc120135)

SLURM Configuration:

- Add account specification (adamgrp) for proper job accounting
- Switch from compute to med2 partition for appropriate resource allocation
- Add login shell support (-l flag) for proper environment initialization

  Module System Support:

- Add robust module system initialization with fallback detection
- Conditionally load rust/1.91.0 and python/3.11 modules with graceful warnings
- Support both /etc/profile.d/modules.sh and Lmod initialization paths
- Gracefully handle missing module command by using existing PATH

  Build Optimization:

- Use node-local scratch for CARGO_TARGET_DIR under SLURM to improve I/O performance
- Make cargo update conditional via CARGO_UPDATE_IN_JOB environment variable
- Add tolerance for cargo clean failures due to NFS .nfs* lock files

  Error Handling and Monitoring:

- Track phase completion status (ok/timeout/failed) for both SlotMap and DenseSlotMap
- Generate overall job status (COMPLETED vs COMPLETED_WITH_ISSUES) in summary
- Add graceful handling for missing Criterion output directories
- Improve error messages with emoji indicators (⚠️, ℹ️) for better visibility
- Bump slotmap in the dependencies group across 1 directory [#137](https://github.com/acgetchell/delaunay/pull/137)
  [`f984b91`](https://github.com/acgetchell/delaunay/commit/f984b9150ca4513e2e6c658356ed78d3295482e3)

Bumps the dependencies group with 1 update in the / directory: [slotmap](https://github.com/orlp/slotmap).

  Updates `slotmap` from 1.1.0 to 1.1.1

- [Changelog](https://github.com/orlp/slotmap/blob/master/RELEASES.md)
- [Commits](https://github.com/orlp/slotmap/compare/v1.1.0...v1.1.1)

  ---
  updated-dependencies:

- dependency-name: slotmap
    dependency-version: 1.1.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.62.56 to 2.62.60 [#134](https://github.com/acgetchell/delaunay/pull/134)
  [`739038f`](https://github.com/acgetchell/delaunay/commit/739038f388e0e42b1cf0f6e4a120f1169716be94)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.62.56 to 2.62.60.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/f79fe7514db78f0a7bdba3cb6dd9c1baa7d046d9...3575e532701a5fc614b0c842e4119af4cc5fd16d)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.62.60
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Release v0.6.0 [#140](https://github.com/acgetchell/delaunay/pull/140)
  [`2c9d6e4`](https://github.com/acgetchell/delaunay/commit/2c9d6e438be7aaee4be17e6c1c7bc1086d9c7394)

- chore(release): release v0.6.0

  - Bump version to v0.6.0
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.6.0

  - fix: correct circumsphere performance analysis and recommendations

  **Issues Fixed:**

  - Performance ranking incorrectly included boundary vertex cases (3-4ns outliers)
  - Ranking claimed insphere was fastest overall, contradicting actual benchmark data
  - Recommendations overstated insphere superiority without supporting data

## [0.5.4] - 2025-11-25

### Merged Pull Requests

- Release v0.5.4 [#131](https://github.com/acgetchell/delaunay/pull/131)
- Bump taiki-e/install-action from 2.62.54 to 2.62.56 [#130](https://github.com/acgetchell/delaunay/pull/130)
- Bump actions/checkout from 5.0.0 to 6.0.0 [#129](https://github.com/acgetchell/delaunay/pull/129)
- Bump astral-sh/setup-uv from 7.1.2 to 7.1.4 [#128](https://github.com/acgetchell/delaunay/pull/128)
- Perf/reduce-heap-allocations [#127](https://github.com/acgetchell/delaunay/pull/127)
- Add stack-allocated SmallBuffer types for D+1 operations [#126](https://github.com/acgetchell/delaunay/pull/126)
- Fix remove_vertex topology consistency and test failures [#124](https://github.com/acgetchell/delaunay/pull/124)
- Refactor/jaccard [#123](https://github.com/acgetchell/delaunay/pull/123)
- Migrate to nalgebra and add adaptive tolerance [#122](https://github.com/acgetchell/delaunay/pull/122)
- Test/improve coverage [#119](https://github.com/acgetchell/delaunay/pull/119)

### Added

- Migrate to nalgebra and add adaptive tolerance [#122](https://github.com/acgetchell/delaunay/pull/122)
  [`1baa1ee`](https://github.com/acgetchell/delaunay/commit/1baa1ee01a4b04d99439a53a81f411cbbcab2910)

- feat(geometry): migrate to nalgebra and add adaptive tolerance

  •  Replace peroxide with nalgebra for matrix ops; introduce Matrix alias
    (na::DMatrix<f64>), update determinant/inverse calls
  •  Add matrix::adaptive_tolerance() that excludes an all-ones last column
    for ∞-norm scaling; apply to orientation/insphere (incl. lifted) and
    add fast path when test point equals a simplex vertex
  •  Update predicates, robust_predicates, util, and debug tools to use the
    new matrix utilities
  •  Derive thiserror::Error for core/geometry error types; remove manual
    Error impls and simplify error handling
  •  Remove peroxide from Cargo.toml; update Cargo.lock
  •  Docs: expand numerical robustness guide (adaptive tolerance details and
    sign conventions) and wrap long lines to satisfy markdownlint (MD013)
  •  Tests: add proptest regression corpus and update affected tests

### Changed

- Test/improve coverage [#119](https://github.com/acgetchell/delaunay/pull/119)
  [`936fc8c`](https://github.com/acgetchell/delaunay/commit/936fc8ce49cd712e35f9a0026166c8e293ba4d9f)
- Increase tarpaulin timeout for 5D integration tests [`8755071`](https://github.com/acgetchell/delaunay/commit/87550712a8a5595b76beb85c92e40f99d850dbf3)

Increases the tarpaulin timeout to 240 seconds. This is done to
  accommodate the longer execution times of the 5D integration tests,
  preventing premature termination of the code coverage analysis.

- Updates dependencies and reduces test samples [`318990a`](https://github.com/acgetchell/delaunay/commit/318990a72d38bdb8e4a83589a45ef7b99b768510)

Updates `derive_builder` and `pastey` dependencies to their
  latest versions. Reduces the sample size in one integration
  test to improve coverage run times. (Internal change)

- Skips 5D tests in CI, adds attributes to tests [`ed4b714`](https://github.com/acgetchell/delaunay/commit/ed4b71438a7f5a9b7928a4be030e2304cffec70b)

Changes the 5D tests to be skipped during continuous integration
  due to their long runtime. Adds attributes to the integration tests
  via macro to allow for skipping.

- Refactor/jaccard [#123](https://github.com/acgetchell/delaunay/pull/123)
  [`9eaa225`](https://github.com/acgetchell/delaunay/commit/9eaa22539e605761a0694e1b7c66be0231708f1d)

### Documentation

- Update CHANGELOG.md [`b6825e9`](https://github.com/acgetchell/delaunay/commit/b6825e99bb48799571aa107c9c8e8585c2fc2d5a)

### Fixed

- Fix remove_vertex topology consistency and test failures [#124](https://github.com/acgetchell/delaunay/pull/124)
  [`da473c8`](https://github.com/acgetchell/delaunay/commit/da473c8c66557b18217ddb45520127dee49bae49)

- fix(core): Fix remove_vertex topology consistency and test failures

  This commit addresses three critical issues:

  1. **Fix remove_vertex to maintain topology consistency**

     - Added logic to clear dangling neighbor references in remaining cells
     - Prevents triangulation corruption when removing vertices
     - Added 3 comprehensive unit tests for remove_vertex functionality
     - Fixes failing doctest at triangulation_data_structure.rs:1784

  2. **Fix proptest quality metric failures**

     - Changed cell comparison from key-order to UUID-based matching
     - Translation/scaling operations can reorder cells in Delaunay triangulation
     - Tests now match cells by their vertex UUIDs instead of assuming same order
     - Fixes prop_normalized_volume and prop_radius_ratio tests in 4D/5D

  3. **Fix convex hull integration test**

     - Updated test_convex_hull_cache_during_construction with proper hull vertices
     - Previous vertices were all interior, causing "No boundary facets" error

  Additional improvements:

  - Improve rollback functionality for missed inserts
  - RobustBowyerWatson now enforces Delaunay property post-insertion
  - Improved duplicate coordinate detection in insertion algorithms
  - Archive completed invariant_validation_plan.md documentation
  - Remove proptest regression files for cleaner repo state

  - fix(triangulation): Fix Delaunay violations in cavity-based insertion

  Fixes a critical bug where the Bowyer-Watson insertion algorithm was
  creating triangulations with violations of the empty circumsphere
  property. The issue manifested in proptest failures and affected
  approximately 1.35% of test cases.

  Root cause: The `determine_strategy_default` method used a faulty
  heuristic for single-cell triangulations, assuming all vertices were
  exterior without proper geometric validation.

- Handle GitHub's 125KB tag annotation limit with CHANGELOG.md references
  [`a74bbdb`](https://github.com/acgetchell/delaunay/commit/a74bbdbaf35d75c589aae25dd89ad74cdb27bf86)

Resolve git tag creation failures when changelog content exceeds GitHub's
  125,000 byte annotation limit (encountered with v0.5.4 at 152KB).

  Tag Creation Strategy:

- Always creates annotated tags for consistency
- Changelogs ≤125KB: Full content in tag annotation
- Changelogs >125KB: Short reference message with CHANGELOG.md link
- Users always use `gh release create "$TAG" --notes-from-tag`

  Implementation (scripts/changelog_utils.py):

- _get_changelog_content(): Returns tuple (message, is_truncated)
  - Checks byte size against 125,000 limit
  - Returns reference message for large changelogs
- _create_tag_with_message(): Always creates annotated tags
  - Renamed parameter: is_lightweight → is_truncated
  - Maintains uniform workflow regardless of size
- _show_success_message(): Simplified to always use --notes-from-tag
  - Notes CHANGELOG.md reference for large changelogs

  Testing (scripts/tests/test_changelog_tag_size_limit.py):

- 7 comprehensive tests using real v0.5.4 data (152KB)
- Tests cover: limit detection, tag creation, success messages, full workflow
- Validates both normal and large changelog handling
- All 452 Python tests passing

### Maintenance

- Update documentation [`283d5e2`](https://github.com/acgetchell/delaunay/commit/283d5e259a92433bda73b3bc02990920920fc08e)
- Bump astral-sh/setup-uv from 7.1.2 to 7.1.4 [#128](https://github.com/acgetchell/delaunay/pull/128)
  [`e3e5977`](https://github.com/acgetchell/delaunay/commit/e3e597779dee3becaeb61430e3bd416352b97ca2)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.2 to 7.1.4.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41...1e862dfacbd1d6d858c55d9b792c756523627244)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.4
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/checkout from 5.0.0 to 6.0.0 [#129](https://github.com/acgetchell/delaunay/pull/129)
  [`747b6ef`](https://github.com/acgetchell/delaunay/commit/747b6efd5c16d335340b02dec4821a1019273588)

Bumps [actions/checkout](https://github.com/actions/checkout) from 5.0.0 to 6.0.0.

- [Release notes](https://github.com/actions/checkout/releases)
- [Commits](https://github.com/actions/checkout/compare/v5...v6)

  ---
  updated-dependencies:

- dependency-name: actions/checkout
    dependency-version: 6.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump taiki-e/install-action from 2.62.54 to 2.62.56 [#130](https://github.com/acgetchell/delaunay/pull/130)
  [`37ccaec`](https://github.com/acgetchell/delaunay/commit/37ccaecab39e7951ee9fbc3da7f7dd1556bdbf95)

Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.62.54 to 2.62.56.

- [Release notes](https://github.com/taiki-e/install-action/releases)
- [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/taiki-e/install-action/compare/62da238c048aa0f865cc5a322082957d34e7fc1a...f79fe7514db78f0a7bdba3cb6dd9c1baa7d046d9)

  ---
  updated-dependencies:

- dependency-name: taiki-e/install-action
    dependency-version: 2.62.56
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Release v0.5.4 [#131](https://github.com/acgetchell/delaunay/pull/131)
  [`bdfc43d`](https://github.com/acgetchell/delaunay/commit/bdfc43dbae7d5dd0109fa7ad6a1fce9d6a01afaa)

- chore(release): release v0.5.4

  - Bump version to v0.5.4
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.5.4

  - fix(build): convert setext headings to ATX in changelog and update SLURM script

  Fixes markdown linting errors (MD003, MD025, MD001) in auto-generated
  CHANGELOG.md caused by setext-style headings from commit messages.

  Changelog fixes:

  - Add _convert_setext_to_atx() to detect and convert setext headings
    (text underlined with === or ---) to ATX style (####)

  - Apply conversion in _format_entry_body() before processing
  - Add "setext", "clonable", "wordlist" to cspell dictionary
  - Regenerate CHANGELOG.md with fixes applied

  All headings in commit message bodies now consistently use #### to:
  1. Avoid conflicts with changelog structure (## for versions, ### for sections)
  2. Maintain proper heading hierarchy
  3. Satisfy markdownlint MD001 (no heading level jumps)

  SLURM script updates:

  - Enable Rust 1.91.0 and Python 3.11 module loads for cluster execution
  - Update from commented examples to active module loads matching MSRV

  - fix(build): lower Python requirement to 3.11 and update documentation

  Reduces Python version requirement from 3.13 to 3.11, making the project
  more accessible while maintaining all modern features.

  Python version changes:

  - Update .python-version from 3.13 to 3.11
  - Update pyproject.toml requires-python to >=3.11
  - Add Python 3.11 and 3.12 classifiers
  - Update ruff target-version to py311
  - Update all documentation references from 3.13+ to 3.11+

  Features used only require Python 3.11+:

  - PEP 585 built-in generics (list[str], dict[str, Any]) - 3.9+
  - PEP 604 union types (X | None) - 3.10+
  - datetime.UTC - 3.11+

  Documentation corrections:

  - CONTRIBUTING.md: Fix MSRV version 1.90.0 → 1.91.0
  - benches/PERFORMANCE_RESULTS.md: Fix misleading "consistently best"
    performance statement to reflect actual test results

  - WARP.md: Add GitHub CLI non-interactive usage requirements

  Coverage test fixes:

  - Remove storage_backend_compatibility from coverage (all tests ignored)
  - Fixes codecov CI "Test failed during run" error

  All 445 Python tests pass with Python 3.11.14. All quality checks pass.

  - docs: improve performance documentation clarity and fix markdown lint

  Performance Documentation:

  - Add average performance times to rankings (161 ns, 177 ns, 181 ns)
  - Clarify relative performance differences (~1.1x slower than fastest)
  - Restructure recommendations with clearer headings and use case guidance
  - Emphasize boundary vertex performance advantage (3-4 ns vs 78-366 ns)
  - Break long lines in conclusion to meet 160-char markdown lint limit

  Coverage Configuration:

  - Exclude test_cavity_boundary_error from coverage runs
  - Document reason: regression test is very slow (~2.5min) and causes tarpaulin issues
  - Improves coverage report generation reliability

  - docs: update just commands and clarify performance recommendations

  Update all documentation files with correct just workflow commands from
  justfile, replacing outdated command references and ensuring consistency
  across the project.

  Documentation Updates:

  - README.md: Replace `just dev`/`just quality` with `just ci` and related commands
  - CONTRIBUTING.md: Update workflow sections with complete command listings
    including test variants, hierarchical linting, benchmark commands with
    timing estimates, and utility commands

  - WARP.md: Update Just Workflow Summary, Code Quality Checks, Testing and
    Validation, and Performance sections with accurate commands

  - docs/code_organization.md: Update Development Workflow section with
    comprehensive command organization and regenerate file layout tree
    (added proptest_duplicates.rs)

  Performance Documentation:

  - benches/PERFORMANCE_RESULTS.md: Clarify performance ranking descriptions
    and recommendation consistency

    - Add specific timings "(3-4 ns)" to performance ranking for clarity
    - Explicitly state `insphere` is both fastest AND most stable upfront
    - Eliminate ambiguity about performance vs stability trade-offs
    - Update wording to emphasize insphere combines best performance with
      numerical reliability

  Key Command Changes:

  - Replaced: `just dev`, `just quality`, `just pre-commit`
  - Updated to: `just ci`, `just commit-check`, `just commit-check-slow`
  - Added: `just help-workflows` references throughout
  - Expanded: Testing commands (test-integration, test-slow, test-slow-release)
  - Expanded: Linting hierarchy (lint, lint-code, lint-docs, lint-config)
  - Added: Benchmark variants with timing estimates (bench-ci, bench-dev, bench-quick)
  - Added: Profiling commands (profile, profile-dev, profile-mem)

  All changes validated with markdown linting, spell checking, and JSON validation.

### Performance

- Add stack-allocated SmallBuffer types for D+1 operations [#126](https://github.com/acgetchell/delaunay/pull/126)
  [`f7203df`](https://github.com/acgetchell/delaunay/commit/f7203df88008dd51431d25708caaf0f06b51861b)

- perf(collections): add stack-allocated SmallBuffer types for D+1 operations

  Replace Vec allocations with stack-allocated SmallBuffer types for
  collections that always contain D+1 elements (typically 3-6 items).
  Eliminates heap allocations for D≤7 (90%+ of use cases).

  New type aliases in src/core/collections.rs:

  - NeighborBuffer<T>: Stack buffer for cell neighbors (D+1)
  - CellVertexBuffer: Stack buffer for vertex keys (D+1)
  - CellVertexUuidBuffer: Stack buffer for vertex UUIDs (D+1)
  - CellToVertexUuidsMap: HashMap using CellVertexUuidBuffer values

  Updated functions:

  - Tds::find_neighbors_by_key() → returns NeighborBuffer<Option<CellKey>>
  - Cell::vertex_uuids() → returns CellVertexUuidBuffer
  - CellVertexKeysMap → uses CellVertexBuffer values
  - Serialization converts SmallVec to Vec for serde compatibility

  Performance benefits:

  - Zero heap allocations for D+1 collections when D≤7
  - Better cache locality with 128-byte stack allocation
  - Maintains same API surface (both implement Deref<[T]>)

  Test improvements:

  - Stabilized 4/5 Delaunay condition property tests (2D-5D empty circumsphere)
  - Documented insertion-order dependency issue in prop_insertion_order_invariance_2d
    (requires separate algorithmic investigation)

  - Removed stale proptest regression file
  - Fixed zero_allocation_iterator_demo to compare as slices

  - refactor(collections): improve type alias consistency and test coverage

  Improve consistency and discoverability of D+1 buffer type aliases:

  - CellNeighborsMap now uses NeighborBuffer<Option<CellKey>> instead of
    spelling out SmallBuffer<..., MAX_PRACTICAL_DIMENSION_SIZE> inline

  - Makes all D+1 neighbor collections share single semantic alias
  - Updates documentation to reference NeighborBuffer consistently

  Add test coverage for CellVertexKeysMap:

  - Extend test_collection_type_instantiation to test CellVertexKeysMap
  - Verify D+1 usage pattern (3 vertices for 2D cell)
  - Validate stack allocation behavior (no heap spill for D≤7)

  Addresses code review feedback for better semantic naming and
  protection against accidental signature changes.

  - refactor(cell): use semantic type aliases for internal buffers

  Replace raw SmallBuffer types with semantic aliases in Cell struct:

  - vertices: SmallBuffer<VertexKey, 8> → CellVertexBuffer
  - neighbors: SmallBuffer<Option<CellKey>, 8> → NeighborBuffer<Option<CellKey>>

  Updated locations:

  - Cell struct field declarations (lines 290, 309)
  - Cell::new() parameter type (line 482)
  - Cell::neighbors() return type (line 652)
  - Cell::ensure_neighbors_buffer_mut() return type (line 727)
  - Deserialization initialization (line 431)
- Perf/reduce-heap-allocations [#127](https://github.com/acgetchell/delaunay/pull/127)
  [`e3def62`](https://github.com/acgetchell/delaunay/commit/e3def623db8dfe98c5d981d82ad97d6c3da913b8)

## [0.5.3] - 2025-10-31

### Merged Pull Requests

- Release v0.5.3 [#117](https://github.com/acgetchell/delaunay/pull/117)
- Handle degenerate configurations and improve test robustness [#116](https://github.com/acgetchell/delaunay/pull/116)

### Fixed

- Prevent duplicate commit entries in generated changelog [`dbcfb87`](https://github.com/acgetchell/delaunay/commit/dbcfb872dbb542b90f5526263f9692073123a74b)

Fixes two bugs in changelog_utils.py that caused duplicate content:

  1. Reused changes_section_index causing multiple insertions

- Mark changes_section_index as used (-1) after insertion to prevent
       the same index from being reused for duplicate insertions

- Add missing pending_expanded_commits.clear() in _finalize_pending_commits

  2. Expanded PR commits duplicating entries from Fixed Issues section

- Track expanded commit SHAs in self.expanded_commit_shas set
- Skip commit lines whose SHAs were already expanded from PRs
- Store both full and short SHA formats to catch all duplicates
- Clear tracking set per release to avoid cross-release conflicts

  Root cause: auto-changelog lists squash commits in both "Merged Pull
  Requests" and "Fixed Issues" sections. PR expansion was processing
  the same commit twice—once from the PR entry and again from the
  commit entry.

- Normalize markdown headers and protect cron expressions [`1f070dd`](https://github.com/acgetchell/delaunay/commit/1f070dd7c6684bbed36f8aec376badaa461b425e)

Fixes markdown formatting issues in commit message bodies:

  1. Normalize header levels to ####

- Convert all markdown headers (# through ######) to #### in commit bodies
- Maintains consistent hierarchy: ## Release > ### Section > #### Detail
- Prevents conflicts with changelog structure (## for releases, ### for sections)
- Satisfies markdownlint MD001 rule (no heading level jumps)
- Previous approach (adding 2 levels) created h3→h5 jumps that violated MD001

  2. Protect cron expressions from markdown interpretation

- Add _protect_cron_expressions() to wrap cron patterns in backticks
- Detects quoted strings with asterisks and digits (e.g., '0 2 * * 0')
- Prevents markdown from treating * as emphasis markers
- Example: '0 2 * * 0' → `0 2 * * 0`
- Handle degenerate configurations and improve test robustness [#116](https://github.com/acgetchell/delaunay/pull/116)
  [`a6ec3fa`](https://github.com/acgetchell/delaunay/commit/a6ec3fae7423e9245018e7fb071bd92fbb70283e)

- fix: Handle degenerate configurations and improve test robustness

  This commit addresses several issues related to degenerate point
  configurations and improves the robustness of property-based tests:

  **Duplicate Vertex Handling**

  - Add duplicate coordinate detection to Tds::new() constructor
  - Silently skip vertices with identical coordinates during batch construction
  - Document behavior difference between new() (permissive) and add() (strict)
  - Matches expected behavior for batch initialization from potentially unfiltered data

  **Property Test Improvements**

  - Filter degenerate configurations in convex hull proptests
    (collinear/coplanar points with no boundary facets)

  - Use approx crate for floating-point comparisons in serialization tests
  - Add BoundaryAnalysis trait import where needed
  - Fix debug-only variable compilation with #[cfg(debug_assertions)]

  **Serialization Test Fixes**

  - Address JSON floating-point precision limitations (not a bug)
  - Use relative_eq! from approx crate instead of exact equality
  - JSON typically preserves 15-17 significant digits for f64
  - Minor coordinate differences after roundtrip are expected and acceptable

  **New Property Tests**

  - Add comprehensive proptest suites for Bowyer-Watson, convex hull,
    cells, facets, geometry utilities, quality metrics, and serialization

  - Include proptest regression files to track minimal failing cases
  - Add serialization vertex preservation integration test

  All tests now pass successfully. Degenerate cases are filtered in
  property tests since they're covered by dedicated edge case tests.

  - fix: Add rollback on cell creation failure and fix property test

  Implement atomic insertion semantics by adding rollback mechanisms when
  cell creation fails during vertex insertion. This prevents TDS
  corruption from partial insertions that leave dangling references.

  Key changes:

  - Add rollback after preventive facet filtering to handle edge cases
    where all boundary facets are unexpectedly removed

  - Add rollback when cell creation yields zero cells despite valid
    facets, preventing inconsistent triangulation state

  - Implement Point Hash + Eq for efficient duplicate detection in
    Tds::new() using HashSet instead of O(n²) sequential scan

  - Add Point import and move HashSet to top-level imports
  - Fix degenerate simplex generator in proptest_quality to create
    full-dimensional thin simplices using safe_usize_to_scalar

  - Add proptest regression file for insertion_algorithm edge cases
  - Improve serialization test coverage for vertex preservation

  The rollback logic ensures failed insertions don't corrupt the
  triangulation, maintaining the invariant that operations either fully
  succeed or leave the TDS unchanged. Property tests now properly
  exercise quality metric comparisons between regular and degenerate
  simplices.

  Fixes edge cases where invalid facet sharing or degenerate geometry
  would previously cause TDS invariant violations.

  - fix: Fix preventive facet filter and UUID/construction_state bugs

  Fix critical bug in preventive facet filtering where all valid interior
  boundary facets were incorrectly rejected, and address UUID duplicate
  detection and construction_state accuracy issues.

  **Preventive Facet Filter Fix (CRITICAL)**

  - Fixed filter_boundary_facets_by_valid_facet_sharing to discount bad
    cell contributions before checking facet sharing limits

  - Bad cells are about to be deleted, but were being counted as if they
    would remain, causing every interior facet to be flagged as over-sharing

  - Added unit test test_preventive_filter_does_not_reject_valid_interior_facets
    to prevent regression of this critical bug

  - Captured bad_cell before moving info to avoid borrow issues

  **UUID Duplicate Detection Fix**

  - Reordered Tds::new() to check UUID collisions before coordinate deduplication
  - Previously duplicate UUIDs were missed when coordinates were also duplicated
  - Added Vertex::new_with_uuid() test helper (marked #[doc(hidden)])
  - Added 4 integration tests for UUID collision detection scenarios

  **Construction State Accuracy Fix**

  - Recalculate construction_state after deduplication loop in Tds::new()
  - Now reflects actual unique vertex count instead of raw input length
  - Added 2 integration tests verifying correct state calculation

  All quality checks pass (979 unit tests, 8 integration tests, clippy pedantic).
  The preventive filter now correctly allows valid cavity reconstructions to proceed.

  - refactor: Improve error handling and performance in insertion algorithm

  Add early exit for empty facet handles, optimize deduplication, clarify
  documentation, and restrict test helper visibility.

  **Early Exit for Empty Input**

  - Add check at start of create_cells_from_facet_handles for empty handles
  - Returns clear error "No facet handles provided" instead of misleading
    "preventive filtering" message

  - Avoids unnecessary vertex insertion and rollback operations

  **Improved Error Messages**

  - Changed post-filtering error to "No boundary facets available after filtering"
  - More accurate since empty result can come from filtering, deduplication, or
    empty input

  **Performance Optimization**

  - Replace std::HashSet with FastHashSet in deduplicate_boundary_facet_info
  - Pre-allocate both collections with boundary_infos.len() capacity
  - Improves deduplication performance and consistency with codebase

  **Documentation Updates**

  - Clarify saturating_add/sub_for_bbox only support floats (not integers)
  - CoordinateScalar is only implemented for f32/f64
  - Update test comments to reflect early exit behavior
  - Add prominent warning to Vertex::new_with_uuid test helper
  - Keep pub visibility for integration tests but enhance safety warnings

  - perf: Optimize deduplication and facet filtering hot paths

  Reduce allocations in deduplication and improve hashing performance in
  the preventive facet filter's inner loop.

  **Deduplication Optimization**

  - Use u64 facet keys instead of Vec<VertexKey> for deduplication
  - Leverages canonical facet_key_from_vertices() hashing
  - Eliminates Vec allocation per facet (N allocations saved)
  - Consistent with hashing used throughout codebase

  **Facet Filter Optimization**

  - Replace std::HashMap with FastHashMap in filter_boundary_facets_by_valid_facet_sharing
  - Pre-allocate with facet_map.len() capacity to avoid rehashing
  - Reduces hashing overhead in N·D inner loop
  - Improves consistency with other hot-path containers

  **Documentation Clarification**

  - Clarify saturating_add/sub_for_bbox perform standard FP arithmetic
  - Add note explaining "saturating" naming is for bbox semantic consistency
  - Not actual integer saturation since CoordinateScalar only supports floats

  **Test Alignment**

  - Fix test_create_cells_from_boundary_facets to assert error on empty input
  - Remove unwrap_or(0) pattern in favor of explicit error assertion
  - Consistent with new API contract where empty handles error immediately

  - docs: add AI tool acknowledgements and clarify bbox helpers

  Add AI-assisted development tool acknowledgements:

  - List AI tools used in project development (ChatGPT, Claude, CodeRabbit,
    GitHub Copilot, KiloCode, WARP) in README.md

  - Add detailed citations and human oversight note in REFERENCES.md
  - Clarify all AI output was reviewed/edited by maintainer

  Clarify floating-point bbox helper documentation:

  - Simplify doc comments for saturating_add/sub_for_bbox functions
  - Emphasize "saturating" naming is for semantic consistency, not actual
    integer saturation behavior

  - Update test comment to reflect deduplication via canonical vertex sets

  - refactor: Surface duplicate boundary facets as errors instead of silent filtering

  Replace silent deduplication with explicit error handling to catch algorithmic
  bugs in cavity boundary detection early.

  **Breaking API Changes**

  - deduplicate_boundary_facet_info: Returns Result<SmallBuffer, InsertionError>
    instead of Vec (errors on duplicates, uses SmallBuffer for performance)

  - filter_boundary_facets_by_valid_facet_sharing: Returns Result<SmallBuffer>
    instead of Vec (propagates errors from facet map building)

  - Add InsertionError::DuplicateBoundaryFacets variant with duplicate_count and
    total_count fields

  **Rationale**
  The cavity boundary should form a topological sphere with no duplicate facets.
  Duplicates indicate:

  - Incorrect neighbor traversal logic
  - Non-manifold mesh connectivity
  - Data structure corruption

  By returning errors instead of silently filtering, we surface these bugs
  immediately rather than allowing corrupted triangulations.

  **Performance Optimization**

  - Use SmallBuffer (stack-allocated, falls back to heap) for typical D+1 facets
  - Faster than Vec for small collections (common case in D dimensions)
  - Maintains O(n) duplicate detection with FastHashSet

  **Implementation Details**

  - Propagate DuplicateBoundaryFacets errors through insert_vertex_cavity_based
    and create_cells_from_facet_handles

  - Update tests to expect errors on duplicates instead of silent filtering
  - Rename prop_deduplication_is_idempotent to prop_deduplication_detects_duplicates
  - Fix test_create_cells_from_facet_handles_duplicate_handles to assert error

  - docs: Rename saturating_* helpers to bbox_* and clarify float semantics

  The "saturating" terminology was misleading for floating-point operations.
  These functions perform plain arithmetic, not saturating arithmetic.

  **Renaming**

  - saturating_sub_for_bbox → bbox_sub
  - saturating_add_for_bbox → bbox_add

  **Updated Documentation**

  - Clarify these are plain float operations, NOT saturating arithmetic
  - Explain float overflow behavior: naturally produces ±infinity
  - Document that this is the desired behavior for bounding box expansion
    (ensures all vertices are contained even with extreme coordinates)

  - Add explicit type constraints: only supports f32/f64 via CoordinateScalar
  - Update all 28 call sites and test names

  **Rationale**
  "Saturating" implies integer-style clamping to min/max bounds, but:

  - Floats don't saturate - they naturally produce ±infinity on overflow
  - This is actually the correct behavior for bbox expansion
  - The old naming suggested these functions did something special, when
    they're just plain +/- operations with overflow-to-infinity semantics

### Maintenance

- Release v0.5.3 [#117](https://github.com/acgetchell/delaunay/pull/117)
  [`fa921a0`](https://github.com/acgetchell/delaunay/commit/fa921a092a8c6514f646274b53734b67845bf8bd)

- Bump version to v0.5.3
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.5.3

## [0.5.2] - 2025-10-29

### Merged Pull Requests

- Release v0.5.2 [#115](https://github.com/acgetchell/delaunay/pull/115)
- Bump actions/download-artifact from 5.0.0 to 6.0.0 [#113](https://github.com/acgetchell/delaunay/pull/113)
- Bump astral-sh/setup-uv from 7.1.1 to 7.1.2 [#112](https://github.com/acgetchell/delaunay/pull/112)
- Bump actions/upload-artifact from 4 to 5 [#111](https://github.com/acgetchell/delaunay/pull/111)
- Refactor/phase 4 [#109](https://github.com/acgetchell/delaunay/pull/109)
- Bump astral-sh/setup-uv from 7.1.0 to 7.1.1 [#108](https://github.com/acgetchell/delaunay/pull/108)
- Feature/geometry quality metrics [#107](https://github.com/acgetchell/delaunay/pull/107)

### Changed

- Feature/geometry quality metrics [#107](https://github.com/acgetchell/delaunay/pull/107)
  [`c707b09`](https://github.com/acgetchell/delaunay/commit/c707b09e6ab78a07bbce873f0c39cf8119dfb076)
- Refactor/phase 4 [#109](https://github.com/acgetchell/delaunay/pull/109)
  [`4ecc21f`](https://github.com/acgetchell/delaunay/commit/4ecc21fbe65c1292165594171c6d137093120076)

### Documentation

- Remove outdated TODOs and update documentation references [`a4213e1`](https://github.com/acgetchell/delaunay/commit/a4213e1bdd44626dea91345f3fad7880a8a048cd)

Cleaned up outdated TODOs and documentation references across the codebase:

  Code cleanup:

- Remove outdated Phase 3 cache migration TODOs from bowyer_watson.rs
- Remove deprecated Cell method references and old commented code from cell.rs
- Update issue reference from #86 to #105 in triangulation_data_structure.rs
- Remove outdated test TODO comments from convex_hull.rs

  Documentation updates:

- Update docs/README.md to reference existing files (property_testing_summary.md, topology.md) instead of non-existent optimization guides
- Update docs/OPTIMIZATION_ROADMAP.md to reference archived Phase 2/3 documentation in docs/archive/ instead of removed files
- Update docs/code_organization.md project structure tree to reflect current files:
  - Add proptest files (proptest_point.rs, proptest_predicates.rs, proptest_triangulation.rs)
  - Add docs/archive/ directory with historical documentation
  - Add property_testing_summary.md and topology.md
- Update test counts from 772/774 to current 781 tests across all documentation

  All changes are internal cleanup with no functional impact. All 781 tests passing.

### Maintenance

- Bump astral-sh/setup-uv from 7.1.0 to 7.1.1 [#108](https://github.com/acgetchell/delaunay/pull/108)
  [`7abf5f9`](https://github.com/acgetchell/delaunay/commit/7abf5f97c142eaf148454daab770042b4cdbfd50)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.0 to 7.1.1.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/3259c6206f993105e3a61b142c2d97bf4b9ef83d...2ddd2b9cb38ad8efd50337e8ab201519a34c9f24)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump actions/download-artifact from 5.0.0 to 6.0.0 [#113](https://github.com/acgetchell/delaunay/pull/113)
  [`9756b59`](https://github.com/acgetchell/delaunay/commit/9756b594b26715b478d2eebdf78aebe016bb48d7)

Bumps [actions/download-artifact](https://github.com/actions/download-artifact) from 5.0.0 to 6.0.0.

- [Release notes](https://github.com/actions/download-artifact/releases)
- [Commits](https://github.com/actions/download-artifact/compare/634f93cb2916e3fdff6788551b99b062d0335ce0...018cc2cf5baa6db3ef3c5f8a56943fffe632ef53)

  ---
  updated-dependencies:

- dependency-name: actions/download-artifact
    dependency-version: 6.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump actions/upload-artifact from 4 to 5 [#111](https://github.com/acgetchell/delaunay/pull/111)
  [`3ffdf6b`](https://github.com/acgetchell/delaunay/commit/3ffdf6b10f37776a40be3c3b047cdac9e44709b6)

Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 4 to 5.

- [Release notes](https://github.com/actions/upload-artifact/releases)
- [Commits](https://github.com/actions/upload-artifact/compare/v4...v5)

  ---
  updated-dependencies:

- dependency-name: actions/upload-artifact
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump astral-sh/setup-uv from 7.1.1 to 7.1.2 [#112](https://github.com/acgetchell/delaunay/pull/112)
  [`febd5f5`](https://github.com/acgetchell/delaunay/commit/febd5f55278f73dfd9e69c7b4a71e86058da2815)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.1 to 7.1.2.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/2ddd2b9cb38ad8efd50337e8ab201519a34c9f24...85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Update CHANGELOG, add storage comparison tooling [`49f5eb5`](https://github.com/acgetchell/delaunay/commit/49f5eb5d90bb0a5eb26166986347b73038100de1)

- Generate comprehensive CHANGELOG.md with AI-categorized Phase 4 entries
  - Add legitimate technical terms to cspell.json dictionary (invariances, kibibytes, unparseable)
  - Document storage backend comparison scripts in code_organization.md and scripts/README.md
  - Add Slurm-based storage comparison script for SlotMap vs DenseSlotMap benchmarking
- Release v0.5.2 [#115](https://github.com/acgetchell/delaunay/pull/115)
  [`2f8b68d`](https://github.com/acgetchell/delaunay/commit/2f8b68dbec13079e024a0cab8780303347823225)

- Bump version to v0.5.2
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.5.2

## [0.5.1] - 2025-10-16

### Merged Pull Requests

- Release v0.5.1 [#104](https://github.com/acgetchell/delaunay/pull/104)
- Refactor/complete phase 3 [#103](https://github.com/acgetchell/delaunay/pull/103)
- Bump astral-sh/setup-uv from 7.0.0 to 7.1.0 [#102](https://github.com/acgetchell/delaunay/pull/102)
- Replaces fxhash with rustc-hash for performance [#101](https://github.com/acgetchell/delaunay/pull/101)
- Update/utilities and roadmap [#100](https://github.com/acgetchell/delaunay/pull/100)
- Bump astral-sh/setup-uv from 6.7.0 to 6.8.0 [#96](https://github.com/acgetchell/delaunay/pull/96)
- Bump actions-rust-lang/setup-rust-toolchain [#95](https://github.com/acgetchell/delaunay/pull/95)
- Bump ordered-float in the dependencies group [#94](https://github.com/acgetchell/delaunay/pull/94)
- Bump the dependencies group with 2 updates [#93](https://github.com/acgetchell/delaunay/pull/93)
- Bump actions-rust-lang/setup-rust-toolchain [#92](https://github.com/acgetchell/delaunay/pull/92)

### Changed

- Update/utilities and roadmap [#100](https://github.com/acgetchell/delaunay/pull/100)
  [`353a46e`](https://github.com/acgetchell/delaunay/commit/353a46e385baa936e5e94f0b3a1b5e7b58a3350e)
- Replaces fxhash with rustc-hash for performance [#101](https://github.com/acgetchell/delaunay/pull/101)
  [`3c56251`](https://github.com/acgetchell/delaunay/commit/3c5625137f7cdd5385d056453860f20083882247)

- Changed: Replaces fxhash with rustc-hash for performance

  Replaces the `fxhash` crate with `rustc-hash` for improved
  performance in hash-based collections. This change enhances the
  speed of internal data structures by utilizing a faster hashing
  algorithm.

  Adds detailed logging for performance comparisons to better
  understand benchmark results and potential regressions.

  - Changed: Replaces fxhash with rustc-hash for performance

  Replaces the fxhash hashing algorithm with rustc-hash.

  rustc-hash provides improved performance characteristics for the
  delaunay triangulation algorithm, leading to faster overall execution.

- Refactor/complete phase 3 [#103](https://github.com/acgetchell/delaunay/pull/103)
  [`2db8acd`](https://github.com/acgetchell/delaunay/commit/2db8acd18e373f4847de73526e387c7f14b5a3e5)

### Maintenance

- Bump actions-rust-lang/setup-rust-toolchain [#92](https://github.com/acgetchell/delaunay/pull/92)
  [`e11ff2e`](https://github.com/acgetchell/delaunay/commit/e11ff2e71caba08020843cb25c38937fe14bdff3)

Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.0 to 1.15.1.

- [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
- [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
- [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/2fcdc490d667999e01ddbbf0f2823181beef6b39...02be93da58aa71fb456aa9c43b301149248829d8)

  ---
  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
    dependency-version: 1.15.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump the dependencies group with 2 updates [#93](https://github.com/acgetchell/delaunay/pull/93)
  [`b0a7051`](https://github.com/acgetchell/delaunay/commit/b0a70518c4594dc97ed2f93316d13c227c4cb861)

Bumps the dependencies group with 2 updates: [serde](https://github.com/serde-rs/serde) and [thiserror](https://github.com/dtolnay/thiserror).

  Updates `serde` from 1.0.227 to 1.0.228

- [Release notes](https://github.com/serde-rs/serde/releases)
- [Commits](https://github.com/serde-rs/serde/compare/v1.0.227...v1.0.228)

  Updates `thiserror` from 2.0.16 to 2.0.17

- [Release notes](https://github.com/dtolnay/thiserror/releases)
- [Commits](https://github.com/dtolnay/thiserror/compare/2.0.16...2.0.17)

  ---
  updated-dependencies:

- dependency-name: serde
    dependency-version: 1.0.228
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: thiserror
    dependency-version: 2.0.17
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Bump ordered-float in the dependencies group [#94](https://github.com/acgetchell/delaunay/pull/94)
  [`1a213f7`](https://github.com/acgetchell/delaunay/commit/1a213f7cd262f3dfc319b72c9bde9836db7119cc)

Bumps the dependencies group with 1 update: [ordered-float](https://github.com/reem/rust-ordered-float).

  Updates `ordered-float` from 5.0.0 to 5.1.0

- [Release notes](https://github.com/reem/rust-ordered-float/releases)
- [Commits](https://github.com/reem/rust-ordered-float/compare/v5.0.0...v5.1.0)

  ---
  updated-dependencies:

- dependency-name: ordered-float
    dependency-version: 5.1.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies
  ...

- Bump actions-rust-lang/setup-rust-toolchain [#95](https://github.com/acgetchell/delaunay/pull/95)
  [`34db79a`](https://github.com/acgetchell/delaunay/commit/34db79a89fefc3e062afb1cb47e9da9d80db4f89)

Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.1 to 1.15.2.

- [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
- [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
- [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/02be93da58aa71fb456aa9c43b301149248829d8...1780873c7b576612439a134613cc4cc74ce5538c)

  ---
  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
    dependency-version: 1.15.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 6.7.0 to 6.8.0 [#96](https://github.com/acgetchell/delaunay/pull/96)
  [`ef8c98d`](https://github.com/acgetchell/delaunay/commit/ef8c98df6fd2399790c5f64e2cbcbdfc19e8bbd9)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 6.7.0 to 6.8.0.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/b75a909f75acd358c2196fb9a5f1299a9a8868a4...d0cc045d04ccac9d8b7881df0226f9e82c39688e)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 6.8.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump astral-sh/setup-uv from 7.0.0 to 7.1.0 [#102](https://github.com/acgetchell/delaunay/pull/102)
  [`8a09e60`](https://github.com/acgetchell/delaunay/commit/8a09e6020ec0dcf55a2cc146d8ddb05a48aee1a2)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.0.0 to 7.1.0.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/eb1897b8dc4b5d5bfe39a428a8f2304605e0983c...3259c6206f993105e3a61b142c2d97bf4b9ef83d)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 7.1.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Release v0.5.1 [#104](https://github.com/acgetchell/delaunay/pull/104)
  [`2675a5e`](https://github.com/acgetchell/delaunay/commit/2675a5e00ae8c983c2e37b5cc115c65e6196a872)

- Bump version to v0.5.1
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.5.1

## [0.5.0] - 2025-09-27

### Merged Pull Requests

- Release v0.5.0 [#91](https://github.com/acgetchell/delaunay/pull/91)
- Refactor/phase 3 [#90](https://github.com/acgetchell/delaunay/pull/90)
- Test/improve coverage [#89](https://github.com/acgetchell/delaunay/pull/89)
- Bump the dependencies group with 4 updates [#88](https://github.com/acgetchell/delaunay/pull/88)

### Changed

- Update documentation for v0.4.4 (skip ci) [`696f740`](https://github.com/acgetchell/delaunay/commit/696f74056e07ad2d3a8db4269f281029df089b33)
- Test/improve coverage [#89](https://github.com/acgetchell/delaunay/pull/89)
  [`b8a2024`](https://github.com/acgetchell/delaunay/commit/b8a2024fa9921141b1a453bb8911fbc2dbf02b5f)
- Refactor/phase 3 [#90](https://github.com/acgetchell/delaunay/pull/90)
  [`2a9b655`](https://github.com/acgetchell/delaunay/commit/2a9b655f73432e3cce52f9c9db58e46d525bc9d3)
- Updates docs, adds Claude config ignore, cspell [`7a1e51f`](https://github.com/acgetchell/delaunay/commit/7a1e51f5af069c35ff840103da691b3d38a4536f)

Updates documentation with progress on Phase 1/2 optimizations,
  justfile usage, development workflow and command examples.

  Adds .claude/ to .gitignore to exclude user-specific Claude AI
  settings.

  Adds "Makefiles" to cspell.json to prevent spell check on
  Makefiles.

### Maintenance

- Increase profiling to 6 hour max runtim [skip ci] [`52c7dfd`](https://github.com/acgetchell/delaunay/commit/52c7dfd51d1a5a1d5cd69bbb424ca3c947a68d46)
- Bump the dependencies group with 4 updates [#88](https://github.com/acgetchell/delaunay/pull/88)
  [`68caee6`](https://github.com/acgetchell/delaunay/commit/68caee6276094e9cb106a7edc43e5ce2a62c9aef)

Bumps the dependencies group with 4 updates: [anyhow](https://github.com/dtolnay/anyhow) , [clap](https://github.com/clap-rs/clap) ,
[nalgebra](https://github.com/dimforge/nalgebra) and [serde](https://github.com/serde-rs/serde) .

  Updates `anyhow` from 1.0.99 to 1.0.100

- [Release notes](https://github.com/dtolnay/anyhow/releases)
- [Commits](https://github.com/dtolnay/anyhow/compare/1.0.99...1.0.100)

  Updates `clap` from 4.5.47 to 4.5.48

- [Release notes](https://github.com/clap-rs/clap/releases)
- [Changelog](https://github.com/clap-rs/clap/blob/master/CHANGELOG.md)
- [Commits](https://github.com/clap-rs/clap/compare/clap_complete-v4.5.47...clap_complete-v4.5.48)

  Updates `nalgebra` from 0.34.0 to 0.34.1

- [Changelog](https://github.com/dimforge/nalgebra/blob/main/CHANGELOG.md)
- [Commits](https://github.com/dimforge/nalgebra/commits/v0.34.1)

  Updates `serde` from 1.0.223 to 1.0.226

- [Release notes](https://github.com/serde-rs/serde/releases)
- [Commits](https://github.com/serde-rs/serde/compare/v1.0.223...v1.0.226)

  ---
  updated-dependencies:

- dependency-name: anyhow
    dependency-version: 1.0.100
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: clap
    dependency-version: 4.5.48
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: nalgebra
    dependency-version: 0.34.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: serde
    dependency-version: 1.0.226
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Release v0.5.0 [#91](https://github.com/acgetchell/delaunay/pull/91)
  [`81f7b88`](https://github.com/acgetchell/delaunay/commit/81f7b889d6ed0d78452505716c7e07ed4bfab514)

- Bump version to v0.5.0
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.5.0

## [0.4.4] - 2025-09-20

### Merged Pull Requests

- Release v0.4.4 [#87](https://github.com/acgetchell/delaunay/pull/87)
- Refactor/phase 2 optimizations [#86](https://github.com/acgetchell/delaunay/pull/86)
- Optimizes internal collections with key-based access [#83](https://github.com/acgetchell/delaunay/pull/83)
- Bump actions-rust-lang/setup-rust-toolchain [#82](https://github.com/acgetchell/delaunay/pull/82)
- Bump the dependencies group with 2 updates [#81](https://github.com/acgetchell/delaunay/pull/81)
- Bump astral-sh/setup-uv from 6.6.1 to 6.7.0 [#80](https://github.com/acgetchell/delaunay/pull/80)
- Bump codecov/codecov-action from 5.5.0 to 5.5.1 [#79](https://github.com/acgetchell/delaunay/pull/79)
- Bump actions/checkout from 4 to 5 [#78](https://github.com/acgetchell/delaunay/pull/78)
- Feature/migrate uuids to slotmap [#77](https://github.com/acgetchell/delaunay/pull/77)
- Fix/benchmarks [#76](https://github.com/acgetchell/delaunay/pull/76)

### Changed

- Reduce verbosity of memory scaling benchmarks [`82c1789`](https://github.com/acgetchell/delaunay/commit/82c178901962fb2fed65bb71cb76671a054aa1bb)

Reduces benchmark output verbosity by introducing a quiet mode controlled by the `CRITERION_QUIET_MODE` environment variable.
  This helps to keep the output clean, especially in automated benchmark runs. Also increases sampling size for 5D benchmarks to the minimum criterion sampling
  size to avoid errors.
  Additionally, fixes a typo in `cspell.json` and `CHANGELOG.md`.

- Feature/migrate uuids to slotmap [#77](https://github.com/acgetchell/delaunay/pull/77)
  [`34ae023`](https://github.com/acgetchell/delaunay/commit/34ae023663edc245139afb20339c27aed3295e1b)
- Optimizes internal collections with key-based access [#83](https://github.com/acgetchell/delaunay/pull/83)
  [`b9a9491`](https://github.com/acgetchell/delaunay/commit/b9a9491a5e86def13697753d4389c511a555ba82)

- Changed: Optimizes internal collections with key-based access

  Migrates internal collections to use direct key-based access instead of UUID lookups. This change enhances performance by eliminating UUID-to-key mapping
  overhead in internal algorithms,
  improving memory efficiency and cache locality. (Internal change)

  - Fixed: Propagates facet errors and avoids UUID lookups

  Fixes a potential issue where facet errors were not propagated during boundary facet retrieval, ensuring errors are correctly handled.

  Improves performance by avoiding redundant UUID→Key lookups in internal data structures, especially within performance-critical algorithms related to cell and
  vertex management. This change
  optimizes data access patterns for `SlotMap` integration and enhances memory efficiency.

  - Fixed: Improves boundary facet retrieval robustness

  Addresses potential errors during boundary facet retrieval by hardening checks for cell and facet existence. This prevents out-of-bounds access and ensures
  that missing cells or vertices do not cause unexpected failures during triangulation processing.
  Also, simplifies logic for single-cell facets.

- Improves Bowyer-Watson triangulation robustness [`f9f2a12`](https://github.com/acgetchell/delaunay/commit/f9f2a12ede4b22e2c7745eb635aed1fd07fe4a77)
- Refactor/phase 2 optimizations [#86](https://github.com/acgetchell/delaunay/pull/86)
  [`f592dd8`](https://github.com/acgetchell/delaunay/commit/f592dd8df54d7c98cfcfc583f62d119f6a476544)

### Fixed

- Fix/benchmarks [#76](https://github.com/acgetchell/delaunay/pull/76)
  [`a090e1e`](https://github.com/acgetchell/delaunay/commit/a090e1e8d9be1140da2e6f5139e3901c06e98f78)

### Maintenance

- Bump actions/checkout from 4 to 5 [#78](https://github.com/acgetchell/delaunay/pull/78)
  [`b847b27`](https://github.com/acgetchell/delaunay/commit/b847b27ff42725438ab8061f18d751bafa2057ff)

Bumps [actions/checkout](https://github.com/actions/checkout) from 4 to 5.

- [Release notes](https://github.com/actions/checkout/releases)
- [Commits](https://github.com/actions/checkout/compare/v4...v5)

  ---
  updated-dependencies:

- dependency-name: actions/checkout
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump codecov/codecov-action from 5.5.0 to 5.5.1 [#79](https://github.com/acgetchell/delaunay/pull/79)
  [`c3b4339`](https://github.com/acgetchell/delaunay/commit/c3b433921f45ee85b6361b71c758e7949f4b9e5e)

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.0 to 5.5.1.

- [Release notes](https://github.com/codecov/codecov-action/releases)
- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/codecov/codecov-action/compare/fdcc8476540edceab3de004e990f80d881c6cc00...5a1091511ad55cbe89839c7260b706298ca349f7)

  ---
  updated-dependencies:

- dependency-name: codecov/codecov-action
    dependency-version: 5.5.1
    dependency-type: direct:production
    update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 6.6.1 to 6.7.0 [#80](https://github.com/acgetchell/delaunay/pull/80)
  [`709bd4d`](https://github.com/acgetchell/delaunay/commit/709bd4d88797cc86db6985af0c522cbc427a32bd)

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 6.6.1 to 6.7.0.

- [Release notes](https://github.com/astral-sh/setup-uv/releases)
- [Commits](https://github.com/astral-sh/setup-uv/compare/557e51de59eb14aaaba2ed9621916900a91d50c6...b75a909f75acd358c2196fb9a5f1299a9a8868a4)

  ---
  updated-dependencies:

- dependency-name: astral-sh/setup-uv
    dependency-version: 6.7.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump actions-rust-lang/setup-rust-toolchain [#82](https://github.com/acgetchell/delaunay/pull/82)
  [`3329d36`](https://github.com/acgetchell/delaunay/commit/3329d3656864f4e2ea2c00730383dd8bf8f23010)

Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.14.1 to 1.15.0.

- [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
- [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
- [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/ac90e63697ac2784f4ecfe2964e1a285c304003a...2fcdc490d667999e01ddbbf0f2823181beef6b39)

  ---
  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
    dependency-version: 1.15.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
  ...

- Bump the dependencies group with 2 updates [#81](https://github.com/acgetchell/delaunay/pull/81)
  [`d259a73`](https://github.com/acgetchell/delaunay/commit/d259a7304e5e395af07d819e32f189b06b5e3541)

Bumps the dependencies group with 2 updates: [serde](https://github.com/serde-rs/serde) and [serde_json](https://github.com/serde-rs/json).

  Updates `serde` from 1.0.219 to 1.0.223

- [Release notes](https://github.com/serde-rs/serde/releases)
- [Commits](https://github.com/serde-rs/serde/compare/v1.0.219...v1.0.223)

  Updates `serde_json` from 1.0.143 to 1.0.145

- [Release notes](https://github.com/serde-rs/json/releases)
- [Commits](https://github.com/serde-rs/json/compare/v1.0.143...v1.0.145)

  ---
  updated-dependencies:

- dependency-name: serde
    dependency-version: 1.0.223
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: serde_json
    dependency-version: 1.0.145
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Release v0.4.4 [#87](https://github.com/acgetchell/delaunay/pull/87)
  [`e71c132`](https://github.com/acgetchell/delaunay/commit/e71c1325d082e62931f40fbb7821e25f63abd608)

- Bump version to v0.4.4
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.4.4

## [0.4.3] - 2025-09-12

### Merged Pull Requests

- Release/v0.4.3 [#75](https://github.com/acgetchell/delaunay/pull/75)
- Bump actions/github-script from 7.0.1 to 8.0.0 [#70](https://github.com/acgetchell/delaunay/pull/70)
- Bump codecov/codecov-action [#69](https://github.com/acgetchell/delaunay/pull/69)
- Memory profiling system and performance optimizations (v0.4.3) [#68](https://github.com/acgetchell/delaunay/pull/68)

### Added

- Memory profiling system and performance optimizations (v0.4.3) [#68](https://github.com/acgetchell/delaunay/pull/68)
  [`2eeb6db`](https://github.com/acgetchell/delaunay/commit/2eeb6db7d2f595834c1a0db92a694a3397bc56aa)

- Changed: Refactors benchmark workflow using Python utils (internal)

  Streamlines benchmark workflow by replacing complex bash scripts with Python utility functions. This simplifies maintenance, improves code readability, and
  reduces the risk of errors.

  The changes encompass baseline generation, comparison, commit extraction, skip determination and result display within GitHub Actions workflows.

  - Added: Comprehensive profiling benchmarks and memory stress tests

  This commit introduces a comprehensive profiling suite for in-depth performance analysis and a separate memory stress test job.

  The profiling suite includes:

  - Large-scale triangulation performance analysis (10³-10⁶ points)
  - Multiple point distributions (random, grid, Poisson disk)
  - Memory allocation tracking (with `count-allocations` feature)
  - Query latency analysis
  - Multi-dimensional scaling (2D-5D)
  - Algorithmic bottleneck identification

  It's integrated into GitHub Actions with scheduled runs and manual triggering, along with uploading profiling results and baselines.

  The memory stress test runs independently to exercise allocation APIs and memory scaling under load.

  Also, ignores the "benches/**" directory in codecov, and adds the profiling suite to the README.md.

  - feat: Add memory profiling system and performance optimizations for v0.4.3

  - Add allocation counter infrastructure with count-allocations feature flag
  - Implement memory tracking for triangulation and convex hull operations
  - Add profiling benchmarks in GitHub Actions workflow
  - Optimize collections with FxHashMap/FxHashSet for better performance
  - Add domain-specific collection types and small buffer optimizations
  - Clean up examples by removing test functions (convert to pure demonstrations)
  - Update WARP.md guidelines and project documentation
  - Add comprehensive memory analysis examples across dimensions (2D-5D)

  - Changed: Improves benchmark workflow and memory profiling

  Refactors benchmark workflows for better performance profiling, improves memory allocation tracking, and enhances numerical stability.

  Updates profiling benchmarks to track actual point counts and adds safety cap to prevent out-of-memory errors.

  Switches `Cell` neighbor storage to `Option` for
  correct positional semantics.
  Changes ConvexHull caching to `ArcSwapOption`.
  These are internal changes to improve performance
  and robustness of the library.

  - Changed: Enhance profiling benchmarks for performance analysis

  Enhances profiling benchmarks with improved memory allocation tracking, optimized query benchmarks, complete dimensional coverage (2D-5D), and environment
  variable control for faster
  iteration. It provides more comprehensive performance analysis for optimization work. This also introduces error handling for grid generation in benchmarks.

  - Changed: Improves benchmark utils with timeout and error handling

  Enhances benchmark utilities to include timeout handling for cargo bench commands, preventing indefinite execution. Introduces `ProjectRootNotFoundError` for
  clearer error reporting when `Cargo.toml` is missing. Exports `BENCHMARK_REGRESSION_DETECTED` to GITHUB_ENV for better CI integration. Protects against
  division by zero in throughput calculations.

  - Changed: Enhances benchmark suite for detailed analysis

  Extends benchmark suite to include 5D circumsphere containment tests and edge cases across all dimensions, improving the profiling suite for more granular
  performance analysis.

  Adds memory profiling with 95th percentile stats and
  optimizes query benchmarks with precomputed simplices, while also introducing env vars for CI tuning.
  Refs: None

  - Improves benchmark suite with new tests and refactoring

  Adds new benchmarks for triangulation creation and circumsphere containment, providing more comprehensive performance analysis.
  Refactors existing benchmark code to reduce duplication and improve maintainability. Updates code coverage configuration and adds mypy
  cache to gitignore.

### Changed

- Release/v0.4.3 [#75](https://github.com/acgetchell/delaunay/pull/75)
  [`1062551`](https://github.com/acgetchell/delaunay/commit/1062551a9152a53e938ddbf94c4152ff6ae4254d)

### Maintenance

- Bump codecov/codecov-action [#69](https://github.com/acgetchell/delaunay/pull/69)
  [`57b9c3c`](https://github.com/acgetchell/delaunay/commit/57b9c3c7e9294c26ae1601b83c4627fe499952e4)

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 8f6ec407a34d5ec2b33d7d8f7f50279493b5efb4 to
fdcc8476540edceab3de004e990f80d881c6cc00.

- [Release notes](https://github.com/codecov/codecov-action/releases)
- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/codecov/codecov-action/compare/8f6ec407a34d5ec2b33d7d8f7f50279493b5efb4...fdcc8476540edceab3de004e990f80d881c6cc00)

  ---
  updated-dependencies:

- dependency-name: codecov/codecov-action
    dependency-version: fdcc8476540edceab3de004e990f80d881c6cc00
    dependency-type: direct:production
  ...

- Bump actions/github-script from 7.0.1 to 8.0.0 [#70](https://github.com/acgetchell/delaunay/pull/70)
  [`18ed4eb`](https://github.com/acgetchell/delaunay/commit/18ed4ebdf819b71135b68857edb8457021e44f81)

Bumps [actions/github-script](https://github.com/actions/github-script) from 7.0.1 to 8.0.0.

- [Release notes](https://github.com/actions/github-script/releases)
- [Commits](https://github.com/actions/github-script/compare/60a0d83039c74a4aee543508d2ffcb1c3799cdea...ed597411d8f924073f98dfc5c65a23a2325f34cd)

  ---
  updated-dependencies:

- dependency-name: actions/github-script
    dependency-version: 8.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

## [0.4.2] - 2025-09-04

### Merged Pull Requests

- Release v0.4.2 [#66](https://github.com/acgetchell/delaunay/pull/66)
- Refactor/benchmarks [#65](https://github.com/acgetchell/delaunay/pull/65)
- Docs/release v0.4.2 [#60](https://github.com/acgetchell/delaunay/pull/60)
- Test/benchmarking [#59](https://github.com/acgetchell/delaunay/pull/59)
- Fix/squashed commit body parsing [#58](https://github.com/acgetchell/delaunay/pull/58)

### Changed

- Test/benchmarking [#59](https://github.com/acgetchell/delaunay/pull/59)
  [`fda43d7`](https://github.com/acgetchell/delaunay/commit/fda43d76af0cf35061d2e70b32f1cfb24d4696b1)
- Docs/release v0.4.2 [#60](https://github.com/acgetchell/delaunay/pull/60)
  [`f10aba3`](https://github.com/acgetchell/delaunay/commit/f10aba34cb9e3fe9418aa549f392aea98eb0d13e)
- Refactor/benchmarks [#65](https://github.com/acgetchell/delaunay/pull/65)
  [`d2e2f86`](https://github.com/acgetchell/delaunay/commit/d2e2f860bd685bbbe8303589fe0687ebedc71229)

### Fixed

- Fix/squashed commit body parsing [#58](https://github.com/acgetchell/delaunay/pull/58)
  [`ea87521`](https://github.com/acgetchell/delaunay/commit/ea87521bc667d66ac9b23f07d1339087ce9f8d6f)

### Maintenance

- Release v0.4.2 [#66](https://github.com/acgetchell/delaunay/pull/66)
  [`272d5b1`](https://github.com/acgetchell/delaunay/commit/272d5b1faa028b3bdeef479eca0a5de1a8252404)

- Bump version to v0.4.2
  - Update changelog with latest changes
  - Update documentation for release
  - Update Cargo.lock

## [0.4.1] - 2025-08-30

### Merged Pull Requests

- Release/0.4.1 fixed [#57](https://github.com/acgetchell/delaunay/pull/57)
- Release v0.4.1 [#56](https://github.com/acgetchell/delaunay/pull/56)
- Docs/update documentation citations benchmarks [#55](https://github.com/acgetchell/delaunay/pull/55)
- Refactors coordinate traits for improved geometry [#54](https://github.com/acgetchell/delaunay/pull/54)

### Added

- Adds DOI badge to README [`a5f0f51`](https://github.com/acgetchell/delaunay/commit/a5f0f51dae0174cdf541f7e556e13441caa2a850)

### Changed

- Refactors coordinate traits for improved geometry [#54](https://github.com/acgetchell/delaunay/pull/54)
  [`c1fe7ee`](https://github.com/acgetchell/delaunay/commit/c1fe7eebfb63cf815740e3b7a82161bcaa251f0f)
- Docs/update documentation citations benchmarks [#55](https://github.com/acgetchell/delaunay/pull/55)
  [`5fb31b1`](https://github.com/acgetchell/delaunay/commit/5fb31b129128e260057d04833b0cf6a97e5fca72)
- Release/0.4.1 fixed [#57](https://github.com/acgetchell/delaunay/pull/57)
  [`d1589c0`](https://github.com/acgetchell/delaunay/commit/d1589c0bf1fb0d7448b8a1f2ac302eaf01ea6e55)
- Update release date for v0.4.1 in CHANGELOG [`13b4df2`](https://github.com/acgetchell/delaunay/commit/13b4df257d102b8ab3608da85434c46db978ddd8)

### Maintenance

- Release v0.4.1 [#56](https://github.com/acgetchell/delaunay/pull/56)
  [`097a1e5`](https://github.com/acgetchell/delaunay/commit/097a1e59349d5be5834be4f6c888b3bf2061a2a6)

- Bump version to v0.4.1
  - Update changelog with latest changes
  - Update documentation for release

## [0.4.0] - 2025-08-23

### Merged Pull Requests

- Release v0.4.0 [#53](https://github.com/acgetchell/delaunay/pull/53)
- Bump actions/checkout from 4 to 5 [#51](https://github.com/acgetchell/delaunay/pull/51)
- Feature/convex hull [#50](https://github.com/acgetchell/delaunay/pull/50)

### Changed

- Feature/convex hull [#50](https://github.com/acgetchell/delaunay/pull/50)
  [`2e31533`](https://github.com/acgetchell/delaunay/commit/2e31533087466293aa99a2d39561a42ec17a76f0)

### Maintenance

- Bump actions/checkout from 4 to 5 [#51](https://github.com/acgetchell/delaunay/pull/51)
  [`c4709e2`](https://github.com/acgetchell/delaunay/commit/c4709e2e81ab933e4c8e2981c5a40b44e0069f08)

Bumps [actions/checkout](https://github.com/actions/checkout) from 4 to 5.

- [Release notes](https://github.com/actions/checkout/releases)
- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
- [Commits](https://github.com/actions/checkout/compare/v4...v5)

  ---
  updated-dependencies:

- dependency-name: actions/checkout
    dependency-version: '5'
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Release v0.4.0 [#53](https://github.com/acgetchell/delaunay/pull/53)
  [`18553aa`](https://github.com/acgetchell/delaunay/commit/18553aa9dd4c11d703cbe02163c548c3600ea50a)

- Bump version to v0.4.0
  - Update changelog with latest changes
  - Update documentation for release

## [0.3.5] - 2025-08-16

### Merged Pull Requests

- Updates delaunay crate to v0.3.5 [#45](https://github.com/acgetchell/delaunay/pull/45)
- Update documentation [#44](https://github.com/acgetchell/delaunay/pull/44)

### Changed

- Updates delaunay crate to v0.3.5 [#45](https://github.com/acgetchell/delaunay/pull/45)
  [`73b5656`](https://github.com/acgetchell/delaunay/commit/73b5656feb3a91699ed5fdb6773104e2308d36b8)

### Documentation

- Update CHANGELOG for v0.3.4 [`458d378`](https://github.com/acgetchell/delaunay/commit/458d37810dd38e4d99cd7db2714e47e3c5a74242)
- Update documentation [#44](https://github.com/acgetchell/delaunay/pull/44)
  [`5bcf318`](https://github.com/acgetchell/delaunay/commit/5bcf31879aa8eb1bfee4c5130b21d8fcfe4dd7a2)

- docs: Update documentation

  Update workflow to generate CHANGELOG.md automatically using commit dates rather than tag dates.

  Fix documentation on bowyer-watson and varous other places.

  Add more examples for use in CONTRIBUTING.md.

  - docs: Refine documentation

  Refactors benchmark setup to improve clarity and reduce code duplication by introducing helper functions for clearing neighbors.

  Updates CI workflow to include tests and adds `ci` to the auto-changelog ignore pattern.

  - Improves changelog generation and formatting

  Refactors the changelog generation script to enhance commit message formatting, and categorize commit messages by type.

  Updates the changelog template to improve presentation, and ensure consistent date formatting and readability.

  Removes the commit message template and updates contributing guidelines to reflect conventional commits usage.

  - Refactors changelog generation for clarity

  Simplifies the changelog generation script by ensuring the "Changes" header is printed only once and streamlining the commit processing logic.

  This improves the script's readability and maintainability by removing redundant checks and simplifying the flow of execution.

### Fixed

- Fix too long keyword [`0c8bc32`](https://github.com/acgetchell/delaunay/commit/0c8bc32bc6ef8996490d1301c6bc7d3247f250ac)
- Fix invalid category slug on crates.io [`cafc7d7`](https://github.com/acgetchell/delaunay/commit/cafc7d7175230642b89aaf941a959be46174eb43)

### Maintenance

- Chore(release): release v0.3.5 [`1f83654`](https://github.com/acgetchell/delaunay/commit/1f83654bfd93e7a8f3716e28b022991c953e8f4e)

## [0.3.4] - 2025-08-15

### Merged Pull Requests

- Improves triangulation performance and validation [#43](https://github.com/acgetchell/delaunay/pull/43)
- Rename delaunay_core to core [#42](https://github.com/acgetchell/delaunay/pull/42)
- Rename project from d-delaunay to delaunay [#41](https://github.com/acgetchell/delaunay/pull/41)
- Update documentation, scripts, benchmarks [#39](https://github.com/acgetchell/delaunay/pull/39)

### Changed

- Update documentation, scripts, benchmarks [#39](https://github.com/acgetchell/delaunay/pull/39)
  [`9d934f7`](https://github.com/acgetchell/delaunay/commit/9d934f7e1da74a14badd1889905eac759cc97651)
- Rename project from d-delaunay to delaunay [#41](https://github.com/acgetchell/delaunay/pull/41)
  [`d9daf45`](https://github.com/acgetchell/delaunay/commit/d9daf45eb39cf96e1d92ec94f213e7f7fc78dd31)
- Rename delaunay_core to core [#42](https://github.com/acgetchell/delaunay/pull/42)
  [`ab9cc99`](https://github.com/acgetchell/delaunay/commit/ab9cc99dea653217e1e2ff99ea71a2b101f15ad9)
- Improves triangulation performance and validation [#43](https://github.com/acgetchell/delaunay/pull/43)
  [`635aee3`](https://github.com/acgetchell/delaunay/commit/635aee35b00b39d9433b6acce75627ec1bff516b)

### Maintenance

- Bump version to v0.3.4 [`9d5feec`](https://github.com/acgetchell/delaunay/commit/9d5feec9d1ecb6603d3860dfde72ce9c58ac7ec8)

## [0.3.3] - 2025-08-05

### Merged Pull Requests

- Refactor/triangulation data structure 2 [#37](https://github.com/acgetchell/delaunay/pull/37)
- Refactor/triangulation data structure [#36](https://github.com/acgetchell/delaunay/pull/36)

### Changed

- Refactor/triangulation data structure [#36](https://github.com/acgetchell/delaunay/pull/36)
  [`8c7d05f`](https://github.com/acgetchell/delaunay/commit/8c7d05fc2c44a0a980f11ea218407920d082880d)
- Refactor/triangulation data structure 2 [#37](https://github.com/acgetchell/delaunay/pull/37)
  [`a2acfec`](https://github.com/acgetchell/delaunay/commit/a2acfec53c7a3c6210cd44519d8810c137cf89ee)

## [0.3.2] - 2025-07-29

### Merged Pull Requests

- Refactors vertex and updates example [#35](https://github.com/acgetchell/delaunay/pull/35)

### Changed

- Refactors vertex and updates example [#35](https://github.com/acgetchell/delaunay/pull/35)
  [`be56d57`](https://github.com/acgetchell/delaunay/commit/be56d577eced6a6d1fe8b0f70e493fd430463962)

## [0.3.1] - 2025-07-26

### Merged Pull Requests

- Improves CodeQL analysis for Rust projects [#34](https://github.com/acgetchell/delaunay/pull/34)

### Changed

- Improves CodeQL analysis for Rust projects [#34](https://github.com/acgetchell/delaunay/pull/34)
  [`bdc206f`](https://github.com/acgetchell/delaunay/commit/bdc206fc322cb963bf6863eee0afa3188b53839f)

### Fixed

- Fix security issues [`e5e8e21`](https://github.com/acgetchell/delaunay/commit/e5e8e21121800bad0091e6e639b2ac629c178a6b)

## [0.3.0] - 2025-07-25

### Merged Pull Requests

- Bump the dependencies group with 2 updates [#33](https://github.com/acgetchell/delaunay/pull/33)
- Introduces `Coordinate` trait for coordinate abstraction [#32](https://github.com/acgetchell/delaunay/pull/32)
- Bump peroxide in the dependencies group [#31](https://github.com/acgetchell/delaunay/pull/31)
- Refactor/predicates [#30](https://github.com/acgetchell/delaunay/pull/30)
- Bump codacy/codacy-analysis-cli-action from 1.1.0 to 4.4.7 [#29](https://github.com/acgetchell/delaunay/pull/29)
- Bump the dependencies group with 3 updates [#28](https://github.com/acgetchell/delaunay/pull/28)
- Refactor/traits [#27](https://github.com/acgetchell/delaunay/pull/27)
- New/predicates [#26](https://github.com/acgetchell/delaunay/pull/26)
- Refactor/carefully [#25](https://github.com/acgetchell/delaunay/pull/25)
- Bump peroxide in the dependencies group [#23](https://github.com/acgetchell/delaunay/pull/23)
- Bump rand from 0.8.5 to 0.9.1 in the dependencies group [#22](https://github.com/acgetchell/delaunay/pull/22)
- Refactor/tds [#21](https://github.com/acgetchell/delaunay/pull/21)
- Tests/improve coverage and lints [#20](https://github.com/acgetchell/delaunay/pull/20)
- Refactors float trait implementations with macros [#19](https://github.com/acgetchell/delaunay/pull/19)
- Bump the dependencies group with 3 updates [#18](https://github.com/acgetchell/delaunay/pull/18)
- Refactors Point comparisons, Vertex and updates examples [#17](https://github.com/acgetchell/delaunay/pull/17)
- Core/consistency [#15](https://github.com/acgetchell/delaunay/pull/15)
- Fix/core data types [#13](https://github.com/acgetchell/delaunay/pull/13)
- Bump crossbeam-channel from 0.5.13 to 0.5.15 [#9](https://github.com/acgetchell/delaunay/pull/9)
- Fix/bowyer watson [#6](https://github.com/acgetchell/delaunay/pull/6)

### Added

- Adds dependabot configuration file [`110048c`](https://github.com/acgetchell/delaunay/commit/110048c6ee4d5e115a8e441e19a44f0a50dcab93)
- Adds comprehensive tests for triangulation data structure [`a3934d4`](https://github.com/acgetchell/delaunay/commit/a3934d4a2c287db18863690b97a6b2beb7afebfa)
- Adds Eq trait implementation for Point struct [`497d5f5`](https://github.com/acgetchell/delaunay/commit/497d5f5d0178dc71022999b5ff055fe334a2f4c2)
- Adds comprehensive tests for Vertex struct [`210063c`](https://github.com/acgetchell/delaunay/commit/210063c7fca5d643400ced84fc262d3dddeb5f05)
- Adds comprehensive facet tests [`8f8e0ae`](https://github.com/acgetchell/delaunay/commit/8f8e0aee2f85e8987ecec186bcfeff479a93e998)
- Adds 5D Bowyer-Watson triangulation test [`7555648`](https://github.com/acgetchell/delaunay/commit/7555648200c7c4664d76fe8fe26b2441c47c119d)
- Adds triangulation data structure validation. [`8e34949`](https://github.com/acgetchell/delaunay/commit/8e34949fed5237afbd2dea84f886c7cb55e0a4ba)

### Changed

- Updates upload-artifact action to v4 [`abed62d`](https://github.com/acgetchell/delaunay/commit/abed62ddb4c6cf268f3084a854f2fd4defcd39cd)
- Updates dependencies and caches Rust dependencies [`e1e99f4`](https://github.com/acgetchell/delaunay/commit/e1e99f4a3acda274bacb914006ab62c6724bd2a7)
- Updates Codecov action and adds words to cspell [`2593c7e`](https://github.com/acgetchell/delaunay/commit/2593c7e2e3203a278bbe096b1acf18bbd0d58cd8)
- Improves CI workflow and code analysis [`f326699`](https://github.com/acgetchell/delaunay/commit/f326699f9b1774195a4a26b6e4b8d62a194fb42d)
- Updates Codecov action to v5 [`c82f152`](https://github.com/acgetchell/delaunay/commit/c82f1527df43b79cd84504590704b74eb65aeb5c)
- Improves CI workflow caching [`e63557e`](https://github.com/acgetchell/delaunay/commit/e63557e4388e2a6ef53fc7a2d0c47f1aa1a473c7)
- Refactors Cell struct for clarity and efficiency [`bc44769`](https://github.com/acgetchell/delaunay/commit/bc44769961f01b73d87882a14568c9a02109c530)
- Improves supercell coordinate calculation [`bbbaa6f`](https://github.com/acgetchell/delaunay/commit/bbbaa6fa50ac78f1495f3546142e28c944e31ea2)
- Improves Delaunay triangulation robustness [`ad716dd`](https://github.com/acgetchell/delaunay/commit/ad716ddc0cb10153fcb3d0d41d696eb2b0375552)
- Copies vertex instead of cloning [`ad0c3b7`](https://github.com/acgetchell/delaunay/commit/ad0c3b7f74efd7d787553e97010cba7dfce364ee)
- Revert "Copies vertex instead of cloning" [`09d35d8`](https://github.com/acgetchell/delaunay/commit/09d35d8c927243ff89854b32c69f721fd9430006)
- Improves simplex creation and adds 4D triangulation test [`40b8209`](https://github.com/acgetchell/delaunay/commit/40b8209e328bac9bae05f644a5bf1947777379a9)
- Marks Delaunay triangulation as implemented [`41a59cd`](https://github.com/acgetchell/delaunay/commit/41a59cd2a068e750a22989b0ff47bc7de992b672)
- Improves Delaunay triangulation robustness [`6fe769b`](https://github.com/acgetchell/delaunay/commit/6fe769b1bf2823cb4181317c1255e8634ad046c8)
- Improves triangulation for small vertex sets [`9420c14`](https://github.com/acgetchell/delaunay/commit/9420c14645d14aec3c6818428c57ece72d113792)
- Improves Delaunay triangulation core logic [`331e856`](https://github.com/acgetchell/delaunay/commit/331e8561db9a7987b366ca42b77378512f804778)
- Core/consistency [#15](https://github.com/acgetchell/delaunay/pull/15)
  [`928f42d`](https://github.com/acgetchell/delaunay/commit/928f42da2e16697423fdbc6385c0441a67e93d5a)
- Refactors Point comparisons, Vertex and updates examples [#17](https://github.com/acgetchell/delaunay/pull/17)
  [`fe44e1a`](https://github.com/acgetchell/delaunay/commit/fe44e1aeadeaef081a6ac692f368f026654b4fd6)
- Refactors float trait implementations with macros [#19](https://github.com/acgetchell/delaunay/pull/19)
  [`1712043`](https://github.com/acgetchell/delaunay/commit/171204332d4eae4481497e948b11751a11473c3c)
- Tests/improve coverage and lints [#20](https://github.com/acgetchell/delaunay/pull/20)
  [`cd13fe4`](https://github.com/acgetchell/delaunay/commit/cd13fe4addee9bb45b97fac8762cfe36418ecb10)
- Refactor/tds [#21](https://github.com/acgetchell/delaunay/pull/21)
  [`ec9a9b2`](https://github.com/acgetchell/delaunay/commit/ec9a9b28afefa923e55f26340d7011f249cd32d7)
- Refactor/carefully [#25](https://github.com/acgetchell/delaunay/pull/25)
  [`0825508`](https://github.com/acgetchell/delaunay/commit/08255083d2f907986aadce962295a5e80ed34c79)
- New/predicates [#26](https://github.com/acgetchell/delaunay/pull/26)
  [`a50c978`](https://github.com/acgetchell/delaunay/commit/a50c978b9b6d9c67ac03ab3593eb8b9ea94102bb)
- Refactor/traits [#27](https://github.com/acgetchell/delaunay/pull/27)
  [`3532ff6`](https://github.com/acgetchell/delaunay/commit/3532ff67ff4fbc91bf55b94d8fb2b8603c8f0019)
- Refactor/predicates [#30](https://github.com/acgetchell/delaunay/pull/30)
  [`2a57833`](https://github.com/acgetchell/delaunay/commit/2a5783304d5c6ce9a76f34b318c2acaaad028fe6)
- Create codeql.yml [`dbba414`](https://github.com/acgetchell/delaunay/commit/dbba414f1eaa8bcc5c4e743c84b1c9c206f664d8)
- Introduces `Coordinate` trait for coordinate abstraction [#32](https://github.com/acgetchell/delaunay/pull/32)
  [`798e5dc`](https://github.com/acgetchell/delaunay/commit/798e5dcb170eb538617e0fb4219c1813ddf5012f)
- Updates to version 0.3.0 with performance improvements [`0be386d`](https://github.com/acgetchell/delaunay/commit/0be386d10b9ef6edd540a3b6f26698160cb9ba71)

### Fixed

- Fix/bowyer watson [#6](https://github.com/acgetchell/delaunay/pull/6)
  [`bd2a079`](https://github.com/acgetchell/delaunay/commit/bd2a0798d7bc8be981159a453c5a9ebdd8b0352a)
- Fix/core data types [#13](https://github.com/acgetchell/delaunay/pull/13)
  [`2826ccf`](https://github.com/acgetchell/delaunay/commit/2826ccf1b14677f4a2bddd2ef26b6aaae2c75e1c)
- Fix deprecated functions [`408d8bd`](https://github.com/acgetchell/delaunay/commit/408d8bd84aadef8bfd81a9256b220a43bf690842)

### Maintenance

- Bump crossbeam-channel from 0.5.13 to 0.5.15 [#9](https://github.com/acgetchell/delaunay/pull/9)
  [`f3ae496`](https://github.com/acgetchell/delaunay/commit/f3ae496c4462d37c641cbcf3987e4e5455c19d75)

Bumps [crossbeam-channel](https://github.com/crossbeam-rs/crossbeam) from 0.5.13 to 0.5.15.

- [Release notes](https://github.com/crossbeam-rs/crossbeam/releases)
- [Changelog](https://github.com/crossbeam-rs/crossbeam/blob/master/CHANGELOG.md)
- [Commits](https://github.com/crossbeam-rs/crossbeam/compare/crossbeam-channel-0.5.13...crossbeam-channel-0.5.15)

  ---
  updated-dependencies:

- dependency-name: crossbeam-channel
    dependency-version: 0.5.15
    dependency-type: indirect
  ...

- Bump the dependencies group with 3 updates [#18](https://github.com/acgetchell/delaunay/pull/18)
  [`8e2b698`](https://github.com/acgetchell/delaunay/commit/8e2b698d682e26e2f561c676039ec049bc51df9b)

---
  updated-dependencies:

- dependency-name: ordered-float
    dependency-version: 5.0.0
    dependency-type: direct:production
    update-type: version-update:semver-major
    dependency-group: dependencies

- dependency-name: peroxide
    dependency-version: 0.39.7
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies

- dependency-name: thiserror
    dependency-version: 2.0.12
    dependency-type: direct:production
    update-type: version-update:semver-major
    dependency-group: dependencies
  ...

- Bump rand from 0.8.5 to 0.9.1 in the dependencies group [#22](https://github.com/acgetchell/delaunay/pull/22)
  [`cb5938c`](https://github.com/acgetchell/delaunay/commit/cb5938ce038ed3f86d0915f461414198c6434238)

Bumps the dependencies group with 1 update: [rand](https://github.com/rust-random/rand).

  Updates `rand` from 0.8.5 to 0.9.1

- [Release notes](https://github.com/rust-random/rand/releases)
- [Changelog](https://github.com/rust-random/rand/blob/master/CHANGELOG.md)
- [Commits](https://github.com/rust-random/rand/compare/0.8.5...rand_core-0.9.1)

  ---
  updated-dependencies:

- dependency-name: rand
    dependency-version: 0.9.1
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies
  ...

- Bump peroxide in the dependencies group [#23](https://github.com/acgetchell/delaunay/pull/23)
  [`7e66937`](https://github.com/acgetchell/delaunay/commit/7e66937fa69ee66e410fb25ef3a5afb9432cae0e)

Bumps the dependencies group with 1 update: [peroxide](https://github.com/Axect/Peroxide).

  Updates `peroxide` from 0.39.8 to 0.39.10

- [Release notes](https://github.com/Axect/Peroxide/releases)
- [Changelog](https://github.com/Axect/Peroxide/blob/master/RELEASES.md)
- [Commits](https://github.com/Axect/Peroxide/compare/v0.39.8...v0.39.10)

  ---
  updated-dependencies:

- dependency-name: peroxide
    dependency-version: 0.39.10
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Bump codacy/codacy-analysis-cli-action from 1.1.0 to 4.4.7 [#29](https://github.com/acgetchell/delaunay/pull/29)
  [`a1bc4a6`](https://github.com/acgetchell/delaunay/commit/a1bc4a6cdb12036eff891bb437b7215c1a994fba)

Bumps [codacy/codacy-analysis-cli-action](https://github.com/codacy/codacy-analysis-cli-action) from 1.1.0 to 4.4.7.

- [Release notes](https://github.com/codacy/codacy-analysis-cli-action/releases)
- [Commits](https://github.com/codacy/codacy-analysis-cli-action/compare/d840f886c4bd4edc059706d09c6a1586111c540b...562ee3e92b8e92df8b67e0a5ff8aa8e261919c08)

  ---
  updated-dependencies:

- dependency-name: codacy/codacy-analysis-cli-action
    dependency-version: 4.4.7
    dependency-type: direct:production
    update-type: version-update:semver-major
  ...

- Bump the dependencies group with 3 updates [#28](https://github.com/acgetchell/delaunay/pull/28)
  [`da292c2`](https://github.com/acgetchell/delaunay/commit/da292c210793264b95f9c2118935f61205e0bddd)

Bumps the dependencies group with 3 updates: [rand](https://github.com/rust-random/rand) , [serde_json](https://github.com/serde-rs/json) and
[criterion](https://github.com/bheisler/criterion.rs) .

  Updates `rand` from 0.9.1 to 0.9.2

- [Release notes](https://github.com/rust-random/rand/releases)
- [Changelog](https://github.com/rust-random/rand/blob/master/CHANGELOG.md)
- [Commits](https://github.com/rust-random/rand/compare/rand_core-0.9.1...rand_core-0.9.2)

  Updates `serde_json` from 1.0.140 to 1.0.141

- [Release notes](https://github.com/serde-rs/json/releases)
- [Commits](https://github.com/serde-rs/json/compare/v1.0.140...v1.0.141)

  Updates `criterion` from 0.5.1 to 0.6.0

- [Changelog](https://github.com/bheisler/criterion.rs/blob/master/CHANGELOG.md)
- [Commits](https://github.com/bheisler/criterion.rs/compare/0.5.1...0.6.0)

  ---
  updated-dependencies:

- dependency-name: rand
    dependency-version: 0.9.2
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: serde_json
    dependency-version: 1.0.141
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies

- dependency-name: criterion
    dependency-version: 0.6.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies
  ...

- Bump peroxide in the dependencies group [#31](https://github.com/acgetchell/delaunay/pull/31)
  [`7c9f037`](https://github.com/acgetchell/delaunay/commit/7c9f037eee34acb10de2c0df3149300d0cdc0afc)

---
  updated-dependencies:

- dependency-name: peroxide
    dependency-version: 0.39.11
    dependency-type: direct:production
    update-type: version-update:semver-patch
    dependency-group: dependencies
  ...

- Bump the dependencies group with 2 updates [#33](https://github.com/acgetchell/delaunay/pull/33)
  [`dcfb7f5`](https://github.com/acgetchell/delaunay/commit/dcfb7f548994157c7cab4c200872ce2e6699fbd2)

Bumps the dependencies group with 2 updates: [peroxide](https://github.com/Axect/Peroxide) and [criterion](https://github.com/bheisler/criterion.rs).

  Updates `peroxide` from 0.39.11 to 0.40.0

- [Release notes](https://github.com/Axect/Peroxide/releases)
- [Changelog](https://github.com/Axect/Peroxide/blob/master/RELEASES.md)
- [Commits](https://github.com/Axect/Peroxide/compare/v0.39.11...v0.40.0)

  Updates `criterion` from 0.6.0 to 0.7.0

- [Changelog](https://github.com/bheisler/criterion.rs/blob/master/CHANGELOG.md)
- [Commits](https://github.com/bheisler/criterion.rs/compare/0.6.0...0.7.0)

  ---
  updated-dependencies:

- dependency-name: peroxide
    dependency-version: 0.40.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies

- dependency-name: criterion
    dependency-version: 0.7.0
    dependency-type: direct:production
    update-type: version-update:semver-minor
    dependency-group: dependencies
  ...

### Removed

- Removes CodeRabbit badge from README [ci skip] [`c2da1ce`](https://github.com/acgetchell/delaunay/commit/c2da1ce43123b62680f2df3b7ca655cf3611b19f)

## [0.2.0] - 2024-09-13

### Merged Pull Requests

- Feature/peroxide [#5](https://github.com/acgetchell/delaunay/pull/5)
- Update GitHub actions [#2](https://github.com/acgetchell/delaunay/pull/2)

### Added

- Add Cell and Vertex structs to delaunay_core [`301611a`](https://github.com/acgetchell/delaunay/commit/301611a856f1351c10c2a6a10a96f4e49d06a9c8)

- Added `Cell` struct with `new` method and associated tests
  - Added `Vertex` struct with `new` method and associated tests
  - Created new files for `cell.rs` and `vertex.rs`
  - Updated imports in other files
- Add CODEOWNERS file, audit-check workflow, and rename CI workflow
  [`3ee6c99`](https://github.com/acgetchell/delaunay/commit/3ee6c995da281b94d746c60d53c68a7030da6a91)
- Add Codecov and Clippy [`5b6d76b`](https://github.com/acgetchell/delaunay/commit/5b6d76b8f63d4faf1c5cf453ceb7ebc1750d4a23)
- Add clippy to cspell.json and update README.md with CI and rust-clippy badges
  [`10cc4bd`](https://github.com/acgetchell/delaunay/commit/10cc4bde0bfbd08c19d5a9e1be0682986371f92f)
- Add incident_cell to Vertex and neighbors to Cells [`0b137ab`](https://github.com/acgetchell/delaunay/commit/0b137ab609655655c8b4aae75391e07ad06b14c0)
- Add documentation [`bc3cf2a`](https://github.com/acgetchell/delaunay/commit/bc3cf2ab89aef32954a95d227471e026d91888d0)
- Add license [skip ci] [`24f6d0c`](https://github.com/acgetchell/delaunay/commit/24f6d0ca887ef9b475ebbc96ddba499d7bb6eb28)
- Add dim and contains_vertex functions to Cell [`4169560`](https://github.com/acgetchell/delaunay/commit/4169560ba4aef98d2b69f2b8f2adc3c46438fa0e)

- Added a new function `dim` that calculates the dimension of the cell by subtracting 1 from the number of vertices.
  - Added a new function `contains_vertex` that checks if a given vertex is present in the cell.

  Example usage:

  ```text
  let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
  let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
  let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
  let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
  let cell = Cell::new_with_data(vec![vertex1, vertex2, vertex3, vertex4], "three-one cell").unwrap();
  assert!(cell.contains_vertex(vertex1));
  ```

- Add circumcenter [`c6c5e0d`](https://github.com/acgetchell/delaunay/commit/c6c5e0d811a6b84021536e09163bdd32e7868d57)
- Add circumradius calculation to Cell struct [`e9a0142`](https://github.com/acgetchell/delaunay/commit/e9a014233d6ab9f4e657fa4d7c15a0e7848b58a7)

This commit adds a new method `circumradius` to the `Cell` struct in the `delaunay_core` module. The `circumradius` method calculates the distance from the
circumcenter of the cell to any vertex. It uses the existing `circumcenter` method and the distance formula from the nalgebra library.

  The commit also includes some minor code cleanup, removing unnecessary imports and debug print statements.

  Unit tests have been added for both successful and failed cases of calculating the circumradius.

  Circumradius calculation has been verified with human-readable output during testing.

- Added circumsphere_contains method [`51f40cd`](https://github.com/acgetchell/delaunay/commit/51f40cd38b4b5e560cd3a1a0953cd89262ad74c4)
- Added more tests [`fd7c05d`](https://github.com/acgetchell/delaunay/commit/fd7c05d260780eff3550e5420236f477e2060cbe)
- Add conversion from `coords` array to `[f64; D]` [`4524fb5`](https://github.com/acgetchell/delaunay/commit/4524fb51c7dbaef980dc984cd60efea77cd246bc)

- Added a new implementation for the `from` function in the `Point` struct that converts the `coords` array to `[f64; D]`.
  - Updated the comments to reflect this change.
- Find_extreme_coordinate [`1c5444e`](https://github.com/acgetchell/delaunay/commit/1c5444e5841a47fad5549db0db0a18fc5995afaf)

- In the `cell.rs` file, changed the type of vertex to match the circumcenter.
  - In the `triangulation_data_structure.rs` file, updated the function call to use `find_extreme_coordinate` instead of `find_min_coordinate`.
  - In the `utilities.rs` file, added a new function called `find_extreme_coordinate` that takes a hashmap of vertices and returns either the minimum or maximum
    coordinates based on the specified ordering.

  This commit makes changes related to changing vertex types and updating utility functions for finding extreme coordinates in different dimensions.

- Supercell [`37ff603`](https://github.com/acgetchell/delaunay/commit/37ff6031e7dbe5546eac8838aab61c1efa24cf01)

Added function that creates a dynamically sized supercell based on the vertices in the triangulation data structure.

- Implement Copy and Default traits [`650f484`](https://github.com/acgetchell/delaunay/commit/650f48462f02da67c98838fe82e6634a2b8d0e25)
- Added Serialization via serde [`a89ff48`](https://github.com/acgetchell/delaunay/commit/a89ff482c4247d0e4834414f8c5b27318bba9285)
- Added Deserialization via serde [`fe3a704`](https://github.com/acgetchell/delaunay/commit/fe3a704fdd852f53caeefb7ba8c66047a2bf8f19)
- Add Facet struct and related functionality [`204a70f`](https://github.com/acgetchell/delaunay/commit/204a70f78c92cb2eeed889c531b6677b0725f79f)

- Added a new file `facet.rs` with the implementation of the `Facet` struct.
  - The `Facet` struct represents a facet of a d-dimensional simplex.
  - It is defined in terms of a cell and the vertex in the cell opposite to it.
  - Provides convenience methods used in the Bowyer-Watson algorithm.
  - Facets are not stored directly in the Triangulation Data Structure (TDS), but created on-the-fly when needed.
- Added Facet vertices method [`5651810`](https://github.com/acgetchell/delaunay/commit/5651810da73d3eb4c71db7429b21e8e0a014be68)
- Add Codacy security scan workflow [`c6a4742`](https://github.com/acgetchell/delaunay/commit/c6a47428d5fa12dd9eacf4d0f1c5745eb3ec55a1)
- Implement PartialEq, Eq, and PartialOrd on Vertex [`1430ce1`](https://github.com/acgetchell/delaunay/commit/1430ce1c96e100142239a5f4a50826e2064656a2)
- Add equality and order implementations for Cell, Facet, and Vertex
  [`0e16c16`](https://github.com/acgetchell/delaunay/commit/0e16c16d2febcc76d8b3c176770ab1f253969d65)
- Add functions to refactor Bowyer-Watson [`e603652`](https://github.com/acgetchell/delaunay/commit/e603652dfc6a6e6d861a403c200b766f09467c7c)
- Use derive_builder [`1d6adc8`](https://github.com/acgetchell/delaunay/commit/1d6adc804dc67fc34a92f4274bbaefcc45535573)

### Changed

- Initial commit [`c6dc73a`](https://github.com/acgetchell/delaunay/commit/c6dc73a4d7443599ef7fa9dbee750e689ecabadc)
- Rename README.md file [`3d58871`](https://github.com/acgetchell/delaunay/commit/3d5887157b5249704dc19e429447732ca86aad07)
- Refactor README.md to include links and clarify project goals
  [`b75bd35`](https://github.com/acgetchell/delaunay/commit/b75bd35333f5cf53eabe26d1c640239c4c4991e0)
- Update README.md formatting and fix hyperlink syntax [`2143a89`](https://github.com/acgetchell/delaunay/commit/2143a893bcefd696c9b1932a0a96065e30620c39)
- Update README.md formatting and fix hyperlinks [`0620071`](https://github.com/acgetchell/delaunay/commit/0620071273a4ec16a0a1b3b34db0ab551677b3c6)
- Create rust.yml [`885f1d4`](https://github.com/acgetchell/delaunay/commit/885f1d40f2537e7983fbac22acb2e76f4e0edfb6)
- Use uuid to identify cells and vertices [`ebc1c61`](https://github.com/acgetchell/delaunay/commit/ebc1c61a06729db927b03d444a889d4c9c6b9d6e)
- Bare-bones Tds ctor, make data optional [`218cfc3`](https://github.com/acgetchell/delaunay/commit/218cfc3a1a2e35e82e7de30809a0a149b41dc762)
- Tell codecov to ignore lib.rs [`7e8ae59`](https://github.com/acgetchell/delaunay/commit/7e8ae59353c3258dc71eb3e66f58c1cf85986751)
- Construct hashmap of vertices in tds [`2559b06`](https://github.com/acgetchell/delaunay/commit/2559b06d0e98dd212329e2f38949baaa955b4026)
- 3D -> d-D [`62359b5`](https://github.com/acgetchell/delaunay/commit/62359b5c932e644ebf4794c53d8adac252442d5e)
- Refactor `dim` method to calculate the dimension based on the number of vertices. Update tests accordingly.
  [`0a080de`](https://github.com/acgetchell/delaunay/commit/0a080ded3630c98385bb33d58f8f1c304fd28afc)
- Constrain cell dimensionality [`693a1df`](https://github.com/acgetchell/delaunay/commit/693a1df2e91a40157626e430c9a351bf053b22ee)
- Tds add function and dimensionality constraints [`7645f34`](https://github.com/acgetchell/delaunay/commit/7645f34e504e5a8a3361b2dec47d87e0738aff7a)
- Refactor Tds struct and add new methods for counting vertices and cells
  [`e701d62`](https://github.com/acgetchell/delaunay/commit/e701d62947e7dfc06c24ebd5abdd3f9e9e98e185)
- More documentation [`94bbbcb`](https://github.com/acgetchell/delaunay/commit/94bbbcb54325a3f3a37ae429fe7aaa850c56a438)
- So Codecov doesn't count Rustdoc tests [`20227be`](https://github.com/acgetchell/delaunay/commit/20227be345fb60fc44e45fae70d94d3e24c0a463)
- Layout structure and future work [`366dbad`](https://github.com/acgetchell/delaunay/commit/366dbad4cea224cf47e616b99e3629fae6d07dc9)
- Partial Bowyer-Watson algorithm implementation [`287f368`](https://github.com/acgetchell/delaunay/commit/287f368e70263bf031807290f4d092b919fc2fa5)
- Test for auto traits [`ba20c84`](https://github.com/acgetchell/delaunay/commit/ba20c846c1659c0161ae9ebaadc09e705df21022)
- Update README [`73a4c0f`](https://github.com/acgetchell/delaunay/commit/73a4c0f7f4e76f3c5e733f6e2d8f983f6947c754)
- Cargo fmt [`f6ab29f`](https://github.com/acgetchell/delaunay/commit/f6ab29f9cb4a8f329484edbb2a0051dd0b156bee)
- Update dependencies in Cargo.toml and fix test assertions [`d2aff26`](https://github.com/acgetchell/delaunay/commit/d2aff26b8fc0e7652d3485de94ae49a28577aa32)
- Update Codacy [`dc1f5b9`](https://github.com/acgetchell/delaunay/commit/dc1f5b9a115e7084be52785c7132180daf9b03fd)
- Documentation fixes [`bb8a8a3`](https://github.com/acgetchell/delaunay/commit/bb8a8a3afd026d9bf8ac0ab611a690b587978a42)
- Debugging Bowyer-Watson [`9e7ba59`](https://github.com/acgetchell/delaunay/commit/9e7ba59b1f4881d36f0423377d72d3d8bdae158f)
- PartialEq, Eq, and PartialOrd on Cells [`530cd68`](https://github.com/acgetchell/delaunay/commit/530cd68b704ba5d2b08e7726055351917a8f8f97)
- Rationalize traits [`2102809`](https://github.com/acgetchell/delaunay/commit/21028093b47997b0e46f766ce05a903ef5e2ba86)
- Update Codacy Analysis CLI version and dependencies [`7d6b6c6`](https://github.com/acgetchell/delaunay/commit/7d6b6c682d32573442141199abfd3bfff0bb97e7)
- Interlink documentation [`8834e5c`](https://github.com/acgetchell/delaunay/commit/8834e5cffa8afac5fc0d7df036f7975980920808)
- Update package versions in Cargo.lock and Cargo.toml, [`ed7206d`](https://github.com/acgetchell/delaunay/commit/ed7206dac6cbea1a5b5d63e4141421c3f8ad6127)
- Update GitHub actions [#2](https://github.com/acgetchell/delaunay/pull/2)
  [`e44ce9f`](https://github.com/acgetchell/delaunay/commit/e44ce9fab33a6c4a8971ef3a80bf15b2e63d4c06)
- Fix libc [`c36bc97`](https://github.com/acgetchell/delaunay/commit/c36bc97cda68db1bbaf4aee25ac39550b7dacbc3)

And a few other trivial changes.

- Tests for cell facets [`f46f2f6`](https://github.com/acgetchell/delaunay/commit/f46f2f6dc435875cac69698d9a9aff7fdef37a2b)
- Refactor Bowyer-Watson [`e122aa9`](https://github.com/acgetchell/delaunay/commit/e122aa9d362e8682a38ca9ebfe602d523d3749cb)
- Deprecate Vector::new and Vector::new_with_data [`c1fc858`](https://github.com/acgetchell/delaunay/commit/c1fc858a3da2a23aba5ced235a07533bf96c443e)
- Remove Vertex::new and Vertex::new_with_data [`3d01927`](https://github.com/acgetchell/delaunay/commit/3d01927397d6ba919c5c5cf71e6be49260dcc70b)
- Deprecate Cell::new and Cell::new_with_data [`fa9f4d5`](https://github.com/acgetchell/delaunay/commit/fa9f4d50b77cfe0eb769b2f5f41fb9fb4c9bf165)
- Entirely replace Cell:new with CellBuilder [`6d75728`](https://github.com/acgetchell/delaunay/commit/6d757285791649d7398a1a437a20171843d5934e)

CellBuilder validator is nice.

- Feature/peroxide [#5](https://github.com/acgetchell/delaunay/pull/5)
  [`196beeb`](https://github.com/acgetchell/delaunay/commit/196beeb1c42539c49dfd11d0803f14462cb6ffac)
- Merge remote-tracking branch 'origin/main' [`a51a6ef`](https://github.com/acgetchell/delaunay/commit/a51a6ef3df18dfc03afc6f7318bbbf23ab0eeef7)
- Better tests, error messages [`6738259`](https://github.com/acgetchell/delaunay/commit/6738259cee47bbcfe6c160121df0081b9ea4ee68)

### Documentation

- Fix audit badge [`4a1ef9c`](https://github.com/acgetchell/delaunay/commit/4a1ef9c7e480d49c4e762647c192b4652b07a584)
- Documentation fixes and dependency updates [`3ca481d`](https://github.com/acgetchell/delaunay/commit/3ca481d444d58488bb74359604837d3bd808a8da)

### Fixed

- Fixing circumcircle [`e7204f0`](https://github.com/acgetchell/delaunay/commit/e7204f03d7f89f549cd8d4ba712d573bf1d45a5f)
- Fix circumsphere function -- it works! [`9fac2c1`](https://github.com/acgetchell/delaunay/commit/9fac2c1223a02aa6ac939784a868acafcf00883f)
- Fix issues identified by CodeQL [`ec1dc54`](https://github.com/acgetchell/delaunay/commit/ec1dc5492a7a7ddefcc6e74d843bb9465d74260a)
- Fix trait bounds and clippy lint [`701e326`](https://github.com/acgetchell/delaunay/commit/701e326f316f5a106dd33493ff730a05184e288c)
- Fix traits to be consistent across Vertex, Cell, Facet, Tds
  [`d55df6f`](https://github.com/acgetchell/delaunay/commit/d55df6f5d0cf42d57e35d42ca7a9cb0906f52ac4)

### Maintenance

- Update Codecov and dependencies [`9538214`](https://github.com/acgetchell/delaunay/commit/95382142398c1e5b96f81679e44e9f83f1916bfd)
- Update dependencies [`581949f`](https://github.com/acgetchell/delaunay/commit/581949fadca235f10bafff991990809fe26e0818)

Resolve Rustsec Advisory for bytemuck 1.16.1, update Codacy.

- Reduce CodeCov expectations [`0f7a2c0`](https://github.com/acgetchell/delaunay/commit/0f7a2c0ced35dbce7b23853346b3364a55bde362)
- Fix ci workflows [`04f680d`](https://github.com/acgetchell/delaunay/commit/04f680d02903a9525b09675610841d49c99ad62f)

Don't need to double-submit jobs for merged PRs.

[unreleased]: https://github.com/acgetchell/delaunay/compare/v0.7.2..HEAD
[0.7.2]: https://github.com/acgetchell/delaunay/compare/v0.7.1..v0.7.2
[0.7.1]: https://github.com/acgetchell/delaunay/compare/v0.7.0..v0.7.1
[0.7.0]: https://github.com/acgetchell/delaunay/compare/v0.6.2..v0.7.0
[0.6.2]: https://github.com/acgetchell/delaunay/compare/v0.6.1..v0.6.2
[0.6.1]: https://github.com/acgetchell/delaunay/compare/v0.6.0..v0.6.1
[0.6.0]: https://github.com/acgetchell/delaunay/compare/v0.5.4..v0.6.0
[0.5.4]: https://github.com/acgetchell/delaunay/compare/v0.5.3..v0.5.4
[0.5.3]: https://github.com/acgetchell/delaunay/compare/v0.5.2..v0.5.3
[0.5.2]: https://github.com/acgetchell/delaunay/compare/v0.5.1..v0.5.2
[0.5.1]: https://github.com/acgetchell/delaunay/compare/v0.5.0..v0.5.1
[0.5.0]: https://github.com/acgetchell/delaunay/compare/v0.4.4..v0.5.0
[0.4.4]: https://github.com/acgetchell/delaunay/compare/v0.4.3..v0.4.4
[0.4.3]: https://github.com/acgetchell/delaunay/compare/v0.4.2..v0.4.3
[0.4.2]: https://github.com/acgetchell/delaunay/compare/v0.4.1..v0.4.2
[0.4.1]: https://github.com/acgetchell/delaunay/compare/v0.4.0..v0.4.1
[0.4.0]: https://github.com/acgetchell/delaunay/compare/v0.3.5..v0.4.0
[0.3.5]: https://github.com/acgetchell/delaunay/compare/v0.3.4..v0.3.5
[0.3.4]: https://github.com/acgetchell/delaunay/compare/v0.3.3..v0.3.4
[0.3.3]: https://github.com/acgetchell/delaunay/compare/v0.3.2..v0.3.3
[0.3.2]: https://github.com/acgetchell/delaunay/compare/v0.3.1..v0.3.2
[0.3.1]: https://github.com/acgetchell/delaunay/compare/v0.3.0..v0.3.1
[0.3.0]: https://github.com/acgetchell/delaunay/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/acgetchell/delaunay/tree/v0.2.0
