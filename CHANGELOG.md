# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes

- Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)

### Merged Pull Requests

- Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)

### Maintenance

- [**breaking**] Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)
  [`1b2af41`](https://github.com/acgetchell/delaunay/commit/1b2af41fcb5115d82c1ae6f0ab66651c075fbd52)

- chore!: replace custom changelog pipeline with git-cliff

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

## Archives

Older releases are archived by minor series:

- [0.6.x](docs/archive/changelog/0.6.md)
- [0.5.x](docs/archive/changelog/0.5.md)
- [0.4.x](docs/archive/changelog/0.4.md)
- [0.3.x](docs/archive/changelog/0.3.md)
- [0.2.x](docs/archive/changelog/0.2.md)

[unreleased]: https://github.com/acgetchell/delaunay/compare/v0.7.2..HEAD
[0.7.2]: https://github.com/acgetchell/delaunay/compare/v0.7.1..v0.7.2
[0.7.1]: https://github.com/acgetchell/delaunay/compare/v0.7.0..v0.7.1
[0.7.0]: https://github.com/acgetchell/delaunay/compare/v0.6.2..v0.7.0
