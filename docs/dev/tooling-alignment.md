# Tooling Alignment

This note records the comparison between `delaunay` and the sibling
`causal-triangulations` repository. Keep it current when changing repository
tooling so future updates are deliberate rather than copied wholesale.

## Current Parity

Both repositories now share the same core Rust and Python support-tooling loop:

- `rust-toolchain.toml`, `rustfmt.toml`, `.taplo.toml`, `clippy.toml`, and
  `ty.toml` pin local tool behavior.
- `pyproject.toml` owns Ruff, Ty, pytest, uv packaging, and uv-managed
  development dependencies. The Python support-tooling baseline is Python 3.13
  through `.python-version`, `requires-python`, and setup-python consumers in
  CI/Codacy; Ruff and Ty infer their analysis target from `requires-python`.
- `pyproject.toml`, `dprint.json`, `.yamllint`, and `typos.toml` define local
  and CI documentation/configuration checks. Markdown linting uses Rust-native
  `rumdl`, and YAML/CFF formatting uses `dprint` with the `pretty_yaml` plugin;
  Node-backed Markdownlint and Prettier are no longer part of the local, CI, or
  Codacy tooling path. `CITATION.cff` is YAML-style-linted with `.yamllint` and
  schema-validated through an
  exact-version `uvx --from cffconvert==2.0.0` invocation, keeping
  `cffconvert`'s old `jsonschema` constraint isolated from Semgrep's newer
  dependency requirements.
- Codacy Markdownlint's MD013 line-length threshold is managed in Codacy's
  Code Patterns UI at 160 columns when that tool is enabled. Local Markdown,
  Python, YAML, and review thresholds are likewise normalized to 160 columns.
  Codacy Markdownlint excludes `docs/RELEASING.md` because that release
  checklist intentionally uses stable absolute step numbers across fenced
  command blocks; local rumdl disables MD029 for the same reason.
- Codacy Python engines are scoped to production scripts and exclude
  `scripts/tests/**`, so Ruff/Bandit feedback stays focused on shipped helper
  code and Bandit does not flag intentional test assertions.
- The Codacy SARIF splitter uses the same Python 3.13 setup as local tooling
  before filtering uploads, and parses raw SARIF JSON into a typed boundary
  model before applying repository-owned rule filtering.
- `justfile` is the local entry point for formatting, linting, tests,
  coverage, Semgrep, changelog, setup commands, and supported Cargo feature
  surface checks.
- `scripts/` contains typed Python helpers for changelog, tag, benchmark,
  coverage, subprocess, and hardware workflows.

## Ported From causal-triangulations

The useful updates ported in this pass are:

- Rust MSRV metadata now follows `causal-triangulations` and `la-stack` at
  Rust 1.96.0. `Cargo.toml`, `rust-toolchain.toml`, `clippy.toml`, contributor
  docs, and agent guidance all use the same baseline so the `la-stack` 0.4.3
  dependency update in #424 has no MSRV conflict.
- The local `cargo-nextest` pin and CI workflow environment variables now use
  0.9.137, matching both sibling repositories. `cargo-llvm-cov` stays on 0.8.7,
  which is still shared across the repositories.
- CI command-runner, Markdown, and spelling tool pins now track the newer
  sibling-repository versions used by `causal-triangulations` and `la-stack`:
  `just` 1.52.0, `rumdl` 0.2.14, and `typos-cli` 1.47.2. The `rumdl` bump keeps
  Delaunay on the sibling repository's current Markdown linter release after
  local `just check` exposed the older 0.2.10 pin as stale.
- uv-managed Python support-tool pins now match `causal-triangulations` exactly
  for the shared dev tools: `ruff` 0.15.14, `semgrep` 1.164.0, and `ty` 0.0.40.
  Delaunay previously used lower-bound specifiers for those tools, which allowed
  local and CI environments to drift away from the reviewed sibling baseline.
- Local just helpers now version-check the pinned `just`, `cargo-nextest`,
  `taplo-cli`, `dprint`, `rumdl`, `typos-cli`, and `zizmor` tools instead of
  accepting any installed version. Delaunay does not currently pin or invoke
  `cargo-instruments`; no `cargo-instruments` alignment was needed in this
  pass.
- GitHub Actions Cargo tool installation now uses
  `taiki-e/cache-cargo-install-action` where Delaunay previously used
  `taiki-e/install-action`, matching the sibling workflow pattern for pinned
  Rust CLI tools while preserving Delaunay's existing matrix shape.
- `.github/workflows/ci.yml` now follows the sibling repository pattern on all
  three operating systems: set up Python and uv, sync the locked dev tool group,
  install pinned Cargo tools, and run `just ci` for Linux, macOS, and Windows.
  uv-managed wrappers provide `actionlint`, `shellcheck`, `shfmt`, and
  `yamllint`, avoiding platform-specific apt/Homebrew/curl installation paths in
  the CI job. The GitHub selected-actions allowlist includes
  `taiki-e/cache-cargo-install-action@*`, so the cached Cargo-tool installer is
  permitted by repository policy.
- `.github/workflows/audit.yml` now installs `cargo-audit` 0.22.1 through the
  same cached Cargo-tool installer, and `.github/workflows/zizmor.yml` imports
  the dedicated GitHub Actions security scan from `la-stack`.
- `semgrep.yaml` now carries the low-noise hardening rules already proven useful
  in sibling repositories: checkout credential persistence, `pull_request_target`
  avoidance, `github-script` expression interpolation, locked `uv sync`, direct
  `subprocess.run` outside the wrapper, public `_unchecked` API exposure,
  prefixed panic macros, doctest unwrap/expect usage, and erased dynamic-error
  doctest examples. It also ports the sibling `assert!(matches!(...))` doctest
  rule and rewrites public documentation examples to `std::assert_matches!` so
  failing examples show the unmatched value instead of a bare boolean.
- Delaunay intentionally keeps its lean `rust-toolchain.toml` profile instead
  of copying the heavier sibling toolchain component and target lists. The
  minimal profile plus `clippy`, `rustfmt`, and `rust-src` remains the better
  default for this larger crate because workflows already install additional
  targets or components when they need them.
- `just check-fast` for a cheap compile-only check.
- `just markdown-check` keeps `rumdl` as the Markdown linter and adds a raw
  active-doc line-length guard so table rows and other Markdown constructs that
  `rumdl` exempts from MD013 still respect the repository's 160-column limit.
  The guard excludes generated changelog files and archived historical docs,
  which remain governed by their regeneration/archive workflows.
- `just changelog-unreleased <version>`, implemented with
  `GIT_CLIFF_OFFLINE=true git-cliff --tag`, so release PR changelogs no longer
  need temporary local release tags.
- `just changelog-tag <version>` as a compatibility alias for the canonical
  `just tag <version>` flow.
- `cliff.toml` improvements for escaped Rust generic angle brackets, GitHub
  `...` compare links, release-prep commit filtering, dependency grouping, and
  `Closes #N` cleanup.
- `scripts/tag_release.py` forced-tag replacement that lets git replace the ref
  transactionally instead of deleting the existing tag first.
- `scripts/archive_changelog.py` uses the la-stack cross-drive fallback for
  archive-directory links, so Windows-style `os.path.relpath` failures fall back
  to absolute POSIX links with a warning instead of aborting changelog
  generation.
- changelog archive sorting for prerelease labels and POSIX archive paths in
  generated links.
- changelog postprocessing improvements for breaking-change summary
  deduplication, star-bullet summaries, rumdl-friendly body headings, and final
  rumdl formatting of generated changelog files.
- `scripts/postprocess_changelog.py` follows the la-stack release-heading
  model, preserving real semver/Unreleased headings while demoting bracketed
  entry-local headings that only look like release boundaries.
- Python support utilities parse invariant-bearing external data at their
  boundaries: Codacy SARIF is recursively checked as strict JSON, and
  `ci_performance_suite` metrics are loaded into typed count records before
  benchmark comparison code consumes them.
- Repository-owned Semgrep now reinforces those Python boundary invariants:
  CI JSON filters must reject non-finite JSON constants on load, emit strict
  JSON with `allow_nan=False`, and keep `ci_performance_suite` construction
  metric counts positive rather than merely non-negative.
- Repository-owned Semgrep now enforces the #406 `FlipError` layout policy:
  scalar/key diagnostics stay inline, while nested typed error payloads in the
  public flip error enum must be boxed sources so future subordinate error-enum
  growth does not silently bloat `Result<_, FlipError>` or hide typed causes
  from `Error::source`.
- Semgrep rule refinements for non-finite fallback patterns, broad Python
  exception catches with bindings, and direct ad hoc subprocess mock
  constructors.
- release documentation describing the no-temporary-tag changelog flow and
  final annotated tag creation from the generated changelog.
- `scripts/tag_release.py` now normalizes archived changelog paths with
  `as_posix()` before rendering GitHub URLs, with a regression test that
  exercises a Windows-style archive path.
- `just semgrep-test` now discovers all fixtures under `tests/semgrep` by
  mirroring fixture paths to a temporary Semgrep config tree backed by the
  repository `semgrep.yaml`, matching the causal-triangulations fixture
  discovery model.
- `just markdown-check` and `just markdown-fix` now use `rumdl`, preserving the
  previous Markdown rule exceptions in `pyproject.toml` while removing the
  `npx markdownlint` dependency from local and CI workflows.
- `just yaml-check` and `just yaml-fix` now use `dprint` with the
  `pretty_yaml` plugin for formatting, while `yaml-lint` remains the yamllint
  style pass for YAML/CFF validation.
- Repository-owned Semgrep rules now guard obvious check/fix command-ordering
  regressions in user-facing docs and enforce SHA-pinned, allowlisted external
  GitHub Actions with readable version comments.
- The GitHub Actions policy rules intentionally exclude `tests/semgrep/**`
  during normal repository scans because those fixtures contain deliberate
  violations; `just semgrep-test` remains the comparison path for proving the
  rules still catch the fixture cases.
- `.github/workflows/rust-clippy.yml` now matches the hardened SARIF pipeline:
  `set -euo pipefail`, `clippy::cargo`, and guarded upload that skips missing
  SARIF files and forked pull-request uploads.
- `.github/workflows/semgrep-sarif.yml` adds the direct repository-rule SARIF
  workflow used by causal-triangulations, pinned to this repository's current
  uv version. It complements the Codacy Opengrep workflow by uploading
  Semgrep-native SARIF and failing the workflow on repository-rule findings.
- `.github/workflows/codacy.yml` defensively filters Codacy Opengrep SARIF to
  repository-owned `delaunay.*` rule IDs before uploading to GitHub Code
  Scanning. Codacy's default maintainability patterns can still run in Codacy,
  but they must not create broad Code Scanning alerts for test-only paths such
  as `scripts/tests/**`.
- Codacy Code Patterns should stay aligned with local validators rather than
  acting as an independent style regime. Keep repository-owned Rust feedback on
  Opengrep/Semgrep rules; keep broad advisory engines such as Lizard disabled
  for PR gating unless a specific baseline audit requests them. If the Codacy
  UI exposes a separate duplicate-code metric, treat it the same way.
- CI and local setup pins should track the same supported tool versions when
  practical. The current workflow pins align coverage and test tooling on
  `cargo-llvm-cov` 0.8.7 and `cargo-nextest` 0.9.137. CI, Codecov, and local
  setup install the same `cargo-nextest` pin with the sibling repositories'
  `taiki-e/cache-cargo-install-action`/`cargo install --locked` pattern before
  nextest-backed recipes run. Pinned Rust CLI tools are installed through Cargo
  rather than Homebrew so local setup cannot drift from CI pins. The CI build
  matrix runs `just ci` on Linux, macOS, and Windows after syncing the locked uv
  dev group and installing the pinned Cargo tools. All uv-backed workflows use
  uv 0.11.20 to match the local Python tooling bootstrap.
- `.codecov.yml` now ratchets Delaunay's coverage policy above the older
  causal-triangulations baseline without copying la-stack's near-total
  threshold. Project coverage targets the current 90% line with only 1%
  tolerated drift. Patch coverage remains at 50% for this cleanup because
  Codecov attributes the `assert_matches!` test-quality migration as uncovered
  macro-invocation churn; once that lands, the next coverage-only ratchet should
  raise the patch target toward 70% without forcing superficial tests.
- Semgrep now ports the sibling repositories' Rust examples/benchmarks hygiene
  checks: examples and benchmarks should avoid panic-only `unwrap`/`expect`
  paths and dynamic error erasure so public usage remains explicit and typed.
- `.config/nextest.toml` keeps `profile.ci` bounded for normal CI while
  `profile.slow` raises the per-test watchdog for `just test-slow`, allowing
  intentional multi-minute correctness regressions to run without making the
  default suite or CI profile permissive.
- `.github/workflows/release-benchmarks.yml` ports the durable release-asset
  benchmark storage pattern used by `la-stack`. Delaunay now packages
  `baseline_results.txt`, `PERFORMANCE_RESULTS.md`, raw Criterion data, and
  metadata as `delaunay-$TAG-criterion-baseline.tar.gz` on each published
  GitHub Release. Release and regression benchmark jobs both use
  `ubuntu-latest`, and `.github/workflows/benchmarks.yml` downloads the latest
  stable release asset to compare CI runs against the released-version CI
  baseline. Local same-machine comparison remains separate through ignored
  `baseline-artifact/` and `baseline-artifacts/` paths, while
  `.github/workflows/generate-baseline.yml` remains manual-only for
  compatibility with ad-hoc CI-runner artifact comparisons.

## CI Shape Evaluation

Issue #402 is treated as an evaluation track, not permission to reduce the
platform matrix. The CI invariant remains that pull requests and pushes run
the complete local `just ci` recipe on Linux, macOS, and Windows. Future
wall-clock reductions should preserve that all-platform command unless a later
design explicitly replaces each lost check with equivalent portability
coverage.

Use the GitHub Actions job and step duration metadata to compare the first run
after a tool pin or cache-key change against later warm-cache runs before
proposing any heavier CI-shape split. That keeps measurement out of the
workflow itself while still making the cold-cache and warm-cache trade-off
visible.

Current measurements from the GitHub Actions UI/API:

| Run | Platform | Job duration | `just ci` duration |
|-----|-----|-----|-----|
| [`27058331033`](https://github.com/acgetchell/delaunay/actions/runs/27058331033) | Ubuntu | 15m 17s | 14m 47s |
| [`27058331033`](https://github.com/acgetchell/delaunay/actions/runs/27058331033) | macOS | 13m 55s | 12m 58s |
| [`27058331033`](https://github.com/acgetchell/delaunay/actions/runs/27058331033) | Windows | 65m 50s | 33m 31s |
| [`27059628821`](https://github.com/acgetchell/delaunay/actions/runs/27059628821) | Ubuntu | 37m 1s | 16m 33s |
| [`27059628821`](https://github.com/acgetchell/delaunay/actions/runs/27059628821) | macOS | 41m 20s | 19m 28s |
| [`27059628821`](https://github.com/acgetchell/delaunay/actions/runs/27059628821) | Windows | 72m 43s | 32m 45s |

## Intentional Differences

Some causal-triangulations tooling remains project-specific and was not ported:

- CDT-specific Semgrep rules for geometry-backend isolation, foliation
  validation, CDT error variants, and `causal_triangulations::prelude` imports.
- CDT's Python test `Path.read_text`/`write_text` encoding rule is deferred to
  #429 because Delaunay has existing Python fixture cleanup to do before that
  rule can become low-noise.
- CDT's `examples-validate` recipe and output-marker checks; Delaunay examples
  are currently validated through `just examples`.
- CDT's `performance-analysis` script; Delaunay keeps its benchmark helpers and
  generated performance-summary workflow.
- Delaunay has a project-specific `just perf-no-regressions` recipe that runs
  the calibrated 2D-5D `ci_performance_suite` canaries against a cached
  same-machine dev-mode baseline for the current GitHub `main` commit. This
  stays local to Delaunay because the benchmark contract, fixture sizes, and
  regression threshold are tied to this library's triangulation performance
  expectations. The cache lives under
  `baseline-artifacts/perf-no-regressions/`, is keyed by the resolved
  `origin/main` commit and local Rust compiler version, and is refreshed when
  either key changes or the artifact no longer matches the benchmark contract.
  The cache/validation logic lives in `benchmark-utils ensure-ref-baseline` and
  `benchmark-utils compare-ref` so workflows can reuse the same behavior without
  depending on justfile shell internals.
  `compare-ref` writes branch/PR-vs-ref reports with explicit names such as
  `benches/worktree_vs_main_compare_results.txt`, while release-baseline
  comparisons use `benches/main_vs_release_compare_results.txt`. Ref
  comparisons fail on total matched-time regressions and report individual
  regressions/improvements as context; release-baseline comparisons remain
  strict for individual regressions.
  `just perf-baseline [ref]` remains the manual persistent-baseline workflow for
  a requested GitHub ref, so ad-hoc comparisons still use the developer's local
  hardware while GitHub Actions publishes shared CI artifacts.
- Delaunay keeps a single `profiling_suite` benchmark target for manual
  large-scale and flamegraph work. The previous standalone large-scale target
  was folded into that harness so `.github/workflows/profiling-benchmarks.yml`
  and `just profile-dev` exercise the same real construction, memory,
  validation, and traversal workloads.
- CDT's concise `docs/dev/commands.md` structure; Delaunay keeps its more
  detailed benchmark-profile guidance because it documents the `perf` profile,
  local performance-regression guard, calibrated benchmark canaries, and release
  benchmark summary workflow.

## Cargo Packaging And Toolchain Hygiene

The Delaunay crate uses an explicit Cargo package allowlist so crates.io
artifacts carry the public library surface, examples, benchmarks, integration
tests, and active documentation without bundling CI-only tooling, Semgrep
fixtures, Python automation, or archived design notes. Keep this allowlist
aligned with files needed by `cargo publish --dry-run`, docs.rs, examples,
benchmarks, and Cargo's package target discovery.

The pinned Rust toolchain intentionally uses the `minimal` profile plus only
the repository-required components (`clippy`, `rustfmt`, and `rust-src`).
Cross-target standard libraries and IDE-only components should be installed by
the workflows or local developers that need them, rather than by every checkout.

GitHub Actions jobs that use `actions-rust-lang/setup-rust-toolchain` keep
dependency and target caching enabled, but explicitly set `cache-bin: false`.
The action restores cache entries after printing the installed Cargo version,
so caching `${CARGO_HOME}/bin` can overwrite rustup's `cargo` shim with a stale
or host-incompatible binary. This is especially visible on macOS, where a
poisoned cache can make `cargo fmt` invoke `rustup-init` instead of Cargo.

## Activated Deferred Updates

The following previously deferred checks are now repository-owned Semgrep rules:

- `delaunay.rust.public-error-enums-non-exhaustive` requires every public
  `*Error` enum to carry `#[non_exhaustive]`, preserving room for more precise
  typed variants without forcing a breaking exhaustive-match change on users.
- `delaunay.rust.no-production-unwrap-panic` blocks bare `unwrap()` and
  `panic!` in non-test `src/` code, with narrowly scoped `expect(...)` matches
  for reviewed public-API panic messages that should become typed errors.
- `delaunay.rust.no-partial-cmp-ordering-defaults` requires source ordering
  code to spell the incomparable branch explicitly instead of silently mapping
  `partial_cmp` failures to an arbitrary `Ordering`.
- `delaunay.rust.no-function-local-use-in-src` keeps non-test source imports at
  module scope so dependency shape is visible during review.
- `delaunay.rust.no-deep-crate-paths-in-functions` keeps long internal module
  paths out of function bodies; use focused imports or a local helper when the
  path is part of the implementation.
- `delaunay.rust.no-ignored-tests` keeps the test taxonomy to explicit
  cfg-gated buckets. Deterministic slow tests should use `slow-tests`,
  benchmark-style measurements should live in `benches/`, and real tests should
  not use `#[ignore]`.
- `delaunay.rust.no-silent-conversion-fallbacks-in-public-samples` extends the
  existing source conversion-fallback check to examples, benchmarks, and public
  API tests so copied usage does not hide numeric conversion failures.
- `delaunay.rust.prefer-prelude-imports-in-examples-benches` and
  `delaunay.rust.prefer-prelude-imports-in-delaunay-doctests` track the
  flattened Delaunay API surface: the removed `delaunay::delaunay::*` facade is
  no longer matched, while focused root modules such as `delaunay::flips::*`
  still trigger guidance toward the orthogonal prelude modules in public
  samples.
- `delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers` keeps local
  repair/topology simplex-key frontiers on `SimplexKeyBuffer` instead of raw
  `Vec<SimplexKey>`. The rule is intentionally name-based and heuristic:
  unbounded full-TDS snapshots, public/error payloads, serialization
  boundaries, and other deliberately heap-backed collections should remain
  `Vec`.

## Public Sample Error Handling

Examples, benchmarks, and public API integration tests should model the same
error-handling style the crate asks users to copy: return or route through
typed errors instead of using `.unwrap()`, `.expect(...)`, or `panic!(...)` as
narrative control flow.

This is now enforced by `delaunay.rust.no-public-surface-unwrap-panic` for:

- `examples/**/*.rs`
- `benches/**/*.rs`
- public API integration tests:
  - `tests/allocation_api.rs`
  - `tests/prelude_exports.rs`
  - `tests/public_*.rs`

The doctest migration remains intentionally separate because Rust doc comments
still have an existing `.unwrap()`/`.expect(...)` baseline that should be
converted with hidden `Result` wrappers in a focused documentation pass. A
broad no-doctest-unwrap Semgrep rule should only be enabled after that baseline
is removed, otherwise `just semgrep` becomes noisy enough to hide actionable
source and public-sample findings.

```bash
just verify-expect-counts
```

The `verify-expect-counts` recipe tracks only that remaining doctest baseline.
When extending the zero-tolerance Semgrep rule to doctests, prefer:

- `fn main() -> Result<(), ExampleError>` in examples, with local
  `#[derive(thiserror::Error)]` enums that wrap the crate's typed errors.
- Setup helpers returning typed `Result` values in benchmarks; Criterion entry
  points may still abort setup explicitly, but benchmark bodies should not hide
  fallible API calls behind `.expect(...)`.
- Doctests that use hidden `Result` wrappers and `?` where the visible API is
  fallible, so examples demonstrate the real error variant users receive.
