# Tooling Alignment

This note records the comparison between `delaunay` and the sibling
`causal-triangulations` repository. Keep it current when changing repository
tooling so future updates are deliberate rather than copied wholesale.

## Current Parity

Both repositories now share the same core Rust and Python support-tooling loop:

- `rust-toolchain.toml`, `rustfmt.toml`, `.taplo.toml`, `clippy.toml`, and
  `ty.toml` pin local tool behavior.
- `pyproject.toml` owns Ruff, Ty, pytest, uv packaging, and uv-managed
  development dependencies.
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
- Codacy Python engines are scoped to production scripts and exclude
  `scripts/tests/**`, so Ruff/Bandit feedback stays focused on shipped helper
  code and Bandit does not flag intentional test assertions.
- `justfile` is the local entry point for formatting, linting, tests,
  coverage, Semgrep, changelog, setup commands, and supported Cargo feature
  surface checks.
- `scripts/` contains typed Python helpers for changelog, tag, benchmark,
  coverage, subprocess, and hardware workflows.

## Ported From causal-triangulations

The useful updates ported in this pass are:

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
- changelog archive sorting for prerelease labels and POSIX archive paths in
  generated links.
- changelog postprocessing improvements for breaking-change summary
  deduplication, star-bullet summaries, rumdl-friendly body headings, and final
  rumdl formatting of generated changelog files.
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
- CI and local setup pins should track the same supported tool versions when
  practical. The current workflow pins align coverage and test tooling on
  `cargo-llvm-cov` 0.8.7 and `cargo-nextest` 0.9.136. Both CI and Codecov
  install the same `cargo-nextest` pin with the repository's pinned binary-tool
  installer and verify `cargo nextest --version` before nextest-backed recipes
  run. The CI build matrix also installs and verifies that pin on Windows, where
  the lightweight direct-Cargo job runs library and integration tests through
  `cargo nextest run` while keeping doctests on `cargo test --doc`. Local
  `just setup-tools` uses `cargo install --locked cargo-nextest` and verifies
  the command is available for developer machines. All uv-backed workflows use
  uv 0.11.15 to match the local Python tooling bootstrap.

## Intentional Differences

Some causal-triangulations tooling remains project-specific and was not ported:

- CDT-specific Semgrep rules for geometry-backend isolation, foliation
  validation, CDT error variants, and `causal_triangulations::prelude` imports.
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
- `delaunay.rust.no-silent-conversion-fallbacks-in-public-samples` extends the
  existing source conversion-fallback check to examples, benchmarks, and public
  API tests so copied usage does not hide numeric conversion failures.
- `delaunay.rust.prefer-prelude-imports-in-examples-benches` and
  `delaunay.rust.prefer-prelude-imports-in-delaunay-doctests` track the
  flattened Delaunay API surface: the removed `delaunay::delaunay::*` facade is
  no longer matched, while focused root modules such as `delaunay::flips::*`
  still trigger guidance toward the orthogonal prelude modules in public
  samples.

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
