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
- `dprint.json`, `.yamllint`, and `typos.toml` define documentation and
  configuration checks.
- `justfile` is the local entry point for formatting, linting, tests,
  coverage, Semgrep, changelog, setup commands, and supported Cargo feature
  surface checks.
- `scripts/` contains typed Python helpers for changelog, tag, benchmark,
  coverage, subprocess, and hardware workflows.

## Ported From causal-triangulations

The useful updates ported in this pass are:

- `just check-fast` for a cheap compile-only check.
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
  deduplication, star-bullet summaries, and blank lines after closing code
  fences.
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
- `.github/workflows/rust-clippy.yml` now matches the hardened SARIF pipeline:
  `set -euo pipefail`, `clippy::cargo`, and guarded upload that skips missing
  SARIF files and forked pull-request uploads.
- `.github/workflows/semgrep-sarif.yml` adds the direct repository-rule SARIF
  workflow used by causal-triangulations, pinned to this repository's current
  uv version. It complements the Codacy Opengrep workflow by uploading
  Semgrep-native SARIF and failing the workflow on repository-rule findings.

## Intentional Differences

Some causal-triangulations tooling remains project-specific and was not ported:

- CDT-specific Semgrep rules for geometry-backend isolation, foliation
  validation, CDT error variants, and `causal_triangulations::prelude` imports.
- CDT's `examples-validate` recipe and output-marker checks; Delaunay examples
  are currently validated through `just examples`.
- CDT's `performance-analysis` script; Delaunay keeps its benchmark helpers and
  generated performance-summary workflow.
- Delaunay has a project-specific `just perf-no-regressions` recipe that runs
  the calibrated 2D-5D `ci_performance_suite` canaries against the current
  dev-mode baseline artifact. This stays local to Delaunay because the
  benchmark contract, fixture sizes, and regression threshold are tied to this
  library's triangulation performance expectations. `just perf-baseline [ref]`
  generates that baseline on the developer's machine from a temporary checkout
  of the requested GitHub ref so the comparison uses the same local hardware as
  the current branch while GitHub Actions still publishes shared CI artifacts.
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
  - `tests/delaunay_public_api_coverage.rs`
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
