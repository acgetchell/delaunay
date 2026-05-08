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
- CDT's `performance-analysis` script; Delaunay keeps its benchmark and
  storage-backend comparison helpers.
- CDT's concise `docs/dev/commands.md` structure; Delaunay keeps its more
  detailed benchmark-profile guidance because it documents the `perf` profile,
  backend compatibility canary, and release benchmark summary workflow.

## Activated Deferred Updates

The following previously deferred checks are now repository-owned Semgrep rules:

- `delaunay.rust.public-error-enums-non-exhaustive` requires every public
  `*Error` enum to carry `#[non_exhaustive]`, preserving room for more precise
  typed variants without forcing a breaking exhaustive-match change on users.
- `delaunay.rust.no-production-unwrap-panic` blocks bare `unwrap()` and
  `panic!` in non-test `src/` code, with narrowly scoped `expect(...)` matches
  for reviewed public-API panic messages that should become typed errors.

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
converted with hidden `Result` wrappers in a focused documentation pass.

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
