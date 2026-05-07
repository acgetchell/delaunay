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

Examples, doctests, and benchmarks should model the same error-handling style
the crate asks users to copy: return or route through typed errors instead of
using `.expect(...)` as narrative control flow.

The next enforcement step is to make this a zero-tolerance Semgrep rule for:

- `examples/**/*.rs`
- `benches/**/*.rs`
- Rust doc comments in `src/**/*.rs`

Current baseline before that cleanup:

- `examples/**/*.rs`: 35 `.expect(...)` calls.
- `benches/**/*.rs`: 64 `.expect(...)` calls.
- public Rust doc comments in `src/**/*.rs`: 17 `.expect(...)` calls.

When removing this baseline, prefer:

- `fn main() -> Result<(), ExampleError>` in examples, with local
  `#[derive(thiserror::Error)]` enums that wrap the crate's typed errors.
- Setup helpers returning typed `Result` values in benchmarks; Criterion entry
  points may still abort setup explicitly, but benchmark bodies should not hide
  fallible API calls behind `.expect(...)`.
- Doctests that use hidden `Result` wrappers and `?` where the visible API is
  fallible, so examples demonstrate the real error variant users receive.
