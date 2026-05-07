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

## Deferred Updates

These were evaluated but deferred:

- A Semgrep rule requiring every public `*Error` enum to be
  `#[non_exhaustive]`. Delaunay already has strong error-enum guidance, but
  enabling the rule should be a focused API audit because it may surface
  historical public enums.
- Production-wide bare `unwrap()` / `panic!` Semgrep rules copied exactly from
  CDT. Delaunay already has a narrower production panic rule; broadening it
  should be handled alongside any intentional invariant-panics it exposes.
