# Scripts Directory

This directory contains Python and shell tooling used by the `delaunay`
repository. Prefer `just` recipes for validation and tests, and use the
`uv run ...` entrypoints documented here when invoking an individual utility
directly.

## Prerequisites

- Python 3.14+
- `uv`

Install dev dependencies:

```bash
uv sync --group dev
```

## CLI entrypoints

These commands are exposed by `pyproject.toml`; all support `--help`.

### Changelog utilities

```bash
just changelog
just changelog-unreleased vX.Y.Z
just tag vX.Y.Z

uv run check-docs-version-sync --help
uv run postprocess-changelog --help
uv run archive-changelog --help
uv run tag-release vX.Y.Z --help
```

`just changelog` runs `git-cliff`, applies markdown hygiene, and archives
completed minor release series under `docs/archive/changelog/`.

`just docs-version-check` runs `check-docs-version-sync`, which compares the
Cargo package version against release-facing docs and metadata.

Use `just changelog-unreleased vX.Y.Z` while preparing a release PR before the
final tag exists. Use `just tag vX.Y.Z` after the release PR is merged to
create the annotated release tag from the matching changelog section.

### Notebook utilities

```bash
just notebook-check
just notebook-execute notebooks/00_quickstart.ipynb
just notebook-reset-from-git
uv run --group dev --group notebooks notebook-check --help
```

`notebook-check` validates notebook JSON, rejects committed outputs and
execution counts, and extracts code cells for Ruff and ty without executing
notebooks. `just notebook-execute` runs one notebook headlessly, writes the
executed notebook and generated artifacts under
`target/notebooks/<notebook-stem>/`, and leaves the source notebook unchanged.
`just notebook-reset-from-git` restores tracked source notebooks from the Git
index, or from an explicit source such as `HEAD`, and removes generated
notebook artifacts and Jupyter checkpoints.

### Benchmark utilities

```bash
uv run benchmark-utils generate-baseline
uv run benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run benchmark-utils bench-compare last
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
uv run benchmark-utils performance-local
uv run benchmark-utils performance-github-assets
uv run benchmark-utils performance-release
```

`benchmark-utils` handles Criterion baseline generation and packaging,
comparison, saved Criterion baseline reports, and release performance summaries.
It formats and compares benchmark evidence; the harnesses being run are
responsible for failing before timings are published when scientific invariants
are violated.
Published releases package `baseline_results.txt` with raw Criterion data as a
GitHub Release asset for Ubuntu GitHub Actions comparisons. Local timing records
should stay in the ignored `baseline-artifact/` or `baseline-artifacts/`
directories. `bench-compare` renders `target/bench-reports/performance.md` from
existing Criterion `new` data and a saved baseline such as `last`.
`performance-local` and `performance-github-assets` generate isolated
release-to-release reports under `target/bench-reports/`, while
`performance-release` promotes the curated report into `docs/PERFORMANCE.md`
and archives the previous one. These release reports are evidence, not routine
pre-`just ci` checks; temp-worktree generation applies tracked checkout changes
but ignores untracked files. The default comparison report for release
baselines is `benches/main_vs_release_compare_results.txt`; the ref-comparison
guard writes `benches/worktree_vs_<ref>_compare_results.txt` and fails only on
total matched-time regressions or execution errors.

### Hardware utilities

```bash
uv run hardware-utils info
uv run hardware-utils kv
uv run hardware-utils info --json
```

### Coverage utilities

```bash
just coverage-ci
uv run coverage-report --help
uv run coverage_report --help
```

`coverage-report` summarizes the Cobertura XML produced by `just coverage-ci`.
`coverage_report` is kept as a backwards-compatible alias.

## Shell helpers

```bash
./scripts/run_all_examples.sh
```

Shell scripts use strict mode and should be linted through the repository
validation commands.

## Linting and tests

```bash
just python-check
just python-typecheck
just test-python
just python-fix
```

## Maintenance expectations

- Keep scripts typed and covered by focused pytest tests.
- Prefer `subprocess_utils.py` wrappers for subprocess execution.
- Use `subprocess.CompletedProcess[str]` in tests instead of ad hoc mocks.
- Catch specific recoverable exception families; avoid broad
  `except Exception`.
- Update this README when adding, renaming, or removing `pyproject.toml`
  script entrypoints.
