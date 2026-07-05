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

uv run postprocess-changelog --help
uv run archive-changelog --help
uv run tag-release vX.Y.Z --help
```

`just changelog` runs `git-cliff`, applies markdown hygiene, and archives
completed minor release series under `docs/archive/changelog/`.

Use `just changelog-unreleased vX.Y.Z` while preparing a release PR before the
final tag exists. Use `just tag vX.Y.Z` after the release PR is merged to
create the annotated release tag from the matching changelog section.

### Notebook utilities

```bash
just notebook-lint
just notebook-check
uv run --group dev --group notebooks notebook-check --help
```

`notebook-check` validates notebook JSON, rejects committed outputs and
execution counts, extracts code cells for Ruff and ty, and can execute notebooks
headlessly. The `just notebook-check` recipe writes executed notebooks under
`target/notebooks/<notebook-stem>/` and leaves source notebooks unchanged.

### Benchmark utilities

```bash
uv run benchmark-utils generate-baseline
uv run benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
```

`benchmark-utils` handles Criterion baseline generation and packaging,
comparison, and release performance summaries. Published releases package
`baseline_results.txt` with raw Criterion data as a GitHub Release asset for
Ubuntu GitHub Actions comparisons. Local timing records should stay in the
ignored `baseline-artifact/` or `baseline-artifacts/` directories. The default
comparison report for release baselines is
`benches/main_vs_release_compare_results.txt`; the ref-comparison guard writes
`benches/worktree_vs_<ref>_compare_results.txt` and fails only on total
matched-time regressions or execution errors.

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
