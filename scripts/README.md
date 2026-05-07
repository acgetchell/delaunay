# Scripts Directory

This directory contains Python and shell tooling used by the `delaunay`
repository. Prefer the `just` recipes and `uv run ...` entrypoints documented
here over invoking Python files directly.

## Prerequisites

- Python 3.12+
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
just changelog-unreleased v0.7.7
just tag v0.7.7

uv run postprocess-changelog --help
uv run archive-changelog --help
uv run tag-release v0.7.7 --help
```

`just changelog` runs `git-cliff`, applies markdown hygiene, and archives
completed minor release series under `docs/archive/changelog/`.

Use `just changelog-unreleased vX.Y.Z` while preparing a release PR before the
final tag exists. Use `just tag vX.Y.Z` after the release PR is merged to
create the annotated release tag from the matching changelog section.

### Benchmark utilities

```bash
uv run benchmark-utils generate-baseline
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
```

`benchmark-utils` handles Criterion baseline generation, comparison, and
release performance summaries.

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
uv run ruff check scripts/ --fix
uv run ruff format scripts/
uv run ty check scripts/ --error all
uv run pytest scripts/tests
```

The usual repository entrypoints are:

```bash
just python-check
just python-fix
just python-typecheck
just test-python
```

## Maintenance expectations

- Keep scripts typed and covered by focused pytest tests.
- Prefer `subprocess_utils.py` wrappers for subprocess execution.
- Use `subprocess.CompletedProcess[str]` in tests instead of ad hoc mocks.
- Catch specific recoverable exception families; avoid broad
  `except Exception`.
- Update this README when adding, renaming, or removing `pyproject.toml`
  script entrypoints.
