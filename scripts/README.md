# Scripts Directory

This directory contains Python and shell tooling used by the `delaunay`
repository. Prefer `just` recipes for validation and tests, and use the
`uv run --locked ...` entrypoints documented here when invoking an individual utility
directly.

## Prerequisites

- Python 3.14+
- `uv`

Install dev dependencies:

```bash
uv sync --group dev
```

Bootstrap the Just version pinned by the root `justfile`:

```bash
bash scripts/bootstrap_just.sh
```

The helper leaves an already-correct installation unchanged and otherwise
installs the pinned release through Cargo. Local setup and CI share the same
version resolver under `.github/actions/setup-just/`.

## CLI entrypoints

These commands are exposed by `pyproject.toml`; all support `--help`.

### Changelog utilities

```bash
just changelog
just changelog-unreleased vX.Y.Z
just release-version-check
just tag vX.Y.Z
just update-version vX.Y.Z

uv run --locked check-docs-version-sync --help
uv run --locked check-docs-version-sync --final-release
uv run --locked postprocess-changelog --help
uv run --locked archive-changelog --help
uv run --locked tag-release vX.Y.Z --help
uv run --locked update-release-version vX.Y.Z
```

`just changelog` runs `git-cliff`, applies markdown hygiene, and archives
completed minor release series under `docs/archive/changelog/`.

`just docs-version-check` runs `check-docs-version-sync`, which compares the
Cargo package version against release-facing docs and metadata.

Use `just changelog-unreleased vX.Y.Z` while preparing a release PR before the
final tag exists. Use `just tag vX.Y.Z` after the release PR is merged to
create the annotated release tag from the matching changelog section.
`just update-version vX.Y.Z` infers the previous stable published GitHub
Release and atomically synchronizes release metadata with the current UTC date.
Same-day retries are content-idempotent; later-day retries advance citation and
existing target changelog dates together.
`just release-version-check` runs the strict final gate, which requires one
current-version changelog heading whose date matches `CITATION.cff`.

### Notebook utilities

```bash
just notebook-check
just notebook-execute notebooks/00_quickstart.ipynb
just notebook-reset-from-git
just validation-doc-figures-check
uv run --locked --group dev --group notebooks notebook-check --help
```

`notebook-check` validates notebook JSON, rejects committed outputs and
execution counts, and extracts code cells for Ruff and ty without executing
notebooks. `just notebook-execute` runs one notebook headlessly, writes the
executed notebook and generated artifacts under
`target/notebooks/<notebook-stem>/`, and leaves the source notebook unchanged.
`just notebook-reset-from-git` restores tracked source notebooks from the Git
index, or from an explicit source such as `HEAD`, and removes generated
notebook artifacts and Jupyter checkpoints.

`just validation-doc-figures-check` executes the validation notebook into
`target/` and compares its complete generated PNG set with the tracked
documentation artifacts without publishing changes. The canonical byte check
is composed into `just ci` on macOS.

`delaunay-scripts` is repository-internal and is not distributed as a PyPI
tool. Run `notebook-check` through the locked project environment or the `just`
recipes above. The repository-managed `dev` and `notebooks` dependency groups
provide its Ruff, ty, and nbclient backends. Notebook-specific imports used by
the notebook being executed remain the notebook author's responsibility.

### Benchmark utilities

```bash
uv run --locked benchmark-utils generate-baseline
uv run --locked benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
uv run --locked benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run --locked benchmark-utils bench-compare last
uv run --locked benchmark-utils run-release-signal
uv run --locked benchmark-utils generate-summary --run-benchmarks --profile perf
uv run --locked benchmark-utils performance-local
uv run --locked benchmark-utils performance-github-assets
uv run --locked benchmark-utils performance-release
uv run --locked benchmark-utils performance-doc
uv run --locked publish-readme-performance
```

`benchmark-utils` handles Criterion baseline generation and packaging,
comparison, saved Criterion baseline reports, and release performance summaries.
`run-release-signal` executes the frozen target/section/group plan used by local
Just recipes, release CI, retained metadata, and strict summary coverage.
It formats and compares benchmark evidence; the harnesses being run are
responsible for failing before timings are published when scientific invariants
are violated.
Published releases package `baseline_results.txt` with raw Criterion data as a
GitHub Release asset for Ubuntu GitHub Actions comparisons. Local timing records
should stay in the ignored `baseline-artifact/` or `baseline-artifacts/`
directories. `bench-compare` renders `target/bench-reports/performance.md` from
existing Criterion `new` data and a saved baseline such as `last`.
`performance-local` and `performance-github-assets` generate isolated
release-to-release Markdown reports plus adjacent CSV and provenance JSON under
`target/bench-reports/`. New GitHub-asset reports require versioned measurement
metadata bound to the requested clean tag; existing legacy assets remain
loadable as provenance-limited absolute timing evidence. Acquisition is retained
separately from measurement provenance, and ratios are suppressed for these
separate hosted measurement sessions.
`performance-release` retains and reload-validates the local bundle before
promoting the curated report into `docs/PERFORMANCE.md`, archiving the previous
report, and copying the exact CSV/provenance bytes into
`docs/archive/performance/data/`. `performance-doc` consumes an existing
validated CSV/JSON pair and performs the same promotion without Cargo or
measurement worktrees; incomplete, invalid, stale, same-version, and
scientifically non-comparable pairs are rejected before documentation changes.
Promotion uses per-file atomic replacement with caught-failure rollback, so a
hard interruption requires inspection and an idempotent rerun. These release reports are evidence, not
routine pre-`just ci` checks; temp-worktree generation applies tracked checkout
changes but ignores untracked files. The default comparison report for release
baselines is `benches/main_vs_release_compare_results.txt`; the ref-comparison
guard writes `benches/worktree_vs_<ref>_compare_results.txt` and fails only on
total matched-time regressions or execution errors.

The versioned CSV is the canonical tabular release artifact: the datasets are
small, human-diffable audit records and remain usable without a dataframe
runtime. Jupyter notebooks may materialize derived Parquet caches for larger
analyses, but those caches are not promotion inputs and must be reproducible
from the validated CSV. Raw Criterion data remains in the release
`delaunay-vX.Y.Z-criterion-baseline.tar.gz` assets.

`publish-readme-performance` consumes the retained bundle after promotion and
atomically publishes the compact README table plus the canonical
`docs/assets/bench/release-performance.{csv,provenance.json}` pair. It never
runs Cargo or Criterion.

### Hardware utilities

```bash
uv run --locked hardware-utils info
uv run --locked hardware-utils kv
uv run --locked hardware-utils info --json
```

### Coverage workflow

```bash
just coverage-ci
```

`just coverage-ci` writes the Cobertura XML consumed by CI to
`coverage/cobertura.xml`.

### Tool-pin maintenance

Prefer the repository recipes for coordinated updates. The direct locked
Python-pin updater is also available for focused diagnosis:

```bash
just update-python-dependencies
uv run --locked update-tool-pins --help
uv run --locked update-python-dev-pins --help
```

`just update` upgrades the Cargo CLI packages owned by `setup-tools`, then runs
`update-tool-pins` to atomically reconcile their installed versions and the
active uv version with the root `justfile`. The updater validates the complete
managed package set and uv version before replacing the pin source, so missing
or malformed tool output leaves the existing declarations unchanged. uv remains
an external prerequisite: update it through its owning system package manager;
the repository update records that active version rather than replacing the uv
installation itself.

Before `uv.lock` is refreshed, `update-python-dev-pins` discovers exact simple
pins under `[dependency-groups].dev`, resolves their latest mutually compatible
universal set for the supported Python version, and replaces them together.
Ranged development requirements and runtime/build requirements remain outside
that rewrite.

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
