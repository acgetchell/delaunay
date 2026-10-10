# Python tooling

This directory contains Delaunay-specific Python modules. Shared infrastructure
comes from the pinned `research-repo-tools` package; shell scripts live in
`../../scripts/`. Prefer `just` recipes for validation and tests, and use the
`uv run --locked ...` entrypoints documented here when invoking an individual utility
directly.

## Prerequisites

Install the declared uv version, Git, and platform build prerequisites, including
Tectonic's native libraries. Initialize the pinned PyPI toolchain with:

```bash
tectonic_environment="$(uv run --locked --managed-python --only-group tooling research-repo-tools tectonic discover --format shell)"
eval "$tectonic_environment"
uv run --locked --managed-python --only-group tooling research-repo-tools setup
```

The installed package and tooling/notebook groups pin
`research-repo-tools==0.1.8`. Shared CLI commands and documented Python APIs own
setup, maintenance, source inventory, subprocesses, archive extraction, worktree
lifecycle, Criterion parsing, file transactions and notebook infrastructure.
Retained adapters own Delaunay science and policy. The
[adoption inventory](../../docs/dev/tooling_adoption.md#remaining-shared-tool-gaps)
records the completed v0.1.8 integrations and remaining performance contracts.

## CLI entrypoints

Consumer-owned commands are exposed by `pyproject.toml`; all support `--help`.

### Shared maintenance

```bash
just changelog
just changelog-preview
just changelog-release vX.Y.Z YYYY-MM-DD
just changelog-archive
just release-notes vX.Y.Z
just docs-version-check
just release-version-check
just update-version vX.Y.Z --date YYYY-MM-DD --previous-release vA.B.C
just tag vX.Y.Z
uv run --locked --group dev research-repo-tools --help
```

Changelog generation normalizes history and retains Unreleased and the latest
minor series in the root, with older series under `docs/archives/changelog/`.
`changelog-unreleased` aliases `changelog-release` and requires the same explicit
date. Metadata preparation is separate. `update-version` defaults to UTC today
and published stable GitHub history; explicit date and predecessor arguments
allow offline preparation. The final release gate requires a matching changelog
heading and citation date, while the DOI remains fixed by consumer policy.

### Notebook utilities

```bash
just notebook-check
just notebook-execute notebooks/00_quickstart.ipynb
just notebook-reset-from-git
just validation-doc-figures-check
uv run --locked --group dev --group notebooks research-repo-tools notebooks --help
```

`just notebook-check` invokes shared JSON/output validation, native Ruff and ty
notebook checks, advice, and the declared lowercase kebab-case ID pattern directly.
The Python CLI wrapper is deleted.
It never executes cells. `just notebook-execute` uses the shared locked kernel,
writes `target/notebooks/notebooks/<notebook-stem>.ipynb` and an adjacent
`.report.json`, and leaves the source notebook unchanged. Notebook-generated
figures and data retain `target/notebooks/<notebook-stem>/`.
`just notebook-reset-from-git` restores tracked source notebooks from the Git
index, or from an explicit source such as `HEAD`, and removes generated
notebook artifacts and Jupyter checkpoints.
`just notebook` launches JupyterLab with shared launch policy and scratch caches
under `target/notebooks/jupyter`.

`just validation-doc-figures-check` executes the validation notebook into
`target/` and compares its complete generated PNG set with the tracked
documentation artifacts without publishing changes. The canonical byte check
is composed into `just ci` on macOS.

`delaunay-scripts` is repository-internal and is not distributed as a PyPI
tool. Run shared notebook commands through the locked project environment or the `just`
recipes above. The repository-managed `dev` and `notebooks` dependency groups
provide its Ruff, ty, and nbclient backends. Notebook-specific imports used by
the notebook being executed remain the notebook author's responsibility.

### Benchmark utilities

```bash
uv run --locked benchmark-utils bench-compare last
uv run --locked benchmark-utils run-release-signal
uv run --locked benchmark-utils generate-summary --run-benchmarks --profile perf
uv run --locked benchmark-utils performance-local
uv run --locked benchmark-utils performance-github-assets
uv run --locked benchmark-utils performance-release
uv run --locked benchmark-utils performance-doc
uv run --locked publish-readme-performance
```

`benchmark-utils` selects Delaunay workloads, validates scientific coverage,
renders saved Criterion diagnostics and retains shared-format comparison evidence.
`run-release-signal` executes the frozen target/section/group plan used by local
Just recipes, release CI, retained metadata, and strict summary coverage.
It formats and compares benchmark evidence; the harnesses being run are
responsible for failing before timings are published when scientific invariants
are violated.
Published releases retain raw Criterion data and versioned metadata as GitHub
Release assets. `bench-compare` renders `target/bench-reports/performance.md` from
existing Criterion `new` data and a saved baseline such as `last`.
`performance-local` and `performance-github-assets` generate isolated
release-to-release Markdown reports plus adjacent shared comparison JSON and evidence under
`target/bench-reports/`. New GitHub-asset reports require versioned measurement
metadata bound to the requested clean tag; existing legacy assets remain
loadable as provenance-limited absolute timing evidence. Acquisition is retained
separately from measurement provenance, and ratios are suppressed for these
separate hosted measurement sessions.
`performance-release` retains and reload-validates the local bundle before
promoting the curated report into `docs/performance.md`, archiving the previous
report, and copying the exact comparison JSON/evidence bytes into
`docs/archive/performance/data/`. `performance-doc` consumes an existing
validated shared JSON evidence pair and performs the same promotion without Cargo or
measurement worktrees; incomplete, invalid, stale, same-version, and
scientifically non-comparable pairs are rejected before documentation changes.
Promotion uses per-file atomic replacement with caught-failure rollback, so a
hard interruption requires inspection and an idempotent rerun. These release reports are evidence, not
routine pre-`just ci` checks; temp-worktree generation uses the shared snapshot
API for tracked changes and nonignored new files, with exact bytes and permissions.

The canonical timing payload is the shared `research-repo-tools/criterion-comparison/v1`
JSON, paired with its digest-bound evidence envelope. Delaunay's context records
coverage and scientific eligibility. Ratios require matching measured harnesses,
plans, toolchains, confidence levels, and hosts. Timing changes are descriptive;
marginal timing intervals are not ratio intervals or significance tests.

Historical CSV/text files stay unchanged as records. Their readers and writers
are retired. New reports use `.comparison.json` and `.evidence.json`; old files
cannot be passed to the new promotion commands.

`publish-readme-performance` consumes the retained bundle after promotion and
atomically publishes the compact README table plus the canonical
`docs/assets/bench/release-performance.{comparison.json,evidence.json}` pair. It never
runs Cargo or Criterion.

### Hardware utilities

```bash
uv run --locked research-repo-tools performance host --output target/host.json
```

The standalone `hardware-utils` command is retired. The shared command captures
versioned host observations. The local hardware module and historical text/tolerance
rules are retired; measurement reports call the shared capture API directly.

### Coverage workflow

```bash
just coverage-ci
```

`just coverage-ci` writes the Cobertura XML consumed by CI to
`coverage/cobertura.xml`.

### Dependency and tool updates

```bash
just update
just update-dependencies
just update-cargo-dependencies
just update-python-dependencies
just update-tools
just tools-check
```

The aggregate upgrades tools before dependencies and stops on failure. Cargo
dependency updates retain both resolution roots. Python updates advance direct
exact dev pins, refresh the full lock, and explicitly sync dev with managed Rust
available. Included tooling pins stay fixed. Dependency-only recipes preserve
tool declarations; tool-only recipes preserve dependency requirements and locks.
uv upgrades through its installation owner. Just follows the shared package's
`rust-just` pin; managed Cargo upgrades replace cargo-update and the legacy
Just-variable reconciler. Change the shared package constraint explicitly through
uv, refresh the lock, review its release notes, and rerun setup.

## Declarative example checks

```bash
just examples
```

The shared validator consumes [`tooling/examples.toml`](../examples.toml),
which declares the release builds, feature policy, per-command timeouts, and
expected output. A Cargo metadata test keeps the example inventory complete.
See [`tooling/README.md`](../README.md) for the configuration contract.

## Linting and tests

```bash
just python-check
just python-typecheck
just test-python
just python-fix
```

## Maintenance expectations

- Keep scripts typed and covered by focused pytest tests.
- Use documented `research_repo_tools.process` runners; byte-preserving Git
  transport is required for binary diffs. Use shared diagnostic formatting for
  captured failures, whose streams may be bytes.
- Use `subprocess.CompletedProcess[str]` in tests instead of ad hoc mocks.
- Catch specific recoverable exception families; avoid broad
  `except Exception`.
- Update this README when adding, renaming, or removing `pyproject.toml`
  script entrypoints.
