# Python tooling

This directory contains Delaunay-specific Python modules. Shared infrastructure
comes from the pinned `research-repo-tools` package. Prefer `just` recipes for validation and tests, and use the
`uv run --locked ...` entrypoints documented here when invoking an individual utility
directly.

## Prerequisites

Install the declared uv version, Git, and platform compiler/linker prerequisites.
Initialize the pinned PyPI toolchain with:

```bash
uv run --locked --managed-python --only-group tooling python tooling/python/shared_tooling.py setup
```

The installed package and tooling/notebook groups pin
`research-repo-tools==0.2.4`. Shared CLI commands and documented Python APIs own
setup, maintenance, source inventory, subprocesses, archive extraction, worktree
lifecycle, Criterion parsing, file transactions and notebook infrastructure.
Retained adapters own Delaunay science and policy. The
[adoption inventory](../../docs/dev/tooling_adoption.md#remaining-shared-tool-gaps)
records the v0.1.8 integrations and v0.1.9 performance adoption.

The temporary `shared_tooling.py` adapter changes only scanner selection for
setup and Semgrep commands, preserving the locked Python 3.15 environment and
all shared validation gates. It rejects other shared-package versions and restores
its process-local overrides. Remove it after the published shared-baseline fix
is adopted under [#673](https://github.com/acgetchell/delaunay/issues/673).

## CLI entrypoints

Consumer-owned commands are exposed by `pyproject.toml`; all support `--help`.

### Shared maintenance

```bash
just changelog
just changelog-preview
just changelog-release vX.Y.Z
just changelog-preview --tag vX.Y.Z
just changelog-archive
just release-notes vX.Y.Z
just docs-version-check
just release-version-check
just update-version vX.Y.Z --date YYYY-MM-DD --previous-release vA.B.C
just tag vX.Y.Z
uv run --locked --group dev research-repo-tools --help
```

Cargo promotion uses `just release-artifact create OUTPUT` in trusted main CI
after the default/all-feature package builds. `just release-artifact verify`
checks exact-commit native evidence and byte-identical repackaging during release
verification and after deployment approval. These commands never upload a crate;
see [release gates and recovery](../../docs/RELEASING.md).
`just release-evidence` inspects the retained package before publication without
Cargo or a release event. `release-publish` requires that inspection, while
workflow verification retains its stricter event and reproduction gates.

`release_changelog.py` is a temporary generic adapter for the citation-date
default used by `changelog-release` and tagged previews.
v0.2.3 already owns generation, but requires an explicit
prospective date. [research-repo-tools #130](https://github.com/acgetchell/research-repo-tools/issues/130)
tracks the shared replacement for this adapter and la-stack's equivalent.
After adopting that published capability, recipes should call it directly and
this adapter should be removed.

Changelog generation normalizes history and retains Unreleased and the latest
minor series in the root, with older series under `docs/archives/changelog/`.
Metadata preparation is separate. `update-version` defaults to UTC today
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
just bench-latest
just bench-save-baseline last
just bench-compare last
just bench-perf-summary
just performance-local
just performance-prepare
just performance-release
just performance-readme --preview
just performance-readme
just performance-doc
just performance-github-assets
```

Fresh paired workflows call the installed `research-repo-tools` schema-4 engine
through `tooling/measurement-release-signal.toml`. Read-only exports preserve
each revision's own native harness. Discovery, gates, exact per-case counts,
saved-copy verification and full raw retention precede publication of
`target/bench-reports/performance.run.json` and its evidence envelope.
The separate topology declaration preserves the 12-sample 4D group.

Both complete phases survive scratch cleanup. Incompatible identities preserve
absolute timings and union coverage while suppressing ratios. Release promotion
is stricter and requires distinct labels and compatible evidence. Preparation
captures independent release/ref assertions; acquisition/promotion and README
publication remain separate retry boundaries. The report and README selections
live in consumer TOML, with immutable history under `docs/performance-runs/`.
Existing reports, timing payloads and archives remain unchanged.

The scientific single-checkout adapter still owns construction counters,
summary interpretation, named Criterion baselines and tagged archive metadata.
The published complete acquisition command requires a pair; running it for
`bench-latest` would repeat measurements. The historical asset adapter retains
Delaunay's nested count/coverage and hosted-session checks. It now reads into
private scratch without creating Git worktrees, and cannot promote old evidence
as a complete run. The old paired execution, promotion and README Python paths
are retired. See the [ownership record](../../docs/dev/tooling_adoption.md#declarative-independent-performance-adoption).

`release_benchmarks.py` retains the workflow's captured release ID and tag commit,
checks the exact benchmarked checkout and refuses to overwrite existing baseline
assets. Upload and explicit publication use the shared `ReleaseTarget` API.

The six declarations ship in the Python wheel/sdist and install under
`share/delaunay/tooling/`. They remain consumer-owned resources; commands run
from a source checkout with its native harnesses and reviewed local tags.

### Profiling metadata

```bash
uv run --locked delaunay-profile-metadata
```

The command captures shared host and source metadata into
`profiling-results/environment.json`. `PROFILE_METADATA_TITLE`,
`PROFILE_METADATA_MODE`, `PROFILE_METADATA_FILTER` (or `BENCH_FILTER_VALUE`),
and `GITHUB_REF_NAME` supply consumer labels. It probes the environment without
running benchmarks.

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

The aggregate upgrades tools before dependencies, audits all maintained final
lockfiles, and stops on failure. Routine tool updates do not rerun bootstrap or
shell setup. Cargo
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
