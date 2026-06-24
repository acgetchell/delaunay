# Development Commands

Development commands and validation steps for the repository.

Agents must run appropriate checks after modifying code.

---

## Contents

- [Core Workflow](#core-workflow)
- [Validation Command Selection](#validation-command-selection)
- [Justfile Usage](#justfile-usage)
- [Formatting](#formatting)
- [Linting](#linting)
- [Documentation Validation](#documentation-validation)
- [Full CI Validation](#full-ci-validation)
- [Benchmark Profiles](#benchmark-profiles)
- [Examples](#examples)
- [Spell Checking](#spell-checking)
- [Markdown Checks](#markdown-checks)
- [TOML Checks](#toml-checks)
- [YAML Checks](#yaml-checks)
- [Shell Script Validation](#shell-script-validation)
- [JSON Validation](#json-validation)
- [CITATION.cff Validation](#citationcff-validation)
- [GitHub Actions Validation](#github-actions-validation)
- [Recommended Command Matrix](#recommended-command-matrix)
- [CI Expectations](#ci-expectations)
- [Changelog](#changelog)

---

## Core Workflow

Typical development loop:

```bash
just check
just check-fast
just fix
just test
just ci
```

These commands ensure:

- formatting
- linting
- static analysis
- tests

## Validation Command Selection

Use the smallest non-mutating validator that covers the files you changed while
iterating. For final handoff validation, Rust changes use `just ci`; docs,
configuration, and Python-only changes stay on the narrower validators below.

| Touched surface | Iteration validation | Final validation |
|-----|-----|-----|
| Documentation or configuration only | `just check` | `just check` |
| Python-only changes under `scripts/` | `just python-check` | `just python-check` |
| Rust/Cargo/examples/benches/tests | Focused checks, `just check`, or targeted tests | `just ci` |

Do not run `just ci` merely because documentation, configuration, or Python
files changed. Run `just ci` before final handoff when Rust code changed or
when the maintainer explicitly asks for full CI.

## Justfile Usage

This repository standardizes development tasks through the `justfile`.

Agents should **prefer running `just` commands instead of invoking the
underlying tools directly**. The justfile ensures the correct flags,
configuration, and tool ordering are used.

Examples:

- prefer `just check` instead of running `cargo clippy` directly
- prefer `just fix` instead of running `cargo fmt` directly
- prefer `just ci` instead of manually running multiple validation steps when
  full CI is the right validation level

Direct tool invocation should only be used when a corresponding `just`
command does not exist.

---

## Formatting

Rust formatting checks are non-mutating:

```bash
just fmt-check
```

Apply formatting through:

```bash
just fix
```

Run checks before mutating fixers; formatting drift should be understood before
`just fix` rewrites files.

---

## Linting

Lint checks include:

```bash
cargo clippy
```

Repository warnings are denied through the manifest lint policy in
`Cargo.toml`; explicitly configured lint exceptions remain warnings. Clippy
invocations also deny warnings.

Run via:

```bash
just check
```

`just check` is the non-mutating lint/validator bundle. It does not run tests,
examples, or benchmarks.

`just check-fast` is the cheapest compile-only check:

```bash
just check-fast
```

`just check` runs the default checks plus an `--all-features` pass. DenseSlotMap
is the only supported storage backend.

---

## Documentation Validation

Documentation must build successfully.

Verify with:

```bash
just doc-check
```

or

```bash
cargo doc
```

---

## Full CI Validation

Before large Rust changes, broad API/test/benchmark changes, release-style
validation, or explicit maintainer requests, run the full CI command:

```bash
just ci
```

This runs:

- formatting checks
- lint checks
- benchmark and release-test compile checks using Cargo's default release profile
- unit tests
- integration tests
- documentation builds
- example builds

---

## Benchmark Profiles

For performance-sensitive code changes, follow
[`perf-tuning.md`](perf-tuning.md): benchmark before editing, add a benchmark
when none covers the hot path, benchmark after editing, and preserve
correctness invariants throughout.

`just ci` is the comprehensive error-catching validation path. It runs the
`check`, `test`, and `examples` recipes. The `test` recipe already depends on
`bench-test-compile`, so CI compiles benchmark harnesses and release tests
through that path; `bench-compile` is a standalone recipe that is **not**
executed by `just ci`:

```bash
just ci
just test              # includes just bench-test-compile
just bench-test-compile
```

Commands that run benchmarks and produce performance data use the `perf`
profile:

```bash
just bench
just bench-ci
just perf-baseline
just perf-compare
just perf-vs-ref
just perf-no-regressions
just bench-perf-summary
cargo bench --profile perf --bench ci_performance_suite
```

The `perf` profile inherits from release and restores ThinLTO with one codegen
unit. Use it for measured benchmark output; `just ci` does not need it to catch
compile, lint, test, documentation, example, or benchmark-harness build errors.
Use `just bench-smoke` only for quick harness validation with minimal samples;
do not treat smoke output as performance data.

Workspace-wide benchmark recipes (`just bench`, `just bench-smoke`,
`just bench-compile`, and the benchmark compile step inside `just test`) enable
`--features bench` so feature-gated benchmark fixtures are compiled.

Some repair benchmarks need feature-gated fixtures that deliberately construct
invalid-but-structurally-coherent topology. Run those harnesses with
`--features bench`; the `bench` feature exists only for benchmark fixtures and
must not expose normal construction escape hatches:

```bash
cargo bench --profile perf --features bench --bench pl_manifold_repair -- --noplot
```

Use `just perf-large-scale-smoke [max_secs]` for a coarse local wall-clock guard
over the release-mode large-scale debug harness. It runs the same 2D-5D defaults
as `just debug-large-scale-{2,3,4,5}d`, caps each test runtime at 60 seconds by
default, and reports all failing dimensions before exiting. It does not compare
against a baseline and should not be treated as benchmark data. Run it before
pushing Rust or benchmark changes to catch obvious local performance drift early.

Use `just bench-perf-summary` from the release PR branch after version and
documentation updates. It runs fresh perf-profile summary benchmarks, records
the current Criterion construction metadata and generated simplex counts, and
regenerates `benches/PERFORMANCE_RESULTS.md`.

Before pushing Rust or benchmark changes, run:

```bash
just ci
just perf-large-scale-smoke
```

For performance-sensitive changes and PR-ready work, also run:

```bash
just perf-no-regressions
```

## Slow Correctness Tests

The routine correctness suite has two buckets:

- `just test` runs default tests that should stay under roughly 10 seconds per
  test.
- `just test-slow` runs tests gated by the `slow-tests` feature when a
  deterministic correctness or regression case exceeds that budget.

`just test-slow` runs in release mode with the repository's `slow` nextest
profile. Debug-mode exact-predicate arithmetic can make high-dimensional tests
look like hangs, so slow correctness timing should be measured with the release
recipe. Deterministic slow tests should use `#[cfg(feature = "slow-tests")]`,
not `#[ignore]`.

`just perf-no-regressions` is the fuller local PR guard. It runs
`ci_performance_suite` with the shared dev-mode Criterion arguments against a
same-machine baseline generated from the current GitHub `main` ref. The guard
reuses a local cache under `baseline-artifacts/perf-no-regressions/` keyed by
the resolved `origin/main` commit and local Rust compiler version, and refreshes
that baseline when `main` or the compiler changes, or when the cached artifact
does not match the benchmark contract. The current worktree benchmark still runs
fresh each time so repeated comparisons can catch local performance drift.
The comparison report is written to
`benches/worktree_vs_main_compare_results.txt` by default so it is visibly a
branch/PR-vs-main check. The local guard exits nonzero only when benchmark
execution fails or total matched benchmark mean time regresses beyond the
threshold; individual benchmark regressions are warnings in the report. The
report also lists total, geomean, median, top regressions, and top improvements,
and the command prints a short terminal status with the report path.
`just clean` removes Criterion data under `target/`, but it does not remove this
local baseline cache.

```bash
just perf-no-regressions
```

To compare the current branch against a specific local release/ref baseline,
use `just perf-vs-ref`:

```bash
just perf-vs-ref v0.7.8
```

It uses the same cached same-machine baseline flow as `just perf-no-regressions`
but resolves and caches the requested ref, writes a
`benches/worktree_vs_<ref>_compare_results.txt` report, and treats overall total
matched-time regressions as failures while keeping individual benchmark
regressions as report warnings.

`just perf-baseline` is optional and intentionally persistent: use it only when
you want to create or refresh `baseline-artifact/baseline_results.txt` for later
manual same-machine comparisons. `baseline-artifact/` and
`baseline-artifacts/` are ignored by git so local timing records stay local. CI
regression checks now download the latest stable GitHub Release asset,
`delaunay-vX.Y.Z-criterion-baseline.tar.gz`, and compare the current
`ubuntu-latest` GitHub Actions run against that released-version Ubuntu
baseline.
`just perf-compare <file>` still writes
`benches/main_vs_release_compare_results.txt` by default. It follows the same
terminal-status convention, but remains stricter: individual benchmark
regressions still make release-style comparisons fail.

For lower-level workflows, `uv run benchmark-utils ensure-ref-baseline --ref
<ref> --dev` prints the cached/generated same-machine baseline path for a branch
or version tag, and `uv run benchmark-utils fetch-baseline --ref <ref>` downloads
the manual compatibility GitHub Actions artifact instead. Use the generated
local baseline for same-machine regression checks; use the downloaded artifact
only when you explicitly want CI-runner parity. `uv run benchmark-utils
compare-ref --ref <ref>` writes
`benches/worktree_vs_<ref>_compare_results.txt` unless `--output` is supplied.

To generate a scratch baseline without replacing the default artifact, write it
somewhere else and compare directly:

```bash
just perf-baseline-to /tmp/delaunay-main-baseline
just perf-compare /tmp/delaunay-main-baseline/baseline_results.txt
```

---

## Examples

Example programs live in:

```text
examples/
```

Validate with:

```bash
just examples
```

Examples must:

- compile
- run successfully
- demonstrate correct API usage

---

## Spell Checking

Documentation and comments are spell‑checked.

Run:

```bash
just spell-check
```

If a legitimate technical word fails:

Add it to:

```text
typos.toml
```

under:

```toml
[default.extend-words]
```

---

## Markdown Checks

Markdown files are checked and fixed with rumdl. Keep the non-mutating check
before the mutating fixer in user-facing command examples.

Commands:

```bash
just markdown-check
just markdown-fix
```

`just markdown-lint` remains as a compatibility alias for `just
markdown-check`.

---

## TOML Checks

TOML files should parse cleanly, pass Taplo linting, and match Taplo
formatting.

Commands:

```bash
just toml-check
just toml-lint
just toml-fmt-check
just toml-fix
```

---

## YAML Checks

YAML and `CITATION.cff` files should match the dprint/pretty_yaml formatting
configuration and pass yamllint.

Commands:

```bash
just yaml-check
just yaml-fix
```

`just yaml-check` runs both `just yaml-fmt-check` and `just yaml-lint`; use
`just yaml-fix` for the mutating dprint formatter.

---

## Shell Script Validation

Shell scripts must pass:

```text
shfmt
shellcheck
```

Run via CI or `just` commands.

---

## JSON Validation

JSON files should be validated after edits.

Run:

```bash
just json-check
```

---

## CITATION.cff Validation

Citation metadata should pass both YAML style linting and CFF schema
validation.

Run:

```bash
just citation-check
```

---

## GitHub Actions Validation

Workflows must pass `actionlint`.

Run with:

```bash
just action-lint
```

---

## Recommended Command Matrix

| Task | Command |
|-----|-----|
| Run lints | `just check` |
| Fast compile check | `just check-fast` |
| Check formatting | `just fmt-check` |
| Apply formatters/auto-fixes | `just fix` |
| Validate documentation/config-only changes | `just check` |
| Validate Python-only changes | `just python-check` |
| Run tests + compile smoke | `just test` |
| Run unit/doc tests only | `just test-unit` |
| Run integration tests | `just test-integration` |
| Run all tests | `just test-all` |
| Run examples | `just examples` |
| Run full CI | `just ci` |
| Run perf-profile benchmarks | `just bench` |

---

## CI Expectations

CI enforces:

- formatting
- clippy lints
- documentation build
- tests

Rust warnings are denied by the manifest lint policy and Clippy warnings are
denied by the `just clippy` invocations. Keep any intentional warning-level
exceptions explicit in `Cargo.toml`.

Agents must ensure changes pass the appropriate local validator before
proposing patches. Rust/Cargo/example/benchmark/test changes should pass
`just ci` for final handoff validation; documentation/config-only changes
should normally pass `just check`, and Python-only changes should normally pass
`just python-check`.

---

## Changelog

The changelog is **auto-generated**.

Never edit manually.

Regenerate with:

```bash
just changelog
```

This runs `git-cliff`, applies the Python postprocessor, archives completed
minor release series under `docs/archive/changelog/`, and applies `rumdl`
formatting to the generated changelog files.

For release PRs, generate the changelog for a version before the final tag
exists with:

```bash
just changelog-unreleased vX.Y.Z
```

Create annotated release tags from the generated changelog after the release PR
is merged with:

```bash
just tag vX.Y.Z
```
