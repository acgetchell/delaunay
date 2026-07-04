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

Treat this as a menu, not a required sequence. The validation matrix below is
the handoff source of truth.

## Validation Command Selection

Use the smallest non-mutating validator that covers the files you changed while
iterating. For final handoff validation, match commands to the changed file
surfaces instead of defaulting all edits to full CI.

Core Rust code means production Rust or manifest changes that can affect library
behavior, public API, features, examples, benchmarks, or downstream users. It
does not include Rust doctest-only, unit-test-only, integration-test-only,
benchmark-only, or example-only edits when the focused validator covers the
changed surface.

| Touched surface | Iteration validation | Final validation |
|-----|-----|-----|
| Markdown documentation (`*.md`) | `just markdown-check` | `just markdown-ci` |
| Python under `scripts/` | Targeted pytest or `just test-python`; add `just python-check` for logic/style | `just python-check` and `just test-python` |
| Jupyter notebooks (`notebooks/**/*.ipynb`) | `just notebook-lint` | `just notebook-check` |
| Configuration only (JSON, TOML, YAML, CFF, workflows) | Matching config validator | `just lint-config` |
| Rust unit tests only (`#[cfg(test)]` in `src/**`) | Targeted `cargo test --lib <filter>` or `just test-unit` | `just test-unit` |
| Rust doctests only (`///` examples or crate docs) | Targeted `cargo test --doc --release <filter>` or `just test-doc` | `just test-doc` |
| Rust integration tests only (`tests/**`) | Targeted `cargo nextest run --test <name>` or `just test-integration-fast` | `just test-integration` |
| Rust benchmark files only (`benches/**`) | Targeted benchmark command or `just bench-smoke` | Matching benchmark validator |
| Rust examples only (`examples/**`) | Targeted `cargo run --example <name>` or `just examples` | `just examples` |
| Core Rust code | Focused checks or targeted tests while iterating | `just ci` |
| Mixed focused surfaces without core Rust | Run each matching focused validator once | Run each matching focused validator once |
| Mixed core Rust plus tests/benches/examples/docs/config | Focused checks while iterating | `just ci` |

Do not run `just ci` merely because documentation, configuration, Python,
notebook, or test-only Rust files changed. Do not run `just test` when a single
focused test bucket covers the change unless you intentionally want the full
default test suite. When a diff touches multiple focused test surfaces, compose
the matching recipes once each; for example, run `just test-doc` and
`just test-integration` for doctest plus integration-test changes. Broad Rust
correctness workflows such as `just test-rust` and `just ci` use
`just test-rust-ci`, which runs Rust lib unit tests and integration tests
together in one release-profile nextest invocation.

During fast code-writing cycles, start with the smallest changed test or
doctest rather than the whole focused bucket. For single-item rustdoc edits, run
`cargo test --doc --release <item-or-module-filter>`; for unit-test edits, run
`cargo test --lib <test-or-module-filter>`; for integration-test crate edits,
run the changed crate with `cargo nextest run --test <crate>`. Cargo's built-in
test-name filter accepts one filter per invocation, so run separate filtered
commands for unrelated changed tests or choose one shared module/name prefix
that covers the intended small group. Use `just test-doc`, `just test-unit`, or
`just test-integration` for final bucket validation or broad changes.

Focused validators own one target class. Avoid adding a compile-only smoke
recipe before a recipe that already compiles and runs the same target class.

For benchmark-only changes, run the changed benchmark with
`cargo bench --profile perf --bench <name>` when the change affects measured
behavior. Use `just bench-smoke` for harness-only edits, and `just bench` for
broad benchmark-suite changes.

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

`rust-core-check` runs all-targets Clippy in the default and all-features
configurations. This intentionally includes the all-targets, all-features
surface uploaded by the PR Clippy SARIF workflow, so `just ci` fails locally on
the same warning classes that would become GitHub code-scanning annotations.

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

Before core Rust changes, broad API-affecting changes, release-style
validation, or explicit maintainer requests, run the full CI command:

```bash
just ci
```

This runs:

- formatting checks
- GitHub Actions checks
- Markdown checks
- JSON/TOML/YAML/CFF checks
- Python lint/typecheck
- notebook hygiene and fast headless execution
- Rust core lint, documentation, and Semgrep checks
- benchmark harness compile checks
- Rust lib unit tests
- Rust doctests
- Rust release integration tests
- Python tests
- example builds

---

## Benchmark Profiles

For performance-sensitive code changes, follow
[`perf-tuning.md`](perf-tuning.md): benchmark before editing, add a benchmark
when none covers the hot path, benchmark after editing, and preserve
correctness invariants throughout.

`just ci` is the comprehensive error-catching validation path used by GitHub
Actions. It is a flat union of leaf validators rather than a nested call to
`just check`. The target classes are kept separate: `rust-core-check` covers
formatting, all-targets Clippy, rustdoc, and Semgrep; `bench-compile` compiles
benchmark harnesses once; `test-rust-ci` compiles and runs Rust lib unit tests
and release integration tests in one release-profile nextest invocation;
`test-doc` compiles and runs Rust doctests once in release profile;
`notebook-check` lints notebooks and executes fast notebooks headlessly once.

`just test` is tests-only. `test-integration-compile` and `bench-test-compile`
are explicit no-run smoke recipes for cases where a compile-only check is the
desired validator; do not run them before `test-integration` unless you
intentionally want a separate compile-only pass. `test-unit` and
`test-integration` stay focused for targeted local validation; broad test and
CI workflows use `test-rust-ci` to avoid a debug-plus-release profile split.

```bash
just ci
just test
just rust-core-check
just test-rust-ci
just notebook-check
just bench-compile
just test-integration-compile
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
just pachner-stress
just pachner-stress-3d
just pachner-stress-4d
cargo bench --profile perf --bench ci_performance_suite
```

The `perf` profile inherits from release and restores ThinLTO with one codegen
unit. Use it for measured benchmark output; `just ci` does not need it to catch
compile, lint, test, documentation, example, or benchmark-harness build errors.
Use `just bench-smoke` only for quick harness validation with minimal samples;
do not treat smoke output as performance data.

Workspace-wide benchmark recipes (`just bench`, `just bench-smoke`,
`just bench-compile`, and the benchmark compile step inside `just ci`) enable
`--features bench` so feature-gated benchmark fixtures are compiled.

Use `just pachner-stress [attempts] [validate_every] [samples]` for the full
manual 3D+4D Pachner Monte Carlo diagnostic run. The dimension-specific
`just pachner-stress-3d` and `just pachner-stress-4d` recipes default to
100,000 attempted moves per Criterion sample and enable report lines so long
chains can be diagnosed without making the workflow part of routine CI.

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

## Notebook Validation

Notebook source files should not commit generated outputs or execution counts.
Notebook code is extracted and checked with Ruff and ty so `.ipynb` cells follow
the same Python standards as repository scripts.

Commands:

```bash
just notebook-lint
just notebook-check
just notebook-check-slow
just notebook-clear-outputs-all
```

`notebook-check` runs notebook hygiene and fast headless execution. Headless
execution writes executed notebooks under `target/notebooks/` and leaves source
notebooks unchanged. Slow notebooks should be named `*_slow.ipynb` or placed
under `notebooks/slow/`; those run through `notebook-check-slow`.

Before notebooks exist, these recipes are clean no-ops so the CI shape can stay
stable ahead of notebook work.

---

## Markdown Checks

Markdown files are checked with rumdl and spell-checking for handoff. Keep the
non-mutating check before the mutating fixer in user-facing command examples.

Commands:

```bash
just markdown-ci
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
| Validate Markdown-only changes | `just markdown-ci` |
| Validate configuration-only changes | `just lint-config` |
| Validate Python scripts/tests | `just python-check` and `just test-python` |
| Validate notebook changes | `just notebook-check` |
| Validate core Rust checks | `just rust-core-check` |
| Run all default test buckets | `just test` |
| Run Rust tests only | `just test-rust` |
| Run Rust CI nextest bucket | `just test-rust-ci` |
| Run Rust lib unit tests only | `just test-unit` |
| Run doctests only | `just test-doc` |
| Run integration tests | `just test-integration` |
| Run all tests | `just test-all` |
| Compile benchmark harnesses | `just bench-compile` |
| Compile release integration tests without running | `just test-integration-compile` |
| Run examples | `just examples` |
| Run full GitHub-equivalent CI | `just ci` |
| Run perf-profile benchmarks | `just bench` |

---

## CI Expectations

CI enforces:

- GitHub Actions checks
- Markdown, JSON, TOML, YAML, CFF, and spell checks
- Python lint, type checks, and tests
- notebook hygiene and fast headless execution
- core Rust formatting, Clippy, rustdoc, and Semgrep checks
- Rust unit, doctest, and integration tests
- benchmark harness compilation
- examples

Rust warnings are denied by the manifest lint policy and Clippy warnings are
denied by the `just clippy` / `just clippy-all-targets` invocations. Keep any
intentional warning-level exceptions explicit in `Cargo.toml`.

Agents must ensure changes pass the appropriate local validator before
proposing patches. Use the validation matrix above for final handoff: core
Rust/Cargo changes require `just ci`, while documentation, configuration,
Python, test-only, benchmark-only, and example-only changes use their focused
validators and compose them once each when multiple surfaces changed.

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
