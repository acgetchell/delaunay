# Contributing to Delaunay

[![CI][ci-badge]][ci-workflow]
[![Coverage][coverage-badge]][coverage-dashboard]
[![OSV-Scanner][osv-badge]][osv-workflow]
[![Gitleaks][gitleaks-badge]][gitleaks-workflow]
[![zizmor][zizmor-badge]][zizmor-workflow]
[![Repository Rule SARIF][semgrep-badge]][semgrep-workflow]

Thank you for your interest in contributing to the [**delaunay**][delaunay-lib]
computational geometry library. This guide is the contributor-facing entry
point; detailed project rules live in the canonical documents linked below so
they do not drift out of sync.

<!-- Preserve links to the former Code of Conduct section. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="code-of-conduct"></a>

This project follows our [Code of Conduct][code-of-conduct]. Please report
unacceptable behavior to the [maintainer][maintainer-email].

## Table of Contents

- [Getting Started](#getting-started)
- [Contributor Workflow](#contributor-workflow)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Types of Contributions](#types-of-contributions)
- [Developer Documentation](#developer-documentation)
- [Release Process](#release-process)
- [Getting Help](#getting-help)
- [AI-Assisted Development](#ai-assisted-development)

## Getting Started

Install Git, the uv version declared in `pyproject.toml`, and the platform
compiler/linker prerequisites described in [the command guide][dev-commands].
The pinned shared package installs Python,
Rust, Cargo tools, and its own Just dependency directly from PyPI.

From the repository root, initialize the declared toolchain:

```bash
uv run --locked --managed-python --only-group tooling research-repo-tools setup
```

Open a new terminal if setup changed PATH. No shared-package checkout or generated
bootstrap launcher is needed. Repository Python tooling is described in
[docs/dev/python.md][dev-python] and [tooling/python/README.md][python-readme].

Build the development profile:

```bash
just setup
```

For library usage and command discovery, see the [README Quickstart](README.md#-quickstart).

## Contributor Workflow

Before starting work, check existing GitHub issues for related bug reports,
feature requests, or design discussion. Create a new issue when the expected
behavior, mathematical context, or proposed implementation needs review first.

Keep changes scoped, update tests and documentation with the behavior they
support, and check performance impact for Rust algorithm or benchmark changes.
During development, use `just check` alongside targeted tests for changed behavior.

| Command | Purpose |
|---------|---------|
| `just check` | Check code, configuration, documentation, formatting, and static analysis during development. |
| `just test` | Run the default Rust and Python test suites. |
| `just security` | Audit Rust and Python dependencies and scan reachable Git history and current files for secrets. |
| `just ci` | Run comprehensive final validation for core Rust/Cargo or public-behavior changes before PR submission. |

Use `just fix` to apply formatters and safe automatic fixes, then rerun the
relevant checks. `just ci` includes `just check` and `just test`; earlier runs
provide feedback while you work. Run `just security` separately: security scans
require network access and a checkout with full Git history.

Before submitting core Rust/Cargo or public-behavior changes, complete `just ci`.
Documentation, configuration, Python, notebook, test-only, benchmark-only, and
example-only changes use their focused validators; for example, Markdown-only
changes use `just check-docs`. See [the command guide][dev-commands] for the
authoritative validation matrix and security scan details.

Before committing edited notebooks, clear generated outputs and execution counts
with `just notebook-clear-outputs-all`; see [notebook policy](docs/dev/notebooks.md).

## Testing

New or changed functionality must include unit tests that validate its behavior,
plus integration tests where public contracts or interactions between components
need coverage. Each test should supply distinct evidence of correctness.

### Unit and Integration Tests

- **Unit tests** belong in the source file that owns the functionality, under
  `src/`, in a `#[cfg(test)] mod tests` block after all production items. Test
  internal algorithms, helpers, and module invariants at the lowest layer that
  owns them.
- **Integration tests** belong under `tests/`. Rust integration tests compile as
  separate crates and exercise the public API, including components used in
  combination, complete workflows, and caller-visible contracts. Keep fixed-bug
  regression cases in `tests/regressions.rs` unless separate crate configuration
  is required.

Test expected results, boundary cases, and error behavior with meaningful
assertions about mathematical, geometric, and topological invariants. Fallible
mutations need evidence for both valid success and complete rollback on failure.
Dimension-generic tests should cover 2D through 5D whenever feasible. Use seeded
randomness and property tests for reproducible geometric coverage. Preserve
regression inputs and assertions when fixing production failures.

Use `just test-unit` and `just test-integration` for the corresponding Rust test
suites, and `just test-doc` for Rustdoc examples. During iteration, run the
specific changed tests; use the matching focused suite for final validation.
See [the testing guide][dev-testing] for test organization, numerical comparisons,
dimension coverage, and slow-test policy.

### Coverage

The checked-in [Codecov configuration](.codecov.yml) sets these coverage targets:

- Project coverage: **90%**, with a **1% threshold**.
- Patch coverage: **70%** for changed lines.
- Benchmark and example files are excluded from Codecov calculations.

Inspect the Codecov PR report for uncovered changes and add tests that verify
the affected behavior. Coverage complements the correctness assertions above.
Use `just coverage` to generate a local HTML report; the Codecov workflow uses
`just coverage-ci` to generate coverage and test results before uploading them.

## Pull Requests

### Branch Names

Create focused branches for your work. Prefer
`{type}/{issue}-descriptor-or-two`, where `{issue}` is the GitHub issue number
when one exists. Use a concise type aligned with the change: `fix`, `feat`,
`perf`, `doc`, `test`, `refactor`, `ci`, `build`, `chore`, or `style`.

```bash
git checkout -b fix/307-oriented-flips
git checkout -b perf/315-bench-profile
git checkout -b doc/329-branch-guidance
```

Follow the [Git and GitHub rules](docs/dev/git.md) for branches, commits, pushes,
and tags; release publication follows the dedicated release guide.

<!-- Preserve links to the former Commit Message Format section. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="commit-message-format"></a>

### PR Titles and Commit Messages

Use conventional PR titles and commit messages so release tooling can generate
useful changelog entries:

```text
type(scope): short description

- Explain the important behavior or maintenance change.
- Include issue or PR references when useful.
```

Common types are `feat`, `fix`, `perf`, `refactor`, `build`, `ci`, `docs`,
`test`, `style`, and `chore`. Use a breaking-change marker when a public API or
behavior changes incompatibly:

```text
feat!: redesign vertex import API

BREAKING CHANGE: callers must now pass validated vertex fixtures.
```

PR titles should use the same conventional format because merge commits appear
in generated changelogs.

<!-- Preserve links to the former Submitting Changes section. -->
<!-- markdownlint-disable-next-line MD033 -->
<a id="submitting-changes"></a>

### PR Descriptions

Open pull requests with a descriptive conventional title and a concise summary
of:

- Problem: what issue or behavior the change addresses
- Solution: how the change handles it
- Testing: which validators, tests, or benchmarks were run
- Performance: relevant measurements or rationale for algorithmic changes

PRs are evaluated for correctness, mathematical accuracy, tests, documentation,
style, and performance impact. Non-substantive whitespace churn or formatting
noise may be declined unless it is part of an intentional tooling cleanup.

## Types of Contributions

We welcome bug fixes, new features, documentation improvements, tests,
benchmarks, performance work, and infrastructure improvements. For algorithmic
or numerical work, cite relevant literature in [REFERENCES.md][references] and
document assumptions, invariants, and known limitations.

## Developer Documentation

| Topic | Canonical reference |
|-------|---------------------|
| Agent rules and repository invariants | [`AGENTS.md`](AGENTS.md) |
| Benchmarks, baselines, and performance workflows | [benches/README.md][benches-readme] |
| Citations and bibliography | [CITATION.cff][citation], [REFERENCES.md][references] |
| Command recipes and validation workflow | [docs/dev/commands.md][dev-commands], `just --list` |
| Construction, validation layers, and invariants | [docs/construction_and_validation.md][construction-validation], [docs/invariants.md][invariants] |
| Examples | [examples/README.md][examples-readme] |
| Project layout and module architecture | [docs/code_organization.md][code-organization] |
| Python tooling and support scripts | [docs/dev/python.md][dev-python], [tooling/python/README.md][python-readme] |
| Releases and changelog generation | [docs/RELEASING.md][releasing], [CHANGELOG.md][changelog] |
| Rust style, docs, and API expectations | [docs/dev/rust.md][dev-rust] |
| Testing conventions and adversarial input guidance | [docs/dev/testing.md][dev-testing] |

## Release Process

The project follows [semantic versioning][semver] and generates changelog files
from commits. Do not update release artifacts by hand; follow the release
workflow in [docs/RELEASING.md][releasing].

## Getting Help

Use GitHub Issues for bug reports and feature requests, and GitHub Discussions
for general questions.

When asking for help, include the Rust version, crate version, operating system,
a minimal reproduction when possible, expected behavior, actual behavior, and
the commands you already tried.

## AI-Assisted Development

This repository contains an [`AGENTS.md`](AGENTS.md) file, which defines the
canonical rules and invariants for AI coding assistants and autonomous agents
working on this codebase.

AI tools, including ChatGPT, Claude, CodeRabbit, Codex, KiloCode, and WARP, are
expected to read and follow `AGENTS.md` and its task-relevant linked guidance
when proposing or applying changes. Agents require explicit maintainer
authorization before version-control mutations; see the
[Git and GitHub rules](docs/dev/git.md).

CodeRabbit review requires explicit maintainer authorization and is not a
readiness requirement. When authorized, run `just review` against the intended
PR base and address valid findings before final validation. See
[local CodeRabbit review](docs/dev/commands.md#local-coderabbit-review) for
scope selection and prerequisites.

Portions of this library were developed with the assistance of these tools:

- [ChatGPT](https://openai.com/chatgpt)
- [Claude](https://www.anthropic.com/claude)
- [CodeRabbit](https://coderabbit.ai/)
- [Codex](https://openai.com/codex/)
- [KiloCode](https://kilocode.ai/)
- [WARP](https://www.warp.dev)

All AI-assisted work must be reviewed and validated by a human maintainer before
it is merged.

For full tool citation metadata, see the
[AI-Assisted Development Tools](REFERENCES.md#ai-assisted-development-tools)
section of [`REFERENCES.md`](REFERENCES.md).

<!-- Links -->
[ci-badge]: https://github.com/acgetchell/delaunay/actions/workflows/ci.yml/badge.svg?branch=main
[ci-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/ci.yml
[coverage-badge]: https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO
[coverage-dashboard]: https://codecov.io/gh/acgetchell/delaunay
[gitleaks-badge]: https://github.com/acgetchell/delaunay/actions/workflows/gitleaks.yml/badge.svg?branch=main
[gitleaks-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/gitleaks.yml
[osv-badge]: https://github.com/acgetchell/delaunay/actions/workflows/osv.yml/badge.svg?branch=main
[osv-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/osv.yml
[semgrep-badge]: https://github.com/acgetchell/delaunay/actions/workflows/semgrep-sarif.yml/badge.svg?branch=main
[semgrep-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/semgrep-sarif.yml
[zizmor-badge]: https://github.com/acgetchell/delaunay/actions/workflows/zizmor.yml/badge.svg?branch=main
[zizmor-workflow]: https://github.com/acgetchell/delaunay/actions/workflows/zizmor.yml
[delaunay-lib]: https://github.com/acgetchell/delaunay
[code-of-conduct]: CODE_OF_CONDUCT.md
[changelog]: CHANGELOG.md
[examples-readme]: examples/README.md
[benches-readme]: benches/README.md
[python-readme]: tooling/python/README.md
[code-organization]: docs/code_organization.md
[dev-commands]: docs/dev/commands.md
[dev-python]: docs/dev/python.md
[dev-rust]: docs/dev/rust.md
[dev-testing]: docs/dev/testing.md
[construction-validation]: docs/construction_and_validation.md
[invariants]: docs/invariants.md
[citation]: CITATION.cff
[references]: REFERENCES.md
[releasing]: docs/RELEASING.md
[maintainer-email]: <mailto:adam@adamgetchell.org>
[semver]: https://semver.org/
