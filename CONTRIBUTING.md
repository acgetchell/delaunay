# Contributing to Delaunay

Thank you for your interest in contributing to the [**delaunay**][delaunay-lib]
computational geometry library. This guide is the contributor-facing entry
point; detailed project rules live in the canonical documents linked below so
they do not drift out of sync.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [AI-Assisted Development](#ai-assisted-development)
- [Contributor Workflow](#contributor-workflow)
- [Project References](#project-references)
- [Commit Message Format](#commit-message-format)
- [Submitting Changes](#submitting-changes)
- [Types of Contributions](#types-of-contributions)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct][code-of-conduct]. By participating, you are expected to uphold
these standards. Please report unacceptable behavior to the
[maintainer][maintainer-email].

Our community is built on:

- Respectful collaboration in computational geometry research and development
- Inclusive participation regardless of background or experience level
- Excellence in scientific computing and algorithm implementation
- Open knowledge sharing about Delaunay triangulations and geometric algorithms

## Getting Started

Install [Rust via rustup][rustup], Git, Python, `uv`, and `just`. The pinned
Rust toolchain is declared in `rust-toolchain.toml`; Python development tooling
is described in [docs/dev/python.md][dev-python] and
[scripts/README.md][scripts-readme].

For the current command list and workflow details, use:

```bash
just --list
just help-workflows
```

A typical first local pass is:

```bash
just setup
just check
just test
```

Run `just ci` before opening or updating a pull request when the change is ready
for full validation. For command details, see [docs/dev/commands.md][dev-commands].

## AI-Assisted Development

This repository contains an [`AGENTS.md`](AGENTS.md) file, which defines the
canonical rules and invariants for AI coding assistants and autonomous agents
working on this codebase.

AI tools, including ChatGPT, Claude, CodeRabbit, Codex, GitHub Copilot,
KiloCode, and WARP, are expected to read and follow `AGENTS.md` when proposing
or applying changes.

Portions of this library were developed with the assistance of these tools:

- [ChatGPT](https://openai.com/chatgpt)
- [Claude](https://www.anthropic.com/claude)
- [CodeRabbit](https://coderabbit.ai/)
- [Codex](https://openai.com/codex/)
- [GitHub Copilot](https://github.com/features/copilot)
- [KiloCode](https://kilocode.ai/)
- [WARP](https://www.warp.dev)

All AI-assisted work must be reviewed and validated by a human maintainer before
it is merged.

For full tool citation metadata, see the
[AI-Assisted Development Tools](REFERENCES.md#ai-assisted-development-tools)
section of [`REFERENCES.md`](REFERENCES.md).

## Contributor Workflow

Before starting work, check existing GitHub issues for related bug reports,
feature requests, or design discussion. Create a new issue when the expected
behavior, mathematical context, or proposed implementation needs review first.

Create focused branches for your work. Prefer
`{type}/{issue}-descriptor-or-two`, where `{issue}` is the GitHub issue number
when one exists. Use a concise type aligned with the change: `fix`, `feat`,
`perf`, `doc`, `test`, `refactor`, `ci`, `build`, `chore`, or `style`.

```bash
git checkout -b fix/307-oriented-flips
git checkout -b perf/315-bench-profile
git checkout -b doc/329-branch-guidance
```

Keep changes scoped, update tests and documentation with the behavior they
support, and check performance impact for Rust algorithm or benchmark changes.
Automation must stop before version-control mutations; contributors should
perform commits, pushes, tags, and release operations manually.

## Project References

Use these documents as the source of truth instead of duplicating their content
in this guide:

| Topic | Canonical reference |
|-------|---------------------|
| Agent rules and repository invariants | [`AGENTS.md`](AGENTS.md) |
| Command recipes and validation workflow | [docs/dev/commands.md][dev-commands], `just --list` |
| Rust style, docs, and API expectations | [docs/dev/rust.md][dev-rust] |
| Testing conventions and adversarial input guidance | [docs/dev/testing.md][dev-testing] |
| Python tooling and support scripts | [docs/dev/python.md][dev-python], [scripts/README.md][scripts-readme] |
| Project layout and module architecture | [docs/code_organization.md][code-organization] |
| Examples | [examples/README.md][examples-readme] |
| Benchmarks, baselines, and performance workflows | [benches/README.md][benches-readme] |
| Tooling and CI alignment rationale | [docs/dev/tooling-alignment.md][tooling-alignment] |
| Validation layers and invariants | [docs/validation.md][validation], [docs/invariants.md][invariants] |
| Citations and bibliography | [CITATION.cff][citation], [REFERENCES.md][references] |
| Releases and changelog generation | [docs/RELEASING.md][releasing], [CHANGELOG.md][changelog] |

## Commit Message Format

Use conventional commits so release tooling can generate useful changelog
entries:

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

## Submitting Changes

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

## Release Process

The project follows [semantic versioning][semver] and generates changelog files
from commits. Do not update release artifacts by hand; follow the release
workflow in [docs/RELEASING.md][releasing].

## Getting Help

Use GitHub Issues for bug reports and feature requests, GitHub Discussions for
general questions, and [email][maintainer-email] for direct maintainer contact.

When asking for help, include the Rust version, crate version, operating system,
a minimal reproduction when possible, expected behavior, actual behavior, and
the commands you already tried.

---

**Questions?** Ask in GitHub Issues or reach out to the
[maintainer][maintainer-email].

<!-- Links -->
[delaunay-lib]: https://github.com/acgetchell/delaunay
[code-of-conduct]: CODE_OF_CONDUCT.md
[changelog]: CHANGELOG.md
[examples-readme]: examples/README.md
[benches-readme]: benches/README.md
[scripts-readme]: scripts/README.md
[code-organization]: docs/code_organization.md
[dev-commands]: docs/dev/commands.md
[dev-python]: docs/dev/python.md
[dev-rust]: docs/dev/rust.md
[dev-testing]: docs/dev/testing.md
[tooling-alignment]: docs/dev/tooling-alignment.md
[validation]: docs/validation.md
[invariants]: docs/invariants.md
[citation]: CITATION.cff
[references]: REFERENCES.md
[releasing]: docs/RELEASING.md
[maintainer-email]: <mailto:adam@adamgetchell.org>
[semver]: https://semver.org/
[rustup]: https://rustup.rs/
