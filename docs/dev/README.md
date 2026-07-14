# Development Guidance Index

This directory contains focused development rules for agents and maintainers.
Keep this file as a navigation index only; put detailed policy in the document
that owns the relevant workflow.

`AGENTS.md` remains the entry point for coding agents. `docs/code_organization.md`
remains the architecture hub.

## Documents

| File | Owns |
|-----|-----|
| [`commands.md`](commands.md) | Validation command selection, `just` recipes, benchmark profiles, and CI expectations |
| [`git.md`](git.md) | Git safety, GitHub CLI usage, issue dependencies, branch names, and commit-message rules |
| [`rust.md`](rust.md) | Rust guidance index; read its focused links before touching Rust code |
| [`testing.md`](testing.md) | Unit, integration, property, doctest, slow-test, and dimension-coverage expectations |
| [`perf-tuning.md`](perf-tuning.md) | Benchmark-before/after workflow for performance-sensitive Rust changes |
| [`docs.md`](docs.md) | Documentation ownership, scientific notation, references, changelog, and crates.io docs |
| [`notebooks.md`](notebooks.md) | Notebook cell IDs, source hygiene, deliberate execution, and tracked artifacts |
| [`python.md`](python.md) | Python support-script typing, subprocess mocks, exceptions, and parser contracts |
| [`debug_env_vars.md`](debug_env_vars.md) | `DELAUNAY_*` diagnostic and debugging environment variables |
| [`tooling-alignment.md`](tooling-alignment.md) | Cross-repository tooling comparison, rationale, and alignment notes |

## Maintenance

- Add new guidance here only as a link and one-line ownership note.
- Prefer updating an existing focused file over growing broad catch-all rules.
- Keep top-level files short enough for agents to parse in one pass; move
  detailed Rust policy under [`rust/`](rust/).
- When a command name or validation policy changes, update
  [`commands.md`](commands.md).
- When architecture or module ownership changes, update the focused docs linked
  from [`../code_organization.md`](../code_organization.md).
