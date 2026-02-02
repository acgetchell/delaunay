# AGENTS.md

Essential guidance for AI assistants working in this repository.

## Core Rules

### Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git commands that modify version control state
- **ALLOWED**: Run read-only git commands (e.g. `git --no-pager status`, `git --no-pager diff`,
  `git --no-pager log`, `git --no-pager show`, `git --no-pager blame`) to inspect changes/history
- **ALWAYS** use `git --no-pager` when reading git output
- Suggest git commands that modify version control state for the user to run manually

### Commit Messages

When user requests commit message generation:

1. Run `git --no-pager diff --cached --stat`
2. Generate conventional commit format: `<type>: <brief summary>`
3. Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include body with organized bullet points and test results
5. Present in code block (no language) - user will commit manually

### Code Quality

- **ALLOWED**: Run formatters/linters: `cargo fmt`, `cargo clippy`, `cargo doc`, `taplo fmt`, `taplo lint`,
  `uv run ruff check --fix`, `uv run ruff format`, `shfmt -w`, `shellcheck -x`, `npx markdownlint --fix`,
  `npx cspell lint`, `actionlint`
- **NEVER**: Use `sed`, `awk`, `perl` for code edits
- **ALWAYS**: Use `apply_patch` for edits (and `create_file` for new files)
- **EXCEPTION**: Shell text tools OK for read-only analysis only

### Validation

- **JSON**: Validate with `jq empty <file>.json` after editing (or `just validate-json`)
- **TOML**: Lint/format with taplo: `just toml-lint`, `just toml-fmt-check`, `just toml-fmt` (or validate parsing with `just validate-toml`)
- **GitHub Actions**: Validate workflows with `just action-lint` (uses `actionlint`)
- **Spell check**: Run `just spell-check` (or `just lint-docs`) after editing; add legitimate technical terms to
  `cspell.json` `words` array (don't spell-check `cspell.json` itself)
- **Shell scripts**: Run `shfmt -w scripts/*.sh` and `shellcheck -x scripts/*.sh` after editing

### Rust

- Integration tests in `tests/*.rs` are separate crates; add a crate-level doc comment (`//! ...`) at the top to satisfy clippy `missing_docs` (CI uses `-D warnings`).

### Python

- Use `uv run` for all Python scripts (never `python3` or `python` directly)
- Use pytest for tests (not unittest)
- **Type checking**: `just python-lint` includes ty + mypy (blocking - all code must pass type checks)
- Add type hints to new code

## Common Commands

```bash
just fix              # Apply formatters/auto-fixes (mutating)
just check            # Lint/validators (non-mutating)
just ci               # Full CI run (checks + all tests + examples + bench compile)
just ci-slow          # CI + slow tests (100+ vertices)
just lint             # All linting
just test             # Lib and doc tests
just test-integration # Integration tests (includes proptests)
just test-all         # All tests (Rust + Python)
just examples         # Run all examples
```

### Changelog

- Never edit `CHANGELOG.md` directly - it's auto-generated from git commits
- Use `just changelog` to regenerate

## Project Context

- **Rust** d-dimensional Delaunay triangulation library (MSRV 1.93.0, Edition 2024)
- **No unsafe code**: `#![forbid(unsafe_code)]`
- **Architecture**: Generic with `const D: usize` for dimension (tested 2D-5D)
- **Modules**: `src/core/` (data structures), `src/geometry/` (predicates)
- **When adding/removing files**: Update `docs/code_organization.md`

## Test Execution

- **tests/ changes**: Run `just test-integration` (or `just ci`)
- **examples/ changes**: Run `just examples`
- **benches/ changes**: Run `just bench-compile`
- **src/ changes**: Run `just test`
- **Any Rust changes**: Run `just doc-check`
