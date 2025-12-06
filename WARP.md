# WARP.md

Essential guidance for AI assistants working in this repository.

## Core Rules

### Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git commands that modify version control state
- **ALWAYS** use `git --no-pager` when reading git output
- Suggest git commands for the user but never execute them

### Commit Messages

When user requests commit message generation:

1. Run `git --no-pager diff --cached --stat`
2. Generate conventional commit format: `<type>: <brief summary>`
3. Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include body with organized bullet points and test results
5. Present in code block (no language) - user will commit manually

### Code Quality

- **ALLOWED**: Run formatters/linters: `cargo fmt`, `cargo clippy`, `uvx ruff format/check --fix`, `shfmt -w`, `markdownlint --fix`
- **NEVER**: Use `sed`, `awk`, `perl` for code edits
- **ALWAYS**: Use `edit_files` tool for code changes
- **EXCEPTION**: Shell text tools OK for read-only analysis only

### Validation

- **JSON**: Validate with `jq empty <file>.json` after editing
- **TOML**: Validate with `uv run python -c "import tomllib; tomllib.load(open('<file>.toml', 'rb'))"` after editing
- **Spell check**: Run after editing any files; add legitimate technical terms to `cspell.json` `words` array (don't spell-check `cspell.json` itself)
- **Shell scripts**: Run `shfmt -w scripts/*.sh` and `shellcheck -x scripts/*.sh` after editing

### Python

- Use `uv run` for all Python scripts (never `python3` or `python` directly)
- Use pytest for tests (not unittest)

## Common Commands

```bash
just commit-check     # Pre-commit validation (use before pushing)
just ci               # Fast iteration (linting + tests + bench compile)
just lint             # All linting
just test             # Lib and doc tests
just test-all         # All tests
just examples         # Run all examples
```

### Changelog

- Never edit `CHANGELOG.md` directly - it's auto-generated from git commits
- Use `just changelog` to regenerate

## Project Context

- **Rust** d-dimensional Delaunay triangulation library (MSRV 1.91.0, Edition 2024)
- **No unsafe code**: `#![forbid(unsafe_code)]`
- **Architecture**: Generic with `const D: usize` for dimension (tested 2D-5D)
- **Modules**: `src/core/` (data structures), `src/geometry/` (predicates)
- **When adding/removing files**: Update `docs/code_organization.md`

## Test Execution

- **tests/ changes**: Run `just test-release` or `just test-debug`
- **examples/ changes**: Run `just examples`
- **benches/ changes**: Run `just bench-compile`
- **src/ changes**: Run `just test`
- **Any Rust changes**: Run `just doc-check`
