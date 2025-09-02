# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## WARP-Specific Rules

### Git Operations

- **DO NOT** issue `git commit` or `git push` commands
- **DO NOT** use `git push --force` or modify tags (`git tag`, `git push --tags`)
- Let the user handle all git operations manually
- You may suggest git commands for the user to run, but never execute them
- This ensures the user maintains full control over version control operations

### Code Quality Tools

- **ALLOWED** to automatically run all code quality and formatting tools
- **ALLOWED** to fix auto-fixable issues (formatting, linting, etc.)
- This includes: `cargo fmt`, `cargo clippy`, `uvx ruff format`, `uvx ruff check --fix`, `markdownlint --fix`, `shfmt`, etc.
- Quality tools improve code without changing functionality or version control state

#### JSON File Validation (AI Assistant Guidance)

- **ALWAYS** validate JSON files after editing them
- **PREFERRED**: Use `jq empty <filename>.json` for quick validation
- **ALTERNATIVE**: Use `npx --yes jsonlint --quiet <filename>.json` for validation
- **REQUIRED** when modifying `cspell.json`, `package.json`, or any other JSON configuration files

#### Spell Check Dictionary Management (AI Assistant Guidance)

- **ALWAYS** run spell check after editing documentation files
- **IF** cspell reports legitimate technical terms, programming keywords, or project-specific terminology as misspelled, add them to the `words` array in `cspell.json`
- **EXAMPLES**: Python terms (`kwargs`, `args`, `asyncio`), Rust terms (`usize`, `clippy`, `rustc`), technical terms
(`triangulation`, `circumsphere`, `delaunay`), project names (`nalgebra`, `serde`, `thiserror`)
- **PURPOSE**: Maintains clean spell check while building comprehensive project dictionary

#### Import Organization (AI Assistant Guidance)

- **ALWAYS** use `uvx ruff check --fix --select F401,F403,I001,I002 scripts/` to fix import issues
- **AUTOMATICALLY** removes duplicate imports, unused imports, and organizes import order
- **PREFERRED** over manual import cleanup - let ruff handle it automatically
- **FOLLOW UP** with `uvx ruff format scripts/` and `uv run pytest` to ensure correctness

### Python Scripts

- **ALWAYS** use `uv run` when executing Python scripts in this project
- **DO NOT** use `python3` or `python` directly
- This ensures correct Python environment (minimum version in `.python-version`, enforced for performance) and dependency management
- Examples: `uv run changelog-utils generate`, `uv run benchmark-utils --help`
- Note: If the `benchmark-utils` console script isn't available, use `uv run python -m scripts.benchmark_utils --help`

## Essential AI Commands

### Code Quality Checks

Run these commands after making changes to ensure code quality:

```bash
# Rust code formatting and linting
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings -D clippy::all -D clippy::pedantic -W clippy::nursery -W clippy::cargo

# Python code quality (for scripts/ directory)
uvx ruff format scripts/
uvx ruff check --fix scripts/

# Markdown linting
npx markdownlint --fix "*.md" "scripts/*.md" "docs/*.md" ".github/*.md"

# Spell checking
npx cspell --config cspell.json --no-progress --gitignore --cache "**/*"

# JSON validation (when JSON files are modified)
jq empty cspell.json
```

### Testing and Validation

```bash
# Build and test
cargo build
cargo test --lib --verbose
cargo test --doc --verbose
cargo test --examples --verbose

# Benchmarks
cargo bench --no-run

# Documentation validation (required for crates.io publishing)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Run all examples
./scripts/run_all_examples.sh

# Python utility testing (if Python scripts modified)
uv run pytest
```

### Performance and Benchmarks

```bash
# Generate performance baseline
uv run benchmark-utils generate-baseline

# Compare performance against baseline
uv run benchmark-utils compare --baseline benches/baseline_results.txt

# Development mode (10x faster for iteration)
uv run benchmark-utils compare --baseline benches/baseline_results.txt --dev
```

### Changelog Management

```bash
# Generate enhanced changelog with AI categorization
uv run changelog-utils generate

# Create git tag with changelog content
uv run changelog-utils tag v0.4.2
```

## Project Context

The `delaunay` library implements d-dimensional Delaunay triangulations in Rust, inspired by CGAL. Key facts for AI assistants:

- **Language**: Rust (MSRV 1.89.0, Edition 2024)
- **Unsafe code**: Forbidden via `#![forbid(unsafe_code)]`
- **Published to**: crates.io (documentation build failures will prevent publishing)
- **CI**: GitHub Actions with strict quality requirements (clippy pedantic mode, rustfmt, no security vulnerabilities)
- **Architecture**: Generic design with `T: CoordinateScalar`, `U: DataType` for vertex data, `V: DataType` for cell data, `const D: usize` for dimensionality

## Ongoing Projects

These items are incomplete and may require future attention:

### Performance Optimization

- **Status**: Not started
- **Scope**: Algorithmic performance improvements, memory optimization, SIMD utilization
- **Dependencies**: Baseline system is complete, ready for optimization work

### Python Code Quality Improvements

- **Status**: Partially complete
- **Remaining**: Replace deprecated `typing.Dict/List/Tuple`, improve error handling patterns, reduce function argument counts
- **Tools**: Uses ruff for comprehensive linting (replaces pylint)

### Benchmark System Validation

- **Status**: Implementation complete, testing in progress
- **Remaining**: Test release flow with git tag generation, validate hardware compatibility warnings
- **Architecture**: GitHub Actions artifacts for baseline storage, 5% regression threshold

### Documentation Maintenance

- **Status**: Ongoing
- **Critical**: When adding/removing files, always update `docs/code_organization.md`
- **Reason**: Serves as authoritative project structure reference for contributors
