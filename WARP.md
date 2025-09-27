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
- **DO NOT** use scripts or automated tools (like `sed`, `awk`) for refactoring or logic-altering edits—only for non-semantic formatting and read-only checks
- **PREFERRED**: Interactive code editing using the `edit_files` tool for precise, reviewed changes
- **IMPORTANT**: Benchmark files (in `benches/`) are Rust code and must follow the same quality standards as core library code
  (e.g., `cargo clippy --benches -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo`)

#### JSON File Validation (AI Assistant Guidance)

- **ALWAYS** validate JSON files after editing them
- **PREFERRED**: Use `jq empty <filename>.json` for quick validation
- **ALTERNATIVE**: Use `npx --yes jsonlint --quiet <filename>.json` for validation
- **REQUIRED** when modifying `cspell.json`, `package.json`, or any other JSON configuration files
- Validate all modified JSON files:
  - `git status --porcelain | awk '/\.json$/ {print $2}' | xargs -r -n1 jq empty`

#### TOML File Validation (AI Assistant Guidance)

- **ALWAYS** validate TOML files after editing them
- **PREFERRED**: Use `uv run python -c "import tomllib; tomllib.load(open('<filename>.toml', 'rb')); print('<filename>.toml is valid TOML')"` for validation
- **REQUIRED** when modifying `pyproject.toml`, `Cargo.toml`, or any other TOML configuration files
- **FAST**: Uses Python's built-in `tomllib` (Python 3.11+) for reliable validation
- Validate all modified TOML files:
  - `git status --porcelain | awk '/\.toml$/ {print $2}' | xargs -r -I {} uv run python -c "import tomllib; tomllib.load(open('{}', 'rb')); print('{} is valid TOML')"`

#### Spell Check Dictionary Management (AI Assistant Guidance)

- **ALWAYS** run spell check after editing ANY files (code, documentation, configuration files, etc.)
- **REQUIRED** when adding or modifying any files to ensure proper spelling throughout the project
- **EXCEPTION**: **DO NOT** spell-check `cspell.json` itself - it contains intentional misspellings in `flagWords` from misspelled git commits
- **IF** cspell reports legitimate technical terms, programming keywords, or project-specific terminology as misspelled, add them to the `words` array in `cspell.json`
- **EXAMPLES**: Python terms (`kwargs`, `args`, `asyncio`), Rust terms (`usize`, `clippy`, `rustc`), technical terms (`triangulation`, `circumsphere`, `delaunay`),
  project/crate names (e.g., `nalgebra`, `serde`, `thiserror`, `pastey`).
- **PURPOSE**: Maintains a clean spell-check while building a comprehensive project dictionary
- Prefer `ignorePaths` for generated files (e.g., build artifacts) instead of adding their tokens to `words`.

#### Import Organization (AI Assistant Guidance)

- **USE COMPREHENSIVE**: `uv run ruff check scripts/ --fix` (recommended - covers all rules)
- **IMPORT-ONLY** (if needed): `uv run ruff check scripts/ --select F401,F811,I,PLC0415 --fix`
  - **F401**: Unused imports
  - **F811**: Redefined/duplicate imports  
  - **I**: Import ordering issues
  - **PLC0415**: Local imports that should be at top-level
- **PREFERRED**: Use comprehensive check over manual cleanup - let ruff handle all fixes
- **FOLLOW UP** with `uv run ruff format scripts/` and `uv run pytest` to ensure correctness
- **NOTE**: The main "Code Quality Checks" section uses the comprehensive approach

#### Shell Script Formatting (AI Assistant Guidance)

- **ALWAYS** run `shfmt` to format shell scripts after editing them
- **REQUIRED**: Use `shfmt -w scripts/*.sh` to format consistently (uses default shfmt settings)
- **LINT**: Use `shellcheck -x scripts/*.sh` to follow sourced files and catch include issues
- **CI CRITICAL**: Shell script formatting failures will cause CI to fail – must use default shfmt options to match CI
- **INDENTATION**: Uses shfmt default settings (tabs for indentation, standard spacing)
- **NOTE**: If the repo ever standardizes on spaces instead of tabs, document the required flags (e.g., `-i 2` for 2-space indentation)
- **POST-EDIT REQUIREMENT**: After any shell script edits, immediately run `shfmt -w` to prevent CI failures
- **EXAMPLES**: `find scripts -type f -name '*.sh' -exec shfmt -w {} +` formats all shell scripts
- **TROUBLESHOOTING**: If CI shows formatting diffs, run `shfmt -w` on the affected scripts

### Python Scripts

- **ALWAYS** use `uv run` when executing Python scripts in this project
- **DO NOT** use `python3` or `python` directly
- This ensures correct Python environment (minimum version in `.python-version`, enforced for performance) and dependency management
- Examples: `uv run changelog-utils generate`, `uv run benchmark-utils --help`
- Note: If the `benchmark-utils` console script isn't available, use `uv run python -m scripts.benchmark_utils --help`

#### Python Testing Framework

- **ALWAYS** use pytest when writing new Python script tests
- **PREFERRED** over unittest for better fixtures, parametrization, and assertion introspection
- **EXAMPLES**: Test files should be named `test_*.py` and use pytest fixtures and assertions
- **EXECUTION**: Use `uv run pytest` to run all Python tests with proper dependency management

## Essential AI Commands

### Just Workflow Summary

This project uses [just](https://github.com/casey/just) for streamlined development workflows. Here are the most commonly used commands:

```bash
# Quick development cycle
just dev            # Format, lint, and test (fastest feedback loop)

# Pre-commit workflow (recommended before pushing)
just pre-commit     # Full quality checks + all tests + examples

# CI simulation
just ci             # Run what CI runs (quality + release tests + benchmark compile)

# Comprehensive quality check
just quality        # All formatting, linting, validation, and spell checking

# Common individual commands
just fmt            # Format code
just clippy         # Rust linting
just test           # Standard tests
just test-all       # All tests (Rust + Python)
just examples       # Run all examples
just coverage       # Generate coverage report

# View all available commands
just --list         # or just help-workflows
```

### ⚠️ **CI Performance Impact Warning**

**CRITICAL**: Any changes to Rust code (`src/**`, `benches/**`, `Cargo.toml/lock`) will trigger lengthy performance regression testing (30-45 minutes) in CI.

**Best Practice**: Keep documentation/Python updates in separate branches from Rust code changes to avoid triggering benchmarks unnecessarily.

### Code Quality Checks

Run these commands after making changes to ensure code quality (the assistant must not execute git-altering commands; present them for the user to run):

**Note**: When asked to run code quality checks on "changed files", use `git status --porcelain` to identify which files have been
modified, added, or staged, and then focus the quality tools on those specific files.

```bash
# Comprehensive quality check (recommended - runs all checks below)
just quality

# Individual quality checks (if you need granular control):
just fmt           # Rust code formatting
just clippy        # Rust linting with pedantic/nursery/cargo warnings
just doc-check     # Rust documentation validation (required for crates.io publishing)
just python-lint   # Python code quality (ruff check + format)
just shell-lint    # Shell script formatting and linting
just markdown-lint # Markdown linting with auto-fixes
just spell-check   # Spell checking (only checks modified files)
just validate-json # JSON validation
just validate-toml # TOML validation

# Legacy commands (for reference, but prefer just recipes above):
# cargo fmt --all
# cargo clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic -W clippy::nursery -W clippy::cargo
# RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
# uv run ruff check scripts/ --fix && uv run ruff format scripts/
# git ls-files -z '*.sh' | xargs -0 -r -n1 shfmt -w
# git ls-files -z '*.sh' | xargs -0 -r -n4 shellcheck -x
# git ls-files -z '*.md' | xargs -0 -r -n100 npx markdownlint --config .markdownlint.json --fix
# (spell checking, JSON/TOML validation commands in justfile)
```

### Testing and Validation

```bash
# Comprehensive testing
just test-all      # Run all tests (Rust + Python)
just pre-commit    # Full pre-commit checks (quality + tests + examples)

# Rust testing
just test          # Standard library and doc tests
just test-release  # All tests in release mode (recommended for performance)
just test-debug    # Debug tools with output (circumsphere_debug_tools)
just test-allocation  # Memory allocation profiling tests

# Python testing
just test-python   # Run pytest for Python scripts

# Examples and validation
just examples      # Run all examples (validates functionality)

# Coverage analysis
just coverage      # Generate HTML coverage report (excludes benchmarks/examples/integration tests)
# View coverage report: open target/tarpaulin/tarpaulin-report.html

# Benchmarks
just bench-compile # Compile benchmarks without running
just bench         # Run all benchmarks

# Legacy commands (for reference, but prefer just recipes above):
# cargo build
# cargo test --lib --verbose && cargo test --doc --verbose
# cargo test --release
# cargo test --test circumsphere_debug_tools -- --nocapture
# bash scripts/run_all_examples.sh
# uv run pytest
# cargo tarpaulin --exclude-files 'benches/**' --exclude-files 'examples/**' --exclude-files 'tests/**' --out Html --output-dir target/tarpaulin
```

### Performance and Benchmarks

```bash
# Performance baseline management
just bench-baseline  # Generate performance baseline

# Performance comparison
just bench-compare   # Compare against baseline
just bench-dev       # Development mode (10x faster for iteration)

# Benchmark execution
just bench-compile   # Compile benchmarks without running
just bench          # Run all benchmarks

# Legacy commands (for reference, but prefer just recipes above):
# uv run benchmark-utils generate-baseline
# uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
# uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt --dev
```

### Changelog Management

```bash
# Generate enhanced changelog with AI categorization
just changelog       # Generate changelog
just changelog-update # Generate changelog with success message

# Create git tag with changelog content (user-only; WARP must not execute)
# Run manually from your terminal:
just changelog-tag <version>  # e.g., just changelog-tag v0.4.2

# Legacy commands (for reference, but prefer just recipes above):
# uv run changelog-utils generate
# uv run changelog-utils tag vX.Y.Z
```

## Project Context

The `delaunay` library implements d-dimensional Delaunay triangulations in Rust, inspired by CGAL. Key facts for AI assistants:

- **Language**: Rust (MSRV 1.90.0, Edition 2024)
- **Unsafe code**: Forbidden via `#![forbid(unsafe_code)]`
- **Published to**: crates.io (documentation build failures will prevent publishing)
- **CI**: GitHub Actions with strict quality requirements (clippy pedantic mode, rustfmt, no security vulnerabilities)
- **Architecture**: Generic design with `T: CoordinateScalar`, `U: DataType` for vertex data, `V: DataType` for cell data, `const D: usize` for dimensionality
- **Core Modules**:
  - `src/core/` - Triangulation data structures (Tds, Cell, Vertex, Facet) and Bowyer-Watson algorithms
  - `src/geometry/` - Geometric predicates, Point abstraction, and convex hull algorithms
- **Key Features**: Arbitrary dimensions (tested 2D-5D), generic coordinate types (f32/f64), serialization/deserialization, convex hull extraction

## Ongoing Projects

These items are incomplete and may require future attention:

### Performance Optimization

- **Status**: Not started
- **Scope**: Algorithmic performance improvements, memory optimization, SIMD utilization
- **Dependencies**: Baseline system is complete and ready for optimization work
- **Tracking**: Create GitHub issue for performance optimization roadmap

### Benchmark System Validation

- **Status**: Implementation complete, testing in progress
- **Remaining**: Test release flow with git tag generation, validate hardware compatibility warnings
- **Architecture**: GitHub Actions artifacts for baseline storage, 5% regression threshold
- **Tracking**: Create GitHub issue for benchmark system testing validation

### Documentation Maintenance

- **Status**: Ongoing
- **Critical**: When adding/removing files, always update `docs/code_organization.md`
- **Reason**: Serves as authoritative project structure reference for contributors
- **Tracking**: This is ongoing maintenance - create issues for specific documentation gaps as discovered

### Dependency Migration

- **Status**: Not started
- **Scope**: Replace peroxide with nalgebra for linear algebra operations
- **Rationale**: nalgebra is more mature, better maintained, and has better ecosystem integration
- **Impact**: Will affect matrix operations, potentially improving performance and reducing compilation times
- **Dependencies**: None - can be done independently when time permits
- **Tracking**: Issue #61

## AI Assistant Guidelines

### Context-Aware Test Execution (AI Assistant Guidance)

- **IF** Rust code changed in `tests/` directory → **MUST** run integration/debug tests:
  - `cargo test --release` (for performance)
  - `cargo test --test circumsphere_debug_tools -- --nocapture` (for debug output)
- **IF** Rust code changed in `examples/` directory → **MUST** run examples validation:
  - See "Testing and Validation → Run all examples" for the canonical command
- **IF** Rust code changed in `benches/` directory → **MUST** run benchmark verification:
  - `cargo bench --no-run` (verifies benchmarks compile without executing them)
- **IF** other Rust code changed (`src/`, etc.) → **MUST** run standard Rust tests:
  - `cargo test --lib --verbose`
  - `cargo test --doc --verbose`
- **FOR ANY** Rust code changes → validate documentation (see "Code Quality Checks → Rust documentation validation")
- **IMPORTANT**: For allocation testing, use `cargo test --test allocation_api --features count-allocations`
- **PURPOSE**: Ensures appropriate validation for the type of code changes made

### Integration Testing Patterns

- **Debug Tools**: Use `cargo test --test circumsphere_debug_tools -- --nocapture` for interactive debugging
- **Performance Tests**: Always run integration tests in `--release` mode for accurate performance measurements
- **Test Categories**: Organize tests by purpose: debugging tools (`*_debug_tools.rs`), integration (`*_integration.rs`), regression (`*_error.rs`), comparison (`*_comparison.rs`)
- **Test Documentation**: Each test file should have clear module documentation explaining purpose, usage, and test coverage
- **Specialized Tests**: Available integration tests include:
  - `circumsphere_debug_tools.rs` - Interactive debugging across dimensions (2D-4D)
  - `robust_predicates_comparison.rs` - Numerical accuracy testing
  - `robust_predicates_showcase.rs` - Focused showcase of robust predicates solving degenerate cases
  - `convex_hull_bowyer_watson_integration.rs` - Algorithm integration testing
  - `coordinate_conversion_errors.rs` - Error handling tests for extreme values, NaN, infinity
  - `test_cavity_boundary_error.rs` - Reproduction tests for cavity boundary facet errors
  - `allocation_api.rs` - Memory allocation profiling (requires `count-allocations` feature)

### Testing Best Practices

- **Performance Considerations**: Integration tests run significantly slower in debug mode - always recommend `--release` flag
- **Verbose Output**: Use `--nocapture` flag for debugging tests that produce detailed analysis output
- **Test Structure**: Convert CLI-style applications in tests to proper `#[test]` functions for better integration with cargo test framework
- **Memory Testing**: Use `--features count-allocations` for allocation profiling tests

### Test-Driven Development (TDD) Guidelines

- **PREFERRED**: Use Test-Driven Development (TDD) approach for new feature development
- **TDD Cycle**: Follow the Red-Green-Refactor cycle:
  1. **Red**: Write failing tests first that define the desired functionality
  2. **Green**: Write minimal code to make tests pass
  3. **Refactor**: Improve code quality while keeping tests passing
- **Test Types**: Apply TDD to both unit tests (`src/` modules with `#[cfg(test)]`) and integration tests (`tests/` directory)
- **Benefits**: TDD ensures better test coverage, cleaner APIs, and more maintainable code architecture
- **Rust-Specific**: Leverage Rust's type system and compiler feedback during the TDD process
- **Documentation**: Use `cargo test --doc` to validate code examples in documentation as part of TDD
- **Performance**: Write performance-focused tests early to catch regressions during development

### Documentation Standards

- **Directory READMEs**: Major directories (`examples/`, `benches/`, `tests/`) should have comprehensive README.md files
- **Usage Instructions**: Include specific command examples with proper flags (e.g., `--nocapture`, `--release`)
- **Test Categories**: Organize documentation by test purpose with clear headings and emoji indicators
- **Cross-References**: Link related documentation files and provide navigation between different documentation types

### File Organization Guidance

- **Test Files**: Place debugging utilities in `tests/` directory as proper test functions, not CLI applications
- **Integration Tests**: Use descriptive naming patterns that indicate test purpose and type
- **Documentation Updates**: When restructuring test files, update both `docs/code_organization.md` and relevant README files
- **Consistency**: Maintain consistent documentation patterns across `examples/README.md`, `benches/README.md`, and `tests/README.md`
