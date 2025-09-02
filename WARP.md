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
- This includes: `cargo fmt`, `cargo clippy`, `ruff format`, `ruff check --fix`, `markdownlint --fix`, `shfmt`, etc.
- Quality tools improve code without changing functionality or version control state

#### Import Organization (AI Assistant Guidance)

- **ALWAYS** use `uvx ruff check --fix --select F401,F403,I001,I002 scripts/` to fix import issues
- **AUTOMATICALLY** removes duplicate imports, unused imports, and organizes import order
- **PREFERRED** over manual import cleanup - let ruff handle it automatically
- **FOLLOW UP** with `uvx ruff format scripts/` and `uv run pytest` to ensure correctness

### Python Scripts

- **ALWAYS** use `uv run` when executing Python scripts in this project
- **DO NOT** use `python3` or `python` directly
- This ensures correct Python environment (minimum version in `.python-version`, enforced for performance) and dependency management
- Examples: `uv run python scripts/changelog_utils.py`, `uv run python -c "import scripts.subprocess_utils"`

## Overview

The `delaunay` library implements d-dimensional Delaunay triangulations in Rust,
inspired by CGAL. It provides a lightweight alternative for computational geometry
applications with support for arbitrary data types associated with vertices and
cells, d-dimensional triangulations, and serialization/deserialization capabilities.

## Essential Development Commands

### Rust Toolchain

The project uses a pinned Rust toolchain via `rust-toolchain.toml`:

- **Version**: 1.89.0 (matches MSRV in Cargo.toml)
- **Automatically enforced** when entering the project directory
- **Includes all necessary components**: clippy, rustfmt, rust-docs, rust-src
- **Cross-platform targets**: macOS (Intel/Apple Silicon), Linux, Windows
- **Team-friendly**: New contributors get the correct setup automatically via `rustup`

The toolchain file ensures consistent Rust versions across development, CI, and deployment
environments, preventing version drift issues and ensuring reproducible builds.

### Building and Testing

```bash
# Build the library
cargo build

# Build in release mode
cargo build --release

# Verify benchmarks compile (without running them)
cargo bench --no-run

# Check documentation for errors (public API)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Check documentation for errors (comprehensive, including private items)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items

# Run all tests (library, doc tests, examples)
cargo test --lib --verbose
cargo test --doc --verbose  
cargo test --examples --verbose

# Run tests with allocation counting feature
cargo test --features count-allocations -- allocation_counting 
```

### Identifying Changed Files

```bash
# Show changed files in machine-readable format
git status --porcelain=v1 -z
```

### Code Quality

#### Rust Code Quality

```bash
# Format Rust code
cargo fmt --all

# Check Rust formatting without modifying files
cargo fmt --all -- --check

# Run clippy (strict pedantic mode configured)
cargo clippy --all-targets --all-features -- -D warnings -D clippy::all -D clippy::pedantic -W clippy::nursery -W clippy::cargo
```

#### Python Code Quality

**IMPORTANT**: Run these commands after any changes to Python scripts in the `scripts/` directory:

```bash
# Format Python code (PEP 8 compliance) - replaces autopep8
uvx ruff format scripts/

# Lint & auto-fix (imports, unused, style per pyproject.toml)
uvx ruff check --fix scripts/

# Alternative: Fix only import-related issues (organize, remove duplicates, fix unused)
uvx ruff check --fix --select F401,F403,I001,I002 scripts/

# Test Python utilities to ensure they still work correctly
uv sync --group dev  # Install test dependencies
uv run pytest       # Run comprehensive Python utility tests

# Comprehensive linting is handled by ruff check above
# No additional linting step needed - ruff provides complete coverage
```

#### Documentation Quality

**IMPORTANT**: Since this crate is published to crates.io, ensure documentation builds correctly:

```bash
# Build documentation with warnings treated as errors (preferred)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

**Note**: Documentation build failures will prevent successful publishing to crates.io. Always verify docs build cleanly before releases.

**Note**: Ruff provides comprehensive Python code quality:

- `ruff format`: Fixes PEP 8 style violations (replaces autopep8)
- `ruff check --fix`: Organizes imports, removes unused code, fixes linting issues, and reports code quality problems (replaces isort, autoflake, and pylint)

**Configuration**: Both tools use `pyproject.toml` settings optimized for CLI scripts, which appropriately ignore:

- Complex control flow patterns natural to command-line tools (many branches, statements)
- CLI-specific patterns like boolean flags, print statements, subprocess calls
- Defensive exception handling and graceful degradation patterns
- Import placement optimizations for CLI startup time

**Installation**: The commands use `uvx` (uv's command runner) to execute Python tools:

- **uv**: Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **macOS**: `brew install uv`
- **Windows**: `powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **pip**: `pip install uv` (if you prefer installing via pip)

`uvx` automatically manages tool dependencies and provides isolated execution environments without requiring global installations.

#### Shell Script Code Quality

**IMPORTANT**: Run these commands after any changes to shell scripts in the `scripts/` directory:

```bash
# Lint shell scripts with shellcheck (detects common issues)
find scripts -type f -name '*.sh' -print0 | xargs -0 shellcheck

# Lint a specific shell script (follow sourced files)
shellcheck -x scripts/run_all_examples.sh

# Show all shellcheck warnings including informational ones
find scripts -type f -name '*.sh' -print0 | xargs -0 shellcheck -S info

# Check scripts with specific shell (if not detected automatically)
find scripts -type f -name '*.sh' -print0 | xargs -0 shellcheck -s bash
```

Additionally, use shfmt for consistent formatting:

```bash
# Format all shell scripts in-place (tabs by default)
find scripts -type f -name '*.sh' -exec shfmt -w {} +

# Check formatting without modifying files (useful in CI)
find scripts -type f -name '*.sh' -exec shfmt -d {} +

# Example: enforce 2-space indentation and named functions style
find scripts -type f -name '*.sh' -exec shfmt -i 2 -fn -w {} +
```

**Note**: shellcheck helps detect:

- Syntax errors and typos
- Quoting issues that could cause word splitting
- Incorrect variable usage patterns
- Potential security vulnerabilities
- POSIX compliance issues
- Performance anti-patterns

shfmt ensures consistent, idiomatic formatting, which reduces diffs and aids readability.

**Installation**: Install tools via:

- shellcheck
  - macOS: `brew install shellcheck`
  - Ubuntu/Debian: `apt install shellcheck`
  - Other platforms: See [shellcheck.net](https://www.shellcheck.net/)
- shfmt
  - macOS: `brew install shfmt`
  - Linux/Windows: See <https://github.com/mvdan/sh#shfmt> for binaries and package options

#### Markdown Code Quality

**IMPORTANT**: Run these commands after any changes to Markdown files:

```bash
# Lint project Markdown files (uses project .markdownlint.json configuration)
npx markdownlint "*.md" "scripts/*.md" "docs/*.md" ".github/*.md"

# Fix auto-fixable Markdown issues
npx markdownlint --fix "*.md" "scripts/*.md" "docs/*.md" ".github/*.md"

# Lint specific files
npx markdownlint README.md CONTRIBUTING.md WARP.md
```

**Note**: markdownlint detects formatting and style issues including inconsistent headings, improper list formatting, missing link formatting,
line length violations, and trailing whitespace.

**Installation**: markdownlint is automatically available via npx.

#### YAML Code Quality

**IMPORTANT**: Run these commands after any changes to YAML files:

```bash
# Lint all YAML files (uses project .yamllint configuration)
find . -type f \( -name '*.yml' -o -name '*.yaml' \) -exec yamllint -c .yamllint {} +

# Lint specific YAML files
yamllint -c .yamllint .github/workflows/ci.yml
```

**Note**: yamllint detects YAML syntax errors, indentation issues, line length violations, and trailing whitespace.

**Installation**: Install yamllint via `brew install yamllint` or `pip install yamllint`.

#### Spell Checking

```bash
# Check spelling with project configuration (respects .gitignore)
npx cspell --config cspell.json --no-progress --gitignore --cache "**/*"
```

#### Codacy Analysis (Local)

**IMPORTANT**: Use absolute paths when running Codacy Analysis CLI locally:

```bash
# Analyze specific directory with Ruff (most common)
codacy-analysis-cli analyze --tool ruff --directory /absolute/path/to/project/scripts/ --format json

# Analyze entire project
codacy-analysis-cli analyze --tool ruff --directory /absolute/path/to/project/ --format json

# Example for this project (adjust path to your system)
codacy-analysis-cli analyze --tool ruff --directory /Users/username/projects/delaunay/scripts/ --format json
```

**Note**: The Codacy CLI requires absolute directory paths and will fail with relative paths like `scripts/` or `./`.
Always use the full path when specifying the `--directory` parameter. Empty JSON array output `[]` indicates no issues found.

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench --bench circumsphere_containment
cargo bench --bench small_scale_triangulation
cargo bench --bench triangulation_creation

# Generate performance baseline for CI
uv run benchmark-utils generate-baseline

# Generate baseline for development (faster)
uv run benchmark-utils generate-baseline --dev

# Compare performance against baseline
uv run benchmark-utils compare --baseline benches/baseline_results.txt

# Compare with development settings (faster)
uv run benchmark-utils compare --baseline benches/baseline_results.txt --dev
```

### Examples and Development Scripts

```bash
# Run specific example
cargo run --example triangulation_3d_50_points
cargo run --example test_circumsphere

# Run all examples
./scripts/run_all_examples.sh
```

### Running Single Tests

```bash
# Run specific test module
cargo test triangulation_validation

# Run specific test function
cargo test test_basic_allocation_counting

# Run tests with output
cargo test -- --nocapture
```

### Changelog Generation

```bash
# Generate changelog with commit dates (recommended)
uv run changelog-utils generate

# Generate changelog with debug output (keeps intermediate files)
uv run changelog-utils generate --debug

# Create git tag with changelog content as tag message
uv run changelog-utils tag v0.4.2

# Force recreate existing tag
uv run changelog-utils tag v0.4.2 --force

# Advanced: direct auto-changelog usage (bypasses project wrappers)
npx auto-changelog --unreleased
npx auto-changelog --latest-version v0.3.4

# Test changelog generation without writing to file (direct auto-changelog)
npx auto-changelog --stdout
```

**Note**: The project uses `uv run changelog-utils generate` to generate changelogs with commit dates instead of tag creation dates.
This provides more accurate release timing that reflects when the actual work was completed rather than when tags were created.
The tool also supports creating git tags with changelog content for GitHub releases.

## Architecture Overview

### Core Module Structure

**`src/core/`** - Primary data structures and algorithms

- **`triangulation_data_structure.rs`** - Main `Tds` struct for triangulation data management
- **`vertex.rs`** - Vertex representation with generic coordinate support
- **`cell.rs`** - D-dimensional cells (simplices) with neighbor relationships
- **`facet.rs`** - (D-1)-dimensional faces derived from cells
- **`boundary.rs`** - Boundary analysis and facet detection
- **`util.rs`** - Helper functions for triangulation operations
- **`algorithms/`** - Triangulation construction algorithms
  - **`bowyer_watson.rs`** - Incremental Bowyer-Watson algorithm implementation
  - **`robust_bowyer_watson.rs`** - Enhanced Bowyer-Watson with robust geometric predicates
- **`traits/`** - Core traits for triangulation operations
  - **`boundary_analysis.rs`** - Boundary detection and analysis trait
  - **`data_type.rs`** - Generic data type constraints
  - **`insertion_algorithm.rs`** - Unified interface for vertex insertion algorithms

**`src/geometry/`** - Geometric algorithms and predicates

- **`point.rs`** - Generic Point struct with NaN-aware equality and hashing
- **`predicates.rs`** - Standard geometric predicates (insphere, orientation tests)
- **`robust_predicates.rs`** - Enhanced predicates with improved numerical stability
- **`matrix.rs`** - Matrix operations for geometric computations
- **`util.rs`** - Geometric utility functions
- **`algorithms/`** - Geometric algorithms
  - **`convex_hull.rs`** - Convex hull extraction from Delaunay triangulations
- **`traits/`** - Coordinate system abstractions
  - **`coordinate.rs`** - Unified coordinate traits including primary coordinate trait, scalar types, finite value validation,
    consistent hashing for floating-point coordinates, and NaN-aware equality comparison

**`src/lib.rs`** - Main library file with module declarations and prelude module

### Key Relationships

- `core` modules use `geometry` for predicates and coordinate operations
- `Tds` (Triangulation Data Structure) is the main entry point containing vertices and cells
- Cells maintain neighbor relationships via SlotMap-based storage
- Facets are computed on-demand from cell data rather than stored explicitly

### Generic Design

The library extensively uses generics:

- `T: CoordinateScalar` for coordinate types (f32, f64, etc.)
- `U: DataType` for vertex-associated data
- `V: DataType` for cell-associated data  
- `const D: usize` for dimensionality

## Development Workflows

### Standard Development Process

1. **Create feature/fix branch** from main
2. **Make changes** following architectural patterns
3. **Build and test** with `cargo build && cargo test`
4. **Format and lint** with `cargo fmt && cargo clippy`
5. **Test benchmarks compile** with `cargo bench --no-run` (avoids long execution time)
6. **Run examples** with `./scripts/run_all_examples.sh`
7. **Check performance impact** with `uv run benchmark-utils compare --baseline benches/baseline_results.txt --dev` (only if performance-critical changes)
8. **Commit and push** for CI validation

**Note**: Use `cargo bench --no-run` to verify benchmarks compile without actually running them, as full benchmark execution takes several minutes.

### Performance-Critical Changes

For changes affecting algorithmic performance:

1. **Generate baseline** before changes: `uv run benchmark-utils generate-baseline`
2. **Make modifications**
3. **Test regression** with `uv run benchmark-utils compare --baseline benches/baseline_results.txt`
4. **If >5% regression detected**, investigate and optimize
5. **Update baseline** if performance change is acceptable

### Adding New Examples

Examples in `examples/` demonstrate library capabilities:

- Use `delaunay::prelude::*` for imports
- Include comprehensive documentation header
- Show error handling patterns
- Validate with `./scripts/run_all_examples.sh`

### Documentation Maintenance

**CRITICAL**: When adding, removing, or significantly restructuring files, always update `docs/code_organization.md`:

#### File Structure Changes

```bash
# After adding new files/directories:
# 1. Update the Complete Directory Tree section in docs/code_organization.md
# 2. Add descriptions for new files with their purpose
# 3. Update Architecture Overview if the change affects core structure
```

**Examples of changes requiring documentation updates:**

- **New modules**: `src/core/new_module.rs` → Add to directory tree with description
- **New directories**: `src/algorithms/` → Update architecture overview
- **Removed files**: Delete from directory tree, update references
- **Moved files**: Update both old and new locations, check all references
- **New examples**: Add to `examples/` section with purpose
- **New scripts**: Add to `scripts/` section with functionality description
- **New workflows**: Add to `.github/workflows/` section

**Why this matters:**

- `docs/code_organization.md` serves as the authoritative project structure reference
- Contributors rely on it to understand the codebase architecture  
- Out-of-date documentation creates confusion and slows development
- The directory tree is used by new contributors for navigation

**Quick check**: If you've added/removed files, run:

```bash
find . -type f -name '*.rs' | wc -l  # Compare with documented file count
```

## Benchmarking and Performance

### Key Benchmarks

- **`circumsphere_containment`** - Geometric predicate performance across methods
- **`small_scale_triangulation`** - Triangulation creation for 10-50 points in 2D/3D/4D
- **`assign_neighbors_performance`** - Neighbor relationship computation
- **`microbenchmarks`** - Fine-grained performance tests for specific operations
- **`triangulation_creation`** - Basic triangulation creation benchmarks

### Performance Characteristics

- **2D triangulations**: Excellent performance, sub-millisecond for ≤10 points

---

## Coding Standards and Patterns

### Language Requirements

- **Rust Edition**: 2024
- **MSRV**: 1.89.0
- **Unsafe code**: Forbidden via `#![forbid(unsafe_code)]`
- **Missing docs**: Warnings enabled

### Code Organization Pattern

Following `docs/code_organization.md`, modules use consistent structure:

1. **Module documentation** (`//!` comments)
2. **Imports** (grouped and ordered)
3. **Error types** (using `thiserror::Error`)
4. **Convenience macros** (with comprehensive docs)
5. **Struct definitions** (with Builder pattern)
6. **Deserialization** (manual implementation)
7. **Core implementations**
8. **Trait implementations**
9. **Tests** (with subsection separators)

### Key Dependencies and Patterns

- **`derive_builder`** - Builder pattern for complex structs
- **`serde`** - Serialization with manual Deserialize for complex types
- **`slotmap`** - Efficient storage for cells and vertices with stable keys
- **`uuid`** - Unique identification for vertices and cells
- **`nalgebra`** - Linear algebra operations
- **`ordered-float`** - Consistent floating-point handling

### Error Handling

- Custom error types inherit from standard patterns
- Use `thiserror::Error` for automatic Display/Error implementations
- Chain errors with `#[from]` attribute for source tracking
- Provide context-rich error messages

## CI and Automation

### GitHub Workflows

- **`.github/workflows/ci.yml`** - Build, test, format, and lint on all platforms
- **`.github/workflows/benchmarks.yml`** - Performance regression testing (main branch only)
- **`.github/workflows/rust-clippy.yml`** - Additional clippy analysis
- **`.github/workflows/audit.yml`** - Security vulnerability scanning
- **`.github/workflows/codecov.yml`** - Test coverage tracking

### CI Requirements

- All tests must pass across Ubuntu, macOS, and Windows
- Clippy pedantic mode must pass without warnings
- Code must be properly formatted with rustfmt
- No security vulnerabilities in dependencies

### Performance CI

- Benchmarks run automatically on main branch changes affecting performance-critical files
- Separate from main CI to avoid blocking development
- Fails if >5% performance regression detected
- Skips gracefully if baseline doesn't exist

## Development Resources

### Documentation

- **`README.md`** - Project overview and features
- **`CONTRIBUTING.md`** - Comprehensive contribution guidelines  
- **`docs/code_organization.md`** - Module organization patterns
- **`scripts/README.md`** - Development script documentation
- **`benches/README.md`** - Benchmarking guide and performance results
- **`examples/README.md`** - Example program descriptions

### Scripts and Tools

- **`uv run benchmark-utils`** - Modern Python utility for performance baselines and regression testing
- **`scripts/run_all_examples.sh`** - Validate all examples

## TODOs - Automated Performance Baseline System

### Implementation Complete ✅

The automated performance baseline system has been successfully implemented with:

- **Automated baseline generation** on git tag creation (`.github/workflows/generate-baseline.yml`)
- **Enhanced benchmark testing** for PRs and main branch commits
- **Python utilities** replacing complex bash scripts (`scripts/hardware_utils.py`, `scripts/benchmark_utils.py`)
- **GitHub Actions artifacts** for baseline storage (no repo commits needed)
- **Backward-compatible shell wrappers** preserving existing CLIs

### Next Steps - Testing and Validation

#### Priority 1: Test the Release Flow

```bash
# Create and push a release tag to test automatic baseline generation
git tag v0.4.2
git push origin v0.4.2

# Verify baseline generation workflow runs successfully
# Check that performance-baseline-v0.4.2 artifact is created
```

#### Priority 2: Test PR Performance Regression

- Create a PR with some benchmark-affecting changes
- Verify that the benchmark workflow downloads the baseline artifact
- Confirm regression detection works with 5% threshold
- Validate hardware compatibility warnings

#### Priority 3: Python Code Quality Improvements

The CI includes Python linting that's currently non-blocking. Address gradually:

```bash
# Fix formatting and linting issues in Python scripts
uvx ruff format scripts/
uvx ruff check --fix scripts/
# Note: pylint has been retired in favor of comprehensive ruff linting
```

Key improvements needed:

- Replace deprecated `typing.Dict/List/Tuple` with modern syntax
- Fix print statements in library code (use logging)
- Address error handling patterns
- Improve function signatures (reduce argument counts)

#### Priority 4: Monitor and Iterate

- Monitor performance baseline accuracy across different hardware
- Adjust regression thresholds if needed (currently 5%)
- Consider expanding to additional benchmark suites
- Optimize CI runtime if 5-minute constraint becomes an issue

### System Architecture Notes

**Baseline Storage Strategy:**

- Uses GitHub Actions artifacts instead of committing to repo
- Artifacts tied to specific releases with 365-day retention
- Fallback generation if no baseline found (dev mode for speed)

**Hardware Compatibility:**

- Cross-platform hardware detection (macOS, Linux, Windows)
- Warnings when comparing across different hardware configurations
- Detailed hardware metadata in all baselines for troubleshooting

**Performance Comparison:**

- 5% regression threshold for time measurements
- Hardware-normalized comparisons where possible
- Comprehensive reporting with improvement/regression indicators

### Mathematical References

The library implements algorithms based on established computational geometry literature:

- Bowyer-Watson algorithm for Delaunay triangulation
- Shewchuk's robust geometric predicates
- Lifted paraboloid method for circumsphere tests
- CGAL-inspired triangulation data structures
