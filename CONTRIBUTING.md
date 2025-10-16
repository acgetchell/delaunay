# Contributing to Delaunay

Thank you for your interest in contributing to the [**delaunay**][delaunay-lib] computational geometry library!
This document provides comprehensive guidelines for contributors, from first-time contributors
to experienced developers looking to contribute significant features.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Just Command Runner](#just-command-runner)
- [CI Performance Testing](#ci-performance-testing)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Citation and References](#citation-and-references)
- [Performance and Benchmarking](#performance-and-benchmarking)
- [Submitting Changes](#submitting-changes)
- [Types of Contributions](#types-of-contributions)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct][code-of-conduct].
By participating, you are expected to uphold these standards.
Please report unacceptable behavior to the [maintainer][maintainer-email].

Our community is built on the principles of:

- **Respectful collaboration** in computational geometry research and development
- **Inclusive participation** regardless of background or experience level
- **Excellence in scientific computing** and algorithm implementation
- **Open knowledge sharing** about Delaunay triangulations and geometric algorithms

## Getting Started

### Prerequisites

Before you begin, ensure you have:

1. **Rust** (latest stable version): Install via [rustup.rs][rustup]
2. **Git** for version control
3. **Python and uv** (for development scripts and automation):
   - **Python**: Minimum version specified in `.python-version` (enforced for performance reasons)
   - **uv**: Fast Python package manager - Install via:
     - **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
     - **Windows**: `powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"`
     - **Alternative**: `pip install uv` (if you prefer using pip)
   - See [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more options
4. **System dependencies** (for shell scripts):
   - **macOS**: `brew install findutils coreutils`
   - **Ubuntu/Debian**: `sudo apt-get install findutils coreutils`
   - **Other systems**: Install equivalent packages for `find` and `sort`

**Note**: Many development tasks now use Python utilities (managed by uv) instead of traditional shell tools, reducing the number of required system dependencies.

### Quick Start

1. **Fork and clone** the repository:
   - Fork this repository to your GitHub account using the "Fork" button
   - Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/delaunay.git
   cd delaunay
   ```

2. **Build the project**:

   ```bash
   cargo build
   ```

3. **Run tests**:

   ```bash
   # Basic tests
   cargo test                # Rust library tests
   uv sync --group dev       # Install Python dev dependencies
   uv run pytest             # Python utility tests
   
   # Or use just for comprehensive testing:
   just test                 # Library and doc tests
   just test-all             # All tests (Rust + Python)
   just test-release         # All tests in release mode (faster performance)
   ```

4. **Try the examples**:

   ```bash
   cargo run --release --example triangulation_3d_50_points
   ./scripts/run_all_examples.sh  # Run all examples
   ```

5. **Run benchmarks** (optional):

   ```bash
   # Compile benchmarks without running (useful for CI)
   just bench-compile
   
   # Run all benchmarks
   just bench
   ```

6. **Code quality checks**:

   ```bash
   just fmt             # Format all code
   just clippy          # Strict clippy with pedantic/nursery/cargo warnings
   just doc-check       # Validate documentation builds
   ```

7. **Use Just for comprehensive workflows** (recommended):

   ```bash
   # Install just command runner
   cargo install just
   
   # See all available commands
   just --list
   
   # Common workflows (from quick to comprehensive)
   just dev             # Quick development cycle (format, lint, test)
   just quality         # All quality checks + tests
   just ci              # CI simulation (quality + release tests + benchmarks)
   just commit-check    # Pre-commit validation (most thorough: CI + examples)
   
   # Granular quality checks
   just lint            # All linting (code + docs + config)
   just lint-code       # Code linting only (Rust, Python, Shell)
   just lint-docs       # Documentation linting only
   just lint-config     # Configuration validation only
   ```

> **Tip**: Use `just help-workflows` for workflow guidance and to see all available commands.

## Development Environment Setup

### Recommended Tools

- **IDE/Editor**: Any editor with Rust Language Server (rust-analyzer) support
- **Linting**: The project uses strict clippy lints - ensure your editor shows clippy warnings
- **Formatting**: Use `rustfmt` for code formatting (configured in `rustfmt.toml`)

### Project Configuration

The project uses:

- **Edition**: Rust 2024
- **MSRV**: Rust 1.90.0
- **Linting**: Strict clippy pedantic mode
- **Testing**: Standard `#[test]` with comprehensive coverage
- **Benchmarking**: Criterion with allocation tracking

### Automatic Toolchain Management

**🔧 This project uses automatic Rust toolchain management via `rust-toolchain.toml`**

When you enter the project directory, `rustup` will automatically:

- **Install the correct Rust version** (1.90.0) if you don't have it
- **Switch to the pinned version** for this project
- **Install required components** (clippy, rustfmt, rust-docs, rust-src)
- **Add cross-compilation targets** for supported platforms

**What this means for contributors:**

1. **No manual setup needed** - Just have `rustup` installed ([rustup.rs][rustup])
2. **Consistent environment** - Everyone uses the same Rust version automatically
3. **Reproducible builds** - Eliminates "works on my machine" issues
4. **CI compatibility** - Your local environment matches our CI exactly

**First time in the project?** You'll see:

```text
info: syncing channel updates for '1.90.0-<your-platform>'
info: downloading component 'cargo'
info: downloading component 'clippy'
...
```

This is normal and only happens once. After that, the correct toolchain is used automatically whenever you work on the project.

**Verification:** Run `rustup show` to confirm you're using the pinned toolchain:

```bash
rustup show
# Should show: active toolchain: 1.89.0-<platform> (overridden by '/path/to/delaunay/rust-toolchain.toml')
```

### Python Development Environment

#### Python Utilities for Development Automation

The project has transitioned from traditional shell scripts to Python-based utilities for better cross-platform compatibility and maintainability.

**Key Python Configuration Files:**

- **`pyproject.toml`**: Defines Python project metadata, dependencies, and tool configurations
- **`uv.lock`**: Lockfile ensuring reproducible Python environments across different machines
- **Python utilities** in `scripts/`: Modern replacements for legacy shell scripts

**Python Dependencies Management:**

The project uses `uv` for fast, reliable Python dependency management:

```bash
# Python dependencies are automatically managed
# No manual installation required - uv handles everything

# If you need to run Python tools directly:
uvx ruff format scripts/     # Code formatting
uvx ruff check --fix scripts/ # Linting with auto-fixes
uvx pylint scripts/          # Code quality analysis
```

**Integration with Development Workflow:**

- **GitHub Actions**: Python utilities integrate seamlessly with CI/CD
- **Hardware Detection**: Cross-platform hardware information gathering
- **Benchmark Processing**: Automated performance regression detection
- **Changelog Management**: Enhanced changelog generation and git tagging with Python parsing

**Migration from Shell Scripts:**

The project has evolved from shell-based to Python-based automation:

- ✅ **New**: Python utilities (`benchmark-utils`, `hardware-utils`, `changelog-utils`) accessible via `uv run`
  with comprehensive benchmark processing, hardware detection, and changelog management functionality
- ❌ **Legacy**: Old shell scripts like `generate_baseline.sh`, `compare_benchmarks.sh`, `tag-from-changelog.sh` (replaced by Python equivalents)
- 🔄 **Hybrid**: Some shell scripts remain as simple wrappers (e.g., `run_all_examples.sh`)

**Benefits of Python Utilities:**

- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Better error handling** and structured data processing
- **Integration with GitHub Actions** for automated workflows
- **Easier maintenance** and testing compared to complex shell scripts

## Project Structure

The project follows a standard Rust library structure with additional tooling for computational geometry research:

### Key Directories

- **`src/`** - Core library code
  - **`core/`** - Triangulation data structures and algorithms (Bowyer-Watson, boundary analysis)
  - **`geometry/`** - Geometric predicates, point operations, and convex hull algorithms
- **`examples/`** - Usage examples and demonstrations (see [examples documentation][examples-readme])
- **`benches/`** - Performance benchmarks with Criterion (see [benchmarks documentation][benches-readme])
- **`tests/`** - Integration tests and regression test suites
- **`docs/`** - Additional documentation and guides
- **`scripts/`** - Development automation (Python utilities, shell scripts)

### Configuration Files

- **`.codacy.yml`** - Code quality analysis configuration
- **`Cargo.toml`** - Package metadata and Rust tooling configuration
- **`pyproject.toml`** - Python development tools configuration
- **`rustfmt.toml`** - Code formatting rules
- **`rust-toolchain.toml`** - Pinned Rust version for reproducible builds

### Development Resources

- **`WARP.md`** - AI development assistant guidance
- **`CONTRIBUTING.md`** - This file
- **`REFERENCES.md`** - Academic citations and bibliography
- **`.github/workflows/`** - CI/CD automation (testing, benchmarks, quality checks)

For detailed code organization patterns and module structure, see [code organization documentation][code-organization].

## CI Performance Testing

### ⚠️ **Important: Rust Code Changes Trigger Lengthy Baseline Comparisons**

**Any changes to Rust code will automatically trigger performance regression testing in CI, which can take 30-45 minutes to complete.**

The benchmark workflow runs on changes to:

- `src/**` - Any core library code
- `benches/**` - Benchmark code  
- `Cargo.toml` or `Cargo.lock` - Dependencies

### **Branch Strategy Recommendation**

To avoid triggering lengthy baseline comparisons unnecessarily:

✅ **Recommended**: Keep documentation and Python utility updates in **separate branches/PRs** from Rust code changes

❌ **Avoid**: Mixing documentation updates with Rust code changes in the same commit/PR

### **Examples**

**Good workflow:**

```bash
# Branch 1: Documentation updates only
git checkout -b docs/update-readme
# Edit README.md, CONTRIBUTING.md, etc.
git commit -m "docs: update contributing guidelines"
# → No benchmarks triggered, fast CI

# Branch 2: Rust code changes (separate PR)
git checkout -b feat/improve-algorithm 
# Edit src/core/triangulation.rs
git commit -m "feat: optimize triangulation algorithm"
# → Benchmarks triggered, but isolated to code changes
```

**Avoid:**

```bash
# Mixed changes (triggers benchmarks for trivial doc fixes)
git add README.md src/core/triangulation.rs
git commit -m "feat: algorithm improvement + doc updates"
# → 45-minute benchmark run for a simple doc fix
```

This strategy helps maintain fast feedback loops for documentation work while ensuring proper performance regression testing for code changes.

## Just Command Runner

### Overview

The project uses [Just](https://github.com/casey/just) as a command runner to provide convenient workflows that combine multiple development tasks.
Just recipes are defined in the [`justfile`](justfile) in the project root.

### Installation

```bash
# Install just
cargo install just

# Verify installation
just --version
```

### Common Workflows

```bash
# See all available commands
just --list
just help-workflows   # Show organized workflow help

# Quick development cycle
just dev              # Format, lint, and test (fast feedback)

# Comprehensive validation
just quality          # All quality checks + tests (Rust + Python)

# CI simulation
just ci               # Quality + release tests + benchmark compilation

# Pre-commit validation (most thorough)
just pre-commit       # Comprehensive checks before pushing

# Quality checks
just quality          # All formatting, linting, validation

# Testing workflows
just test-all         # All tests (Rust + Python)
just test-release     # Tests in release mode (performance)
just coverage         # Generate HTML coverage report

# Benchmark workflows
just bench-baseline   # Generate performance baseline
just bench-compare    # Compare against baseline

# CI simulation
just ci               # Run what CI runs locally
```

### Individual Task Recipes

#### Code Quality

- `just fmt` - Format all code
- `just clippy` - Run strict clippy
- `just lint` - Format + clippy + doc validation
- `just python-lint` - Format and lint Python scripts
- `just spell-check` - Check spelling across project files

#### Testing

- `just test` - Standard library and doc tests
- `just test-debug` - Debug tools with output
- `just test-allocation` - Memory allocation profiling
- `just examples` - Run all examples to verify functionality

#### Validation

- `just validate-json` - Validate all JSON files
- `just validate-toml` - Validate all TOML files
- `just shell-lint` - Format and lint shell scripts
- `just markdown-lint` - Lint markdown files

#### Utilities

- `just setup` - Set up development environment
- `just clean` - Clean build artifacts
- `just changelog` - Generate enhanced changelog
- `just help-workflows` - Show workflow guidance

### Workflow Recommendations

**During active development:**

```bash
just dev              # Quick cycle: format, lint, test
```

**Before committing:**

```bash
just commit-check    # Most comprehensive: CI + examples
```

**When working on performance:**

```bash
just bench-baseline   # Generate baseline
# Make changes...
just bench-compare    # Check for regressions
```

**Testing CI locally:**

```bash
just ci               # Simulate what CI runs
```

## Development Workflow

### 1. Issue-Driven Development

Before starting work:

1. **Check existing issues** for similar problems or feature requests
2. **Create an issue** if none exists, describing:
   - The problem or feature request
   - Expected behavior vs. actual behavior
   - Relevant mathematical/algorithmic context
   - Proposed solution approach (for features)

### 2. Branch Strategy

Create focused branches for your work:

```bash
# For bug fixes
git checkout -b fix/issue-description

# For new features  
git checkout -b feature/feature-name

# For documentation
git checkout -b docs/doc-improvement
```

### 3. Development Process

1. **Make focused commits** with clear messages (see [Commit Message Format](#commit-message-format))
2. **Write or update tests** for your changes
3. **Update documentation** as needed
4. **Run the full test suite** before pushing
5. **Check performance impact** for algorithmic changes
6. **Push to your fork** and create a pull request to the main repository

**Important Note on Git Operations:**

Per project rules (see [WARP.md](WARP.md)), **DO NOT** include `git commit` or `git push` commands in
development scripts. All git operations should be handled manually by contributors to maintain full control over
version control operations. This ensures:

- **User control** over commit messages and timing
- **Prevention of accidental commits** during automated processes
- **Compliance with project security policies**
- **Flexibility** in branching and merging strategies

Any automation scripts should stop at the point where git operations would be needed, allowing contributors to handle version control manually.

### 4. Continuous Integration

The project uses comprehensive CI workflows:

- **Main CI** (`.github/workflows/ci.yml`): Build, test, lint on every PR
- **Benchmarks** (`.github/workflows/benchmarks.yml`): Performance regression testing
- **Security** (`.github/workflows/audit.yml`): Dependency vulnerability scanning
- **Code Quality** (`.github/workflows/rust-clippy.yml`): Strict linting
- **Codacy** (`.github/workflows/codacy.yml`): Code quality analysis using project configurations
- **Coverage** (`.github/workflows/codecov.yml`): Test coverage tracking

All PRs must pass CI checks before merging.

### 5. Code Quality Analysis

The project uses **Codacy** for automated code quality analysis across both Rust and Python code:

- **Configuration**: `.codacy.yml` in the project root
- **Rust Analysis**: Uses Clippy (configured via `Cargo.toml`) and rustfmt (configured via `rustfmt.toml`)
- **Python Analysis**: Uses Ruff and Pylint (configured via `pyproject.toml`)
- **Additional Tools**: ShellCheck for shell scripts, markdownlint for documentation, yamllint for config files

**Key Benefits:**

- **Unified Quality Dashboard**: Single view of code quality across all languages
- **Uses Project Settings**: Respects your local tool configurations (no duplicate/conflicting rules)
- **Pull Request Integration**: Quality feedback directly in PR reviews
- **Trend Tracking**: Monitor code quality improvements over time

**For Contributors:**

- Codacy analysis runs automatically on all PRs
- Quality issues are reported as PR comments
- The same tools and rules used locally in development (following WARP.md guidelines)
- No additional setup required - uses existing project configurations

## Commit Message Format

This project uses [conventional commits](https://www.conventionalcommits.org/) to generate meaningful changelogs automatically.

### Format

```text
type(scope): short description (50 chars or less)

Optional longer explanation (wrap at 72 chars)
- Why this change was made
- What problem it solves
- Any side effects or considerations

Reference issues: Fixes #123, Closes #456
Breaking changes: BREAKING CHANGE: description
```

### Types

| Type | Description | Appears in Changelog |
|------|-------------|----------------------|
| feat | New features | ✅ Yes |
| fix | Bug fixes | ✅ Yes |
| perf | Performance improvements | ✅ Yes |
| refactor | Code refactoring | ✅ Yes |
| build | Build system changes | ✅ Yes |
| ci | CI/CD changes | ✅ Yes |
| revert | Reverting changes | ✅ Yes |
| chore | Maintenance tasks | ❌ No (filtered) |
| style | Formatting changes | ❌ No (filtered) |
| docs | Documentation only | ❌ No (filtered) |
| test | Test changes only | ❌ No (filtered) |

### Scopes (Optional)

- `core` - Core triangulation structures
- `geometry` - Geometric algorithms and predicates
- `benchmarks` - Performance benchmarks
- `examples` - Usage examples
- `docs` - Documentation

### Examples

```bash
# Features
feat(core): implement d-dimensional boundary analysis
feat(geometry): add robust circumsphere predicates
feat: add 4D triangulation support

# Bug fixes
fix(core): prevent infinite loop in degenerate triangulations
fix(geometry): handle NaN coordinates in point validation
fix: resolve memory leak in vertex insertion

# Performance
perf(core): optimize Bowyer-Watson algorithm with cell caching
perf: reduce allocations in neighbor assignment

# Breaking changes
feat!: redesign Vertex API for better type safety
fix!: change Point::coordinates() to Point::to_array()

# Maintenance (filtered from changelog)
chore: update dependencies
style: fix clippy warnings
docs: update README examples
test: add edge case coverage
```

### PR Titles

Since PR merges appear prominently in the changelog, use the same conventional format for PR titles:

```text
feat: implement 4D triangulation support
fix: resolve edge case in Bowyer-Watson algorithm
perf: optimize boundary facet detection
```

## Code Style and Standards

### Rust Style Guidelines

The project follows strict Rust coding standards:

```toml
# From Cargo.toml - all code must pass these lints
[lints.clippy]
pedantic = { level = "warn", priority = -1 }
extra_unused_type_parameters = "warn"
```

Key standards:

- **Documentation**: All public APIs must have comprehensive doc comments
- **Error Handling**: Use proper `Result` types, avoid `unwrap()` in library code
- **Type Safety**: Leverage Rust's type system for algorithmic correctness
- **Performance**: Consider algorithmic complexity and memory allocation patterns

### Mathematical Documentation

Given the mathematical nature of computational geometry:

- **Algorithm References**: Cite relevant papers or textbooks
- **Complexity Analysis**: Document time/space complexity where relevant
- **Geometric Intuition**: Explain the geometric meaning of operations
- **Numerical Stability**: Note floating-point considerations

### Code Organization

Follow the patterns documented in [code organization documentation][code-organization]:

1. Module documentation (`//!` comments)
2. Imports (organized by source)
3. Error types (using `thiserror`)
4. Convenience macros and helpers
5. Struct definitions (with Builder pattern)
6. Core implementations
7. Trait implementations
8. Tests (comprehensive with subsections)

## Testing

### Test Categories

The project maintains comprehensive test coverage:

```bash
# Unit tests (embedded in source files)
cargo test

# Integration tests
cargo test --tests

# Python utility tests (development scripts)
uv sync --group dev  # Install test dependencies
uv run pytest       # Run Python tests

# Example tests (ensure examples compile and run)
./scripts/run_all_examples.sh

# Benchmark tests (performance verification)
cargo bench
```

### Writing Tests

Follow these testing patterns:

1. **Unit Tests**: Test individual functions and methods

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_specific_functionality() {
           // Test implementation
       }
   }
   ```

2. **Property-Based Tests**: For geometric algorithms

   ```rust
   #[test]
   fn test_geometric_property() {
       // Test that geometric invariants hold
   }
   ```

3. **Edge Case Tests**: Boundary conditions and special cases

   ```rust
   #[test]
   fn test_degenerate_cases() {
       // Test edge cases like collinear points
   }
   ```

### Test Data and Reproducibility

- Use **fixed random seeds** for reproducible tests
- Include tests for **various dimensions** (2D, 3D, 4D, etc.)
- Test with **different data distributions** (uniform, clustered, etc.)
- Include **regression tests** for fixed bugs

## Documentation

### Documentation Types

1. **API Documentation**: Rust doc comments on all public items
2. **Examples**: Comprehensive examples in `examples/` directory
3. **User Guides**: High-level documentation in `docs/`
4. **Contributing Guides**: Development-focused documentation

### Writing Good Documentation

- **Start with purpose**: What does this function/struct/module do?
- **Explain parameters**: What do generic parameters represent?
- **Provide examples**: Show typical usage patterns
- **Note constraints**: Preconditions, postconditions, invariants
- **Reference theory**: Link to relevant mathematical concepts

### Documentation Commands

```bash
# Generate and view documentation
cargo doc --open

# Test documentation examples
just test             # Includes doc tests

# Check documentation coverage and validate
just doc-check        # Validates documentation builds for crates.io
```

## Citation and References

### Academic Citations

This library is designed for research and academic use in computational geometry. If you use this library in your research, please cite it appropriately.

#### How to Cite This Library

The project provides standardized citation metadata in [CITATION.cff](CITATION.cff) that can be automatically
processed by GitHub and academic tools. For the most up-to-date citation information, see [REFERENCES.md](REFERENCES.md).

**Quick citation (ACM format):**

```text
Adam Getchell (https://orcid.org/0000-0002-0797-0021). 2025. delaunay: A d-dimensional Delaunay triangulation library.
Zenodo. DOI: https://doi.org/10.5281/zenodo.16931097
```

**BibTeX:**

```bibtex
@software{getchell_delaunay_2025,
  author  = {Adam Getchell},
  title   = {delaunay: A d-dimensional Delaunay triangulation library},
  year    = {2025},
  doi     = {10.5281/zenodo.16931097},
  url     = {https://doi.org/10.5281/zenodo.16931097},
  orcid   = {0000-0002-0797-0021}
}
```

Note: The canonical citation is maintained in [CITATION.cff](CITATION.cff); prefer that as the source of truth.

#### Adding Academic References

When contributing algorithmic improvements or new features based on academic literature:

1. **Update REFERENCES.md**: Add new citations to the appropriate section
2. **Follow the existing format**: Use consistent bibliographic style
3. **Include DOI links**: When available, provide DOI URLs for easy access
4. **Categorize appropriately**: Place references under relevant sections:
   - Core Delaunay Triangulation Algorithms and Data Structures
   - Geometric Predicates and Numerical Robustness
   - Convex Hull Algorithms
   - Advanced Computational Geometry Topics

#### Reference Format Guidelines

Use this format for academic papers:

```text
- Author, A. "Paper Title." *Journal Name* Volume, no. Issue (Year): Pages.
  DOI: [10.xxxx/xxxx](https://doi.org/10.xxxx/xxxx)
```

For books:

```text
- Author, A. "Book Title." Publisher, Year.
```

For online resources:

```text
- [Resource Name](URL)
```

#### Documentation in Code

When implementing algorithms from academic sources:

- **Reference the source** in module or function documentation
- **Explain the algorithm** in computational geometry terms
- **Note any modifications** you made from the original
- **Include complexity analysis** when relevant

Example:

```rust
/// Implements the Bowyer-Watson algorithm for incremental Delaunay triangulation.
/// 
/// Based on:
/// - Bowyer, A. "Computing Dirichlet tessellations." The Computer Journal 24, no. 2 (1981): 162-166.
/// - Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes."
///   The Computer Journal 24, no. 2 (1981): 167-172.
///
/// This implementation extends the original algorithm to d-dimensions and includes
/// robust geometric predicates for numerical stability.
```

### Maintaining Academic Standards

As contributors to a computational geometry library:

- **Respect intellectual property**: Always cite sources for algorithms and ideas
- **Verify mathematical correctness**: Ensure implementations match published algorithms
- **Test against known results**: Use standard test cases from literature when possible
- **Document assumptions**: Note any mathematical assumptions or constraints

For comprehensive bibliographic information, see [REFERENCES.md](REFERENCES.md).

## Performance and Benchmarking

Performance is crucial for computational geometry algorithms.

### Benchmark Infrastructure

The project includes comprehensive benchmarking:

- **Location**: `benches/` directory with detailed [README][benches-readme]
- **Framework**: Criterion with allocation tracking
- **Coverage**: Small-scale triangulations across dimensions
- **Automated Baselines**: Performance baselines are automatically generated on releases

### Performance Testing Workflow

**For development and manual testing:**

```bash
# Run benchmarks directly
just bench

# Run all examples to verify performance
just examples
```

**Note**: The project uses an **automated performance baseline system**:

- **Automatic baseline generation**: Baselines are created automatically when git tags are pushed via GitHub Actions
- **CI regression testing**: Performance regressions are detected automatically in PRs against the latest baseline
- **Hardware compatibility**: The system detects hardware differences and provides warnings when comparing across different configurations
- **5% regression threshold**: CI fails if performance degrades by more than 5%

The old shell scripts (`generate_baseline.sh`, `compare_benchmarks.sh`) mentioned in some documentation have been
**replaced** with Python utilities that integrate with GitHub Actions for automated baseline management.

### Performance Guidelines

- **Algorithmic Complexity**: Document and optimize time/space complexity
- **Memory Allocation**: Minimize unnecessary allocations
- **Numerical Stability**: Balance performance with numerical accuracy
- **Regression Detection**: CI fails on >5% performance regressions
- **Hardware Awareness**: Consider performance implications across different hardware configurations

See [scripts documentation][scripts-readme] for detailed benchmarking workflows and the [WARP.md](WARP.md) file
for implementation details of the automated baseline system.

## Submitting Changes

### Pull Request Process

1. **Create a descriptive PR title**:
   - `feat: add 4D triangulation support`
   - `fix: resolve vertex insertion edge case`
   - `docs: improve boundary analysis examples`

2. **Write a comprehensive PR description**:
   - **Problem**: What issue does this solve?
   - **Solution**: How does your change address it?
   - **Testing**: How did you verify the fix?
   - **Performance**: Any performance implications?

3. **Ensure CI passes**: All checks must be green

4. **Request review**: Tag relevant reviewers

### PR Review Criteria

PRs are evaluated on:

- **Correctness**: Does the code solve the stated problem?
- **Testing**: Are there adequate tests for the changes?
- **Documentation**: Is the code properly documented?
- **Performance**: Are there any performance regressions?
- **Style**: Does the code follow project conventions?
- **Mathematical Accuracy**: Are geometric algorithms correct?

### Handling Feedback

- **Respond to all comments**: Address each piece of feedback
- **Ask for clarification**: If feedback is unclear
- **Make focused updates**: Address feedback in separate commits
- **Re-request review**: After making significant changes

## Types of Contributions

We welcome various types of contributions:

### 🐛 Bug Fixes

- Report bugs with minimal reproduction cases
- Fix algorithmic errors in geometric computations
- Resolve edge cases in triangulation algorithms
- Improve numerical stability

### ✨ Features

- New triangulation algorithms or optimizations
- Additional geometric predicates
- Higher-dimensional support
- Performance improvements

### 📚 Documentation

- Improve API documentation
- Add usage examples (see [examples documentation][examples-readme])
- Write tutorials or guides
- Fix typos and improve clarity

### 🧪 Testing

- Add test cases for edge conditions
- Improve test coverage
- Add property-based tests
- Create benchmark tests

### 🚀 Performance

- Algorithmic optimizations
- Memory usage improvements  
- Parallel processing support
- SIMD optimizations

### 🔧 Infrastructure

- CI/CD improvements
- Build system enhancements
- Development tooling
- Script improvements

## Release Process

The project follows [semantic versioning][semver] and maintains a detailed [CHANGELOG.md][changelog].

### Version Numbering

- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Workflow

For a detailed, copy-pastable, step-by-step workflow (including a clean release PR flow with exact commands), see docs/RELEASING.md.

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions and conversations
- **Email**: [maintainer][maintainer-email] for direct contact

### Resources

- **Documentation**: `cargo doc --open`
- **Examples**: [examples documentation][examples-readme]
- **Benchmarks**: [benchmarks documentation][benches-readme]
- **Scripts**: [scripts documentation][scripts-readme]
- **Code Organization**: [code organization documentation][code-organization]

### When to Ask for Help

- **Algorithm Questions**: Need help with geometric algorithms
- **Performance Issues**: Experiencing unexpected performance problems
- **API Design**: Unsure about API design decisions
- **Testing Strategy**: Need guidance on testing approaches
- **Contribution Process**: Confused about the contribution workflow

### Providing Context

When asking for help, please provide:

- **Version information**: Rust version, crate version
- **Minimal example**: Code that reproduces the issue
- **Expected vs. actual behavior**
- **System information**: OS, hardware (for performance issues)
- **Steps attempted**: What have you tried already?

## Acknowledgments

This project builds upon decades of computational geometry research. We acknowledge:

- The mathematical foundations developed by researchers worldwide
- The Rust community for providing excellent tools and libraries
- Contributors who help improve the library through code, documentation, and feedback
- Users who provide valuable bug reports and feature requests

Thank you for contributing to the advancement of computational geometry in Rust! 🦀

---

**Questions?** Don't hesitate to ask in GitHub Issues or reach out to the [maintainer][maintainer-email].

*This document is living and evolves with the project. Suggestions for improvements are always welcome!*

<!-- Links -->
[delaunay-lib]: https://github.com/acgetchell/delaunay
[code-of-conduct]: CODE_OF_CONDUCT.md
[changelog]: CHANGELOG.md
[examples-readme]: examples/README.md
[benches-readme]: benches/README.md
[scripts-readme]: scripts/README.md
[code-organization]: docs/code_organization.md
[maintainer-email]: mailto:adam@adamgetchell.org
[semver]: https://semver.org/
[rustup]: https://rustup.rs/
