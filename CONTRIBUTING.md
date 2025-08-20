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
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
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
3. **System dependencies** (for running scripts):
   - **macOS**: `brew install jq findutils coreutils`
   - **Ubuntu/Debian**: `sudo apt-get install jq findutils coreutils bc`
   - **Other systems**: Install equivalent packages for `jq`, `find`, `sort`, and `bc`

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
   cargo test
   ```

4. **Try the examples**:

   ```bash
   cargo run --example triangulation_3d_50_points
   ./scripts/run_all_examples.sh  # Run all examples
   ```

5. **Run benchmarks** (optional):

   ```bash
   cargo bench
   ./scripts/generate_baseline.sh  # Create performance baseline
   ```

## Development Environment Setup

### Recommended Tools

- **IDE/Editor**: Any editor with Rust Language Server (rust-analyzer) support
- **Linting**: The project uses strict clippy lints - ensure your editor shows clippy warnings
- **Formatting**: Use `rustfmt` for code formatting (configured in `rustfmt.toml`)

### Project Configuration

The project uses:

- **Edition**: Rust 2024
- **MSRV**: Rust 1.85.0
- **Linting**: Strict clippy pedantic mode
- **Testing**: Standard `#[test]` with comprehensive coverage
- **Benchmarking**: Criterion with allocation tracking

## Project Structure

Understanding the project layout will help you navigate and contribute effectively:

```text
delaunay/
├── src/                                          # Core library code
│   ├── core/                                     # Core triangulation structures
│   │   ├── algorithms/                           # Triangulation algorithms
│   │   │   ├── bowyer_watson.rs                  # Incremental Bowyer-Watson algorithm
│   │   │   └── robust_bowyer_watson.rs           # Robust geometric predicates version
│   │   ├── boundary.rs                           # Boundary analysis and facet detection
│   │   ├── cell.rs                               # Cell (simplex) implementation
│   │   ├── facet.rs                              # Facet implementation
│   │   ├── triangulation_data_structure.rs       # Main Tds struct
│   │   ├── util.rs                               # Helper functions for triangulation operations
│   │   ├── vertex.rs                             # Vertex implementation with generic support
│   │   └── traits/                               # Core traits for data types and algorithms
│   │       ├── boundary_analysis.rs              # Boundary analysis traits
│   │       ├── data_type.rs                      # DataType trait definitions
│   │       └── insertion_algorithm.rs            # Insertion algorithm traits
│   ├── geometry/                                 # Geometric algorithms and predicates
│   │   ├── algorithms/                           # Geometric algorithms
│   │   │   └── convex_hull.rs                    # Convex hull computation
│   │   ├── matrix.rs                             # Matrix operations for geometric computations
│   │   ├── point.rs                              # Generic Point struct with NaN-aware operations
│   │   ├── predicates.rs                         # Geometric predicates (insphere, orientation)
│   │   ├── robust_predicates.rs                  # Robust geometric predicates
│   │   ├── util.rs                               # Geometric utility functions
│   │   └── traits/                               # Coordinate abstractions and floating-point traits
│   │       ├── coordinate.rs                     # Core Coordinate trait abstraction
│   │       ├── finitecheck.rs                    # Finite value validation traits
│   │       ├── hashcoordinate.rs                 # Floating-point hashing traits
│   │       └── orderedeq.rs                      # Ordered equality comparison traits
│   └── lib.rs                                    # Main library file with module declarations and prelude
├── examples/                                     # Usage examples and demonstrations
│   ├── README.md                                 # Examples documentation
│   ├── boundary_analysis_trait.rs                # Boundary analysis examples
│   ├── check_float_traits.rs                     # Floating-point trait examples
│   ├── implicit_conversion.rs                    # Type conversion examples
│   ├── point_comparison_and_hashing.rs           # Point operations examples
│   ├── test_alloc_api.rs                         # Allocation API examples
│   ├── test_circumsphere.rs                      # Circumsphere computation examples
│   └── triangulation_3d_50_points.rs             # 3D triangulation example
├── benches/                                      # Performance benchmarks
│   ├── README.md                                 # Benchmarking guide and performance results
│   ├── assign_neighbors_performance.rs           # Neighbor assignment benchmarks
│   ├── baseline_results.txt                      # Performance baseline data
│   ├── circumsphere_containment.rs               # Circumsphere predicate benchmarks
│   ├── helpers.rs                                # Benchmark helper functions
│   ├── microbenchmarks.rs                        # Fine-grained performance tests
│   ├── small_scale_triangulation.rs              # Small triangulation benchmarks
│   └── triangulation_creation.rs                 # Triangulation creation benchmarks
├── tests/                                        # Integration tests
│   ├── bench_helpers_test.rs                     # Tests for benchmark helper functions
│   ├── convex_hull_bowyer_watson_integration.rs  # Integration tests for convex hull and Bowyer-Watson
│   ├── robust_predicates_comparison.rs           # Robust vs standard predicates comparison tests
│   ├── robust_predicates_showcase.rs             # Robust predicates demonstration tests
│   └── test_cavity_boundary_error.rs             # Cavity boundary error reproduction tests
├── docs/                                         # Additional documentation
│   ├── templates/                                # Templates for automated generation
│   │   ├── README.md                             # Templates documentation
│   │   └── changelog.hbs                         # Custom changelog template
│   ├── code_organization.md                      # Code organization patterns
│   └── optimization_recommendations.md           # Performance optimization guide
├── scripts/                                      # Development and CI scripts
│   ├── README.md                                 # Scripts documentation
│   ├── benchmark_parser.sh                       # Shared benchmark parsing utilities
│   ├── compare_benchmarks.sh                     # Performance regression testing
│   ├── generate_baseline.sh                      # Create performance baselines
│   ├── generate_changelog.sh                     # Generate changelog with commit dates
│   ├── hardware_info.sh                          # Hardware information and system capabilities
│   ├── run_all_examples.sh                       # Validate all examples
│   └── tag-from-changelog.sh                     # Create git tags from changelog content
├── .cargo/                                       # Cargo configuration
│   └── config.toml                               # Build configuration
├── .github/                                      # GitHub configuration
│   ├── CODEOWNERS                                # Code ownership definitions
│   ├── dependabot.yml                            # Dependency update configuration
│   └── workflows/                                # CI/CD workflows
│       ├── audit.yml                             # Security vulnerability scanning
│       ├── benchmarks.yml                        # Performance regression testing
│       ├── ci.yml                                # Main CI pipeline
│       ├── codacy.yml                            # Code quality analysis
│       ├── codecov.yml                           # Test coverage tracking
│       ├── codeql.yml                            # Security analysis
│       └── rust-clippy.yml                       # Additional clippy analysis
├── .auto-changelog                               # Auto-changelog configuration
├── .codecov.yml                                  # CodeCov configuration
├── .coderabbit.yml                               # CodeRabbit AI review configuration
├── .gitignore                                    # Git ignore patterns
├── .markdownlint.json                            # Markdown linting configuration
├── .yamllint                                     # YAML linting configuration
├── CHANGELOG.md                                  # Version history
├── CODE_OF_CONDUCT.md                            # Community guidelines
├── CONTRIBUTING.md                               # This file
├── Cargo.lock                                    # Dependency lockfile
├── Cargo.toml                                    # Package configuration and dependencies
├── cspell.json                                   # Spell checking configuration
├── LICENSE                                       # MIT License
├── README.md                                     # Project overview and getting started
├── rustfmt.toml                                  # Code formatting configuration
└── WARP.md                                       # WARP AI development guidance
```

For detailed code organization patterns, see [code organization documentation][code-organization].

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

### 4. Continuous Integration

The project uses comprehensive CI workflows:

- **Main CI** (`.github/workflows/ci.yml`): Build, test, lint on every PR
- **Benchmarks** (`.github/workflows/benchmarks.yml`): Performance regression testing
- **Security** (`.github/workflows/audit.yml`): Dependency vulnerability scanning
- **Code Quality** (`.github/workflows/rust-clippy.yml`): Strict linting
- **Coverage** (`.github/workflows/codecov.yml`): Test coverage tracking

All PRs must pass CI checks before merging.

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
|------|-------------|-----------------------|
| `feat` | New features | ✅ Yes |
| `fix` | Bug fixes | ✅ Yes |
| `perf` | Performance improvements | ✅ Yes |
| `refactor` | Code refactoring | ✅ Yes |
| `build` | Build system changes | ✅ Yes |
| `ci` | CI/CD changes | ✅ Yes |
| `revert` | Reverting changes | ✅ Yes |
| `chore` | Maintenance tasks | ❌ No (filtered) |
| `style` | Formatting changes | ❌ No (filtered) |
| `docs` | Documentation only | ❌ No (filtered) |
| `test` | Test changes only | ❌ No (filtered) |

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
cargo test --doc

# Check documentation coverage
cargo doc --document-private-items
```

## Performance and Benchmarking

Performance is crucial for computational geometry algorithms.

### Benchmark Infrastructure

The project includes comprehensive benchmarking:

- **Location**: `benches/` directory with detailed [README][benches-readme]
- **Framework**: Criterion with allocation tracking
- **Coverage**: Small-scale triangulations across dimensions

### Performance Testing Workflow

```bash
# Generate performance baseline (first time)
./scripts/generate_baseline.sh

# Test for performance regressions
./scripts/compare_benchmarks.sh

# Development mode (faster iteration)
./scripts/compare_benchmarks.sh --dev
```

### Performance Guidelines

- **Algorithmic Complexity**: Document and optimize time/space complexity
- **Memory Allocation**: Minimize unnecessary allocations
- **Numerical Stability**: Balance performance with numerical accuracy
- **Regression Detection**: CI fails on >5% performance regressions

See [scripts documentation][scripts-readme] for detailed benchmarking workflows.

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

1. **Update version** in `Cargo.toml` and any documentation if needed
2. **Create temporary tag** to enable changelog generation:

   ```bash
   git tag -a v0.3.5 -m "delaunay v0.3.5"
   ```

3. **Generate changelog**:

   ```bash
   ./scripts/generate_changelog.sh
   ```

   Note: do not push this temporary tag, it will be recreated later with full changelog content

4. **Commit all changes together**:

   ```bash
   git add Cargo.toml CHANGELOG.md docs/ # Add any updated files
   git commit -m "chore(release): release v0.3.5
   
   - Bump version to v0.3.5
   - Update changelog with latest changes  
   - Update documentation for release"
   ```

5. **Move tag to final release commit with changelog content**:

   ```bash
   # Delete temporary tag and recreate with changelog content
   git tag -d v0.3.5
   ./scripts/tag-from-changelog.sh v0.3.5 --force
   ```

   The `tag-from-changelog.sh` script extracts the changelog section that matches
   the specified version (v0.3.5) from CHANGELOG.md and uses it as the tag message.
   It supports common changelog formats including `## [v0.3.5] ...`, `## v0.3.5 ...`,
   and `## 0.3.5 ...`. The script ensures you get the correct version's changelog
   content rather than an unrelated section.

6. **Verify tag annotation** (optional but recommended):

   ```bash
   # View the tag message content that will be used for GitHub release
   git tag -l --format='%(contents)' v0.3.5
   ```

   This shows exactly what content will be used when creating the GitHub release.

7. **Push changes and tag**:

   ```bash
   git push --atomic origin main v0.3.5
   ```

8. **Publish to crates.io** (maintainer only):

   ```bash
   cargo publish --dry-run # Validate package
   cargo publish
   ```

9. **Create GitHub release** (maintainer only):

   ```bash
   # Create release using changelog content from tag message
   gh release create v0.3.5 --notes-from-tag
   ```

**Note**: The project uses `./scripts/generate_changelog.sh` to generate changelogs with commit dates instead of tag creation dates,
providing more accurate release timing that reflects when development work was completed.

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
