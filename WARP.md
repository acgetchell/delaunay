# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

The `delaunay` library implements d-dimensional Delaunay triangulations in Rust,
inspired by CGAL. It provides a lightweight alternative for computational geometry
applications with support for arbitrary data types associated with vertices and
cells, d-dimensional triangulations, and serialization/deserialization capabilities.

## Essential Development Commands

### Building and Testing

```bash
# Build the library
cargo build

# Build in release mode
cargo build --release

# Run all tests (library, doc tests, examples)
cargo test --lib --verbose
cargo test --doc --verbose  
cargo test --examples --verbose

# Run tests with allocation counting feature
cargo test allocation_counting --features count-allocations
```

### Code Quality

```bash
# Format code
cargo fmt --all

# Check formatting without modifying files
cargo fmt --all -- --check

# Run clippy (strict pedantic mode configured)
cargo clippy --all-targets --all-features -- -D warnings -D clippy::all -D clippy::pedantic -W clippy::nursery -W clippy::cargo
```

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench --bench circumsphere_containment
cargo bench --bench small_scale_triangulation
cargo bench --bench triangulation_creation

# Generate performance baseline for CI
./scripts/generate_baseline.sh

# Generate baseline for development (faster)
./scripts/generate_baseline.sh --dev

# Compare performance against baseline
./scripts/compare_benchmarks.sh

# Compare with development settings (faster)
./scripts/compare_benchmarks.sh --dev
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
# Generate complete changelog from all commits and update CHANGELOG.md
git-cliff --output CHANGELOG.md

# Generate changelog for latest version only and update CHANGELOG.md
git-cliff --latest --output CHANGELOG.md

# Generate changelog for unreleased changes since last tag
git-cliff --unreleased --output CHANGELOG.md

# Prepend new entries to existing CHANGELOG.md (useful for releases)
git-cliff --prepend CHANGELOG.md

# Generate changelog for specific tag range
git-cliff v0.3.0..HEAD --output CHANGELOG.md
```

## Architecture Overview

### Core Module Structure

**`src/core/`** - Primary data structures and algorithms

- **`triangulation_data_structure.rs`** - Main `Tds` struct implementing Bowyer-Watson algorithm
- **`vertex.rs`** - Vertex representation with generic coordinate support
- **`cell.rs`** - D-dimensional cells (simplices) with neighbor relationships
- **`facet.rs`** - (D-1)-dimensional faces derived from cells
- **`boundary.rs`** - Boundary analysis and facet detection
- **`utilities.rs`** - Helper functions for triangulation operations
- **`traits/`** - Core traits for data types and boundary analysis

**`src/geometry/`** - Geometric algorithms and predicates

- **`point.rs`** - Generic Point struct with NaN-aware equality and hashing
- **`predicates.rs`** - Geometric predicates (insphere, orientation tests)
- **`matrix.rs`** - Matrix operations for geometric computations
- **`traits/`** - Coordinate abstractions and floating-point traits

**`src/prelude.rs`** - Re-exports commonly used types and macros for convenience

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
5. **Run examples** with `./scripts/run_all_examples.sh`
6. **Check performance impact** with `./scripts/compare_benchmarks.sh --dev`
7. **Commit and push** for CI validation

### Performance-Critical Changes

For changes affecting algorithmic performance:

1. **Generate baseline** before changes: `./scripts/generate_baseline.sh`
2. **Make modifications**
3. **Test regression** with `./scripts/compare_benchmarks.sh`
4. **If >5% regression detected**, investigate and optimize
5. **Update baseline** if performance change is acceptable

### Adding New Examples

Examples in `examples/` demonstrate library capabilities:

- Use `delaunay::prelude::*` for imports
- Include comprehensive documentation header
- Show error handling patterns
- Validate with `./scripts/run_all_examples.sh`

## Benchmarking and Performance

### Key Benchmarks

- **`circumsphere_containment`** - Geometric predicate performance across methods
- **`small_scale_triangulation`** - Triangulation creation for 10-50 points in 2D/3D/4D
- **`boundary_facets_performance`** - Boundary detection algorithm performance
- **`assign_neighbors_performance`** - Neighbor relationship computation

### Performance Characteristics

- **2D triangulations**: Excellent performance, sub-millisecond for ≤10 points
- **3D triangulations**: ~10x slower than 2D, exponential scaling beyond 30 points  
- **4D triangulations**: Highest complexity, suitable for small-scale problems (≤30 points)
- **Boundary detection**: O(N·F) complexity via HashMap-based optimization

### Regression Testing

- Baselines stored in `benches/baseline_results.txt` with git commit tracking
- CI automatically detects >5% performance degradation
- Development mode (`--dev`) provides 10x faster feedback during iteration

## Coding Standards and Patterns

### Language Requirements

- **Rust Edition**: 2024
- **MSRV**: 1.85.0  
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

- **`scripts/generate_baseline.sh`** - Create performance baselines
- **`scripts/compare_benchmarks.sh`** - Performance regression testing
- **`scripts/run_all_examples.sh`** - Validate all examples
- **`scripts/benchmark_parser.sh`** - Shared benchmark parsing utilities

### Mathematical References

The library implements algorithms based on established computational geometry literature:

- Bowyer-Watson algorithm for Delaunay triangulation
- Shewchuk's robust geometric predicates
- Lifted paraboloid method for circumsphere tests
- CGAL-inspired triangulation data structures
