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
cargo test --features count-allocations -- allocation_counting 
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
# Generate changelog with commit dates (recommended)
./scripts/generate_changelog.sh

# Generate changelog with unreleased changes included
npx auto-changelog --unreleased

# Generate changelog for specific version
npx auto-changelog --latest-version v0.3.4

# Generate changelog with custom commit limit per release
npx auto-changelog --commit-limit 10

# Generate changelog using Keep a Changelog format overrides the project template configured in .auto-changelog config)
npx auto-changelog --template keepachangelog

# Test changelog generation without writing to file
npx auto-changelog --stdout
```

**Note**: The project uses `./scripts/generate_changelog.sh` to generate changelogs with commit dates instead of tag creation dates.
This provides more accurate release timing that reflects when the actual work was completed rather than when tags were created.

## Architecture Overview

### Core Module Structure

**`src/core/`** - Primary data structures and algorithms

- **`triangulation_data_structure.rs`** - Main `Tds` struct implementing Bowyer-Watson algorithm
- **`vertex.rs`** - Vertex representation with generic coordinate support
- **`cell.rs`** - D-dimensional cells (simplices) with neighbor relationships
- **`facet.rs`** - (D-1)-dimensional faces derived from cells
- **`boundary.rs`** - Boundary analysis and facet detection
- **`util.rs`** - Helper functions for triangulation operations
- **`traits/`** - Core traits for data types and boundary analysis

**`src/geometry/`** - Geometric algorithms and predicates

- **`point.rs`** - Generic Point struct with NaN-aware equality and hashing
- **`predicates.rs`** - Geometric predicates (insphere, orientation tests)
- **`matrix.rs`** - Matrix operations for geometric computations
- **`traits/`** - Coordinate abstractions and floating-point traits

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
- **`assign_neighbors_performance`** - Neighbor relationship computation
- **`microbenchmarks`** - Fine-grained performance tests for specific operations
- **`triangulation_creation`** - Basic triangulation creation benchmarks

### Performance Characteristics

- **2D triangulations**: Excellent performance, sub-millisecond for ≤10 points

---

## CURRENT REFACTORING PROJECT: Pure Incremental Delaunay Triangulation

### Overview

We're refactoring from a batch-oriented approach using supercells to a pure incremental approach. This will provide better performance,
cleaner code organization, and support for future vertex deletion operations.

### Current Issues (Pre-Refactor)

1. **Mixed Approaches**: The current `bowyer_watson` method is designed for batch processing (all vertices at once) while the `add` method tries to do
   incremental insertion
2. **Inefficient Hull Recomputation**: Each vertex insertion recomputes the convex hull from scratch, resulting in O(N²) complexity instead of O(N log N)
3. **Complex Supercell Logic**: Batch approach requires complex supercell creation, insertion, and cleanup
4. **Inconsistent State Handling**: Different code paths for D+1 vertices vs subsequent vertices

### New Architecture (Pure Incremental)

#### Core Algorithm Module

- **Location**: `src/core/algorithms/bowyer_watson.rs`
- **Main struct**: `IncrementalBoyerWatson<T, U, V, D>`
- **Key methods**:
  - `initialize_triangulation()` - Handle first D+1 vertices
  - `insert_vertex()` - Core incremental insertion
  - `insert_inside_vertex()` - Standard Bowyer-Watson for interior points
  - `insert_outside_vertex()` - Hull extension for exterior points

#### Cached Hull System

- **Simple ConvexHull struct** with boundary facets and bounding box
- **Incremental updates** instead of full recomputation
- **Efficient inside/outside testing** using cached data

#### Clean TDS Integration

- Remove all supercell methods: `supercell()`, `create_default_supercell()`, `remove_cells_containing_supercell_vertices()`
- Simplify `add()` method to consistent incremental approach
- Better construction state tracking

### Benefits

1. **Performance**: O(N log N) instead of O(N²) for N vertex insertions
2. **Consistency**: Same algorithm path for all vertices after D+1
3. **Simplicity**: No complex supercell machinery
4. **Maintainability**: Clear separation between algorithm and data structure
5. **Future-Ready**: Foundation for vertex deletion operations
6. **Memory Efficiency**: Cached hull avoids expensive recomputation

### Implementation Status

#### Completed ✅

- ✅ **Core Algorithm Implementation**: Complete `IncrementalBoyerWatson` struct in `src/core/algorithms/bowyer_watson.rs`
- ✅ **Trait Bounds Resolution**: Fixed const generics trait bounds (`[f64; D]: Default + DeserializeOwned + Serialize + Sized`)
- ✅ **Geometric Predicates Integration**: Working `simplex_orientation` and `insphere` tests with proper error handling
- ✅ **Multiple Insertion Strategies**:
  - **Cavity-based insertion** for interior vertices (standard Bowyer-Watson)
  - **Hull extension** for exterior vertices (convex hull expansion)
  - **Fallback insertion** for edge cases
- ✅ **Visibility Testing**: Complete implementation of facet visibility predicates for hull extension
- ✅ **Boundary Facet Detection**: Robust algorithm to find cavity boundaries after bad cell removal
- ✅ **Cell Creation and Management**: Proper cell creation from facets with UUID mapping
- ✅ **Algorithm Statistics**: Insertion tracking with cells created/removed counts
- ✅ **Comprehensive Testing**: All diagnostic tests passing with proper invariant validation
- ✅ **Warning-Free Codebase**: Consolidated tests, eliminated all compilation warnings
- ✅ **Triangulation Validation**: All geometric invariants satisfied (boundary facets, neighbor relationships)

#### Algorithm Features ✅

- ✅ **Strategy Selection**: Automatic interior vs exterior vertex detection via circumsphere tests
- ✅ **Bad Cell Detection**: Conservative circumsphere containment testing to prevent over-removal
- ✅ **Cavity Triangulation**: Proper boundary facet identification and new cell creation
- ✅ **Hull Extension**: Geometric visibility testing for exterior vertex insertion
- ✅ **Robust Fallback**: Aggressive fallback trying all facets when standard methods fail
- ✅ **Topology Preservation**: Maintains valid neighbor relationships and facet sharing
- ✅ **Dimension Generic**: Works for arbitrary dimensions with const generic `D`

#### Test Results ✅

- ✅ **Simple Tetrahedron**: 4 vertices → 1 cell, 4 boundary facets (perfect)
- ✅ **Hull Extension**: 5 vertices → 2 cells, 6 boundary facets, 1 internal facet (optimal)
- ✅ **Complex Geometry**: 5-point challenging configurations handled correctly
- ✅ **Invariant Preservation**: Zero invalid facet sharing, proper topology maintained
- ✅ **Algorithm Components**: Strategy selection, bad cell detection, insertion all working

#### August 16, 2025 Update ✅

- ✅ **TDS Refactoring Complete**: Successfully removed all legacy methods and integrated IncrementalBoyerWatson
- ✅ **Legacy Code Cleanup**: Removed outdated find_bad_cells, find_boundary_facets, and vertex insertion methods  
- ✅ **Buffer Architecture**: Moved algorithm buffers from TDS struct to IncrementalBoyerWatson for better separation
- ✅ **Deserialization Updates**: Fixed custom Deserialize implementation for removed buffer fields
- ✅ **Import Cleanup**: Resolved unused import warnings by moving test-only imports to #[cfg(test)] sections
- ✅ **Full Test Coverage**: All 448 tests passing with zero compilation warnings
- ✅ **Clean Architecture**: Achieved proper separation between data structure (TDS) and algorithms (IncrementalBoyerWatson)

#### Next Phase 🔄

- 🔄 **Performance Optimization**: Add hull caching for O(N log N) complexity improvements
- 🔄 **Deletion Support**: Foundation for vertex deletion operations
- 🔄 **Benchmark Integration**: Performance testing against previous implementation
- 🔄 **Documentation Updates**: Update API documentation to reflect new architecture

### Notes

- **No mod.rs files**: Per project preference, modules are defined in lib.rs
- **Type Safety**: All existing type safety and validation preserved
- **API Compatibility**: External API remains the same, internal implementation improved
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
