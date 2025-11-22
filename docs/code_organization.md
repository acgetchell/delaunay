# Code Organization Guide

This document provides a comprehensive guide to the delaunay project's code organization, from the overall project architecture to detailed individual module patterns.

## Table of Contents

- [Project Structure](#project-structure)
  - [Complete Directory Tree](#complete-directory-tree)
  - [Architecture Overview](#architecture-overview)
  - [Architectural Principles](#architectural-principles)
- [Module Organization Patterns](#module-organization-patterns)
  - [Canonical Section Sequence](#canonical-section-sequence)
  - [Comment Separators](#comment-separators)
  - [Section-by-Section Analysis](#section-by-section-analysis)
  - [Module-Specific Variations](#module-specific-variations)
  - [Key Conventions](#key-conventions)

---

## Project Structure

The delaunay project follows a standard Rust library structure with additional tooling for computational geometry research.

### Complete Directory Tree

> **Tip**: Generate this tree in CI
>
> ```bash
> # Requires tree command (install with: brew install tree or apt-get install tree)
> git ls-files | sed 's/^/delaunay\//' | tree -I 'target|.git|**/*.png|**/*.svg' -F --fromfile
>
> # Alternative using find (when tree is not available):
> find . -type f -name "*.rs" -o -name "*.md" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" | sort
> ```
>
> This keeps the directory tree automatically synchronized with the actual project structure.

```text
delaunay/
├── benches/
│   ├── assign_neighbors_performance.rs
│   ├── ci_performance_suite.rs
│   ├── circumsphere_containment.rs
│   ├── large_scale_performance.rs
│   ├── microbenchmarks.rs
│   ├── PERFORMANCE_RESULTS.md
│   ├── profiling_suite.rs
│   ├── README.md
│   └── triangulation_creation.rs
├── Cargo.lock
├── Cargo.toml
├── CHANGELOG.md
├── CITATION.cff
├── clippy.toml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── cspell.json
├── docs/
│   ├── archive/
│   │   ├── fix-delaunay.md
│   │   ├── invariant_validation_plan.md
│   │   ├── jaccard.md
│   │   ├── optimization_recommendations_historical.md
│   │   ├── phase_3a_implementation_guide.md
│   │   ├── phase_3c_action_plan.md
│   │   ├── phase2_bowyer_watson_optimization.md
│   │   ├── phase2_uuid_iter_optimization.md
│   │   └── testing.md
│   ├── code_organization.md
│   ├── numerical_robustness_guide.md
│   ├── OPTIMIZATION_ROADMAP.md
│   ├── phase4.md
│   ├── property_testing_summary.md
│   ├── README.md
│   ├── RELEASING.md
│   ├── templates/
│   │   ├── changelog.hbs
│   │   └── README.md
│   └── topology.md
├── examples/
│   ├── convex_hull_3d_20_points.rs
│   ├── into_from_conversions.rs
│   ├── memory_analysis.rs
│   ├── point_comparison_and_hashing.rs
│   ├── README.md
│   ├── triangulation_3d_20_points.rs
│   └── zero_allocation_iterator_demo.rs
├── justfile
├── LICENSE
├── pyproject.toml
├── README.md
├── REFERENCES.md
├── rust-toolchain.toml
├── rustfmt.toml
├── scripts/
│   ├── benchmark_models.py
│   ├── benchmark_utils.py
│   ├── changelog_utils.py
│   ├── compare_storage_backends.py
│   ├── enhance_commits.py
│   ├── hardware_utils.py
│   ├── README.md
│   ├── run_all_examples.sh
│   ├── slurm_storage_comparison.sh
│   ├── subprocess_utils.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_benchmark_models.py
│       ├── test_benchmark_utils.py
│       ├── test_changelog_utils.py
│       ├── test_compare_storage_backends.py
│       ├── test_enhance_commits.py
│       ├── test_hardware_utils.py
│       └── test_subprocess_utils.py
├── src/
│   ├── core/
│   │   ├── algorithms/
│   │   │   ├── bowyer_watson.rs
│   │   │   ├── robust_bowyer_watson.rs
│   │   │   └── unified_insertion_pipeline.rs
│   │   ├── boundary.rs
│   │   ├── cell.rs
│   │   ├── collections.rs
│   │   ├── facet.rs
│   │   ├── traits/
│   │   │   ├── boundary_analysis.rs
│   │   │   ├── data_type.rs
│   │   │   ├── facet_cache.rs
│   │   │   └── insertion_algorithm.rs
│   │   ├── triangulation_data_structure.rs
│   │   ├── util.rs
│   │   └── vertex.rs
│   ├── geometry/
│   │   ├── algorithms/
│   │   │   └── convex_hull.rs
│   │   ├── matrix.rs
│   │   ├── point.rs
│   │   ├── predicates.rs
│   │   ├── quality.rs
│   │   ├── robust_predicates.rs
│   │   ├── traits/
│   │   │   └── coordinate.rs
│   │   └── util.rs
│   └── lib.rs
├── tests/
│   ├── allocation_api.rs
│   ├── circumsphere_debug_tools.rs
│   ├── convex_hull_bowyer_watson_integration.rs
│   ├── coordinate_conversion_errors.rs
│   ├── COVERAGE.md
│   ├── integration_robust_bowyer_watson.rs
│   ├── proptest_bowyer_watson.rs
│   ├── proptest_cell.rs
│   ├── proptest_convex_hull.rs
│   ├── proptest_delaunay_condition.rs
│   ├── proptest_facet_cache.rs
│   ├── proptest_facet.rs
│   ├── proptest_geometry.rs
│   ├── proptest_invariants.rs
│   ├── proptest_point.rs
│   ├── proptest_predicates.rs
│   ├── proptest_quality.rs
│   ├── proptest_robust_bowyer_watson.rs
│   ├── proptest_safe_conversions.rs
│   ├── proptest_serialization.rs
│   ├── proptest_triangulation.rs
│   ├── proptest_vertex.rs
│   ├── README.md
│   ├── regression_delaunay_known_configs.rs
│   ├── robust_predicates_comparison.rs
│   ├── robust_predicates_showcase.rs
│   ├── serialization_vertex_preservation.rs
│   ├── storage_backend_compatibility.rs
│   ├── tds_basic_integration.rs
│   ├── test_cavity_boundary_error.rs
│   ├── test_convex_hull_error_paths.rs
│   ├── test_facet_cache_integration.rs
│   ├── test_geometry_util.rs
│   ├── test_insertion_algorithm_trait.rs
│   ├── test_insertion_algorithm_utils.rs
│   ├── test_robust_fallbacks.rs
│   └── test_tds_edge_cases.rs
└── WARP.md
```

**Note**: `tests/circumsphere_debug_tools.rs` contains interactive debugging test functions that can be run with:

```bash
# Run debug tests with interactive output (just command)
just test-debug

# Or run specific test functions with verbose output (direct cargo)
cargo test --test circumsphere_debug_tools test_2d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools test_3d_circumsphere_debug -- --nocapture
cargo test --test circumsphere_debug_tools test_all_debug -- --nocapture
# Or run all debug tests at once
cargo test --test circumsphere_debug_tools -- --nocapture
```

**Note**: Delaunay-specific regression and stepwise debug harnesses live in:

- `tests/regression_delaunay_known_configs.rs` – macro-based regression tests
  for canonical Delaunay-violation point sets (currently 5D), kept `#[ignore]`
  until the insertion pipeline is fixed.
- `src/core/algorithms/unified_insertion_pipeline.rs` (test module) – contains
  the stepwise unified-insertion debug test
  `debug_5d_stepwise_insertion_of_seventh_vertex` and the generic helper
  `run_stepwise_unified_insertion_debug` for 2D–5D.

Example invocations:

```bash
# Run the 5D unified insertion stepwise debug test (ignored by default)
cargo test --lib unified_insertion_pipeline::tests::debug_5d_stepwise_insertion_of_seventh_vertex -- --ignored --nocapture

# Run all known-config regression tests (ignored by default)
cargo test --test regression_delaunay_known_configs -- --ignored --nocapture
```

**Note**: Memory allocation profiling is available through the `count-allocations` feature:

```bash
# Run allocation profiling tests (just command)
just test-allocation

# Run benchmarks with allocation counting (direct cargo for specific bench)
cargo bench --bench profiling_suite --features count-allocations
```

> **Allocator Requirements**: Results depend on the system allocator (typically the default allocator on stable Rust).
> For consistent results across environments, ensure the same allocator is used. The `allocation-counter` crate works
> with the global allocator interface.

**Note**: Benchmark-style tests are available through the `bench` feature for performance analysis and demonstrations:

```bash
# Run regular tests (just command)
just test

# Run all tests including benchmark-style performance analysis
cargo test --lib --features bench
```

> **CI Stability**: The `bench` feature gates timing-based tests that may be flaky in CI environments.
> These tests are designed for local performance analysis and ergonomics validation rather than
> deterministic unit testing. Use `--features bench` when conducting performance investigations.

**Note**: Basic TDS integration tests validate fundamental triangulation data structure operations:

```bash
# Run basic TDS creation, neighbor assignment, and boundary tests
cargo test --test tds_basic_integration
# Or with detailed output
cargo test --test tds_basic_integration -- --nocapture
```

**Note**: Robust predicates testing demonstrates cases where enhanced numerical stability prevents triangulation failures:

```bash
# Run robust predicates showcase (demonstrates real problem solving)
cargo test --test robust_predicates_showcase -- --nocapture
# Run numerical accuracy comparisons
cargo test --test robust_predicates_comparison
# Run coordinate conversion error handling tests
cargo test --test coordinate_conversion_errors
```

**Note**: Python tests in `scripts/tests/` are executed via pytest (use `uv run pytest` for reproducible envs) and discovered via `pyproject.toml`. Run with:

```bash
# Run all Python utility tests (just command)
just test-python

# Or run specific test files directly
uv run pytest scripts/tests/test_benchmark_utils.py

# Without uv:
pytest scripts/tests/test_benchmark_utils.py
```

**Note**: Performance summary generation is available through the benchmark utilities CLI:

```bash
# Generate performance summary in benches/PERFORMANCE_RESULTS.md (just command)
just bench-baseline

# Or use the CLI directly for more options
uv run benchmark-utils generate-summary
```

The `benchmark-utils` CLI provides integrated benchmark workflow functionality including performance summary generation,
with convenient `just` shortcuts for common workflows.

### Architecture Overview

#### Core Library (`src/`)

**`src/core/`** - Triangulation data structures and algorithms:

- `triangulation_data_structure.rs` - Main `Tds` struct
- `vertex.rs`, `cell.rs`, `facet.rs` - Core geometric primitives
- `collections.rs` - Optimized collection types and utilities
- `boundary.rs` - Boundary detection and analysis
- `algorithms/` - Bowyer-Watson implementations (standard and robust)
- `traits/` - Core trait definitions including FacetCacheProvider for performance optimization

**`src/geometry/`** - Geometric algorithms and predicates:

- `point.rs` - NaN-aware Point operations
- `predicates.rs`, `robust_predicates.rs` - Geometric tests (see [Numerical Robustness Guide](numerical_robustness_guide.md))
- `quality.rs` - Cell quality metrics (radius ratio, normalized volume) for d-dimensional simplices; provides mesh quality analysis to identify
  poorly-shaped cells (supports 2D-6D)
- `matrix.rs` - Linear algebra support
- `algorithms/convex_hull.rs` - Hull extraction
- `traits/coordinate.rs` - Coordinate abstractions

#### Development Infrastructure

- **`examples/`** - Usage demos and trait examples, including memory profiling
  (see: [examples/memory_analysis.rs](../examples/README.md#5-memory-analysis-across-dimensions-memory_analysisrs)) and zero-allocation iterator demonstrations
- **`benches/`** - Performance benchmarks with automated baseline management (2D-5D coverage) and memory allocation tracking
  (see: [benches/profiling_suite.rs](../benches/README.md#profiling-suite-comprehensive))
- **`tests/`** - Integration tests including basic TDS validation (creation, neighbor assignment, boundary analysis),
  debugging utilities, regression testing, allocation profiling tools
  (see: [tests/allocation_api.rs](../tests/README.md#allocation_apirs)), and robust predicates validation
- **`docs/`** - Architecture guides, performance documentation, numerical robustness guide, and templates
- **`scripts/`** - Python utilities for automation and CI integration

#### Configuration

- **Quality Control**: `.codacy.yml`, `rustfmt.toml`, `pyproject.toml`, linting configurations
- **Environment**: `rust-toolchain.toml`, `.python-version`, `.cargo/config.toml`, GitHub Actions workflows
- **Development Workflow**: `justfile` with automated commands for common development tasks (see [Development Workflow](#development-workflow) below)
- **Memory Profiling**: `count-allocations` feature flag, allocation-counter dependency, profiling benchmarks
- **Performance Analysis**: `bench` feature flag for timing-based tests and performance demos (see "Benchmark-style tests" note above)
- **Project Metadata**: `CITATION.cff`, `REFERENCES.md`, `WARP.md`

### Architectural Principles

The project structure reflects several key architectural decisions:

1. **Separation of Concerns**: Clear boundaries between data structures (`core/`) and algorithms (`geometry/`)
2. **Generic Design**: Extensive use of generics for coordinate types, data associations, and dimensionality
3. **Trait-Based Architecture**: Heavy use of traits for extensibility and code reuse
4. **Performance Focus**: Dedicated benchmarking infrastructure, performance regression detection, and memory allocation profiling
5. **Memory Profiling**: Comprehensive allocation tracking with `count-allocations` feature for detailed memory analysis
6. **Performance Analysis (opt-in)**: `bench` feature for timing-based tests and ergonomics checks; distinct from CI-driven regression detection in item 4
7. **Academic Integration**: Strong support for research use with comprehensive citations and references
8. **High-Performance Optimizations**: Phase 1-2 optimizations completed in v0.4.4+ with significant performance gains
9. **Enhanced Robustness**: Rollback mechanisms, atomic operations, and comprehensive error handling
10. **Cross-Platform Development**: Modern Python tooling alongside traditional Rust development
11. **Quality Assurance**: Multiple layers of automated quality control and testing

This structure supports both library users (through examples and documentation) and contributors (through comprehensive
development tooling and clear architectural guidance).

#### Memory Profiling System

Version 0.4.4 includes comprehensive memory profiling capabilities:

- **Allocation Tracking**: Optional `count-allocations` feature using the `allocation-counter` crate
- **Memory Benchmarks**: Dedicated benchmarks for memory scaling analysis (`profiling_suite.rs`) - comprehensive profiling suite
  with typical runtime of 1-2 hours (10³-10⁶ points). **Recommended for manual profiling runs** rather than CI due to
  long execution time. Use `PROFILING_DEV_MODE=1` for faster iteration (10x speedup).
- **Profiling Examples**: `memory_analysis.rs` demonstrates allocation counting across different operations
- **Integration Testing**: `allocation_api.rs` provides utilities for testing memory usage in various scenarios
- **CI Integration**: Automated profiling benchmarks with detailed allocation reports

#### Performance Optimization System (v0.4.4+)

Version 0.4.4+ completes Phase 1-2 of the comprehensive optimization roadmap:

- **Collection Optimization** (Phase 1): FxHasher-based collections for 2-3x faster hash operations
- **Key-Based Internal APIs** (Phase 2): Direct SlotMap key operations eliminating UUID lookups
- **FacetCacheProvider**: 50-90% reduction in facet mapping computation time
- **Zero-Allocation Iterators**: 1.86x faster iteration with `vertex_uuid_iter()`
- **Enhanced Collections**: 15-30% additional gains from FastHashSet/SmallBuffer optimizations
- **Thread Safety**: RCU-based caching and atomic operations for concurrent operations
- **Enhanced Error Handling**: Comprehensive `InsertionError` enum with rollback mechanisms
- **Robustness Infrastructure**: Atomic TDS operations with validation and rollback capabilities

#### Development Workflow

The project uses [`just`](https://github.com/casey/just) as a command runner to simplify common development tasks. Key workflows include:

**Quick Development Cycle:**

```bash
just dev           # Format, lint, and test (fast feedback loop)
```

**Pre-Commit Checks:**

```bash
just pre-commit    # Full quality checks, tests, and examples
```

**Testing Workflows:**

```bash
just test          # Run library and doc tests
just test-python   # Run Python utility tests
just test-debug    # Run debug tools with output
just test-allocation  # Run allocation profiling tests
just test-all      # All tests (Rust + Python)
```

**Quality and Linting:**

```bash
just quality       # All quality checks (formatting, linting, validation)
just fmt          # Format Rust code
just clippy       # Run Clippy with strict settings
just python-lint  # Format and lint Python scripts
```

**Benchmarks and Performance:**

```bash
just bench         # Run all benchmarks
just bench-baseline # Generate performance baseline
just bench-compare # Compare against baseline
```

**Documentation Maintenance:**

```bash
just update-tree   # Update directory tree in this document
```

**CI Simulation:**

```bash
just ci           # Run what CI runs (quality + release tests + bench compile)
```

**Complete Command Reference:**

```bash
just --list       # Show all available commands
just help-workflows # Show common workflow patterns
```

This `justfile`-based workflow provides consistent, cross-platform development commands and integrates seamlessly with the existing tooling ecosystem.

---

## Module Organization Patterns

The canonical organizational patterns found across key modules in the codebase: `cell.rs`, `vertex.rs`, `facet.rs`, `boundary.rs`, and `util.rs`.

### Canonical Section Sequence

Based on analysis of the modules, the standard ordering follows this sequence:

1. **Module Documentation** (`//!` doc comments)
2. **Imports** (with section separator)
3. **Error Types** (with section separator)
4. **Convenience Macros and Helpers** (with section separator)
5. **Struct Definitions** (with section separator)
6. **Deserialization Implementation** (with section separator)
7. **Core Implementation Blocks** (with section separator)
8. **Advanced Implementation Blocks** (specialized trait bounds)
9. **Standard Trait Implementations** (with section separator)
10. **Specialized Trait Implementations** (e.g., Hashing, Equality)
11. **Tests** (with section separator)

### Comment Separators

#### Primary Section Separators

All modules consistently use this pattern for major sections:

```rust
// =============================================================================
// SECTION NAME
// =============================================================================
```

#### Subsection Separators

Within test modules, subsections use consistent formatting:

```rust
    // =============================================================================
    // SUBSECTION NAME TESTS
    // =============================================================================
```

### Section-by-Section Analysis

#### 1. Module Documentation (`//!` comments)

**Pattern**: Comprehensive module-level documentation with:

- Brief description of the module's purpose
- Key features (bulleted list with `**bold**` headings)
- Usage examples with code blocks
- References to external concepts (linked where appropriate)

**Example Structure**:

```rust
//! Brief description of the module
//!
//! Detailed explanation of what the module provides
//!
//! # Key Features
//!
//! - **Feature 1**: Description
//! - **Feature 2**: Description
//!
//! # Examples
//!
//! ```rust
//! // Code example
//! ```
```

#### 2. Imports Section

**Pattern**: Organized into logical groups with clear hierarchy:

1. `super::` imports (internal crate modules)
2. `crate::` imports (other crate modules)
3. External crate imports (alphabetically ordered)
4. Standard library imports

**Consistent Elements**:

- Section header: `// IMPORTS`
- Clear grouping with spacing
- Trait imports explicitly named

#### 3. Error Types Section

**Pattern**: Custom error enums using `thiserror::Error`:

```rust
/// Errors that can occur during [operation] validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum [Module]ValidationError {
    /// Description of error variant
    #[error("Error message: {source}")]
    VariantName {
        /// Description of source field
        #[from]
        source: SourceErrorType,
    },
}
```

**Consistent Elements**:

- Descriptive enum names ending in `ValidationError` or `Error`
- Full derive macro set: `Clone, Debug, Error, PartialEq, Eq`
- Detailed documentation for each variant
- `#[from]` attribute for error chaining

#### 4. Convenience Macros and Helpers Section

**Pattern**: Procedural macros with comprehensive documentation:

```rust
/// Convenience macro for creating [items] with less boilerplate.
///
/// Detailed description of macro functionality
///
/// # Returns
/// Description of return type
///
/// # Panics
/// Description of panic conditions
///
/// # Usage
/// ```rust
/// // Usage examples
/// ```
#[macro_export]
macro_rules! item_name {
    // Pattern definitions
}

// Re-export at crate level
pub use crate::macro_name;
```

**Helper Function Pattern**:

```rust
/// Helper function description
fn helper_function<generics>(parameters) -> ReturnType
where
    // trait bounds
{
    // implementation
}
```

#### 5. Struct Definitions Section

**Pattern**: Builder pattern with comprehensive documentation:

```rust
#[derive(Builder, Clone, Debug, Default, Serialize)]
#[builder(build_fn(validate = "Self::validate"))]
/// Comprehensive struct documentation
///
/// # Generic Parameters
/// * `T` - Description
/// * `U` - Description
/// * `const D` - Description
///
/// # Properties
/// - **field**: Description
///
/// # Usage
/// ```rust
/// // Usage example
/// ```
pub struct StructName<generics>
where
    // trait bounds
{
    /// Field documentation
    field: Type,
    
    #[builder(setter(skip), default = "default_value()")]
    auto_field: Type,
}
```

#### 6. Deserialization Implementation Section

**Pattern**: Manual `Deserialize` implementation with visitor pattern:

```rust
/// Manual implementation of Deserialize for [Type]
impl<'de, G> serde::Deserialize<'de> for Type<G>
where
    G: /* trait bounds as needed */,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Visitor pattern implementation
        // Ok(Self { /* ... */ })
    }
}
```

#### 7. Core Implementation Blocks

**Pattern**: Primary functionality with clear method groupings:

```rust
impl<generics> StructName<generics>
where
    // basic trait bounds
{
    /// Method documentation with examples
    ///
    /// # Arguments
    /// * `param` - Description
    ///
    /// # Returns
    /// Description
    ///
    /// # Example
    /// ```rust
    /// // Example code
    /// ```
    pub fn method_name(self) -> ReturnType {
        // implementation
    }
}
```

#### 8. Advanced Implementation Blocks

**Pattern**: Specialized implementations with additional trait bounds:

```rust
// Advanced implementation block for methods requiring ComplexField
impl<generics> StructName<generics>
where
    T: CoordinateScalar + Clone + ComplexField<generics> + PartialEq + PartialOrd + Sum,
    // additional specialized bounds
{
    /// Advanced method requiring specialized traits
    pub fn advanced_method(self) -> ReturnType {
        // implementation
    }
}
```

#### 9. Standard Trait Implementations Section

**Pattern**: Standard Rust traits with clear documentation:

```rust
/// Description of trait implementation behavior
impl<generics> TraitName for StructName<generics>
where
    // trait bounds
{
    /// Implementation documentation
    #[inline]
    fn trait_method(self, other: Self) -> ReturnType {
        // implementation
    }
}
```

**Common Standard Traits**:

- `PartialEq` - based on core data, excluding metadata
- `PartialOrd` - lexicographic ordering
- `Eq` - marker trait
- `From`/`Into` conversions

#### 10. Specialized Trait Implementations

**Pattern**: Complex traits like `Hash` with detailed contract documentation:

```rust
/// Custom Hash implementation using only [criteria] for consistency with `PartialEq`.
///
/// This ensures that items with the same [criteria] have the same hash,
/// maintaining the Eq/Hash contract: if a == b, then hash(a) == hash(b).
///
/// Note: [excluded fields] are excluded from hashing to match
/// the `PartialEq` implementation.
impl<G> core::hash::Hash for StructName<G>
where
    G: /* trait bounds as needed */,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // implementation with explanation comments
    }
}
```

#### 11. Tests Section

**Pattern**: Comprehensive test organization with multiple subsections:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // additional test imports
    
    // Type aliases for commonly used types to reduce repetition
    type TestType = StructName<generics>;
    
    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================
    
    /// Helper function for common test setup
    fn helper_function() -> TestType {
        // setup code
    }
    
    // =============================================================================
    // CATEGORY TESTS
    // =============================================================================
    // Tests covering [specific functionality]
    
    #[test]
    fn test_function_name() {
        // test implementation
    }
}
```

**Test Categories** (in order of appearance):

1. **Helper Functions** - Common test utilities
2. **Convenience Macro Tests** - Macro functionality
3. **Trait Implementation Tests** - Core Rust traits
4. **Core Methods Tests** - Primary functionality
5. **Dimensional Tests** - Multi-dimensional support
6. **Serialization Tests** - Serde functionality
7. **Geometric Properties Tests** - Domain-specific logic
8. **Error Handling Tests** - Validation and error cases
9. **Edge Case Tests** - Boundary conditions

### Module-Specific Variations

#### `cell.rs` (large module)

- Most comprehensive implementation
- Multiple specialized implementation blocks
- Extensive geometric predicates integration
- Detailed Hash/Eq contract documentation

#### `vertex.rs` (large module)

- Strong focus on coordinate validation
- Comprehensive equality testing
- Multiple numeric type support
- Detailed serialization testing

#### `facet.rs` (medium module)

- Geometric relationship focus
- Key generation utilities
- Adjacency testing
- Error handling for geometric constraints

#### `boundary.rs` (small module)

- Trait implementation focused
- Algorithm-specific testing
- Performance benchmarking
- Integration with TDS

#### `util.rs` (large module)

- Function-focused (not struct-focused)
- UUID generation and validation utilities with comprehensive error handling
- Extreme coordinate finding functions for SlotMap-based vertex collections
- Supercell simplex creation for triangulation initialization
- Hash utilities for stable, deterministic hash computation
- Facet adjacency checking and geometric utilities
- Combination generation for k-simplex vertex combinations
- **Jaccard similarity testing utilities** (v0.5.4+):
  - Set extraction helpers: `extract_vertex_coordinate_set()`, `extract_edge_set()`, `extract_hull_facet_set()`
  - Comparison utilities: `jaccard_index()`, `jaccard_distance()`, `format_jaccard_report()`
  - Assertion macro: `assert_jaccard_gte!` for automatic diagnostics on failure
  - Safe f64 conversion with overflow detection (2^53 limit)
- Multi-dimensional testing across 1D-5D with both f32 and f64 coordinate types
- Extensive edge case testing and error handling validation with systematic test organization

### Key Conventions

#### Documentation Standards

- Always include examples in public API documentation
- Use `///` for item documentation, `//!` for module documentation
- Include `# Arguments`, `# Returns`, `# Errors`, `# Panics` sections where applicable
- Reference other types using `[Type]` notation

#### Naming Conventions

- Error types: `[Module]ValidationError` or `[Module]Error`
- Test functions: `test_[functionality]_[specific_case]`
- Helper functions: `create_[item]` or `assert_[property]`
- Type aliases in tests: `Test[Type][Dimension]` (e.g., `TestCell3D`)

#### Code Organization

- Group related functionality within implementation blocks
- Separate basic and advanced functionality into different impl blocks
- Use consistent indentation and spacing
- Include inline comments for complex logic

#### Testing Patterns

- Comprehensive edge case coverage
- Both positive and negative test cases
- Type parameter variation testing (f32, f64, different dimensions)
- Serialization round-trip testing
- Error message validation

This organizational pattern provides a consistent, maintainable structure that scales well across different module
complexities while maintaining readability and discoverability.
