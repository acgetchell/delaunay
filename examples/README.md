# Examples

This directory contains examples demonstrating various features and
capabilities of the delaunay library.

## Performance Considerations

⚠️ **Important**: Examples may run noticeably slower in debug mode. For optimal performance, run examples in release mode:

```bash
# Recommended: Run in release mode for better performance
cargo run --release --example <example_name>

# Debug mode (slower, but includes debug symbols)
cargo run --example <example_name>
```

Release mode provides:

- **Faster execution** (often up to 10-100x speedup for geometric algorithms)
- **More accurate performance measurements**
- **Better representation of production performance**
- **Optimized floating-point operations**

## Available Examples

### 1. Point Comparison and Hashing (`point_comparison_and_hashing.rs`)

Demonstrates the robust comparison and hashing behavior of the Point struct,
with special emphasis on handling of NaN (Not a Number) and infinity values.

**Key Features:**

- **NaN-aware equality**: Unlike the IEEE 754 standard where NaN ≠ NaN, our
  Point implementation treats NaN values as equal to themselves for
  consistent behavior in data structures.
- **Consistent hashing**: Points with identical coordinates (including NaN)
  produce the same hash value, enabling reliable use in HashMap and HashSet.
- **Mathematical properties**: Equality satisfies reflexivity, symmetry, and
  transitivity.
- **Special value handling**: Proper comparison of infinity, negative infinity,
  and zero values.

**Run with:** `cargo run --release --example point_comparison_and_hashing`
[View source](./point_comparison_and_hashing.rs)

### 2. Circumsphere Containment and Simplex Orientation Testing (`test_circumsphere.rs`)

Demonstrates and compares two methods for determining if a point lies inside
the circumsphere of a 4D simplex (5-cell/hypertetrahedron), plus comprehensive
testing of simplex orientation across multiple dimensions.

**Key Features:**

- **Distance-based method** (`circumsphere_contains`): Computes the circumcenter
  and circumradius explicitly, then checks if the test point is within that
  distance from the circumcenter.
- **Determinant-based method** (`insphere`): Uses a matrix
  determinant approach that avoids explicit circumcenter calculation and
  provides superior numerical stability.
- **4D simplex testing**: Uses a unit 4D simplex with vertices at:
  - `[0,0,0,0]` (origin)
  - `[1,0,0,0]` (unit vector along x-axis)
  - `[0,1,0,0]` (unit vector along y-axis)
  - `[0,0,1,0]` (unit vector along z-axis)
  - `[0,0,0,1]` (unit vector along w-axis)
- **Comprehensive testing**: Tests various categories of points including:
  - Inside points (well within the circumsphere)
  - Outside points (clearly beyond the circumsphere)
  - Boundary points (on edges and faces of the 4D simplex)
  - Vertex points (the simplex vertices themselves)
- **Simplex orientation testing**: Tests simplex orientation across dimensions:
  - 4D simplex orientation with positive and negative variants
  - 3D tetrahedron orientation for comparison
  - 2D triangle orientation with normal and reversed vertex ordering
  - Degenerate cases (collinear points)
- **Orientation impact demonstration**: Shows how the determinant-based method
  automatically handles orientation differences while maintaining consistent results.
- **Method comparison**: Shows how both methods perform on the same test cases,
  demonstrating where they agree and where numerical differences may occur.

**Run with:** `cargo run --release --example test_circumsphere`
[View source](./test_circumsphere.rs)

### 3. Implicit Conversion Example (`implicit_conversion.rs`)

Demonstrates the implicit conversion capabilities of the delaunay library,
showing how vertices and points can be automatically converted to coordinate
arrays using Rust's `From` trait.

**Key Features:**

- **Vertex to coordinate conversion**: Both owned vertices and vertex references
  can be implicitly converted to coordinate arrays
- **Point to coordinate conversion**: Both owned points and point references
  can be implicitly converted to coordinate arrays
- **Type safety**: All conversions are compile-time checked and type-safe
- **Zero-cost abstractions**: No runtime overhead for conversions
- **Ergonomic syntax**: Cleaner, more readable code compared to explicit
  coordinate access

**Usage Examples:**

```rust
// From owned vertex
let coords: [f64; 3] = vertex.into();

// From vertex reference (preserves original)
let coords: [f64; 3] = (&vertex).into();

// From owned point
let coords: [f64; 3] = point.into();

// From point reference (preserves original)
let coords: [f64; 3] = (&point).into();
```

**Run with:** `cargo run --release --example implicit_conversion`
[View source](./implicit_conversion.rs)

### 4. 3D Triangulation with 50 Points (`triangulation_3d_50_points.rs`)

A comprehensive example demonstrating the creation and analysis of a 3D Delaunay
triangulation using 50 randomly generated points. This example showcases the
full triangulation workflow from vertex generation to validation and analysis.

**Key Features:**

- **Random vertex generation**: Creates 50 random 3D points with reproducible
  seeding for consistent results across runs
- **Delaunay triangulation construction**: Uses the Bowyer-Watson algorithm to
  build a valid 3D Delaunay triangulation
- **Comprehensive analysis**: Detailed examination of triangulation properties:
  - Vertex and cell counts
  - Dimension verification  
  - Cell-to-vertex ratios
  - Individual cell analysis (vertices per cell, neighbor counts)
- **Validation testing**: Thorough validation of the triangulation including:
  - Geometric validity of all cells
  - Neighbor relationship consistency
  - Absence of duplicate cells
  - Vertex mapping consistency
  - Facet sharing validation
- **Boundary analysis**: Computation and analysis of boundary facets that form
  the convex hull of the point set
- **Performance benchmarking**: Detailed performance analysis including:
  - Validation timing across multiple runs
  - Boundary computation performance
  - Memory usage estimation
  - Performance per vertex/cell ratios
- **Error handling**: Demonstrates proper error handling and debugging
  information for triangulation failures
- **Reproducibility**: Uses fixed random seeds to ensure consistent results
  for testing and comparison purposes

**Sample Output:**

```text
3D Delaunay Triangulation Example - 50 Random Points
=================================================================

Generated 50 vertices:
  v 0: [   4.123,   -2.456,    7.890]
  v 1: [  -1.234,    5.678,   -3.210]
  ... and 48 more vertices

Creating Delaunay triangulation...
✓ Triangulation created successfully in 2.345ms

Triangulation Analysis:
======================
  Number of vertices: 50
  Number of cells:    234
  Dimension:          3
  Vertex/Cell ratio:  0.21

✓ Triangulation is VALID
  Validation completed in 156μs
```

**Run with:** `cargo run --release --example triangulation_3d_50_points`
[View source](./triangulation_3d_50_points.rs)

### 5. Boundary Analysis Trait Demonstration (`boundary_analysis_trait.rs`)

Demonstrates the clean separation of boundary analysis functionality from the main
`Tds` struct using a trait-based approach. This example showcases the modular
design of the boundary analysis system.

**Key Features:**

- **Trait-based Design**: Shows how the `BoundaryAnalysis` trait cleanly separates
  boundary operations from the core triangulation data structure
- **Multiple Boundary Methods**: Demonstrates different approaches to boundary analysis:
  - `boundary_facets()` - Get all boundary facets as a collection
  - `number_of_boundary_facets()` - Efficient counting without creating vectors
  - `is_boundary_facet()` - Check individual facets
- **Modular Architecture**: Illustrates benefits of trait-based design including
  better testability, extensibility, and separation of concerns
- **Complex Examples**: Shows boundary analysis on both simple (tetrahedron) and
  complex (multiple tetrahedra) triangulations
- **Performance Considerations**: Demonstrates efficient boundary detection methods

**Benefits Highlighted:**

- Clean separation of concerns in the codebase
- Easy extensibility for future boundary algorithms
- Better IDE support and discoverability
- Consistent interface across different triangulation types

**Run with:** `cargo run --release --example boundary_analysis_trait`
[View source](./boundary_analysis_trait.rs)

### 6. Float Traits Validation (`check_float_traits.rs`)

A technical example that validates and demonstrates which traits are included
in the `Float` trait from `num_traits`, helping you understand the trait bounds
used throughout the library.

**Run with:** `cargo run --release --example check_float_traits`
[View source](./check_float_traits.rs)

**Key Features:**

- **Trait Analysis**: Compile-time verification of traits included in `Float`
- **Included Traits**: Confirms `Float` includes `PartialEq`, `PartialOrd`, `Copy`, and `Clone`
- **Missing Traits**: Demonstrates that `Default` is NOT included in `Float`
- **Design Rationale**: Explains why `Default` must be explicitly specified in
  trait bounds even when using `Float`
- **Educational Value**: Helps developers understand the library's type system
  and floating-point abstractions

**Technical Insights:**

```rust
// Float includes these traits automatically:
fn requires_partial_eq<U: PartialEq>() {}
fn requires_partial_ord<U: PartialOrd>() {}
fn requires_copy<U: Copy>() {}
fn requires_clone<U: Clone>() {}

// But Default must be specified separately:
// fn requires_default<U: Default>() {} // This would NOT compile with just Float
```

**Run with:** `cargo run --release --example check_float_traits`

### 7. Memory Allocation Testing API (`test_alloc_api.rs`)

Comprehensive testing utilities and examples for memory allocation tracking
in Delaunay triangulation operations. This example provides both a testing
framework and a demonstration of memory usage patterns.

**Key Features:**

- **Allocation Counter Integration**: Shows how to use the `allocation-counter`
  feature for memory profiling
- **Comprehensive Test Helpers**: Provides utilities for:
  - Measuring allocations with error handling
  - Creating test points in 2D and 3D
  - Building test triangulations
  - Printing detailed memory summaries
- **Feature-Aware Design**: Gracefully handles both enabled and disabled
  allocation counting modes
- **Real-World Testing**: Demonstrates allocation patterns for actual
  triangulation operations
- **Performance Analysis**: Tools for understanding memory usage in
  computational geometry operations

**Allocation Testing Categories:**

- Basic vector operations baseline
- Point creation in multiple dimensions
- Triangulation data structure initialization
- Complex triangulation workflows

**Usage Examples:**

```sh
// Test with allocation counting enabled:
cargo test --example test_alloc_api --features count-allocations
```

```rust
// View memory usage for operations:
let (result, info) = measure_with_result(|| {
    create_test_points_3d(100)
});
print_alloc_summary(&info, "3D point creation");
```

**Run with:** `cargo run --release --example test_alloc_api`

**Test with allocation tracking:** `cargo test --example test_alloc_api --features count-allocations`
[View source](./test_alloc_api.rs)
