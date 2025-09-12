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

### 2. Into/From Conversions Example (`into_from_conversions.rs`)

Demonstrates ergonomic conversions using Rust's `Into`/`From` traits,
showing how vertices and points can be converted to coordinate arrays
using Rust's `Into`/`From` traits.

**Key Features:**

- **Vertex to coordinate conversion**: Owned vertices and references
  convert to coordinate arrays via `Into`/`From`
- **Point to coordinate conversion**: Owned points and references
  convert to coordinate arrays via `Into`/`From`
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

**Run with:** `cargo run --release --example into_from_conversions`
[View source](./into_from_conversions.rs)

### 3. 3D Triangulation with 50 Points (`triangulation_3d_50_points.rs`)

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

### 4. 3D Convex Hull with 50 Points (`convex_hull_3d_50_points.rs`)

Demonstrates convex hull extraction and analysis from a 3D Delaunay triangulation.
This example showcases the extraction of convex hulls from triangulations and
their geometric properties and analysis.

**Key Features:**

- **Triangulation to convex hull**: Extracts convex hull from existing triangulation
- **Hull validation**: Comprehensive validation of convex hull properties
- **Point containment testing**: Tests various points for containment within the hull
- **Visible facet analysis**: Determines which hull facets are visible from external points
- **Performance analysis**: Benchmarks hull extraction and query operations
- **Geometric analysis**: Detailed analysis of hull properties and facet structure

**Run with:** `cargo run --release --example convex_hull_3d_50_points`
[View source](./convex_hull_3d_50_points.rs)

### 5. Memory Analysis Across Dimensions (`memory_analysis.rs`)

Demonstrates memory usage analysis for Delaunay triangulations across different dimensions (2D-5D) using the memory profiling system introduced in v0.4.3.

**Key Features:**

- **Multi-dimensional analysis**: Tests memory usage patterns from 2D to 5D triangulations
- **Allocation tracking**: Uses the `count-allocations` feature to provide detailed memory metrics
- **Performance profiling**: Measures both construction time and memory usage
- **Convex hull analysis**: Compares triangulation vs hull memory consumption
- **Reproducible results**: Uses fixed random seeds for consistent analysis
- **Memory scaling insights**: Provides empirical observations about memory scaling patterns

**Memory Profiling Features:**

- **Detailed allocation tracking**: Shows bytes allocated per vertex/operation
- **Memory efficiency ratios**: Compares hull memory to triangulation memory
- **Cross-dimensional comparison**: Analyzes how memory scales with dimension
- **Feature-aware**: Works with or without `count-allocations` feature

**Sample Output (with count-allocations feature enabled):**

```text
Memory Analysis for Delaunay Triangulations Across Dimensions
=============================================================
✓ Allocation counter enabled - detailed memory tracking available

=== Memory Analysis with 25 Points ===

--- 3D Triangulation ---
  Analyzing 3D triangulation with 25 points
    Triangulation: 25 vertices, 94 cells
    Convex hull: 46 facets
    Construction time: 1.234ms
    Hull extraction time: 456μs
    Triangulation memory: 12.4 KiB (524 bytes/vertex)
    Hull memory: 3.8 KiB (30.6% of triangulation)
```

**Usage:**

```bash
# Basic analysis (without detailed memory tracking)
cargo run --release --example memory_analysis

# Detailed analysis with allocation tracking
cargo run --release --example memory_analysis --features count-allocations
```

**Complementary Benchmarks:**

For comprehensive memory scaling analysis, see:

- `cargo bench --bench memory_scaling --features count-allocations`  
- `cargo bench --bench triangulation_vs_hull_memory --features count-allocations`

**Run with:** `cargo run --release --example memory_analysis --features count-allocations`  
[View source](./memory_analysis.rs)
