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

### 3. 3D Triangulation with 20 Points (`triangulation_3d_20_points.rs`)

A comprehensive example demonstrating the creation and analysis of a 3D Delaunay
triangulation using a stable 20-point random configuration. This example
showcases the full triangulation workflow from vertex generation to validation
and analysis.

**Key Features:**

- **Random vertex generation**: Creates 20 random 3D points with reproducible
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

**Sample Output (typical):**

```text
=================================================================
3D Delaunay Triangulation Example - 20 Random Points
=================================================================

Creating 3D Delaunay triangulation with 20 random points in [-3, 3]^3 (seed = 666)...
✓ Triangulation created successfully in 142.43ms
Generated 6 vertices
First few vertices:
  v 0: [...]
  v 1: [...]
  v 2: [...]
  ...

Triangulation Analysis:
======================
  Number of vertices: 6
  Number of cells:    1
  Dimension:          3

Triangulation Validation:
========================
✓ Triangulation is VALID
  Validation completed in 26.75µs

Boundary Analysis:
=================
  Boundary facets:     4

Performance Analysis:
====================
  Validation Performance (5 runs):
    • Average time: < 1 ms

=================================================================
Example completed successfully!
=================================================================
```

**Run with:** `cargo run --release --example triangulation_3d_20_points`
[View source](./triangulation_3d_20_points.rs)

### 4. 3D Convex Hull with 20 Points (`convex_hull_3d_20_points.rs`)

**Sample Output (typical):**

```text
      Vertices: 4
    Valid cells:     2211/2211
    Avg neighbors:   3.42

Triangulation Validation
========================

✓ Triangulation is VALID
  Validation completed in 953.292µs

  Validation Details:
    • All cells have valid geometry
    • Neighbor relationships are consistent
    • No duplicate cells detected
    • Vertex mappings are consistent
    • Facet sharing is valid

Boundary Analysis
=================

  Boundary facets:     4880
  Boundary computation: 630.125µs

  Boundary Details:
    • Boundary facets form the convex hull
    • Each boundary facet belongs to exactly one cell
    • Facet 1: key = Ok(8560930012034061606)
    • Facet 2: key = Ok(6274251604205431426)
    • Facet 3: key = Ok(11636944546669252672)
    • ... and 4877 more boundary facets

  Topological Properties:
    • Vertices (V): 100
    • Cells (C):    2211
    • Boundary facets: 4880

Performance Analysis
====================

  Validation Performance (5 runs):
    • Average time: 835.408µs
    • Min time:     788.125µs
    • Max time:     902.792µs

  Boundary Computation Performance (3 runs):
    • Average time: 552.43µs

  Memory Usage Estimation (stack only, excludes heap allocations):
    • Vertex memory: ~5600 bytes
    • Cell memory:   ~406824 bytes
    • Total memory:  ~412424 bytes (402.8 KB)
    Note: This excludes heap-owned data like neighbors and internal collections

  Performance Ratios:
    • Validation per vertex: 8354.08 ns
    • Validation per cell:   377.84 ns

=================================================================
Example completed successfully
=================================================================

```

**Run with:** `cargo run --release --example triangulation_3d_100_points`
[View source](./triangulation_3d_100_points.rs)

### 4. 3D Convex Hull with 100 Points (`convex_hull_3d_100_points.rs`)

Demonstrates convex hull extraction and analysis from a 3D Delaunay triangulation.
This example uses the same stable 20-point configuration as
`triangulation_3d_20_points` and showcases the extraction of convex hulls from
triangulations and their geometric properties and analysis.

**Key Features:**

- **Triangulation to convex hull**: Extracts convex hull from existing triangulation
- **Hull validation**: Comprehensive validation of convex hull properties
- **Point containment testing**: Tests various points for containment within the hull
- **Visible facet analysis**: Determines which hull facets are visible from external points
- **Performance analysis**: Benchmarks hull extraction and query operations
- **Geometric analysis**: Detailed analysis of hull properties and facet structure

**Run with:** `cargo run --release --example convex_hull_3d_20_points`
[View source](./convex_hull_3d_20_points.rs)

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

--- 2D Triangulation ---
  Analyzing 2D triangulation with 25 points
    Triangulation: 25 vertices, 26 cells
    Convex hull: 32 facets
    Construction time: 2.811791ms
    Hull extraction time: 5.625µs
    Triangulation memory: 1233.7 KiB (1.20 MiB, 50532 bytes/vertex)
    Hull memory: 7.1 KiB (0.6% of triangulation)

--- 3D Triangulation ---
  Analyzing 3D triangulation with 25 points
    Triangulation: 25 vertices, 100 cells
    Convex hull: 60 facets
    Construction time: 2.016208ms
    Hull extraction time: 24.917µs
    Triangulation memory: 3502.7 KiB (3.42 MiB, 143471 bytes/vertex)
    Hull memory: 26.8 KiB (0.8% of triangulation)

--- 4D Triangulation ---
  Analyzing 4D triangulation with 25 points
    Triangulation: 25 vertices, 211 cells
    Convex hull: 341 facets
    Construction time: 4.791167ms
    Hull extraction time: 74.666µs
    Triangulation memory: 10315.2 KiB (10.07 MiB, 422509 bytes/vertex)
    Hull memory: 111.7 KiB (1.1% of triangulation)

--- 5D Triangulation ---
  Analyzing 5D triangulation with 25 points
    Triangulation: 25 vertices, 532 cells
    Convex hull: 1290 facets
    Construction time: 10.621375ms
    Hull extraction time: 220.959µs
    Triangulation memory: 22676.3 KiB (22.14 MiB, 928823 bytes/vertex)
    Hull memory: 248.2 KiB (1.1% of triangulation)

=== Key Insights (empirical) ===
• On random 3D inputs, memory tends to scale between O(n) and O(n log n), distribution-dependent
• Convex hull memory is often a fraction of triangulation memory (ballpark 10–30%, varies)
• Hull extraction is typically faster than triangulation construction
• Use --features count-allocations to see detailed allocation metrics

For comprehensive scaling analysis, run:
  cargo bench --bench profiling_suite --features count-allocations -- memory_profiling
  cargo bench --bench triangulation_vs_hull_memory --features count-allocations
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

- `cargo bench --bench profiling_suite --features count-allocations -- memory_profiling`  
- `cargo bench --bench triangulation_vs_hull_memory --features count-allocations`

**Run with:** `cargo run --release --example memory_analysis`
[View source](./memory_analysis.rs)

### 6. Zero-Allocation Iterator Demo (`zero_allocation_iterator_demo.rs`)

Demonstrates the performance benefits and API usage patterns of zero-allocation
iterators compared to traditional Vec allocation approaches. This example showcases
the `vertex_uuid_iter()` method which provides iteration over vertex UUIDs without
heap allocations.

**Key Features:**

- **Performance comparison**: Direct benchmarking between allocation vs zero-allocation approaches
- **Functional equivalence**: Demonstrates that both methods produce identical results
- **Iterator capabilities**: Shows ExactSizeIterator implementation and iterator trait support
- **API usage patterns**: Practical examples of when to use zero-allocation iterators
- **Memory efficiency**: Zero heap allocations during iteration
- **Ergonomic usage**: Full iterator trait support (map, filter, collect, etc.)

**Sample Output:**

```text
=================================================================
Zero-Allocation Iterator Demo
=================================================================

Using a 4D cell with 5 vertices from triangulation

Functional Equivalence Test:
===========================
  vertex_uuids() returned 5 UUIDs
  vertex_uuid_iter().collect() returned 5 UUIDs
  Results are identical: true

Performance Comparison:
======================
  Method 1 (vertex_uuids): 512.46µs (10000 iterations)
  Method 2 (vertex_uuid_iter):   7.54µs (10000 iterations)
  Speedup: 67.95x faster
  Counts match: true

Iterator Capabilities:
=====================
  Length via ExactSizeIterator: 5
  Non-nil UUIDs via for loop: 5
  Valid UUIDs via iterator chain: 5
  First 3 UUIDs: 3 collected

=================================================================
Key Benefits of vertex_uuid_iter():
- Zero heap allocations (no Vec created)
- Implements ExactSizeIterator (O(1) len())
- Full iterator trait support (map, filter, etc.)
- Lazy evaluation (only compute what you need)
- Better performance for iteration-only use cases
=================================================================
```

**Run with:** `cargo run --release --example zero_allocation_iterator_demo`
[View source](./zero_allocation_iterator_demo.rs)

---

## Stable Random-Triangulation Parameter Sets

Several examples share parameter sets with the core random-triangulation tests
in `src/geometry/util.rs::test_generate_random_triangulation_dimensions`:

- `triangulation_3d_20_points` and `convex_hull_3d_20_points` both use
  `(n_points = 20, bounds = [-3.0, 3.0], seed = 666)` in 3D.
- `zero_allocation_iterator_demo` uses the 4D configuration
  `(n_points = 12, bounds = [-1.0, 1.0], seed = 777)`.

These configurations are intentionally reused between tests and examples so that:

- CI exercises the same nontrivial point sets in both unit tests and examples.
- Examples remain robust against extreme Delaunay-repair paths while still
  demonstrating realistic triangulations in 3D and 4D.
