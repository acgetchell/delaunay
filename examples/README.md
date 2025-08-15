# Examples

This directory contains examples demonstrating various features and
capabilities of the delaunay library.

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

**Run with:** `cargo run --example point_comparison_and_hashing`

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

**Run with:** `cargo run --example test_circumsphere`

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

**Run with:** `cargo run --example implicit_conversion`

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

**Run with:** `cargo run --example triangulation_3d_50_points`
