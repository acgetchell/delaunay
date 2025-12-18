# Numerical Robustness Guide for Delaunay Triangulation

This guide explains the numerical robustness improvements implemented in the delaunay library to address geometric predicate stability
issues, including the "No cavity boundary facets found" error and other precision-related problems.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Current Implementation Status](#current-implementation-status)
3. [Implemented Solutions](#implemented-solutions)
4. [Error Handling and Retry Logic](#error-handling-and-retry-logic)
5. [Robust Predicates](#robust-predicates)
6. [Matrix Conditioning](#matrix-conditioning)
7. [Usage Examples](#usage-examples)
8. [Configuration Selection Guide](#configuration-selection-guide)
9. [Convex Hull Robustness](#convex-hull-robustness)
10. [Testing and Validation](#testing-and-validation)
11. [Performance Considerations](#performance-considerations)
12. [Migration Strategy](#migration-strategy)

## Problem Overview

The "No cavity boundary facets found for X bad cells" error occurs in the Bowyer-Watson triangulation algorithm when:

1. **Bad cells are correctly identified** (cells whose circumsphere contains the new vertex)
2. **Cavity boundary detection fails** (no valid boundary facets are found around the cavity)
3. **Algorithm cannot proceed** (cannot create new cells to fill the cavity)

This typically happens with:

- Random point configurations that create degenerate geometry
- Points that are nearly coplanar or cospherical
- Very large or very small coordinate values
- Points at circumsphere boundaries (numerical precision issues)

## Current Implementation Status

As of version 0.4.3, the delaunay library includes comprehensive robustness improvements:

### ✅ Implemented Features

1. **Robust Predicates Module** (`src/geometry/robust_predicates.rs`)
   - Enhanced `insphere` predicate with adaptive tolerances
   - Matrix conditioning for improved numerical stability
   - Multiple fallback strategies for degenerate cases
   - Scale-factor recovery for proper determinant interpretation
   - Symbolic perturbation for handling extreme degeneracies
     (implements Simulation of Simplicity-style epsilon ordering to break ties
     deterministically when geometric predicates fail)

2. **Configuration System** (`src/geometry/robust_predicates.rs`)
   - `RobustPredicateConfig` for customizable tolerance settings
   - Predefined configuration presets for different use cases
   - Adaptive tolerance computation based on input scale
   - Performance-optimized preset selections

3. **Convex Hull Robustness** (`src/geometry/algorithms/convex_hull.rs`)
   - Fallback visibility tests for degenerate orientation cases
   - Distance-based heuristics when geometric predicates fail
   - Comprehensive error handling for edge cases
   - Multi-dimensional support (tested in 2D-5D)

4. **Enhanced Error Handling** (`src/core/algorithms/incremental_insertion.rs`, `src/core/triangulation.rs`)
   - Structured `InsertionError` enum with geometric degeneracy classification
   - `NonManifoldTopology` variant for facet sharing violations (retryable via perturbation)
   - Automatic retry logic with progressive coordinate perturbation (1e-4 to 5e-2)
   - Direct error propagation avoiding unnecessary unwrapping
   - Transactional insertion with automatic rollback on failure
   - Detailed error diagnostics with facet hash and cell count information

5. **Kernel-Based Robust Predicates** (`src/geometry/kernel.rs`)
   - `FastKernel`: Direct floating-point predicates for well-conditioned inputs
   - `RobustKernel`: Adaptive tolerance + matrix conditioning for degeneracies
   - Unified insertion algorithm with kernel-based predicate dispatch
   - Statistics tracking for insertion performance analysis
   - Support for degenerate point configurations via automatic retry

### 🚧 In Progress

1. **Exact Arithmetic Fallback**
   - High-precision arithmetic for extreme degenerate cases
   - Automatic switching thresholds based on condition numbers

### 📋 Planned Features

1. **Adaptive Algorithm Selection**
   - Automatic switching between standard and robust predicates
   - Performance-based predicate selection

2. **Advanced Matrix Conditioning**
   - Pivoting strategies for better numerical stability
   - Iterative refinement for high-precision results

## Implemented Solutions

The library now includes several robust numerical methods to address these issues:

### 1. Enhanced Geometric Predicates

Location: `src/geometry/robust_predicates.rs`

The robust predicates module provides improved versions of geometric tests:

```rust
use delaunay::geometry::robust_predicates::{robust_insphere, RobustPredicateConfig};
use delaunay::geometry::robust_predicates::config_presets;

// Create configuration for your use case
let config = config_presets::general_triangulation();

// Use robust insphere test
let result = robust_insphere(&simplex_points, &test_point, &config)?;
```

### 2. Matrix Conditioning System

Implemented matrix conditioning improves numerical stability:

```rust
// The robust predicates automatically apply conditioning:
// 1. Row scaling to normalize matrix entries
// 2. Scale factor tracking for determinant recovery
// 3. Adaptive tolerance computation based on matrix scale
```

### 3. Convex Hull Robustness

Location: `src/geometry/algorithms/convex_hull.rs`

Robust convex hull operations with fallback strategies:

```rust
use delaunay::geometry::algorithms::convex_hull::ConvexHull;

// Robust point-in-hull testing with fallback for degenerate cases
let is_outside = hull.is_point_outside(&test_point, &tds)?;

### Fallback visibility testing when orientation predicates fail
let visible_facets = hull.find_visible_facets(&external_point, &tds)?;
```

## Error Handling and Retry Logic

### Structured Error Classification

Location: `src/core/algorithms/incremental_insertion.rs`

The `InsertionError` enum provides structured error variants for geometric degeneracies:

```rust
use delaunay::core::algorithms::incremental_insertion::InsertionError;

// Structured error for non-manifold topology
match insertion_result {
    Err(InsertionError::NonManifoldTopology { facet_hash, cell_count }) => {
        eprintln!("Non-manifold: facet {:x} shared by {} cells", facet_hash, cell_count);
        // Retryable via perturbation
    }
    Err(InsertionError::ConflictRegion(e)) => {
        // Duplicate boundary facets or ridge fans - also retryable
    }
    Err(e) if e.is_retryable() => {
        // Automatic retry with perturbation
    }
    Err(e) => {
        // Non-retryable structural error
        return Err(e);
    }
    Ok(result) => result,
}
```

### Automatic Retry with Perturbation

Location: `src/core/triangulation.rs`

The `insert_transactional` method provides automatic retry logic:

```rust
use delaunay::core::triangulation::Triangulation;
use delaunay::core::algorithms::incremental_insertion::InsertionOutcome;
use delaunay::vertex;

// Transactional insertion with automatic rollback and retry
let vertex = vertex!([0.5, 0.5, 0.5]);
let (outcome, stats) = triangulation.insert_with_statistics(vertex, None, None)?;

println!("Insertion statistics:");
println!("  Attempts: {}", stats.attempts);
println!("  Used perturbation: {}", stats.used_perturbation);
println!("  Cells repaired: {}", stats.cells_removed_during_repair);
println!("  Success: {}", stats.success);
println!("  Skipped: {}", stats.skipped);

match outcome {
    InsertionOutcome::Inserted { vertex_key, hint } => {
        println!("Inserted vertex {vertex_key:?}, hint={hint:?}");
    }
    InsertionOutcome::Skipped { error } => {
        eprintln!("Vertex skipped: {error}");
    }
}
```

### Progressive Perturbation Schedule

The retry mechanism uses a progressive perturbation schedule:

1. **Attempt 0**: Original coordinates (no perturbation)
2. **Attempt 1**: ε = 1e-4 (0.01% perturbation)
3. **Attempt 2**: ε = 1e-3 (0.1% perturbation)
4. **Attempt 3**: ε = 1e-2 (1% perturbation)
5. **Attempt 4**: ε = 2e-2 (2% perturbation)
6. **Attempt 5**: ε = 5e-2 (5% perturbation)

Each attempt:

- Clones the TDS for rollback (transactional semantics)
- Applies coordinate perturbation
- Attempts insertion
- On failure: restores TDS from snapshot and increases perturbation
- On success: returns result with statistics

### Retryable Error Detection

The `is_retryable()` method classifies errors:

```rust
// Retryable errors (geometric degeneracies)
- InsertionError::NonManifoldTopology { .. }        // Facet sharing violation
- InsertionError::Location(CycleDetected { .. })    // Point location cycle
- InsertionError::ConflictRegion(DuplicateBoundaryFacets { .. })
- InsertionError::ConflictRegion(RidgeFan { .. })
- InsertionError::TopologyValidation(_)             // Repair failure

// Non-retryable errors (structural failures)
- InsertionError::DuplicateUuid { .. }              // UUID conflict
- InsertionError::DuplicateCoordinates { .. }       // Coordinate conflict
- InsertionError::Construction(_)                   // Generic construction error
- InsertionError::CavityFilling { .. }              // Cavity filling error
- InsertionError::NeighborWiring { .. }             // Wiring error (legacy)
```

### Benefits

1. **Type Safety**: Structured error variants eliminate string parsing
2. **Automatic Recovery**: Retry logic resolves most geometric degeneracies
3. **Transactional Semantics**: TDS always remains in valid state
4. **Diagnostic Information**: Detailed error context for debugging
5. **Progressive Resolution**: Increasing perturbation scales resolve degeneracies

## Robust Predicates

### Core Implementation

The `robust_insphere` function in `src/geometry/robust_predicates.rs` provides:

1. **Adaptive Tolerance Computation**

   ```rust
   use delaunay::geometry::matrix;
   let adaptive_tolerance = matrix::adaptive_tolerance(&matrix, config.base_tolerance);
   ```

   Why exclude the constant-1 column?
   - Orientation and the standard insphere determinants use an augmented matrix with a trailing column of 1.0s.
   - If we include that column in the matrix magnitude (∞-norm) estimate,
     small-coordinate problems (e.g., 1e-9 scale) get a magnitude dominated by
     the 1.0s, making the tolerance too large and masking boundary signal.
   - The helper detects a last column that is approximately all 1.0 and
     excludes it for the magnitude estimate only. The determinant itself is
     unchanged. This keeps the tolerance proportional to the geometric scale
     (coordinates and squared norms) while preserving robustness.

   Note:
   - Only the tolerance scaling excludes the constant-1 column.
   - The determinant is always computed on the full matrix, including the
     constant-1 column. This preserves the correct predicate geometry while
     making the tolerance scale-aware.

2. **Matrix Conditioning**

   ```rust
   let (conditioned_matrix, scale_factor) = condition_matrix(matrix, &config);
   let determinant = conditioned_matrix.determinant() * scale_factor;
   ```

3. **Multiple Fallback Strategies**

- Standard determinant with adaptive tolerance
- Matrix conditioning when standard method fails
- Symbolic perturbation for extreme degenerate cases

Sign conventions for the insphere predicate

- Standard determinant form (`predicates::insphere`) and robust form
  (`robust_predicates::robust_insphere`): interpret the determinant sign
  relative to the simplex orientation. No dimension-parity adjustment is needed.
- Lifted paraboloid form (`predicates::insphere_lifted`): the lifted
  determinant has opposite sign in even dimensions. We apply a parity factor of
  −1 for even D (and +1 for odd D) before the orientation normalization so
  results match the standard form across 2D–5D.

### Configuration Options

Three predefined configurations are available:

```rust
// Balanced performance and robustness
let config = config_presets::general_triangulation();

// Maximum precision for critical applications
let config = config_presets::high_precision();

// Robust handling of degenerate inputs
let config = config_presets::degenerate_robust();
```

### Usage in Triangulation

The robust predicates are integrated into the triangulation kernels and automatically applied
during insertion. Users can choose between different kernel types based on their needs:

```rust
use delaunay::prelude::*;
use delaunay::geometry::kernel::{FastKernel, RobustKernel};

// Fast predicates (default) - best for well-conditioned inputs
let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
    DelaunayTriangulation::empty();

// Robust predicates - handles degeneracies and extreme coordinates
let mut dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::with_empty_kernel(RobustKernel::new());

// Insert vertices - kernel predicates used automatically
dt_robust.insert(vertex!([1e10, 1e10, 1e10])).unwrap();
dt_robust.insert(vertex!([1e10 + 1.0, 1e10, 1e10])).unwrap();
```

## Matrix Conditioning

### Current Implementation

The `condition_matrix` function in `src/geometry/robust_predicates.rs` implements row scaling:

```rust
use delaunay::geometry::matrix::Matrix;

/// Apply conditioning to improve matrix stability
fn condition_matrix(
    mut matrix: Matrix,
    _config: &RobustPredicateConfig<f64>,
) -> (Matrix, f64) {
    let mut scale_factor = 1.0;
    
    // Row scaling - normalize each row by its maximum element
    for i in 0..matrix.nrows() {
        let mut max_element = 0.0;
        for j in 0..matrix.ncols() {
            max_element = max_element.max(matrix[(i, j)].abs());
        }
        
        if max_element > 1e-100 {
            // Scale the row and track the scale factor
            for j in 0..matrix.ncols() {
                matrix[(i, j)] /= max_element;
            }
            scale_factor *= max_element;
        }
    }
    
    (matrix, scale_factor)
}
```

### Benefits

1. **Improved Condition Numbers**: Row scaling reduces condition numbers
2. **Scale Factor Recovery**: Proper determinant calculation after conditioning
3. **Numerical Stability**: Reduces amplification of floating-point errors
4. **Zero Division Protection**: Handles near-zero matrix elements safely

### Integration with Cast Function

The implementation uses the `cast` function for clean type conversions:

```rust
use num_traits::cast::cast;

// Safe type conversion with fallback
let tolerance_f64 = cast(config.base_tolerance)
    .unwrap_or(f64::EPSILON * 1000.0);

// Proper scale factor application
let final_determinant = conditioned_determinant * scale_factor;
let result_determinant = cast(final_determinant)
    .unwrap_or_else(T::zero);
```

## Usage Examples

### Basic Robust Triangulation

```rust
use delaunay::prelude::*;
use delaunay::geometry::kernel::{FastKernel, RobustKernel};

// For problematic point sets, use robust kernel
let vertices = vec![
    vertex!([1e10, 1e10, 1e10]),           // Large coordinates
    vertex!([1e10 + 1.0, 1e10, 1e10]),     // Small relative differences
    vertex!([1e10, 1e10 + 1.0, 1e10]),
    vertex!([1e10, 1e10, 1e10 + 1.0]),
    vertex!([1e10 + 0.5, 1e10 + 0.5, 1e10 + 0.5]),
];

// Try standard triangulation with fast predicates first
let dt_result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 3>, _> = 
    DelaunayTriangulation::new(&vertices);

if dt_result.is_err() {
    println!("Standard triangulation failed, using robust approach");
    
    // Use robust kernel for numerical stability
    let dt = DelaunayTriangulation::with_kernel(
        RobustKernel::new(),
        &vertices
    ).expect("Robust triangulation failed");
    
    println!("Robust triangulation completed with {} cells", dt.number_of_cells());
}
```

### Robust Predicate Testing

```rust
use delaunay::geometry::robust_predicates::{robust_insphere, config_presets, InSphere};
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;

// Test nearly coplanar points
let simplex = vec![
    Point::new([0.0, 0.0, 0.0]),
    Point::new([1.0, 0.0, 0.0]),
    Point::new([0.0, 1.0, 0.0]),
    Point::new([0.5, 0.5, 1e-15]), // Very small z-coordinate
];

let test_point = Point::new([0.25, 0.25, 0.25]);

// Use robust configuration for stability
let config = config_presets::degenerate_robust::<f64>();
let result = robust_insphere(&simplex, &test_point, &config)?;

match result {
    InSphere::INSIDE => println!("Point is inside circumsphere"),
    InSphere::OUTSIDE => println!("Point is outside circumsphere"),
    InSphere::BOUNDARY => println!("Point is on circumsphere boundary"),
}
```

## Configuration Selection Guide

### Available Configurations

The library provides robust predicates integrated into kernel types. Choose the appropriate
kernel based on your input characteristics:

```rust
use delaunay::prelude::*;
use delaunay::geometry::kernel::{FastKernel, RobustKernel};

// Fast predicates (default) - best for well-conditioned inputs
let dt_fast: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
    DelaunayTriangulation::empty();

// Robust predicates - handles degeneracies, extreme coordinates
let dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::with_empty_kernel(RobustKernel::new());
```

### Kernel Selection Comparison

| Kernel Type | Predicate Strategy | Best Use Case |
|---|---|---|
| `FastKernel<f64>` | Direct floating-point determinants | Well-conditioned inputs, performance-critical |
| `RobustKernel<f64>` | Adaptive tolerance + conditioning | Degeneracies, extreme coordinates, error recovery |

### Kernel Selection Examples

```rust
use delaunay::prelude::*;
use delaunay::geometry::kernel::{FastKernel, RobustKernel};

// For general applications - use default fast kernel
let vertices = vec![vertex!([0.0, 0.0, 0.0]), /* ... */];
let dt = DelaunayTriangulation::new(&vertices).unwrap();

// For high-precision scientific computing - use robust kernel
let dt_precise: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices).unwrap();

// For problematic/degenerate inputs - use robust kernel
let dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices).unwrap();
```

### Error Recovery in Applications

```rust
use delaunay::prelude::*;
use delaunay::geometry::kernel::{FastKernel, RobustKernel};
use delaunay::core::vertex::Vertex;

pub fn create_triangulation_with_fallback(
    vertices: &[Vertex<f64, (), 3>],
) -> Result<
    DelaunayTriangulation<RobustKernel<f64>, (), (), 3>,
    DelaunayTriangulationConstructionError,
> {
    // Strategy 1: Try fast kernel first for performance
    let fast_result: Result<
        DelaunayTriangulation<FastKernel<f64>, (), (), 3>,
        DelaunayTriangulationConstructionError,
    > = DelaunayTriangulation::new(vertices);
    
    if fast_result.is_ok() {
        println!("Fast kernel succeeded");
        // Convert to robust kernel type for uniform return type
        return DelaunayTriangulation::with_kernel(RobustKernel::new(), vertices);
    }
    
    println!("Fast kernel failed, trying robust kernel");
    
    // Strategy 2: Try robust kernel with automatic retry logic
    match DelaunayTriangulation::with_kernel(RobustKernel::new(), vertices) {
        Ok(dt) => {
            println!("Robust kernel succeeded");
            Ok(dt)
        }
        Err(e) => {
            // Strategy 3: Preprocess points and retry with robust kernel
            println!("Robust kernel failed, trying with filtered vertices");
            let filtered_vertices = remove_duplicate_and_near_duplicate_points(vertices);
            DelaunayTriangulation::with_kernel(RobustKernel::new(), &filtered_vertices)
        }
    }
}
```

```rust
fn remove_duplicate_and_near_duplicate_points(
    vertices: &[Vertex<f64, Option<()>, 3>]
) -> Vec<Vertex<f64, Option<()>, 3>> {
    // Simple deduplication based on coordinate proximity
    let mut filtered = Vec::new();
    let tolerance = 1e-10;
    
    for vertex in vertices {
        let is_duplicate = filtered.iter().any(|existing: &Vertex<f64, Option<()>, 3>| {
            let existing_coords: [f64; 3] = existing.point().into();
            let vertex_coords: [f64; 3] = vertex.point().into();
            
            existing_coords.iter().zip(vertex_coords.iter())
                .all(|(a, b)| (a - b).abs() < tolerance)
        });
        
        if !is_duplicate {
            filtered.push(*vertex);
        }
    }
    
    filtered
}
```

## Convex Hull Robustness

### Fallback Visibility Testing

The convex hull implementation includes robust visibility tests:

```rust
// In ConvexHull::is_facet_visible_from_point()
// When geometric orientation predicates fail:
match (orientation_inside, orientation_test) {
    (Orientation::NEGATIVE, Orientation::POSITIVE)
    | (Orientation::POSITIVE, Orientation::NEGATIVE) => Ok(true),
    (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
        // Fallback to distance-based heuristic
        Ok(Self::fallback_visibility_test(facet, point))
    }
    _ => Ok(false),
}
```

### Distance-Based Fallback

When orientation predicates become degenerate, the hull uses distance-based heuristics:

```rust
/// Fallback visibility test for degenerate cases
fn fallback_visibility_test(
    facet: &Facet<T, U, V, D>, 
    point: &Point<T, D>
) -> bool {
    // Calculate facet centroid
    let facet_vertices = facet.vertices();
    let mut centroid_coords = [T::zero(); D];
    
    for vertex_point in &facet_vertices {
        let coords: [T; D] = vertex_point.into();
        for (i, &coord) in coords.iter().enumerate() {
            centroid_coords[i] += coord;
        }
    }
    
    let num_vertices = T::from_usize(facet_vertices.len()).unwrap_or_else(T::one);
    for coord in &mut centroid_coords {
        *coord /= num_vertices;
    }
    
    // Use distance threshold for visibility determination
    let point_coords: [T; D] = point.into();
    let mut diff_coords = [T::zero(); D];
    for i in 0..D {
        diff_coords[i] = point_coords[i] - centroid_coords[i];
    }
    let distance_squared = squared_norm(diff_coords);
    
    // Simple threshold-based visibility
    // Note: NumCast::from requires num_traits::NumCast
    let threshold = NumCast::from(1.0f64).unwrap_or_else(T::one);
    distance_squared > threshold
}
```

### Robust Hull Operations

All convex hull operations include error handling for edge cases:

1. **Point-in-Hull Testing**: Graceful handling when visibility tests fail
2. **Facet Enumeration**: Robust iteration over boundary facets
3. **Visible Facet Finding**: Multiple strategies for facet visibility
4. **Nearest Facet Search**: Distance-based selection with numerical stability

### Multi-Dimensional Support

The robustness improvements work across all dimensions:

```rust
// 2D triangulations with robust hull extraction
let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> = 
    ConvexHull::from_triangulation(&tds_2d)?;

// 4D and higher dimensions
let hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> = 
    ConvexHull::from_triangulation(&tds_4d)?;

// All use the same robust visibility algorithms
let is_outside_2d = hull_2d.is_point_outside(&point_2d, &tds_2d)?;
let is_outside_4d = hull_4d.is_point_outside(&point_4d, &tds_4d)?;
```

## Testing and Validation

### Current Test Coverage

The robust predicates system has comprehensive test coverage demonstrating real-world effectiveness:

**Integration Tests:**

- [`tests/robust_predicates_showcase.rs`](../tests/robust_predicates_showcase.rs) - Demonstrates cases where robust predicates solve failures
- [`tests/robust_predicates_comparison.rs`](../tests/robust_predicates_comparison.rs) - Numerical accuracy testing across dimensions
- [`tests/coordinate_conversion_errors.rs`](../tests/coordinate_conversion_errors.rs) - Error handling for extreme values

**Test Results Summary:**

- Robust predicates successfully handle degenerate test cases that cause standard predicates to fail:
  - `test_nearly_coplanar_points` - Nearly coplanar point configurations
  - `test_large_coordinate_values` - Extreme coordinate values and precision issues
  - `test_consistency_verification` - Boundary case consistency
  - Additional showcase tests in `robust_predicates_showcase.rs`
- Demonstrates consistent results across dimensions (2D-5D)
- Validates error recovery for "No cavity boundary facets found" scenarios
- Confirms that structural invariants checked by `Tds::is_valid()` remain intact
  and can be further inspected via `DelaunayTriangulation::validation_report()` and
  (when needed) `DelaunayTriangulation::is_valid()` / `DelaunayTriangulation::validate()` in targeted tests.

### Degenerate Case Tests

```rust
#[cfg(test)]
mod robustness_tests {
    use super::*;

    #[test]
    fn test_nearly_coplanar_points() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1e-15]), // Very slightly off-plane
        ];
        
        let config = config_presets::degenerate_robust();
        
        // Should handle gracefully without panicking
        let test_point = Point::new([0.25, 0.25, 1e-16]);
        let result = robust_insphere(&points, &test_point, &config);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_coordinate_values() {
        let points = vec![
            Point::new([1e10, 1e10, 1e10]),
            Point::new([1e10 + 1.0, 1e10, 1e10]),
            Point::new([1e10, 1e10 + 1.0, 1e10]),
            Point::new([1e10, 1e10, 1e10 + 1.0]),
        ];
        
        let config = config_presets::general_triangulation();
        let test_point = Point::new([1e10 + 0.5, 1e10 + 0.5, 1e10 + 0.5]);
        
        let result = robust_insphere(&points, &test_point, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consistency_verification() {
        let points = create_test_simplex_3d();
        let test_point = Point::new([0.5, 0.5, 0.5]);
        
        let config = config_presets::general_triangulation();
        
        // Test that different methods give consistent results
        let robust_result = robust_insphere(&points, &test_point, &config).unwrap();
        let standard_result = insphere(&points, test_point).unwrap();
        
        // Results should be consistent (allowing for boundary differences)
        match (robust_result, standard_result) {
            (InSphere::INSIDE, InSphere::INSIDE) => {}
            (InSphere::OUTSIDE, InSphere::OUTSIDE) => {}
            (InSphere::BOUNDARY, _) | (_, InSphere::BOUNDARY) => {}
            _ => panic!("Inconsistent results: {:?} vs {:?}", robust_result, standard_result),
        }
    }
}
```

### Performance Benchmarks

```rust
// Note: Add [[bench]] entries in Cargo.toml or use cargo bench --features criterion
#[cfg(test)]
mod performance_tests {
    use criterion::{Criterion, black_box};

    fn benchmark_predicate_performance(c: &mut Criterion) {
        let points = create_test_simplex_3d();
        let test_point = Point::new([0.5, 0.5, 0.5]);
        let config = config_presets::general_triangulation();

        let mut group = c.benchmark_group("geometric_predicates");
        
        group.bench_function("standard_insphere", |b| {
            b.iter(|| {
                black_box(insphere(&points, test_point))
            })
        });
        
        group.bench_function("robust_insphere", |b| {
            b.iter(|| {
                black_box(robust_insphere(&points, &test_point, &config))
            })
        });
        
        group.finish();
    }
}
```

## Performance Considerations

### Computational Overhead

The robust predicates and algorithms add computational overhead, but provide significant reliability gains:

1. **Adaptive tolerance computation**: ~10-20% overhead per predicate call (RobustKernel only)
2. **Automatic retry with perturbation**: ~50-100% overhead (only for retryable insertion errors)
3. **Matrix conditioning**: ~30-50% overhead for problematic cases (RobustKernel only)
4. **RobustKernel vs FastKernel**: ~20-40% slower for normal cases, but succeeds on cases that would fail
5. **Transactional insertion**: Minimal overhead (~5%) for maintaining valid state during errors

> **Note**: These percentages are based on benchmark measurements from [`benches/robust_predicates_comparison.rs`](../benches/robust_predicates_comparison.rs)
> and [`tests/robust_predicates_showcase.rs`](../tests/robust_predicates_showcase.rs). For detailed results, see [`benches/PERFORMANCE_RESULTS.md`](../benches/PERFORMANCE_RESULTS.md).
> Actual overhead may vary based on input complexity and geometry.

### When to Use Robust Kernels

**Use `RobustKernel` when:**

- Standard triangulation fails with insertion errors (TopologyValidation, NonManifoldTopology)
- Working with nearly degenerate point configurations
- Points have extreme coordinate values (very large or very small)
- Input data comes from measurements with numerical precision issues
- Reliability is more important than maximum performance

**Use standard `FastKernel` (default) when:**

- Working with well-conditioned point sets
- Maximum performance is critical
- Input points are known to be in general position

### Optimization Strategies

1. **Tiered approach**: Try `FastKernel` first, fall back to `RobustKernel` on failure
2. **Automatic retry**: Insertion errors trigger automatic retry with random perturbation (built-in)
3. **Early termination**: Return as soon as a reliable result is found
4. **Preprocessing**: Remove duplicate and near-duplicate points before triangulation
5. **Error handling**: Structured error types enable precise error recovery strategies

### Memory Usage

- **Configuration objects**: Minimal overhead (~16-32 bytes)
- **Additional matrices**: ~2x memory for conditioning operations during predicate evaluation
- **Statistics tracking**: ~64-128 bytes per algorithm instance
- **Buffer reuse**: Both algorithms reuse internal buffers to minimize allocations

## Migration Strategy

### ✅ Completed Phases

#### Phase 1: Robust Predicates Implementation

- ✅ Implemented robust predicate functions (`robust_insphere`, `robust_orientation`)
- ✅ Added comprehensive tests and validation suite
- ✅ Benchmarked performance impact and documented overhead

#### Phase 2: Algorithm Consolidation and Kernel Integration

- ✅ Consolidated incremental insertion into unified cavity-based algorithm
- ✅ Integrated robust predicates into kernel architecture (`FastKernel`, `RobustKernel`)
- ✅ Added automatic retry logic with progressive perturbation for transient errors
- ✅ Implemented structured error types for precise error handling
- ✅ Demonstrated significant improvement in success rates for problematic cases

### 🚧 Current Phase: Selective Adoption

**Recommended Migration Path:**

1. **Immediate**: Use `RobustKernel` for applications encountering triangulation failures

   ```rust
   use delaunay::prelude::*;
   use delaunay::geometry::kernel::RobustKernel;
   
   // Replace failed standard triangulations
   let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
       DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices)?;
   ```

2. **Short-term**: Implement tiered approach for new applications

   ```rust
   use delaunay::prelude::*;
   use delaunay::geometry::kernel::{FastKernel, RobustKernel};
   
   // Try fast kernel first, fall back to robust
   let dt = DelaunayTriangulation::new(&vertices)
       .or_else(|_| {
           println!("Fast kernel failed, using robust kernel");
           DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices)
       })?;
   ```

3. **Medium-term**: Evaluate kernel performance for your specific input characteristics

### 📋 Future Phases

#### Phase 3: Performance Optimization

- Profile performance bottlenecks in production usage
- Add specialized fast paths for common cases
- Implement adaptive algorithm selection based on input analysis

#### Phase 4: Ecosystem Integration

- Integrate robust predicates with other geometric algorithms
- Add automatic robustness level selection
- Provide migration tools for legacy codebases

### Current Status Summary

The robust predicates system is **production-ready** and integrated into the kernel architecture.
Users experiencing insertion errors should switch to `RobustKernel` for improved numerical stability.
The system provides automatic retry logic with progressive perturbation for transient degeneracies,
combined with kernel-level robust predicates for persistent numerical issues.
