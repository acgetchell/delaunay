# Numerical Robustness Guide for Delaunay Triangulation

This guide explains the numerical robustness improvements implemented in the delaunay library to address geometric predicate stability
issues, including the "No cavity boundary facets found" error and other precision-related problems.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Current Implementation Status](#current-implementation-status)
3. [Implemented Solutions](#implemented-solutions)
4. [Robust Predicates](#robust-predicates)
5. [Matrix Conditioning](#matrix-conditioning)
6. [Usage Examples](#usage-examples)
7. [Configuration Selection Guide](#configuration-selection-guide)
8. [Convex Hull Robustness](#convex-hull-robustness)
9. [Testing and Validation](#testing-and-validation)
10. [Performance Considerations](#performance-considerations)
11. [Migration Strategy](#migration-strategy)

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

4. **Enhanced Error Handling**
   - Detailed error types with diagnostic information
   - Graceful degradation for numerical edge cases
   - Comprehensive validation and consistency checks
   - Recovery strategies for boundary detection failures

5. **Robust Bowyer-Watson Algorithm** (`src/core/algorithms/robust_bowyer_watson.rs`)
   - Complete integration of robust predicates into triangulation construction
   - Recovery strategies for cavity boundary detection failures
   - Statistics tracking for algorithm performance analysis
   - Configurable robustness settings for different use cases
   - Support for degenerate point configurations

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

// Fallback visibility testing when orientation predicates fail
let visible_facets = hull.find_visible_facets(&external_point, &tds)?;
```

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

The robust predicates are designed for integration with the Bowyer-Watson algorithm:

```rust
// In bowyer_watson.rs - find_bad_cells() can use robust predicates
for cell_key in &tds.cells().indices().collect::<Vec<_>>() {
    let cell = &tds.cells()[*cell_key];
    
    // Use robust insphere test for stability
    let config = config_presets::general_triangulation();
    match robust_insphere(&cell.vertices_as_points(), vertex.point(), &config) {
        Ok(InSphere::INSIDE) => bad_cells.push(*cell_key),
        Ok(InSphere::OUTSIDE) => continue,
        Ok(InSphere::BOUNDARY) => {
            // Handle boundary case with additional logic
            if should_include_boundary_cell(cell, vertex) {
                bad_cells.push(*cell_key);
            }
        }
        Err(e) => {
            eprintln!("Insphere test failed for cell {:?}: {}", cell_key, e);
            continue;
        }
    }
}
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
use delaunay::core::{
    algorithms::robust_bowyer_watson::RobustBoyerWatson,
    traits::insertion_algorithm::InsertionAlgorithm,
    triangulation_data_structure::Tds,
};
use delaunay::geometry::robust_predicates::config_presets;
use delaunay::vertex;

// For problematic point sets, use robust triangulation
let vertices = vec![
    vertex!([1e10, 1e10, 1e10]),           // Large coordinates
    vertex!([1e10 + 1.0, 1e10, 1e10]),     // Small relative differences
    vertex!([1e10, 1e10 + 1.0, 1e10]),
    vertex!([1e10, 1e10, 1e10 + 1.0]),
    vertex!([1e10 + 0.5, 1e10 + 0.5, 1e10 + 0.5]),
];

// Try standard construction first
let tds_result = Tds::new(&vertices);
if tds_result.is_err() {
    println!("Standard triangulation failed, using robust approach");
    
    // Use robust Bowyer-Watson algorithm
    let mut robust_algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();
    let mut tds = Tds::empty();
    
    for vertex in &vertices {
        if let Err(e) = robust_algorithm.insert_vertex(&mut tds, vertex) {
            eprintln!("Failed to insert vertex {:?}: {}", vertex, e);
        }
    }
    
    println!("Robust triangulation completed with {} cells", tds.cells().len());
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

The library provides three preset configurations optimized for different scenarios:

```rust
use delaunay::geometry::robust_predicates::config_presets;

// General-purpose triangulation (balanced performance and robustness)
let general_config = config_presets::general_triangulation::<f64>();

// High-precision applications (maximum accuracy)
let precision_config = config_presets::high_precision::<f64>();

// Degenerate input handling (maximum robustness)
let robust_config = config_presets::degenerate_robust::<f64>();
```

### Configuration Comparison

| Configuration | Base Tolerance | Refinement Iterations | Best Use Case |
|---|---|---|---|
| `general_triangulation()` | 1e-12 | 3 | Well-conditioned inputs, performance-critical |
| `high_precision()` | 1e-15 | 5 | Scientific computing, high accuracy required |
| `degenerate_robust()` | 1e-10 | 2 | Nearly degenerate inputs, error recovery |

### Algorithm Selection Examples

```rust
use delaunay::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;

// For general applications
let general_algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

// For high-precision scientific computing
let precision_config = config_presets::high_precision::<f64>();
let precise_algorithm = RobustBoyerWatson::with_config(precision_config);

// For problematic/degenerate inputs
let robust_algorithm = RobustBoyerWatson::for_degenerate_cases();
```

### Error Recovery in Applications

```rust
use delaunay::core::{
    algorithms::robust_bowyer_watson::RobustBoyerWatson,
    traits::insertion_algorithm::InsertionAlgorithm,
    triangulation_data_structure::Tds,
    vertex::Vertex,
};
use delaunay::geometry::robust_predicates::config_presets;

pub fn create_triangulation_with_fallback(
    vertices: &[Vertex<f64, Option<()>, 3>]
) -> Result<Tds<f64, Option<()>, Option<()>, 3>, String> {
    // Strategy 1: Try standard triangulation
    if let Ok(tds) = Tds::new(vertices) {
        println!("Standard triangulation succeeded");
        return Ok(tds);
    }
    
    // Strategy 2: Try with general robust configuration
    let mut robust_algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();
    let mut tds = Tds::empty();
    
    for vertex in vertices {
        if let Err(_) = robust_algorithm.insert_vertex(&mut tds, vertex) {
            // Strategy 3: Try with maximum robustness configuration
            let mut max_robust_algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();
            let mut max_robust_tds = Tds::empty();
            
            for v in vertices {
                if let Err(e) = max_robust_algorithm.insert_vertex(&mut max_robust_tds, v) {
                    // Strategy 4: Preprocess points and retry with max robustness
                    let filtered_vertices = remove_duplicate_and_near_duplicate_points(vertices);
                    return try_with_filtered_vertices(&filtered_vertices);
                }
            }
            return Ok(max_robust_tds);
        }
    }
    
    println!("Robust triangulation succeeded");
    Ok(tds)
}

fn try_with_filtered_vertices(
    vertices: &[Vertex<f64, Option<()>, 3>]
) -> Result<Tds<f64, Option<()>, Option<()>, 3>, String> {
    let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();
    let mut tds = Tds::empty();
    
    for vertex in vertices {
        algorithm.insert_vertex(&mut tds, vertex)
            .map_err(|e| format!("All triangulation strategies failed: {}", e))?;
    }
    
    println!("Filtered robust triangulation succeeded");
    Ok(tds)
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
  and can be further inspected via `Tds::validation_report()` and
  `Tds::validate_delaunay()` in targeted tests.

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

1. **Adaptive tolerance computation**: ~10-20% overhead per predicate call
2. **Multiple strategy attempts**: ~50-100% overhead (only for difficult cases that would otherwise fail)
3. **Matrix conditioning**: ~30-50% overhead for problematic cases
4. **Consistency verification**: ~100% overhead (double computation) when enabled
5. **RobustBoyerWatson vs IncrementalBoyerWatson**: ~20-40% slower for normal cases, but succeeds on cases that would fail

> **Note**: These percentages are based on benchmark measurements from [`benches/robust_predicates_comparison.rs`](../benches/robust_predicates_comparison.rs)
> and [`tests/robust_predicates_showcase.rs`](../tests/robust_predicates_showcase.rs). For detailed results, see [`benches/PERFORMANCE_RESULTS.md`](../benches/PERFORMANCE_RESULTS.md).
> Actual overhead may vary based on input complexity and geometry.

### When to Use Robust Algorithms

**Use `RobustBoyerWatson` when:**

- Standard triangulation fails with "No cavity boundary facets found" errors
- Working with nearly degenerate point configurations
- Points have extreme coordinate values (very large or very small)
- Input data comes from measurements with numerical precision issues
- Reliability is more important than maximum performance

**Use standard `IncrementalBoyerWatson` when:**

- Working with well-conditioned point sets
- Maximum performance is critical
- Input points are known to be in general position

### Optimization Strategies

1. **Tiered approach**: Try standard algorithm first, fall back to robust on failure
2. **Caching**: Cache computed tolerances and conditioned matrices
3. **Early termination**: Return as soon as a reliable result is found
4. **Selective robustness**: Use `RobustBoyerWatson::new()` for general cases, `::for_degenerate_cases()` only when needed
5. **Preprocessing**: Remove duplicate and near-duplicate points before triangulation

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

#### Phase 2: Algorithm Integration

- ✅ Integrated robust predicates into `RobustBoyerWatson` algorithm
- ✅ Added configuration options (`general_triangulation`, `high_precision`, `degenerate_robust`)
- ✅ Demonstrated significant improvement in success rates for problematic cases

### 🚧 Current Phase: Selective Adoption

**Recommended Migration Path:**

1. **Immediate**: Use `RobustBoyerWatson` for applications encountering triangulation failures

   ```rust
   // Replace failed standard triangulations
   let robust_algorithm = RobustBoyerWatson::for_degenerate_cases();
   ```

2. **Short-term**: Implement tiered approach for new applications

   ```rust
   use delaunay::core::error::TriangulationError;
   
   // Try standard first, fall back to robust
   match Tds::new(&vertices) {
       Ok(tds) => tds,
       Err(_) => create_with_robust_algorithm(&vertices)?
           .map_err(|e: TriangulationError| e)?,
   }
   ```

3. **Medium-term**: Evaluate robust algorithms for performance-critical applications based on your specific input characteristics

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

The robust predicates system is **production-ready** and addresses the core numerical stability issues.
Users experiencing "No cavity boundary facets found" errors should migrate to `RobustBoyerWatson` immediately.
The system provides a mature, well-tested solution with comprehensive error recovery strategies.
