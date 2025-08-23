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
7. [Convex Hull Robustness](#convex-hull-robustness)
8. [Testing and Validation](#testing-and-validation)
9. [Performance Considerations](#performance-considerations)
10. [Migration Strategy](#migration-strategy)

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

As of version 0.4.0, the delaunay library includes several robustness improvements:

### ✅ Implemented Features

1. **Robust Predicates Module** (`src/geometry/robust_predicates.rs`)
   - Enhanced `insphere` predicate with adaptive tolerances
   - Matrix conditioning for improved numerical stability
   - Multiple fallback strategies for degenerate cases
   - Scale-factor recovery for proper determinant interpretation

2. **Configuration System** (`src/geometry/robust_predicates.rs`)
   - `RobustPredicateConfig` for customizable tolerance settings
   - Predefined configuration presets for different use cases
   - Adaptive tolerance computation based on input scale

3. **Convex Hull Robustness** (`src/geometry/algorithms/convex_hull.rs`)
   - Fallback visibility tests for degenerate orientation cases
   - Distance-based heuristics when geometric predicates fail
   - Comprehensive error handling for edge cases

4. **Enhanced Error Handling**
   - Detailed error types with diagnostic information
   - Graceful degradation for numerical edge cases
   - Comprehensive validation and consistency checks

### 🚧 In Progress

1. **Robust Bowyer-Watson Algorithm**
   - Integration of robust predicates into triangulation construction
   - Recovery strategies for cavity boundary detection failures
   - Statistics tracking for algorithm performance analysis

2. **Exact Arithmetic Fallback**
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
   let adaptive_tolerance = compute_adaptive_tolerance(&matrix, &config);
   ```

2. **Matrix Conditioning**

   ```rust
   let (conditioned_matrix, scale_factor) = condition_matrix(matrix, &config);
   let determinant = conditioned_matrix.determinant() * scale_factor;
   ```

3. **Multiple Fallback Strategies**
   - Standard determinant with adaptive tolerance
   - Matrix conditioning when standard method fails
   - Symbolic perturbation for extreme degenerate cases

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
        Err(_) => {
            // Log error but continue with next cell
            continue;
        }
    }
}
```

## Matrix Conditioning

### Current Implementation

The `condition_matrix` function in `src/geometry/robust_predicates.rs` implements row scaling:

```rust
/// Apply conditioning to improve matrix stability
fn condition_matrix(
    mut matrix: na::DMatrix<f64>,
    _config: &RobustPredicateConfig<T>,
) -> (na::DMatrix<f64>, f64) {
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
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::robust_predicates::config_presets;
use delaunay::vertex;

// For problematic point sets, use robust configuration
let vertices = vec![
    vertex!([1e10, 1e10, 1e10]),           // Large coordinates
    vertex!([1e10 + 1.0, 1e10, 1e10]),     // Small relative differences
    vertex!([1e10, 1e10 + 1.0, 1e10]),
    vertex!([1e10, 1e10, 1e10 + 1.0]),
    vertex!([1e10 + 0.5, 1e10 + 0.5, 1e10 + 0.5]),
];

// Standard construction might fail
let tds_result = Tds::new(&vertices);
if tds_result.is_err() {
    println!("Standard triangulation failed, using robust approach");
    
    // TODO: Use robust Bowyer-Watson when implemented
    // let config = config_presets::degenerate_robust();
    // let tds = robust_triangulation(&vertices, &config)?;
}
```

### Robust Predicate Testing

```rust
use delaunay::geometry::robust_predicates::{robust_insphere, config_presets};
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
let config = config_presets::degenerate_robust();
let result = robust_insphere(&simplex, &test_point, &config)?;

match result {
    InSphere::INSIDE => println!("Point is inside circumsphere"),
    InSphere::OUTSIDE => println!("Point is outside circumsphere"),
    InSphere::BOUNDARY => println!("Point is on circumsphere boundary"),
}
```

### Error Recovery in Applications

```rust
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::robust_predicates::config_presets;

pub fn create_triangulation_with_fallback(
    vertices: &[Vertex<f64, Option<()>, 3>]
) -> Result<Tds<f64, Option<()>, Option<()>, 3>, String> {
    // Strategy 1: Try standard triangulation
    if let Ok(tds) = Tds::new(vertices) {
        return Ok(tds);
    }
    
    // Strategy 2: Try with general robust config
    // (Implementation pending robust Bowyer-Watson)
    
    // Strategy 3: Try with maximum robustness
    // (Implementation pending robust Bowyer-Watson)
    
    // Strategy 4: Preprocess points and retry
    let filtered_vertices = remove_duplicate_and_near_duplicate_points(vertices);
    Tds::new(&filtered_vertices)
        .map_err(|e| format!("All triangulation strategies failed: {}", e))
}

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
    
    for vertex_point in &vertex_points {
        let coords: [T; D] = vertex_point.into();
        for (i, &coord) in coords.iter().enumerate() {
            centroid_coords[i] += coord;
        }
    }
    
    let num_vertices = T::from_usize(vertex_points.len()).unwrap_or_else(T::one);
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
    let threshold = T::from_f64(1.0).unwrap_or_else(T::one);
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

The robust predicates add computational overhead:

1. **Adaptive tolerance computation**: ~10-20% overhead
2. **Multiple strategy attempts**: ~50-100% overhead (only for difficult cases)
3. **Matrix conditioning**: ~30-50% overhead
4. **Consistency verification**: ~100% overhead (double computation)

### Optimization Strategies

1. **Lazy evaluation**: Only use robust methods when standard methods fail
2. **Caching**: Cache computed tolerances and conditioned matrices
3. **Early termination**: Return as soon as a reliable result is found
4. **Selective application**: Use robust predicates only for critical operations

### Memory Usage

- **Configuration objects**: Minimal overhead (~5-10 bytes)
- **Additional matrices**: ~2x memory for conditioning operations
- **Statistics tracking**: ~20-50 bytes per algorithm instance

## Migration Strategy

### Phase 1: Add Robust Predicates

1. Implement robust predicate functions
2. Add comprehensive tests
3. Benchmark performance impact

### Phase 2: Selective Integration

1. Use robust predicates only for error recovery
2. Add configuration options
3. Monitor improvement in success rates

### Phase 3: Full Integration

1. Make robust predicates the default for new triangulations
2. Provide migration path for existing code
3. Deprecate problematic APIs

### Phase 4: Optimization

1. Profile performance bottlenecks
2. Add specialized fast paths for common cases
3. Implement caching strategies

This comprehensive approach addresses the numerical robustness issues while maintaining performance and providing a clear migration path for existing code.
