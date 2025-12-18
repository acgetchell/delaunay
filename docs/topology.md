# Topology Integration Design

This document outlines the design and implementation strategy for introducing topology analysis and validation into the delaunay triangulation library.

> Note: Euler characteristic validation is implemented in `Triangulation::is_valid()` (Level 3) as of v0.6.x.
> Some sections below describe earlier plans and are marked as historical/superseded.

## Table of Contents

- [Overview](#overview)
- [Proposed Architecture](#proposed-architecture)
- [Module Structure](#module-structure)
- [Core Design Patterns](#core-design-patterns)
- [Integration Points](#integration-points)
- [Implementation Details](#implementation-details)
- [Testing Strategy](#testing-strategy)
- [Implementation Phases](#implementation-phases)
- [Benefits and Rationale](#benefits-and-rationale)

---

## Overview

### Motivation

Delaunay triangulations possess well-defined topological properties that can be validated to ensure correctness. By introducing topology analysis, we can:

- **Validate Triangulation Integrity**: Verify that triangulations conform to expected topological invariants
- **Detect Construction Errors**: Identify when algorithmic errors produce topologically invalid results
- **Support Multiple Topologies**: Provide framework for planar, spherical, toroidal, and other topological spaces
- **Enable Robust Testing**: Validate randomly generated triangulations against theoretical expectations

### Design Goals

1. **Default Planar Topology**: Keep planar topology as the default, maintaining backward compatibility
2. **Euler Characteristic Validation**: Implement Euler characteristic calculation and validation
3. **Extensible Architecture**: Design for future topology types (spherical, toroidal)
4. **Dimensional Genericity**: Support topology validation across all dimensions D ≥ 2
5. **Integration with Existing Validation**: Build on
   `Tds::is_valid()` / `Tds::validate()` (Levels 1–2) via the Triangulation layer (Level 3)
6. **Comprehensive Testing**: Validate randomly generated triangulations

---

## Proposed Architecture

### High-Level Design

The topology system introduces a new `topology` module alongside the existing `core` and `geometry` modules, following the established
architectural patterns in the codebase.

```text
delaunay/
├── src/
│   ├── core/                          # Existing triangulation structures
│   ├── geometry/                      # Existing geometric algorithms  
│   ├── topology/                      # NEW: Topology analysis and validation
│   │   ├── traits/                    # Topology-related traits
│   │   ├── characteristics/           # Topological invariants (Euler, etc.)
│   │   ├── spaces/                    # Different topological spaces
│   │   └── util.rs                    # Topology utility functions
│   └── lib.rs                         # Updated with topology exports
├── tests/                             # Extended with topology tests
├── examples/                          # New topology examples
└── docs/                              # This document
```

---

## Module Structure

### Complete Topology Module Organization

```text
src/topology/
├── mod.rs                             # Module declarations and re-exports
├── traits/                            # Topology-related traits
│   ├── mod.rs                         # Trait module declarations
│   ├── topological_space.rs           # Core topology traits
│   └── euler_characteristic.rs        # Euler characteristic computation trait
├── characteristics/                   # Topological invariants
│   ├── mod.rs                         # Characteristics module declarations  
│   ├── euler.rs                       # Euler characteristic implementation
│   └── validation.rs                  # Topological validation functions
├── spaces/                            # Different topological spaces
│   ├── mod.rs                         # Spaces module declarations
│   ├── planar.rs                      # Planar topology (default)
│   ├── spherical.rs                   # Future: Spherical topology
│   └── toroidal.rs                    # Future: Toroidal topology
└── util.rs                            # Topology utility functions
```

### Integration with Existing Structure

The topology module integrates with existing modules through:

- **Core Integration**: Extend `Tds` with topology validation methods
- **Geometry Integration**: Use existing geometric predicates for topological calculations
- **Trait System**: Follow established trait patterns from `core::traits`
- **Testing Integration**: Extend existing validation and testing frameworks

Topology validation is layered on top of the existing structural invariants
validated by `Tds::is_valid()` / `Tds::validate()` (Levels 1–2) and exposed at the
Triangulation layer (Level 3), rather than duplicating those checks.

---

## Core Design Patterns

### 1. Topological Space Trait

The central abstraction for different topology types:

```rust
// src/topology/traits/topological_space.rs

use crate::core::{triangulation_data_structure::Tds, traits::DataType};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Core trait for topological spaces that triangulations can inhabit
pub trait TopologicalSpace<T, U, V, const D: usize> 
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Get the expected Euler characteristic for this topology
    fn expected_euler_characteristic(&self) -> i64;
    
    /// Validate that the triangulation conforms to this topology
    fn validate_topology(&self, tds: &Tds<T, U, V, D>) -> Result<(), TopologyError>;
    
    /// Get the genus of this topological space (for orientable surfaces)
    /// Returns None for non-surface topologies or when genus is undefined
    fn genus(&self) -> Option<u64>;
    
    /// Check if this topology supports the given dimension
    fn supports_dimension(&self, dimension: usize) -> bool;
    
    /// Get a human-readable name for this topology
    fn topology_name(&self) -> &'static str;
}
```

### 2. Error Handling

Topology-specific error types that integrate with existing error handling:

```rust
// src/topology/traits/mod.rs

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum TopologyError {
    #[error("Euler characteristic mismatch: expected {expected}, calculated {calculated}")]
    EulerCharacteristicMismatch { expected: i64, calculated: i64 },
    
    #[error("Topology {topology} does not support dimension {dimension}")]
    UnsupportedDimension { topology: String, dimension: usize },
    
    #[error("Invalid topological configuration: {message}")]
    InvalidConfiguration { message: String },
    
    #[error("Topological validation failed: {details}")]
    ValidationFailed { details: String },
}
```

### 3. Euler Characteristic Calculator

Dimensional-generic Euler characteristic computation:

```rust
// src/topology/characteristics/euler.rs

use crate::core::{triangulation_data_structure::Tds, traits::DataType};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Euler characteristic calculator for triangulated spaces
pub struct EulerCharacteristic;

impl EulerCharacteristic {
    /// Calculate Euler characteristic: χ = Σ(-1)^k * N_k
    /// Where N_k is the number of k-dimensional simplices
    ///
    /// For D-dimensional triangulations:
    /// - N_0 = vertices
    /// - N_1 = edges (calculated from cells)
    /// - N_2 = faces/facets (calculated from cells) 
    /// - ...
    /// - N_D = D-dimensional cells
    pub fn calculate<T, U, V, const D: usize>(
        tds: &Tds<T, U, V, D>
    ) -> i64 
    where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        match D {
            2 => Self::calculate_2d(tds),
            3 => Self::calculate_3d(tds), 
            4 => Self::calculate_4d(tds),
            _ => Self::calculate_generic(tds),
        }
    }
    
    /// Optimized 2D calculation: χ = V - E + F
    pub fn calculate_2d<T, U, V>(tds: &Tds<T, U, V, 2>) -> i64 
    where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        let vertices = tds.number_of_vertices() as i64;
        let edges = Self::count_edges_2d(tds) as i64;
        let faces = tds.number_of_cells() as i64;
        
        vertices - edges + faces
    }
    
    /// Optimized 3D calculation: χ = V - E + F - C
    pub fn calculate_3d<T, U, V>(tds: &Tds<T, U, V, 3>) -> i64
    where
        T: CoordinateScalar,
        U: DataType, 
        V: DataType,
    {
        let vertices = tds.number_of_vertices() as i64;
        let edges = Self::count_edges_3d(tds) as i64;
        let faces = Self::count_faces_3d(tds) as i64;
        let cells = tds.number_of_cells() as i64;
        
        vertices - edges + faces - cells
    }
    
    /// Generic D-dimensional calculation
    fn calculate_generic<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>) -> i64
    where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        let mut euler = 0i64;
        
        // Alternate signs: (-1)^k
        for k in 0..=D {
            let sign = if k % 2 == 0 { 1 } else { -1 };
            let count = Self::count_k_simplices(tds, k) as i64;
            euler += sign * count;
        }
        
        euler
    }
    
    // Helper methods for counting different dimensional simplices
    fn count_edges_2d<T, U, V>(tds: &Tds<T, U, V, 2>) -> usize { /* Implementation */ }
    fn count_edges_3d<T, U, V>(tds: &Tds<T, U, V, 3>) -> usize { /* Implementation */ }
    fn count_faces_3d<T, U, V>(tds: &Tds<T, U, V, 3>) -> usize { /* Implementation */ }
    fn count_k_simplices<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>, k: usize) -> usize { /* Implementation */ }
}
```

### 4. Planar Topology Implementation

Default topology for standard Euclidean triangulations:

```rust
// src/topology/spaces/planar.rs

use super::super::traits::{TopologicalSpace, TopologyError};
use super::super::characteristics::euler::EulerCharacteristic;
use crate::core::{triangulation_data_structure::Tds, traits::DataType};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Planar topology for standard Euclidean triangulations
///
/// This represents triangulations embedded in Euclidean space with:
/// - Genus 0 (no holes)
/// - Closed (D-1)-sphere boundary semantics for Euler characteristic
/// - Convex boundary (for finite triangulations)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanarTopology;

impl<T, U, V, const D: usize> TopologicalSpace<T, U, V, D> for PlanarTopology
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn expected_euler_characteristic(&self) -> i64 {
        match D {
            2 => 0, // Boundary is S^1 (circle): χ = 0
            3 => 2, // Boundary is S^2: χ = 2
            4 => 0, // Boundary is S^3: χ = 0
            _ => self.calculate_expected_for_dimension(D),
        }
    }
    
    fn validate_topology(&self, tds: &Tds<T, U, V, D>) -> Result<(), TopologyError> {
        // Check if this topology supports the given dimension
        if !self.supports_dimension(D) {
            return Err(TopologyError::UnsupportedDimension { 
                topology: self.topology_name().to_string(), 
                dimension: D 
            });
        }
        
        // Calculate actual Euler characteristic
        let calculated = EulerCharacteristic::calculate(tds);
        let expected = self.expected_euler_characteristic();
        
        if calculated == expected {
            Ok(())
        } else {
            Err(TopologyError::EulerCharacteristicMismatch { 
                expected, 
                calculated 
            })
        }
    }
    
    fn genus(&self) -> Option<u64> {
        Some(0) // Planar topology has genus 0
    }
    
    fn supports_dimension(&self, dimension: usize) -> bool {
        dimension >= 2 && dimension <= 10 // Practical limitation
    }
    
    fn topology_name(&self) -> &'static str {
        "Planar"
    }
}

impl PlanarTopology {
    /// Calculate expected Euler characteristic for arbitrary dimension.
    ///
    /// We validate the closed (D-1)-dimensional boundary S^{D-1} of the
    /// triangulation. For spheres, χ(S^n) = 1 + (-1)^n.
    fn calculate_expected_for_dimension(&self, d: usize) -> i64 {
        let n = d - 1; // boundary dimension
        1 + if n % 2 == 0 { 1 } else { -1 }
    }
}
```

---

## Integration Points

### 1. Extended Triangulation Data Structure

Integration with the existing `Tds` struct:

```rust
// Addition to src/core/triangulation_data_structure.rs

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Get the Euler characteristic of this triangulation
    pub fn euler_characteristic(&self) -> i64 {
        EulerCharacteristic::calculate(self)
    }
    
    /// Validate topology using the specified topological space
    pub fn validate_topology<S>(&self, space: &S) -> Result<(), TopologyError>
    where
        S: TopologicalSpace<T, U, V, D>,
    {
        space.validate_topology(self)
    }
    
    /// Validate with default planar topology
    pub fn validate_planar_topology(&self) -> Result<(), TopologyError> {
        let planar = PlanarTopology;
        self.validate_topology(&planar)
    }
    
    /// Get the default topology for this triangulation
    pub fn default_topology(&self) -> PlanarTopology {
        PlanarTopology
    }
    
    /// Extended is_valid() that includes topological validation
    pub fn is_valid_with_topology(&self) -> Result<(), ValidationError> {
        // First run existing geometric/structural validation
        self.is_valid()?;
        
        // Then validate topology
        self.validate_planar_topology()
            .map_err(ValidationError::from)
    }
    
    /// Get topology information as a summary
    pub fn topology_summary(&self) -> TopologySummary {
        let euler_char = self.euler_characteristic();
        let topology = self.default_topology();
        
        TopologySummary {
            topology_type: topology.topology_name(),
            euler_characteristic: euler_char,
            expected_euler: topology.expected_euler_characteristic(),
            genus: topology.genus(),
            dimension: D,
            is_valid: self.validate_planar_topology().is_ok(),
        }
    }
}

/// Summary of topological properties for debugging and analysis
#[derive(Debug, Clone, PartialEq)]
pub struct TopologySummary {
    pub topology_type: &'static str,
    pub euler_characteristic: i64,
    pub expected_euler: i64,
    pub genus: Option<u64>,
    pub dimension: usize,
    pub is_valid: bool,
}
```

### 2. Library Module Exports

Update to `src/lib.rs`:

```rust
// Addition to src/lib.rs module declarations

/// Topology analysis and validation for triangulated spaces
///
/// This module provides traits, algorithms, and data structures for analyzing
/// and validating the topological properties of Delaunay triangulations.
/// 
/// # Features
/// 
/// - **Euler Characteristic Calculation**: Compute topological invariants
/// - **Multiple Topology Support**: Planar, spherical, toroidal spaces
/// - **Validation Framework**: Verify triangulation topological correctness
/// - **Dimensional Generic**: Works across all supported dimensions
pub mod topology {
    /// Traits for topological spaces and characteristics
    pub mod traits {
        pub mod euler_characteristic;
        pub mod topological_space;
        pub use euler_characteristic::*;
        pub use topological_space::*;
    }
    /// Topological invariants and their computation
    pub mod characteristics {
        pub mod euler;
        pub mod validation;
        pub use euler::*;
        pub use validation::*;
    }
    /// Different types of topological spaces
    pub mod spaces {
        pub mod planar;
        // Future: spherical, toroidal
        pub use planar::*;
    }
    pub mod util;
    
    // Re-export commonly used types
    pub use characteristics::*;
    pub use spaces::*;
    pub use traits::*;
}
```

### 3. Prelude Integration

Update to the prelude for convenience:

```rust
// Addition to src/lib.rs prelude module

pub mod prelude {
    // ... existing exports ...
    
    // Re-export topology essentials
    pub use crate::topology::{
        characteristics::euler::EulerCharacteristic,
        spaces::planar::PlanarTopology,
        traits::topological_space::TopologicalSpace,
    };
}
```

---

## Implementation Details

### Simplex Counting Algorithms

The core challenge is efficiently counting k-dimensional simplices from the stored D-dimensional cells.

#### Edge Counting (1-simplices)

```rust
impl EulerCharacteristic {
    /// Count unique edges from all cells
    /// Each edge is defined by a pair of vertices
    fn count_edges<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>) -> usize 
    where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        let mut edge_set = FastHashSet::default();
        
        for cell in tds.cells() {
            let vertices = cell.vertices();
            // Generate all vertex pairs (edges) from this D-simplex
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    let edge = if vertices[i] < vertices[j] {
                        (vertices[i], vertices[j])
                    } else {
                        (vertices[j], vertices[i])
                    };
                    edge_set.insert(edge);
                }
            }
        }
        
        edge_set.len()
    }
}
```

#### Face Counting (2-simplices)

```rust
impl EulerCharacteristic {
    /// Count unique triangular faces from all cells
    /// Each face is defined by a triple of vertices  
    fn count_faces<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>) -> usize
    where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        let mut face_set = FastHashSet::default();
        
        for cell in tds.cells() {
            let vertices = cell.vertices();
            // Generate all vertex triples (triangular faces) from this D-simplex
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    for k in (j + 1)..vertices.len() {
                        let mut face = [vertices[i], vertices[j], vertices[k]];
                        face.sort(); // Canonical ordering
                        face_set.insert(face);
                    }
                }
            }
        }
        
        face_set.len()
    }
}
```

### Performance Considerations

#### Caching Strategy

For large triangulations, simplex counting can be expensive. Consider caching:

```rust
// Future optimization: cached simplex counts
pub struct CachedEulerCalculator<T, U, V, const D: usize> {
    cached_edges: Option<usize>,
    cached_faces: Option<usize>,
    cache_valid: bool,
    _phantom: PhantomData<(T, U, V)>,
}

impl<T, U, V, const D: usize> CachedEulerCalculator<T, U, V, D> {
    pub fn invalidate_cache(&mut self) {
        self.cache_valid = false;
        self.cached_edges = None;
        self.cached_faces = None;
    }
    
    pub fn calculate_cached(&mut self, tds: &Tds<T, U, V, D>) -> i64 {
        if !self.cache_valid {
            self.recalculate_cache(tds);
        }
        // Use cached values...
    }
}
```

---

## Testing Strategy

### 1. Unit Tests for Individual Components

#### Euler Characteristic Tests

```rust
// tests/euler_characteristic_tests.rs

#[cfg(test)]
mod euler_tests {
    use super::*;
    use delaunay::prelude::*;
    
    #[test]
    fn test_2d_triangle_euler_characteristic() {
        // Single triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
        ];
        
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let euler = tds.euler_characteristic();
        assert_eq!(euler, 1);
    }
    
    #[test] 
    fn test_3d_tetrahedron_euler_characteristic() {
        // Single tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
        // Wait, this should be χ = 2 for a convex 3D polytope
        // Let me recalculate: for boundary of tetrahedron χ = 2
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let euler = tds.euler_characteristic();
        assert_eq!(euler, 2);
    }
    
    #[test]
    fn test_4d_simplex_euler_characteristic() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let euler = tds.euler_characteristic();
        // Expected for 4D convex polytope
        assert_eq!(euler, 0);
    }
}
```

#### Topology Validation Tests

```rust
// tests/topology_validation.rs

#[cfg(test)]
mod topology_validation_tests {
    use super::*;
    use delaunay::prelude::*;
    
    #[test]
    fn test_planar_topology_validation_success() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        
        // Should validate successfully with planar topology
        assert!(tds.validate_planar_topology().is_ok());
        assert!(tds.is_valid_with_topology().is_ok());
    }
    
    #[test]
    fn test_topology_summary() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let summary = tds.topology_summary();
        
        assert_eq!(summary.topology_type, "Planar");
        assert_eq!(summary.dimension, 3);
        assert_eq!(summary.genus, Some(0));
        assert_eq!(summary.euler_characteristic, summary.expected_euler);
        assert!(summary.is_valid);
    }
}
```

### 2. Random Triangulation Validation

#### Property-Based Testing

```rust
// tests/random_triangulation_topology.rs

#[cfg(test)]
mod random_topology_tests {
    use super::*;
    use delaunay::prelude::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_random_2d_triangulations_euler_characteristic(
            vertices in prop::collection::vec(
                prop::array::uniform2(-10.0f64..10.0), 
                5..20
            )
        ) {
            let vertex_vec: Vec<_> = vertices.into_iter()
                .map(|coords| vertex!(coords))
                .collect();
                
            if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertex_vec) {
                // Every valid 2D triangulation should have correct topology
                prop_assert!(tds.validate_planar_topology().is_ok());
                
                let summary = tds.topology_summary();
                prop_assert!(summary.is_valid);
            }
        }
        
        #[test]
        fn test_random_3d_triangulations_euler_characteristic(
            vertices in prop::collection::vec(
                prop::array::uniform3(-10.0f64..10.0), 
                8..25
            )
        ) {
            let vertex_vec: Vec<_> = vertices.into_iter()
                .map(|coords| vertex!(coords))
                .collect();
                
            if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertex_vec) {
                // Every valid 3D triangulation should have χ = 2
                prop_assert!(tds.validate_planar_topology().is_ok());
                prop_assert_eq!(tds.euler_characteristic(), 2);
            }
        }
    }
}
```

### 3. Integration with Existing Test Framework

#### Extended Circumsphere Debug Tools

```rust
// Addition to tests/circumsphere_debug_tools.rs

#[test]
fn test_topology_debug_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Topology Debug Analysis ===");
    
    // Test across multiple dimensions
    for d in 2..=4 {
        println!("\n--- {}D Topology Analysis ---", d);
        
        match d {
            2 => test_2d_topology_debug()?,
            3 => test_3d_topology_debug()?,
            4 => test_4d_topology_debug()?,
            _ => {}
        }
    }
    
    Ok(())
}

fn test_2d_topology_debug() -> Result<(), Box<dyn std::error::Error>> {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.2, 0.3]),
    ];
    
    let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices)?;
    let summary = tds.topology_summary();
    
    println!("2D Triangulation:");
    println!("  Vertices: {}", tds.number_of_vertices());
    println!("  Cells: {}", tds.number_of_cells());
    println!("  Euler Characteristic: {} (expected: {})", 
             summary.euler_characteristic, summary.expected_euler);
    println!("  Topology Valid: {}", summary.is_valid);
    
    Ok(())
}
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Deliverables:**

- [ ] `src/topology/` module structure
- [ ] `TopologicalSpace` trait definition
- [ ] `TopologyError` error types
- [ ] Basic `EulerCharacteristic` struct
- [ ] Module exports in `lib.rs`

**Testing:**

- [ ] Trait compilation tests
- [ ] Error type tests
- [ ] Module import tests

### Phase 2: Euler Characteristic Implementation (Week 2-3)

**Deliverables:**

- [ ] Complete `EulerCharacteristic::calculate()` implementation
- [ ] Optimized 2D, 3D calculations
- [ ] Generic D-dimensional calculation
- [ ] Simplex counting algorithms (edges, faces)

**Testing:**

- [ ] Unit tests for known geometric configurations
- [ ] 2D triangle, 3D tetrahedron, 4D simplex tests
- [ ] Edge/face counting validation

### Phase 3: Planar Topology and Tds Integration (Week 3-4)

**Deliverables:**

- [ ] Complete `PlanarTopology` implementation
- [ ] `Tds` method extensions
- [ ] `validate_topology()` and `is_valid_with_topology()`
- [ ] `TopologySummary` struct

**Testing:**

- [ ] Topology validation tests
- [ ] Integration with existing `is_valid()` tests
- [ ] Topology summary tests

### Phase 4: Comprehensive Testing (Week 4-5)

**Deliverables:**

- [ ] Random triangulation property tests
- [ ] Integration with debug tools
- [ ] Performance benchmarking
- [ ] Edge case testing

**Testing:**

- [ ] Property-based random testing
- [ ] Multi-dimensional validation
- [ ] Performance regression tests
- [ ] Documentation examples testing

### Phase 5: Documentation and Examples (Week 5-6)

**Deliverables:**

- [ ] Complete API documentation
- [ ] Usage examples
- [ ] Integration guide
- [ ] Performance characterization

**Testing:**

- [ ] Documentation example tests
- [ ] Integration guide validation
- [ ] Public API completeness check

---

## Benefits and Rationale

### 1. **Correctness Assurance**

- **Invariant Validation**: Ensures triangulations maintain topological correctness
- **Early Error Detection**: Catches algorithmic bugs through topology validation
- **Robust Testing**: Property-based testing validates random configurations

### 2. **Extensibility**

- **Multiple Topologies**: Framework ready for spherical, toroidal extensions
- **Dimensional Generic**: Works across all supported dimensions
- **Trait-Based Design**: Easy to add new topological spaces

### 3. **Integration with Existing Codebase**

- **Backward Compatible**: Does not break existing API
- **Consistent Patterns**: Follows established architectural patterns  
- **Performance Aware**: Designed with caching and optimization hooks

### 4. **Developer Experience**

- **Rich Debugging**: `TopologySummary` provides comprehensive analysis
- **Clear Error Messages**: Descriptive topology validation errors
- **Optional Validation**: Can be enabled/disabled based on performance needs

### 5. **Academic and Research Value**

- **Theoretical Foundation**: Based on well-established topological theory
- **Validation Tool**: Useful for computational geometry research
- **Educational**: Demonstrates topological concepts in practice

### 6. **Future-Proofing**

- **Modular Design**: Easy to extend with new topology types
- **Performance Hooks**: Ready for optimization and caching strategies
- **Testing Framework**: Comprehensive validation for future changes

---

## Conclusion

This topology integration design provides a solid foundation for topological analysis in the delaunay triangulation library. By keeping planar
topology as the default while building an extensible framework, we maintain backward compatibility while adding powerful validation capabilities.

The phased implementation approach ensures steady progress with comprehensive testing at each stage. The design follows established patterns
in the codebase and integrates cleanly with existing validation and testing frameworks.

The result will be a more robust, validated, and theoretically sound triangulation library that serves both practical applications and computational geometry research.

---

## Euler-Poincaré Validation: Detailed Implementation Plan

> Note: The core Level 3 Euler characteristic check is implemented in `Triangulation::is_valid()` as of v0.6.x.
> The remainder of this section is retained as historical design notes for future topology work.

### Status and Metadata

**Status:** Partially implemented  
**Created:** 2025-10-16  
**Implemented:** v0.6.x (Level 3: `Triangulation::is_valid()`)  
**Original Target Release:** v0.6.0 (historical)  
**Target Release (remaining work):** TBD  
**Priority:** High  
**Complexity:** High  

### Motivation and Scope

The **Euler-Poincaré characteristic** (χ) is a fundamental topological invariant that provides a global consistency check for
simplicial complexes. Unlike local checks (neighbor symmetry, facet sharing), the Euler characteristic catches:

- **Global topological corruption**: Errors that preserve local invariants but violate global topology
- **Combinatorial inconsistencies**: Wrong counts of simplices at different dimensions
- **Degenerate configurations**: Non-manifold or disconnected components
- **Construction errors**: Bugs in insertion algorithms that create invalid topology

**Why this matters for Delaunay triangulations:**

- Finite Delaunay triangulations in R^D have a convex-hull boundary that is a **closed (D-1)-sphere**, and this boundary complex is what we validate topologically
- The full complex (including interior cells) is a **topological D-ball** with χ = 1, but for Euler-Poincaré invariants we care about the closed boundary sphere
- Violations indicate serious algorithmic failures that may not be caught by local checks

### Background: Mathematical Foundations

#### Definitions

**Simplicial Complex:** A collection of simplices closed under taking faces.

**k-simplex:** The convex hull of k+1 affinely independent points:

- 0-simplex: vertex
- 1-simplex: edge
- 2-simplex: triangle
- 3-simplex: tetrahedron
- D-simplex: D-dimensional cell with D+1 vertices

**f-vector:** (f₀, f₁, ..., f_D) where f_k = number of k-simplices

**Boundary:** A (D-1)-facet shared by exactly one D-cell (not two)

**Closed complex:** No boundary (every facet shared by exactly 2 cells)

#### The Euler-Poincaré Formula

For a finite simplicial complex:

```text
χ = Σ(k=0 to D) (-1)^k · f_k
  = f₀ - f₁ + f₂ - f₃ + ... ± f_D
```

**Examples:**

- 2D triangle: χ = 3 - 3 + 1 = 1 (with boundary)
- 3D tetrahedron: χ = 4 - 6 + 4 - 1 = 1 (with boundary)
- 4D simplex: χ = 5 - 10 + 10 - 5 + 1 = 1 (with boundary)

#### Expected χ by Topological Classification

|| Topology | Has Boundary | χ (General) | χ (D=2) | χ (D=3) | χ (D=4) |
||----------|--------------|-------------|---------|---------|----------|
|| **Empty** | N/A | 0 | 0 | 0 | 0 |
|| **Single Simplex** | Yes | 1 | 1 | 1 | 1 |
|| **Ball (D-ball)** | Yes | 1 | 1 | 1 | 1 |
|| **Closed Sphere (S^D)** | No | 1+(-1)^D | 2 | 0 | 2 |
|| **Unknown/Invalid** | Varies | N/A | ? | ? | ? |

**Key insight:** For Euler-Poincaré validation we treat finite Delaunay
triangulations as triangulations of the closed (D-1)-sphere boundary, so the
expected Euler characteristic is **χ(S^{D-1}) = 1 + (-1)^{D-1}** (e.g., χ=0 in
2D, χ=2 in 3D, χ=0 in 4D).

### Design Overview: New Topology Module

#### Module Location and Structure

```text
src/core/topology.rs          # New module for topological invariants
```

**Rationale:** Place in `core` (not `geometry`) because topology is about combinatorial structure, not geometric predicates.

#### Public API Design

```rust
// src/core/topology.rs

use crate::core::{
    collections::{FastHashSet, SmallBuffer, MAX_PRACTICAL_DIMENSION_SIZE},
    triangulation_data_structure::Tds,
    traits::data_type::DataType,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use thiserror::Error;

/// Counts of k-simplices for all dimensions 0 ≤ k ≤ D
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplexCounts {
    /// by_dim[k] = f_k = number of k-simplices
    pub by_dim: Vec<usize>,
}

impl SimplexCounts {
    /// Get the number of k-simplices
    #[inline]
    pub fn count(&self, k: usize) -> usize {
        self.by_dim.get(k).copied().unwrap_or(0)
    }
    
    /// The dimension (maximum k where f_k > 0)
    pub fn dimension(&self) -> usize {
        self.by_dim.len().saturating_sub(1)
    }
}

/// Topological classification of a triangulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyClassification {
    /// Empty triangulation (no cells)
    Empty,
    
    /// Single D-simplex
    SingleSimplex(usize),
    
    /// Topological D-ball (has boundary)
    Ball(usize),
    
    /// Closed D-sphere (no boundary)
    ClosedSphere(usize),
    
    /// Cannot determine or doesn't fit known categories
    Unknown,
}

/// Result of Euler characteristic validation
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyCheckResult {
    /// Computed Euler characteristic
    pub chi: isize,
    
    /// Expected χ based on classification (None if unknown)
    pub expected: Option<isize>,
    
    /// Topological classification
    pub classification: TopologyClassification,
    
    /// Full simplex counts
    pub counts: SimplexCounts,
    
    /// Diagnostic notes or warnings
    pub notes: Vec<String>,
}

impl TopologyCheckResult {
    /// Whether χ matches expectation
    pub fn is_valid(&self) -> bool {
        self.expected.map_or(true, |exp| self.chi == exp)
    }
}

/// Errors in topology computation or validation
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyError {
    #[error("Failed to count simplices: {0}")]
    Counting(String),
    
    #[error("Failed to classify triangulation: {0}")]
    Classification(String),
    
    #[error("Euler characteristic mismatch: computed χ={chi}, expected χ={expected} for {classification:?}")]
    Mismatch {
        chi: isize,
        expected: isize,
        classification: TopologyClassification,
    },
}

// ========================================================================
// Primary Public API Functions
// ========================================================================

/// Count all k-simplices in the triangulation
pub fn count_simplices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<SimplexCounts, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Implementation deferred - see counting strategy below
    unimplemented!("count_simplices: to be implemented")
}

/// Compute Euler characteristic from simplex counts
pub fn euler_characteristic(counts: &SimplexCounts) -> isize {
    counts
        .by_dim
        .iter()
        .enumerate()
        .map(|(k, &f_k)| {
            let sign = if k % 2 == 0 { 1 } else { -1 };
            sign * (f_k as isize)
        })
        .sum()
}

/// Classify the triangulation topologically
pub fn classify_triangulation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<TopologyClassification, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Implementation deferred - see classification strategy below
    unimplemented!("classify_triangulation: to be implemented")
}

/// Get expected χ for a topological classification
pub fn expected_chi_for(classification: &TopologyClassification) -> Option<isize> {
    match classification {
        TopologyClassification::Empty => Some(0),
        TopologyClassification::SingleSimplex(_) => Some(1),
        TopologyClassification::Ball(_) => Some(1),
        TopologyClassification::ClosedSphere(d) => {
            // χ(S^D) = 1 + (-1)^D
            Some(1 + if d % 2 == 0 { -1 } else { 1 })
        }
        TopologyClassification::Unknown => None,
    }
}

/// Validate triangulation Euler characteristic
pub fn validate_triangulation_euler<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<TopologyCheckResult, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let counts = count_simplices(tds)?;
    let chi = euler_characteristic(&counts);
    let classification = classify_triangulation(tds)?;
    let expected = expected_chi_for(&classification);
    
    let mut notes = Vec::new();
    
    // Add diagnostic notes
    if let Some(exp) = expected {
        if chi != exp {
            notes.push(format!(
                "Euler characteristic mismatch: computed {}, expected {}",
                chi, exp
            ));
        }
    }
    
    Ok(TopologyCheckResult {
        chi,
        expected,
        classification,
        counts,
        notes,
    })
}

/// Validate that a single D-cell has χ = 1
///
/// A single D-simplex with D+1 vertices has:
/// f_k = C(D+1, k+1) and χ = 1 for all D ≥ 0
pub fn validate_cell_euler<T, U, V, const D: usize>(
    _cell: &crate::core::cell::Cell<T, U, V, D>,
) -> Result<isize, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // For a single simplex, we can compute χ combinatorially
    // without enumerating all sub-simplices
    // χ = Σ(k=0 to D) (-1)^k · C(D+1, k+1) = 1
    Ok(1)
}
```

### Efficient k-Simplex Counting Strategy

#### Overview

- **f₀ (vertices):** `tds.number_of_vertices()` — O(1) from stored count
- **f_D (cells):** `tds.number_of_cells()` — O(1) from stored count
- **f_{D-1} (facets):** `tds.build_facet_to_cells_map().len()` — O(N·D²) where N = cells
- **Intermediate k (0 < k < D-1):** Enumerate combinations from each cell — O(N · Σ C(D+1, k+1))

#### Detailed Algorithm

```rust
fn count_simplices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<SimplexCounts, TopologyError> {
    let mut by_dim = vec![0usize; D + 1];
    
    // f₀: vertices
    by_dim[0] = tds.number_of_vertices();
    
    // f_D: D-cells
    by_dim[D] = tds.number_of_cells();
    
    // Handle empty triangulation
    if by_dim[D] == 0 {
        return Ok(SimplexCounts { by_dim });
    }
    
    // f_{D-1}: (D-1)-facets
    by_dim[D - 1] = tds
        .build_facet_to_cells_map()
        .map_err(|e| TopologyError::Counting(format!("Failed to build facet map: {}", e)))?
        .len();
    
    // Intermediate dimensions: enumerate combinations
    for k in 1..D - 1 {
        by_dim[k] = count_k_simplices(tds, k)?;
    }
    
    Ok(SimplexCounts { by_dim })
}

fn count_k_simplices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    k: usize,
) -> Result<usize, TopologyError> {
    use crate::core::util::generate_combinations;
    
    let mut k_simplex_set = FastHashSet::default();
    let simplex_size = k + 1; // k-simplex has k+1 vertices
    
    for (_cell_key, cell) in tds.cells() {
        let vertices = cell.vertices();
        
        // Generate all C(D+1, k+1) combinations of size k+1
        for combo in generate_combinations(vertices.len(), simplex_size) {
            // Extract vertex keys for this k-simplex
            let mut k_simplex: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                combo.iter().map(|&idx| vertices[idx]).collect();
            
            // Canonicalize by sorting for deduplication
            k_simplex.sort();
            
            k_simplex_set.insert(k_simplex);
        }
    }
    
    Ok(k_simplex_set.len())
}
```

**Complexity Analysis:**

- For D=3: count 1-simplices (edges) and 2-simplices (faces)
- For each cell: C(4,2)=6 edges, C(4,3)=4 faces
- Total: O(N·6) + O(N·4) = O(N) with small constants
- For D=4: C(5,2)=10 edges, C(5,3)=10 faces, C(5,4)=5 facets
- Still practical for D ≤ 5 used in this project

**Implementation Notes:**

- Use `SmallBuffer` to avoid heap allocations for D ≤ 7
- Canonicalize via sorting (orientation-agnostic)
- Fast path for single simplex: f_k = C(D+1, k+1) computed directly
- Empty triangulation: all f_k = 0, χ = 0

### Classification Strategy

```rust
fn classify_triangulation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<TopologyClassification, TopologyError> {
    let num_cells = tds.number_of_cells();
    
    // Empty triangulation
    if num_cells == 0 {
        return Ok(TopologyClassification::Empty);
    }
    
    // Single simplex
    if num_cells == 1 {
        return Ok(TopologyClassification::SingleSimplex(D));
    }
    
    // Check boundary
    let has_boundary = tds
        .number_of_boundary_facets()
        .map_err(|e| TopologyError::Classification(format!("Failed to count boundary: {}", e)))?
        > 0;
    
    if has_boundary {
        // Has boundary → topological ball
        Ok(TopologyClassification::Ball(D))
    } else {
        // No boundary → closed manifold (assume sphere for now)
        Ok(TopologyClassification::ClosedSphere(D))
    }
}
```

**Caveats:**

- Finite Delaunay triangulations almost always have boundary (convex hull)
- Closed manifolds would require special construction (e.g., periodic boundary conditions)
- Non-manifold or disconnected components → classify as `Unknown` (future enhancement)

**Configuration Option (Future):**

```rust
pub struct TopologyConfig {
    pub assume_ball: bool,  // Assume triangulation is a ball
    pub assume_closed: bool, // Assume closed manifold
}
```

### Current Integration (implemented)

As of v0.6.x, Euler characteristic validation is integrated at the **Triangulation** layer (Level 3):

- `Tds::is_valid()` / `Tds::validate()` cover element + structural invariants (Levels 1–2).
- `Triangulation::is_valid()` adds topology checks (manifold-with-boundary + Euler characteristic).
- `Triangulation::validate()` runs `Tds::validate()` first, then `Triangulation::is_valid()`.

The implementation uses the topology module's helper:

```rust
// Simplified from src/core/triangulation.rs
let topology = validate_triangulation_euler(&self.tds)?;
```

For the current user-facing behavior and error-type layering, see `docs/validation.md`.

### Integration Points (historical / superseded)

#### 1. Tds Integration

```rust
// Addition to src/core/triangulation_data_structure.rs

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Validate the Euler characteristic of this triangulation
    pub fn validate_euler_characteristic(&self) -> Result<TopologyCheckResult, TopologyError> {
        crate::core::topology::validate_triangulation_euler(self)
    }
    
    /// Get the Euler characteristic (convenience wrapper)
    pub fn euler_characteristic(&self) -> Result<isize, TopologyError> {
        let counts = crate::core::topology::count_simplices(self)?;
        Ok(crate::core::topology::euler_characteristic(&counts))
    }
}
```

#### 2. is_valid() Integration

```rust
// Modified Tds::is_valid() method

pub fn is_valid(&self) -> Result<(), TriangulationValidationError>
where
    [T; D]: DeserializeOwned + Serialize + Sized,
{
    // Existing validation steps
    self.validate_vertex_mappings()?;
    self.validate_cell_mappings()?;
    self.validate_no_duplicate_cells()?;
    
    for (cell_id, cell) in &self.cells {
        cell.is_valid().map_err(|source| {
            // ... existing error mapping ...
        })?;
    }
    
    self.validate_facet_sharing()?;
    self.validate_neighbors()?;
    
    // NEW: Euler characteristic validation
    // Only run if enabled via feature flag or debug mode
    #[cfg(any(debug_assertions, feature = "topology-validation"))]
    {
        let topology_result = self.validate_euler_characteristic()
            .map_err(|e| TriangulationValidationError::Topology {
                message: format!("Euler characteristic validation failed: {}", e),
            })?;
        
        if !topology_result.is_valid() {
            return Err(TriangulationValidationError::EulerCharacteristic {
                computed: topology_result.chi,
                expected: topology_result.expected.unwrap_or(0),
                classification: format!("{:?}", topology_result.classification),
                notes: topology_result.notes.join("; "),
            });
        }
    }
    
    Ok(())
}
```

#### 3. TriangulationValidationError Extension

```rust
// Addition to TriangulationValidationError enum

#[derive(Debug, Error, PartialEq)]
pub enum TriangulationValidationError {
    // ... existing variants ...
    
    #[error("Euler characteristic mismatch: computed {computed}, expected {expected} for {classification}. Notes: {notes}")]
    EulerCharacteristic {
        computed: isize,
        expected: isize,
        classification: String,
        notes: String,
    },
    
    #[error("Topology validation failed: {message}")]
    Topology { message: String },
}
```

### Design Contract: gather_boundary_facet_info Integration

**Critical Design Requirement:** When `gather_boundary_facet_info` compiles boundary statistics, it **must** also perform Euler characteristic validation.

#### Extended BoundaryFacetInfo Structure

```rust
// Addition to BoundaryFacetInfo or new BoundaryReport struct

pub struct BoundaryReport {
    /// Per-facet information (existing)
    pub facet_infos: Vec<BoundaryFacetInfo>,
    
    /// Euler characteristic validation
    pub chi: isize,
    pub expected_chi: Option<isize>,
    pub classification: TopologyClassification,
    pub euler_ok: bool,
    
    /// Diagnostic notes
    pub notes: Vec<String>,
}
```

#### Implementation Checklist

- [ ] Update `gather_boundary_facet_info` to return `BoundaryReport` instead of `Vec<BoundaryFacetInfo>`
- [ ] Call `topology::count_simplices` within the function
- [ ] Call `topology::classify_triangulation` to get classification
- [ ] Compute χ and compare with expected value
- [ ] Populate `BoundaryReport` with all diagnostic information
- [ ] Update all call sites to handle new return type
- [ ] Update debug printouts to show χ and validation status

**Example Integration:**

```rust
fn gather_boundary_facet_info(
    tds: &Tds<T, U, V, D>,
    boundary_facet_handles: &[FacetHandle],
) -> Result<BoundaryReport, InsertionError> {
    // Existing facet info gathering
    let facet_infos = /* ... existing code ... */;
    
    // NEW: Topology validation
    let counts = topology::count_simplices(tds)
        .map_err(|e| /* ... */)?;
    let chi = topology::euler_characteristic(&counts);
    let classification = topology::classify_triangulation(tds)
        .map_err(|e| /* ... */)?;
    let expected_chi = topology::expected_chi_for(&classification);
    let euler_ok = expected_chi.map_or(true, |exp| chi == exp);
    
    let mut notes = Vec::new();
    if !euler_ok {
        notes.push(format!(
            "WARNING: Euler characteristic mismatch (χ={}, expected={})",
            chi, expected_chi.unwrap_or(0)
        ));
    }
    
    Ok(BoundaryReport {
        facet_infos,
        chi,
        expected_chi,
        classification,
        euler_ok,
        notes,
    })
}
```

### Testing Strategy

#### 1. Unit Tests by Dimension

**2D Tests:**

```rust
#[test]
fn test_2d_single_triangle_euler() {
    // V=3, E=3, F=1 → χ = 3-3+1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    
    let result = tds.validate_euler_characteristic().unwrap();
    assert_eq!(result.chi, 1);
    assert_eq!(result.classification, TopologyClassification::Ball(2));
    assert!(result.is_valid());
}

#[test]
fn test_2d_two_triangles_euler() {
    // Two triangles sharing an edge
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, -1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    
    let result = tds.validate_euler_characteristic().unwrap();
    assert_eq!(result.chi, 1); // Still a topological disk
    assert!(result.is_valid());
}
```

**3D Tests:**

```rust
#[test]
fn test_3d_tetrahedron_euler() {
    // Single tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    
    let result = tds.validate_euler_characteristic().unwrap();
    assert_eq!(result.chi, 1);
    assert_eq!(result.classification, TopologyClassification::Ball(3));
}

#[test]
fn test_3d_with_interior_vertex() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25]), // Interior
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    
    // Should still be χ = 1 (topological ball)
    assert_eq!(tds.euler_characteristic().unwrap(), 1);
}
```

**4D Tests:**

```rust
#[test]
fn test_4d_simplex_euler() {
    // V=5, E=10, F=10, Tet=5, Cell=1
    // χ = 5 - 10 + 10 - 5 + 1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
    
    let result = tds.validate_euler_characteristic().unwrap();
    assert_eq!(result.chi, 1);
}
```

#### 2. Edge Cases

```rust
#[test]
fn test_empty_triangulation_euler() {
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
    
    let result = tds.validate_euler_characteristic().unwrap();
    assert_eq!(result.chi, 0);
    assert_eq!(result.classification, TopologyClassification::Empty);
}

#[test]
fn test_single_vertex_euler() {
    let vertices = vec![vertex!([0.0, 0.0, 0.0])];
    // This may fail to create a valid triangulation
    // Document expected behavior
}
```

#### 3. Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_random_2d_always_chi_one(
        points in prop::collection::vec(
            prop::array::uniform2(-100.0f64..100.0),
            4..50
        )
    ) {
        let vertices: Vec<_> = points.into_iter()
            .map(|p| vertex!(p))
            .collect();
        
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            let chi = tds.euler_characteristic().unwrap();
            prop_assert_eq!(chi, 1, "2D triangulation should have χ=1");
        }
    }
    
    #[test]
    fn test_random_3d_always_chi_one(
        points in prop::collection::vec(
            prop::array::uniform3(-100.0f64..100.0),
            5..50
        )
    ) {
        let vertices: Vec<_> = points.into_iter()
            .map(|p| vertex!(p))
            .collect();
        
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            let chi = tds.euler_characteristic().unwrap();
            prop_assert_eq!(chi, 1, "3D triangulation should have χ=1");
        }
    }
}
```

#### 4. gather_boundary_facet_info Integration Test

```rust
#[test]
fn test_boundary_report_includes_euler() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    
    // Test that boundary report includes topology info
    // (requires gather_boundary_facet_info update)
    let boundary_facets = tds.boundary_facets().unwrap().collect::<Vec<_>>();
    let report = gather_boundary_facet_info(&tds, &boundary_facets).unwrap();
    
    assert_eq!(report.chi, 1);
    assert_eq!(report.expected_chi, Some(1));
    assert!(report.euler_ok);
    assert_eq!(report.classification, TopologyClassification::Ball(3));
}
```

### Academic References

#### Primary References

1. **Hatcher, A.** *Algebraic Topology.* Cambridge University Press, 2002.
   - **Chapter 2:** Euler characteristic fundamentals, CW complexes
   - **Relevance:** Theoretical foundation for χ in simplicial complexes
   - **URL:** <https://pi.math.cornell.edu/~hatcher/AT/ATpage.html>

2. **Munkres, J. R.** *Elements of Algebraic Topology.* Addison-Wesley, 1984.
   - **Chapter 1:** Simplicial complexes and chain complexes
   - **Relevance:** Combinatorial definition of Euler characteristic
   - **ISBN:** 978-0201045864

3. **Edelsbrunner, H., Harer, J. L.** *Computational Topology: An Introduction.* American Mathematical Society, 2010.
   - **Chapter 3:** Simplicial homology and Euler characteristic
   - **Relevance:** Computational algorithms for topology
   - **URL:** <https://www.maths.ed.ac.uk/~v1ranick/papers/edelcomp.pdf>
   - **ISBN:** 978-0821849255

4. **Zomorodian, A.** *Topology for Computing.* Cambridge University Press, 2005.
   - **Chapter 2:** Combinatorial structures and invariants
   - **Relevance:** Persistent homology and computational topology
   - **ISBN:** 978-0521136099

#### Combinatorics References

5. **Ziegler, G. M.** *Lectures on Polytopes.* Springer, 1995.
   - **Chapter 8:** Euler characteristic of convex polytopes
   - **Relevance:** f-vectors and combinatorics of simplices
   - **ISBN:** 978-0387943657

6. **Stillwell, J.** *Euler's Gem: The Polyhedron Formula and the Birth of Topology.* Princeton University Press, 2010.
   - **Historical context** for Euler's formula V - E + F = 2
   - **Relevance:** Intuition and historical development
   - **ISBN:** 978-0691154572

#### Practical References

7. **CGAL Documentation:** Triangulations and Topological Invariants
   - **Section:** 3D Triangulation Validation
   - **Relevance:** Practical implementation in computational geometry library
   - **URL:** <https://doc.cgal.org/latest/Triangulation_3/index.html>

#### Reference Verification Checklist

- [ ] Verify all ISBN numbers and DOIs are correct
- [ ] Ensure URLs are stable (prefer DOIs or archive.org links)
- [ ] Cross-check formulas with multiple sources
- [ ] Document specific sections/chapters referenced
- [ ] Add bibtex entries for LaTeX documentation (future)

### Documentation Updates Checklist

#### Before Publishing to crates.io

- [ ] Update `docs/code_organization.md`:
  - Add `src/core/topology.rs` under "Core Modules"
  - Brief description: "Topological invariants (Euler characteristic) and validation"

- [ ] Update crate-level documentation (`src/lib.rs`):
  - Add section on Euler-Poincaré validation
  - Document how to enable/disable via feature flags
  - Explain performance implications

- [ ] Create examples:
  - `examples/euler_characteristic_2d.rs`
  - `examples/euler_characteristic_3d.rs`
  - `examples/topology_validation.rs`

- [ ] Update README.md:
  - Add "Topological Validation" to features list
  - Mention Euler characteristic checking

- [ ] Create user guide:
  - `docs/topology_validation_guide.md`
  - When to use, performance considerations, interpretation of results

### Implementation Workflow

#### Development Sequence

1. **Create module skeleton**

   ```bash
   touch src/core/topology.rs
   # Update src/core/mod.rs to include pub mod topology;
   ```

2. **Implement core types** (SimplexCounts, TopologyClassification, etc.)
   - No dependencies yet
   - Focus on data structures

3. **Implement counting functions**
   - Start with `count_simplices`
   - Test individually before integration

4. **Implement classification**
   - `classify_triangulation`
   - `expected_chi_for`

5. **Implement validation**
   - `validate_triangulation_euler`
   - `validate_cell_euler`

6. **Integrate with Tds**
   - Add methods to `Tds`
   - Update `is_valid()`
   - Add `TriangulationValidationError` variants

7. **Update gather_boundary_facet_info**
   - Create `BoundaryReport` struct
   - Add topology fields
   - Update all call sites

8. **Write tests**
   - Unit tests per dimension
   - Edge cases
   - Property-based tests
   - Integration tests

9. **Documentation**
   - API docs
   - Examples
   - User guide
   - Update organizational docs

10. **Quality checks**

    ```bash
    just fmt
    just clippy
    just test
    just doc-check
    just examples
    just coverage
    ```

#### Feature Flag Strategy

Currently, Euler characteristic validation runs as part of `Triangulation::is_valid()` (Level 3) and is not feature-gated.

If we later decide to make it optional, a potential feature could look like:

```toml
# Cargo.toml
[features]
default = ["dense-slotmap"]  # Current default

# NOTE: `topology-validation` is proposed; only document it here once it exists in Cargo.toml.
# topology-validation = []  # Gate Level 3 Euler characteristic validation
```

Gate expensive checks:

```rust
#[cfg(any(debug_assertions, feature = "topology-validation"))]
{
    // Euler characteristic validation
}
```

After validation in production, consider making it default.

### Limitations and Future Work

#### Current Limitations

- **Classification heuristic** assumes simple topologies (ball or sphere)
- **No detection** of non-manifold boundaries
- **No handling** of disconnected components
- **Performance** not optimized for very large triangulations (>100K cells)

#### Future Enhancements

- **Caching:** Cache simplex counts, invalidate on modification
- **Incremental:** Update counts incrementally during insertion/deletion
- **Advanced classification:** Detect genus, handle non-manifolds
- **Homology:** Compute full homology groups (Betti numbers)
- **Parallel:** Parallelize k-simplex enumeration for large meshes

### Summary

This design provides a **comprehensive, mathematically sound, and practically implementable** approach to Euler-Poincaré validation
for the Delaunay triangulation library. The plan:

- ✅ Captures all API signatures precisely
- ✅ Documents counting algorithms with complexity analysis
- ✅ Integrates with existing validation framework
- ✅ Includes gather_boundary_facet_info contract
- ✅ Provides comprehensive testing strategy
- ✅ Lists academic references for verification
- ✅ Outlines implementation workflow
- ✅ Notes limitations and future work

**Next Action:** Begin implementation following the workflow above, starting with module skeleton and core data structures.

---

<citations>
<document>
    <document_type>RULE</document_type>
    <document_id>/Users/adam/projects/delaunay/WARP.md</document_id>
</document>
</citations>
