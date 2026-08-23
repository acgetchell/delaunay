//! Stack-friendly buffer aliases for topology and geometry algorithms.
//!
//! The aliases in this module document the expected cardinality of common
//! intermediate collections while keeping hot paths allocation-conscious.

use super::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, Uuid};
use crate::core::facet::FacetHandle;
use crate::core::tds::{SimplexKey, VertexKey};

// =============================================================================
// ALGORITHM-SPECIFIC BUFFER TYPES
// =============================================================================

/// Size constant for operations that may affect multiple simplices during cleanup.
/// 16 provides generous headroom for duplicate removal and topology repair operations.
///
/// This constant is publicly exposed to allow external modules to derive buffer sizes
/// from it, ensuring consistent sizing across the codebase.
pub const CLEANUP_OPERATION_BUFFER_SIZE: usize = 16;

/// Collection for tracking simplices to remove during cleanup operations.
/// Most cleanup operations affect a small number of simplices.
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: Up to 16 simplices (covers most cleanup scenarios)
/// - **Use Case**: Duplicate simplex removal, invalid facet cleanup
/// - **Performance**: Avoids heap allocation for typical cleanup operations
pub type SimplexRemovalBuffer = SmallBuffer<SimplexKey, CLEANUP_OPERATION_BUFFER_SIZE>;

/// Collection for tracking Delaunay violations during iterative refinement.
/// Most violation checks find a small number of violating simplices.
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: Up to 16 simplices (covers most violation scenarios)
/// - **Use Case**: Iterative cavity refinement, Delaunay validation
/// - **Performance**: Avoids heap allocation in hot paths during insertion
/// - **Typical Size**: 0-4 violations in well-conditioned triangulations
pub type ViolationBuffer = SmallBuffer<SimplexKey, CLEANUP_OPERATION_BUFFER_SIZE>;

/// Collection for tracking simplex keys during insertion operations.
/// Most insertion operations create a small number of simplices.
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: Up to 16 simplices (covers most insertion scenarios)
/// - **Use Case**: Cavity-based insertion, simplex creation tracking
/// - **Performance**: Avoids heap allocation during simplex creation
/// - **Typical Size**: 4-8 simplices in well-conditioned triangulations (D+1 for simple cavity)
pub type SimplexKeyBuffer = SmallBuffer<SimplexKey, CLEANUP_OPERATION_BUFFER_SIZE>;

/// Buffer for storing cavity boundary facets during insertion/removal operations.
///
/// This is used by cavity extraction and filling routines. Inline capacity 64 avoids heap
/// allocation for typical cavities while still allowing growth for large conflict regions.
pub type CavityBoundaryBuffer = SmallBuffer<FacetHandle, 64>;

/// Buffer for storing simplices that share a facet.
/// Facets are incident to at most 2 simplices (one-sided=1, two-sided=2).
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: Exactly 2 simplices (no heap allocation for valid triangulations)
/// - **Use Case**: Facet-to-simplices mapping validation, cavity boundary detection
/// - **Performance**: Eliminates heap allocation when invariant holds (≤2 simplices per facet)
/// - **Memory Efficiency**: 2 × 8 bytes = 16 bytes on stack per facet
///
/// # Invariant
///
/// Valid triangulations have the following facet sharing invariants:
/// - **Boundary facets**: Shared by exactly 1 simplex (hull facets)
/// - **Interior facets**: Shared by exactly 2 simplices (adjacent simplices)
/// - **Invalid**: Shared by >2 simplices (indicates TDS corruption)
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::FacetSharingSimplicesBuffer;
///
/// let mut sharing_simplices: FacetSharingSimplicesBuffer = FacetSharingSimplicesBuffer::new();
/// assert!(sharing_simplices.is_empty());
/// ```
pub type FacetSharingSimplicesBuffer = SmallBuffer<SimplexKey, 2>;

// =============================================================================
// SEMANTIC SIZE CONSTANTS AND TYPE ALIASES
// =============================================================================

/// Buffer sized for vertex collections in D-dimensional simplices.
/// A D-dimensional simplex has D+1 vertices, so this handles up to 7D simplices on stack.
///
/// # Use Cases
/// - Simplex vertex operations
/// - Simplex construction
/// - Geometric predicate vertex lists
pub type SimplexVertexBuffer<T> = SmallBuffer<T, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Buffer sized for UUID collections in vertex operations.
/// Optimized for storing vertex UUIDs from a single simplex (D+1 UUIDs).
///
/// # Use Cases
/// - Duplicate simplex detection
/// - Vertex uniqueness checks
/// - Simplex vertex UUID collections
pub type VertexUuidBuffer = SimplexVertexBuffer<Uuid>;

/// Buffer sized for `VertexKey` collections in validation and internal operations.
/// Handles vertex keys from a single D-dimensional simplex.
///
/// # Use Cases
/// - Validation algorithms
/// - Internal vertex key tracking
/// - Simplex vertex key collections
pub type VertexKeyBuffer = SimplexVertexBuffer<VertexKey>;

/// Buffer for storing simplex neighbors (D+1 neighbors for a D-dimensional simplex).
/// Uses stack allocation for typical dimensions (2D-7D).
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: D+1 neighbors fit on stack for D ≤ 7
/// - **Use Case**: Neighbor queries, neighbor assignment, validation
/// - **Performance**: Avoids heap allocation in 90%+ of cases
/// - **Memory Layout**: Better cache locality than heap-allocated Vec
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::NeighborBuffer;
/// use delaunay::prelude::tds::SimplexKey;
///
/// let mut neighbors: NeighborBuffer<Option<SimplexKey>> = NeighborBuffer::new();
/// assert!(neighbors.is_empty());
/// ```
pub type NeighborBuffer<T> = SmallBuffer<T, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Buffer for vertex key collections from a single simplex (D+1 vertices).
/// Avoids heap allocation for typical triangulation dimensions.
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: D+1 vertex keys fit on stack for D ≤ 7
/// - **Use Case**: Simplex vertex storage, validation, geometric operations
/// - **Performance**: Eliminates heap allocation for typical dimensions
/// - **Ordering**: Preserves vertex order for positional semantics
pub type SimplexVertexKeyBuffer = SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Buffer for vertex UUID collections from a single simplex (D+1 vertex UUIDs).
/// Uses stack allocation to avoid heap overhead for simplex operations.
///
/// # Optimization Rationale
///
/// - **Stack Allocation**: D+1 vertex UUIDs fit on stack for D ≤ 7
/// - **Use Case**: Extracting vertex UUIDs from a simplex, validation, duplicate detection
/// - **Performance**: Avoids allocation for temporary UUID collections
/// - **Memory Efficiency**: For D=7, D+1=8 UUIDs → 16 bytes × 8 = 128 bytes on stack
pub type SimplexVertexUuidBuffer = SmallBuffer<Uuid, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Buffer for periodic lattice offsets aligned with a simplex's vertex slots.
///
/// Periodic simplices store one offset per vertex slot so the inline capacity
/// matches [`SimplexVertexKeyBuffer`] and avoids a heap allocation for supported
/// small-to-medium dimensions.
pub type PeriodicOffsetBuffer<const D: usize> = SmallBuffer<[i8; D], MAX_PRACTICAL_DIMENSION_SIZE>;

/// Buffer sized for Point collections in geometric operations.
/// Generic over coordinate type T and dimension D, with practical size limit.
///
/// # Use Cases
/// - Geometric predicate operations
/// - Simplex coordinate collections
/// - Temporary point storage during algorithms
pub type GeometricPointBuffer<T, const D: usize> =
    SmallBuffer<[T; D], MAX_PRACTICAL_DIMENSION_SIZE>;

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    #[test]
    fn test_simplex_vertex_buffer_stack_allocation_boundary() {
        let mut vertex_slots: SlotMap<VertexKey, i32> = SlotMap::default();

        // Test D=7 case: 8 vertices (D+1) should stay on stack
        // MAX_PRACTICAL_DIMENSION_SIZE is 8, so inline capacity is 8
        let mut buffer_d7: SimplexVertexKeyBuffer = SimplexVertexKeyBuffer::new();
        for _ in 0..8 {
            buffer_d7.push(vertex_slots.insert(1));
        }
        assert_eq!(buffer_d7.len(), 8);
        assert!(
            !buffer_d7.spilled(),
            "D=7 (8 vertices) should stay on stack"
        );

        // Test D=8 case: 9 vertices (D+1) should spill to heap
        let mut buffer_d8: SimplexVertexKeyBuffer = SimplexVertexKeyBuffer::new();
        for _ in 0..9 {
            buffer_d8.push(vertex_slots.insert(1));
        }
        assert_eq!(buffer_d8.len(), 9);
        assert!(buffer_d8.spilled(), "D=8 (9 vertices) should spill to heap");

        // Validate the constant MAX_PRACTICAL_DIMENSION_SIZE=8 is correctly sized
        // for practical use cases (D=0 through D=7)
    }
}
