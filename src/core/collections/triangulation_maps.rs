//! Triangulation-specific map aliases for adjacency and incidence bookkeeping.
//!
//! These aliases encode the small-buffer capacities expected by common topology
//! relationships such as facets, neighbors, simplices, and vertex incidence.

use super::{
    FacetIndex, FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer,
    SimplexVertexKeyBuffer, SimplexVertexUuidBuffer, SmallBuffer, Uuid, VertexUuidSet,
};
use crate::core::facet::FacetHandle;
use crate::core::tds::{SimplexKey, VertexKey};

// =============================================================================
// TRIANGULATION-SPECIFIC OPTIMIZED TYPES
// =============================================================================

/// Internal facet-to-simplices mapping optimized for typical triangulation patterns.
///
/// Public APIs expose [`FacetToSimplicesIndex`](crate::prelude::tds::FacetToSimplicesIndex)
/// instead, tying this derived map to the [`Tds`](crate::prelude::tds::Tds) that produced it.
/// Most facets are incident to 1 or 2 simplices (one-sided or two-sided incidence).
///
/// # Optimization Rationale
///
/// - **Key**: `u64` facet hash (from vertex combination)
/// - **Value**: `SmallBuffer<FacetHandle, 2>` - stack allocated for typical case
/// - **Typical Pattern**: 1 simplex (one-sided) or 2 simplices (two-sided)
/// - **Performance**: Avoids heap allocation for >95% of facets
/// - **Memory Efficiency**: `FacetHandle` uses u8 for facet index, same size as raw tuple
pub(crate) type FacetToSimplicesMap = FastHashMap<u64, SmallBuffer<FacetHandle, 2>>;

/// Map of over-shared facets detected during localized validation.
///
/// Used for O(k) facet validation of newly created simplices, avoiding O(N) global scans.
/// Maps facet hash to simplices sharing that facet (only includes facets shared by > 2 simplices).
///
/// # Optimization Rationale
///
/// - **Key**: `u64` facet hash (from sorted vertex keys)
/// - **Value**: `SmallBuffer<(SimplexKey, FacetIndex), 4>` - handles up to 4 over-sharing simplices on stack
/// - **Typical Pattern**: 3-4 simplices in most over-sharing cases
/// - **Performance**: Stack allocation for common over-sharing patterns
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::FacetIssuesMap;
///
/// let issues: FacetIssuesMap = FacetIssuesMap::default();
/// assert!(issues.is_empty());
/// ```
pub type FacetIssuesMap = FastHashMap<u64, SmallBuffer<(SimplexKey, FacetIndex), 4>>;

/// Simplex neighbor mapping optimized for typical simplex degrees.
/// Most simplices have a small number of neighbors (D+1 faces, so at most D+1 neighbors).
///
/// # Optimization Rationale
///
/// - **Key**: `SimplexKey` identifying the simplex
/// - **Value**: `NeighborBuffer<Option<SimplexKey>>` - handles up to 8 neighbors on stack
/// - **Typical Pattern**: 2D=3 neighbors, 3D=4 neighbors, 4D=5 neighbors
/// - **Performance**: Stack allocation for dimensions up to ~7D
///
/// # Note
///
/// This type mirrors `Simplex::neighbors()` which returns `Option<&NeighborBuffer<Option<SimplexKey>>>`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SimplexNeighborsMap;
///
/// let neighbors: SimplexNeighborsMap = SimplexNeighborsMap::default();
/// assert!(neighbors.is_empty());
/// ```
pub type SimplexNeighborsMap = FastHashMap<SimplexKey, NeighborBuffer<Option<SimplexKey>>>;

/// Vertex-to-simplices mapping optimized for typical vertex degrees.
/// Most vertices are incident to a small number of simplices in well-conditioned triangulations.
///
/// # Optimization Rationale
///
/// - **Key**: `VertexKey` identifying the vertex
/// - **Value**: `SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>` - handles up to 8 incident simplices on stack
/// - **Typical Pattern**: Well-conditioned triangulations have low vertex degrees
/// - **Performance**: Avoids heap allocation for most vertices
/// - **Spill Behavior**: Degrees above `MAX_PRACTICAL_DIMENSION_SIZE` spill to heap. This is expected
///   for high-degree vertices (e.g., higher dimensions or boundary configurations) and is not an error.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::VertexToSimplicesMap;
///
/// let vertex_simplices: VertexToSimplicesMap = VertexToSimplicesMap::default();
/// assert!(vertex_simplices.is_empty());
/// ```
pub type VertexToSimplicesMap =
    FastHashMap<VertexKey, SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>>;

/// Simplex vertices mapping optimized for validation operations.
/// Each simplex typically has D+1 vertices, stored as a fast set for efficient intersection operations.
///
/// # Optimization Rationale
///
/// - **Key**: `SimplexKey` identifying the simplex
/// - **Value**: `FastHashSet<VertexKey>` - optimized for set operations
/// - **Use Case**: Validation algorithms that need fast intersection/membership testing
/// - **Performance**: `FastHasher` provides fast hashing for `VertexKey`
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SimplexVerticesMap;
///
/// let simplex_vertices: SimplexVerticesMap = SimplexVerticesMap::default();
/// assert!(simplex_vertices.is_empty());
/// ```
pub type SimplexVerticesMap = FastHashMap<SimplexKey, FastHashSet<VertexKey>>;

/// Simplex vertex keys mapping optimized for validation operations requiring positional access.
/// Each simplex typically has D+1 vertices, stored in a stack-allocated buffer for efficiency.
///
/// # Optimization Rationale
///
/// - **Key**: `SimplexKey` identifying the simplex
/// - **Value**: `SimplexVertexKeyBuffer` - stack-allocated for D ≤ 7, preserves vertex order
/// - **Use Case**: Validation algorithms that need positional vertex access (e.g., neighbors\[i\] opposite vertices\[i\])
/// - **Performance**: Eliminates heap allocation for typical dimensions, better cache locality
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SimplexVertexKeysMap;
///
/// let simplex_vertex_keys: SimplexVertexKeysMap = SimplexVertexKeysMap::default();
/// assert!(simplex_vertex_keys.is_empty());
/// ```
pub type SimplexVertexKeysMap = FastHashMap<SimplexKey, SimplexVertexKeyBuffer>;

/// Mapping from facet keys to vertex sets for hull algorithms.
/// Used in convex hull and Voronoi diagram construction.
///
/// # Optimization Rationale
///
/// - **Key**: `u64` facet hash for O(1) lookup
/// - **Value**: `VertexUuidSet` for fast set operations
/// - **Use Case**: Hull algorithms, visibility determination
/// - **Performance**: Optimized for geometric algorithm patterns
pub type FacetVertexMap = FastHashMap<u64, VertexUuidSet>;

/// Mapping from simplex UUIDs to their vertex UUIDs (optimized for internal operations).
/// Uses stack-allocated buffers for vertex UUID storage.
///
/// # Optimization Rationale
///
/// - **Key**: Simplex UUID for stable identification
/// - **Value**: `SimplexVertexUuidBuffer` for stack-allocated vertex UUID storage (D+1 UUIDs)
/// - **Use Case**: Internal operations, temporary mappings, validation
/// - **Performance**: Stack allocation for typical simplex vertex counts, avoids heap for D ≤ 7
///
/// # Serialization Note
///
/// For serialization/deserialization, use `FastHashMap<Uuid, Vec<Uuid>>` instead,
/// as serde doesn't natively serialize `SmallVec`. Convert using `.to_vec()` when serializing.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::collections::SimplexToVertexUuidsMap;
///
/// let mapping: SimplexToVertexUuidsMap = SimplexToVertexUuidsMap::default();
/// assert!(mapping.is_empty());
/// ```
pub type SimplexToVertexUuidsMap = FastHashMap<Uuid, SimplexVertexUuidBuffer>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::{SimplexKey, VertexKey};
    use slotmap::SlotMap;

    #[test]
    fn test_triangulation_map_type_instantiation() {
        // Test domain-specific UUID-based types compile and instantiate
        let _facet_map: FacetToSimplicesMap = FacetToSimplicesMap::default();
        let _neighbors: SimplexNeighborsMap = SimplexNeighborsMap::default();
        let _vertex_simplices: VertexToSimplicesMap = VertexToSimplicesMap::default();
        let _simplex_vertices: SimplexVerticesMap = SimplexVerticesMap::default();

        // Test SimplexVertexKeysMap with SmallBuffer for D+1 usage pattern
        let mut simplex_vertex_keys: SimplexVertexKeysMap = SimplexVertexKeysMap::default();
        let mut simplex_slots: SlotMap<SimplexKey, i32> = SlotMap::default();
        let mut vertex_slots: SlotMap<VertexKey, i32> = SlotMap::default();

        let simplex_key = simplex_slots.insert(1);
        let mut vertex_buffer: crate::core::collections::SimplexVertexKeyBuffer =
            crate::core::collections::SimplexVertexKeyBuffer::new();
        // Simulate D+1 vertices for a 2D simplex (3 vertices)
        for _ in 0..3 {
            vertex_buffer.push(vertex_slots.insert(1));
        }
        assert!(!vertex_buffer.spilled()); // Should be on stack for D=2
        simplex_vertex_keys.insert(simplex_key, vertex_buffer);
        assert_eq!(simplex_vertex_keys.len(), 1);
    }
}
