use super::{
    CellVertexBuffer, CellVertexUuidBuffer, FacetIndex, FastHashMap, FastHashSet,
    MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, SmallBuffer, Uuid, VertexUuidSet,
};
use crate::core::facet::FacetHandle;
use crate::core::triangulation_data_structure::{CellKey, VertexKey};

// =============================================================================
// TRIANGULATION-SPECIFIC OPTIMIZED TYPES
// =============================================================================

/// Facet-to-cells mapping optimized for typical triangulation patterns.
/// Most facets are shared by at most 2 cells (boundary facets = 1, interior facets = 2).
///
/// # Optimization Rationale
///
/// - **Key**: `u64` facet hash (from vertex combination)
/// - **Value**: `SmallBuffer<FacetHandle, 2>` - stack allocated for typical case
/// - **Typical Pattern**: 1 cell (boundary) or 2 cells (interior facet)
/// - **Performance**: Avoids heap allocation for >95% of facets
/// - **Memory Efficiency**: `FacetHandle` uses u8 for facet index, same size as raw tuple
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::FacetToCellsMap;
///
/// let facet_map: FacetToCellsMap = FacetToCellsMap::default();
/// assert!(facet_map.is_empty());
/// ```
pub type FacetToCellsMap = FastHashMap<u64, SmallBuffer<FacetHandle, 2>>;

/// Map of over-shared facets detected during localized validation.
///
/// Used for O(k) facet validation of newly created cells, avoiding O(N) global scans.
/// Maps facet hash to cells sharing that facet (only includes facets shared by > 2 cells).
///
/// # Optimization Rationale
///
/// - **Key**: `u64` facet hash (from sorted vertex keys)
/// - **Value**: `SmallBuffer<(CellKey, FacetIndex), 4>` - handles up to 4 over-sharing cells on stack
/// - **Typical Pattern**: 3-4 cells in most over-sharing cases
/// - **Performance**: Stack allocation for common over-sharing patterns
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::FacetIssuesMap;
///
/// let issues: FacetIssuesMap = FacetIssuesMap::default();
/// assert!(issues.is_empty());
/// ```
pub type FacetIssuesMap = FastHashMap<u64, SmallBuffer<(CellKey, FacetIndex), 4>>;

/// Cell neighbor mapping optimized for typical cell degrees.
/// Most cells have a small number of neighbors (D+1 faces, so at most D+1 neighbors).
///
/// # Optimization Rationale
///
/// - **Key**: `CellKey` identifying the cell
/// - **Value**: `NeighborBuffer<Option<CellKey>>` - handles up to 8 neighbors on stack
/// - **Typical Pattern**: 2D=3 neighbors, 3D=4 neighbors, 4D=5 neighbors
/// - **Performance**: Stack allocation for dimensions up to ~7D
///
/// # Note
///
/// This type mirrors `Cell::neighbors()` which returns `Option<&NeighborBuffer<Option<CellKey>>>`.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::CellNeighborsMap;
///
/// let neighbors: CellNeighborsMap = CellNeighborsMap::default();
/// assert!(neighbors.is_empty());
/// ```
pub type CellNeighborsMap = FastHashMap<CellKey, NeighborBuffer<Option<CellKey>>>;

/// Vertex-to-cells mapping optimized for typical vertex degrees.
/// Most vertices are incident to a small number of cells in well-conditioned triangulations.
///
/// # Optimization Rationale
///
/// - **Key**: `VertexKey` identifying the vertex
/// - **Value**: `SmallBuffer<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>` - handles up to 8 incident cells on stack
/// - **Typical Pattern**: Well-conditioned triangulations have low vertex degrees
/// - **Performance**: Avoids heap allocation for most vertices
/// - **Spill Behavior**: Degrees above `MAX_PRACTICAL_DIMENSION_SIZE` spill to heap. This is expected
///   for high-degree vertices (e.g., higher dimensions or boundary configurations) and is not an error.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::VertexToCellsMap;
///
/// let vertex_cells: VertexToCellsMap = VertexToCellsMap::default();
/// assert!(vertex_cells.is_empty());
/// ```
pub type VertexToCellsMap =
    FastHashMap<VertexKey, SmallBuffer<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>>;

/// Cell vertices mapping optimized for validation operations.
/// Each cell typically has D+1 vertices, stored as a fast set for efficient intersection operations.
///
/// # Optimization Rationale
///
/// - **Key**: `CellKey` identifying the cell
/// - **Value**: `FastHashSet<VertexKey>` - optimized for set operations
/// - **Use Case**: Validation algorithms that need fast intersection/membership testing
/// - **Performance**: `FastHasher` provides fast hashing for `VertexKey`
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::CellVerticesMap;
///
/// let cell_vertices: CellVerticesMap = CellVerticesMap::default();
/// assert!(cell_vertices.is_empty());
/// ```
pub type CellVerticesMap = FastHashMap<CellKey, FastHashSet<VertexKey>>;

/// Cell vertex keys mapping optimized for validation operations requiring positional access.
/// Each cell typically has D+1 vertices, stored in a stack-allocated buffer for efficiency.
///
/// # Optimization Rationale
///
/// - **Key**: `CellKey` identifying the cell
/// - **Value**: `CellVertexBuffer` - stack-allocated for D ≤ 7, preserves vertex order
/// - **Use Case**: Validation algorithms that need positional vertex access (e.g., neighbors\[i\] opposite vertices\[i\])
/// - **Performance**: Eliminates heap allocation for typical dimensions, better cache locality
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::CellVertexKeysMap;
///
/// let cell_vertex_keys: CellVertexKeysMap = CellVertexKeysMap::default();
/// assert!(cell_vertex_keys.is_empty());
/// ```
pub type CellVertexKeysMap = FastHashMap<CellKey, CellVertexBuffer>;

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

/// Mapping from cell UUIDs to their vertex UUIDs (optimized for internal operations).
/// Uses stack-allocated buffers for vertex UUID storage.
///
/// # Optimization Rationale
///
/// - **Key**: Cell UUID for stable identification
/// - **Value**: `CellVertexUuidBuffer` for stack-allocated vertex UUID storage (D+1 UUIDs)
/// - **Use Case**: Internal operations, temporary mappings, validation
/// - **Performance**: Stack allocation for typical cell vertex counts, avoids heap for D ≤ 7
///
/// # Serialization Note
///
/// For serialization/deserialization, use `FastHashMap<Uuid, Vec<Uuid>>` instead,
/// as serde doesn't natively serialize `SmallVec`. Convert using `.to_vec()` when serializing.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::collections::CellToVertexUuidsMap;
///
/// let mapping: CellToVertexUuidsMap = CellToVertexUuidsMap::default();
/// assert!(mapping.is_empty());
/// ```
pub type CellToVertexUuidsMap = FastHashMap<Uuid, CellVertexUuidBuffer>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::{CellKey, VertexKey};
    use slotmap::SlotMap;

    #[test]
    fn test_triangulation_map_type_instantiation() {
        // Test domain-specific UUID-based types compile and instantiate
        let _facet_map: FacetToCellsMap = FacetToCellsMap::default();
        let _neighbors: CellNeighborsMap = CellNeighborsMap::default();
        let _vertex_cells: VertexToCellsMap = VertexToCellsMap::default();
        let _cell_vertices: CellVerticesMap = CellVerticesMap::default();

        // Test CellVertexKeysMap with SmallBuffer for D+1 usage pattern
        let mut cell_vertex_keys: CellVertexKeysMap = CellVertexKeysMap::default();
        let mut cell_slots: SlotMap<CellKey, i32> = SlotMap::default();
        let mut vertex_slots: SlotMap<VertexKey, i32> = SlotMap::default();

        let cell_key = cell_slots.insert(1);
        let mut vertex_buffer: crate::core::collections::CellVertexBuffer =
            crate::core::collections::CellVertexBuffer::new();
        // Simulate D+1 vertices for a 2D cell (3 vertices)
        for _ in 0..3 {
            vertex_buffer.push(vertex_slots.insert(1));
        }
        assert!(!vertex_buffer.spilled()); // Should be on stack for D=2
        cell_vertex_keys.insert(cell_key, vertex_buffer);
        assert_eq!(cell_vertex_keys.len(), 1);
    }
}
