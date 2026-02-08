use crate::core::triangulation_data_structure::{CellKey, VertexKey};
use slotmap::SparseSecondaryMap;

// =============================================================================
// SLOTMAP SECONDARY MAPS FOR AUXILIARY DATA
// =============================================================================

/// Sparse secondary map for tracking auxiliary data associated with cells.
///
/// This is the idiomatic way to associate temporary data with `SlotMap` keys during algorithms.
/// Only stores entries for cells that have associated data (sparse representation).
///
/// # Performance Benefits
///
/// - **Memory efficient**: Only allocates for cells with data (vs dense array)
/// - **Type safe**: Can only use valid `CellKey` from the primary `SlotMap`
/// - **Cache friendly**: Better locality than separate `HashMap<CellKey, V>`
/// - **`SlotMap` integration**: Designed specifically for this use case
///
/// # Use Cases
///
/// - **Phase 3 algorithms**: Conflict region finding, cavity extraction
/// - **Algorithm state**: Marking cells as "visited", "in conflict", "processed"
/// - **Temporary data**: Associating algorithm-specific data with cells
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// use delaunay::core::collections::CellSecondaryMap;
/// let mut in_conflict: CellSecondaryMap<bool> = CellSecondaryMap::new();
/// for (cell_key, _) in tds.cells() {
///     in_conflict.insert(cell_key, true);
/// }
/// ```
pub type CellSecondaryMap<V> = SparseSecondaryMap<CellKey, V>;

/// Sparse secondary map for tracking auxiliary data associated with vertices.
///
/// This is the idiomatic way to associate temporary data with `SlotMap` keys during algorithms.
/// Only stores entries for vertices that have associated data (sparse representation).
///
/// # Performance Benefits
///
/// - **Memory efficient**: Only allocates for vertices with data (vs dense array)
/// - **Type safe**: Can only use valid `VertexKey` from the primary `SlotMap`
/// - **Cache friendly**: Better locality than separate `HashMap<VertexKey, V>`
/// - **`SlotMap` integration**: Designed specifically for this use case
///
/// # Use Cases
///
/// - **Algorithm state**: Marking vertices as "visited", "processed", "boundary"
/// - **Distance tracking**: Shortest path, geodesic distance computations
/// - **Temporary data**: Associating algorithm-specific data with vertices
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// use delaunay::core::collections::VertexSecondaryMap;
/// let mut processing_order: VertexSecondaryMap<usize> = VertexSecondaryMap::new();
/// for (idx, (vertex_key, _)) in tds.vertices().enumerate() {
///     processing_order.insert(vertex_key, idx);
/// }
/// ```
pub type VertexSecondaryMap<V> = SparseSecondaryMap<VertexKey, V>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secondary_maps_compile_and_instantiate() {
        let _cell_aux: CellSecondaryMap<bool> = CellSecondaryMap::new();
        let _vertex_aux: VertexSecondaryMap<usize> = VertexSecondaryMap::new();
    }
}
