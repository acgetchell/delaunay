//! Optional, opt-in adjacency index for fast topology queries.
//!
//! The triangulation does **not** store a persistent adjacency cache internally.
//! Instead, downstream code can build an `AdjacencyIndex` on demand for repeated
//! queries (e.g. mesh analysis, FEM assembly, graph algorithms).
//!
//! This index is:
//! - immutable once built
//! - built from the current triangulation snapshot
//! - never stored inside the triangulation (no interior mutability)

use crate::core::collections::{
    FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToCellsMap,
};
use crate::core::edge::EdgeKey;
use crate::core::triangulation_data_structure::{CellKey, VertexKey};
use thiserror::Error;

/// Errors that can occur while building an [`AdjacencyIndex`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdjacencyIndexBuildError {
    /// A cell references a vertex key that does not exist in vertex storage.
    #[error("Cell {cell_key:?} references missing vertex key {vertex_key:?}")]
    MissingVertexKey {
        /// The cell that referenced the missing vertex.
        cell_key: CellKey,
        /// The missing vertex key.
        vertex_key: VertexKey,
    },

    /// A cell references a neighbor key that does not exist in cell storage.
    #[error("Cell {cell_key:?} references missing neighbor cell {neighbor_key:?}")]
    MissingNeighborCell {
        /// The cell that referenced the missing neighbor.
        cell_key: CellKey,
        /// The missing neighbor cell key.
        neighbor_key: CellKey,
    },
}

/// Immutable adjacency maps for topology traversal.
///
/// This is an *opt-in* structure returned by
/// [`Triangulation::build_adjacency_index`](crate::core::triangulation::Triangulation::build_adjacency_index).
///
/// ## Notes
/// - No sorted-order guarantees are provided for the values.
/// - The collections are optimized for performance (FxHasher-backed).
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// // Two tetrahedra sharing a triangular facet.
/// let vertices: Vec<_> = vec![
///     // Shared triangle
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([2.0, 0.0, 0.0]),
///     vertex!([1.0, 2.0, 0.0]),
///     // Two apices
///     vertex!([1.0, 0.7, 1.5]),
///     vertex!([1.0, 0.7, -1.5]),
/// ];
///
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let tri = dt.triangulation();
///
/// let index = tri.build_adjacency_index().unwrap();
///
/// // Query incident cells for some vertex key from the triangulation.
/// let vk = tri.vertices().next().unwrap().0;
/// let incident_cells = index.vertex_to_cells.get(&vk).unwrap();
/// assert!(!incident_cells.is_empty());
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AdjacencyIndex {
    /// Vertex → incident edges.
    pub vertex_to_edges: FastHashMap<VertexKey, SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>>,

    /// Vertex → incident cells.
    pub vertex_to_cells: VertexToCellsMap,

    /// Cell → neighboring cells (boundary facets omitted).
    pub cell_to_neighbors: FastHashMap<CellKey, SmallBuffer<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    #[test]
    fn adjacency_index_build_error_display_is_informative() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let mut cells: SlotMap<CellKey, ()> = SlotMap::with_key();

        let vk = vertices.insert(());
        let ck = cells.insert(());

        let err = AdjacencyIndexBuildError::MissingVertexKey {
            cell_key: ck,
            vertex_key: vk,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing vertex key"));

        let err = AdjacencyIndexBuildError::MissingNeighborCell {
            cell_key: ck,
            neighbor_key: ck,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing neighbor cell"));
    }

    #[test]
    fn adjacency_index_is_send_sync_unpin() {
        fn assert_auto_traits<T: Send + Sync + Unpin>() {}
        assert_auto_traits::<AdjacencyIndex>();
    }
}
