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
/// let tri = dt.as_triangulation();
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

impl AdjacencyIndex {
    /// Returns an iterator over all cells incident to `v`.
    ///
    /// If `v` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
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
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let shared_vertex_key = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    /// assert_eq!(index.adjacent_cells(shared_vertex_key).count(), 2);
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn adjacent_cells(&self, v: VertexKey) -> impl Iterator<Item = CellKey> + '_ {
        self.vertex_to_cells
            .get(&v)
            .into_iter()
            .flat_map(|cells| cells.iter().copied())
    }

    /// Returns the number of cells incident to `v`.
    ///
    /// If `v` is not present in this index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let shared_vertex_key = tri
    /// #     .vertices()
    /// #     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
    /// #     .unwrap();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// assert_eq!(index.number_of_adjacent_cells(shared_vertex_key), 2);
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_adjacent_cells(&self, v: VertexKey) -> usize {
        self.vertex_to_cells.get(&v).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all unique edges incident to `v`.
    ///
    /// If `v` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(index.incident_edges(v0).count(), 3);
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.vertex_to_edges
            .get(&v)
            .into_iter()
            .flat_map(|edges| edges.iter().copied())
    }

    /// Returns the number of unique edges incident to `v`.
    ///
    /// If `v` is not present in this index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(index.number_of_incident_edges(v0), 3);
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.vertex_to_edges.get(&v).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all neighbors of a cell.
    ///
    /// If `c` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let cell_key = tri.cells().next().unwrap().0;
    /// assert_eq!(index.cell_neighbors(cell_key).count(), 1);
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn cell_neighbors(&self, c: CellKey) -> impl Iterator<Item = CellKey> + '_ {
        self.cell_to_neighbors
            .get(&c)
            .into_iter()
            .flat_map(|neighbors| neighbors.iter().copied())
    }

    /// Returns the number of neighbors of a cell.
    ///
    /// If `c` is not present in this index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let cell_key = tri.cells().next().unwrap().0;
    /// assert_eq!(index.number_of_cell_neighbors(cell_key), 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_cell_neighbors(&self, c: CellKey) -> usize {
        self.cell_to_neighbors.get(&c).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all unique edges in the triangulation snapshot.
    ///
    /// This emits each edge exactly once by only yielding it from its canonical `v0()` endpoint.
    ///
    /// Iteration order is not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let edges: std::collections::HashSet<_> = index.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.vertex_to_edges.iter().flat_map(|(vk, edges)| {
            let vk = *vk;
            edges.iter().copied().filter(move |edge| edge.v0() == vk)
        })
    }

    /// Returns the number of unique edges in the triangulation snapshot.
    ///
    /// This is equivalent to `self.edges().count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// assert_eq!(index.number_of_edges(), 6);
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.edges().count()
    }
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

    #[test]
    fn adjacency_index_query_helpers_are_consistent() {
        use crate::core::delaunay_triangulation::DelaunayTriangulation;
        use crate::geometry::kernel::FastKernel;
        use crate::vertex;

        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            // Two apices
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        // Shared vertex is incident to both cells.
        let shared_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();

        assert_eq!(
            index.adjacent_cells(shared_vertex_key).count(),
            index.number_of_adjacent_cells(shared_vertex_key)
        );

        assert_eq!(
            index.incident_edges(shared_vertex_key).count(),
            index.number_of_incident_edges(shared_vertex_key)
        );
        assert!(index.number_of_incident_edges(shared_vertex_key) > 0);

        // Each cell has exactly one neighbor across the shared facet.
        let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
        for &ck in &cell_keys {
            assert_eq!(
                index.cell_neighbors(ck).count(),
                index.number_of_cell_neighbors(ck)
            );
            assert_eq!(index.number_of_cell_neighbors(ck), 1);
        }

        // Global edges view is consistent.
        let edges: std::collections::HashSet<_> = index.edges().collect();
        assert_eq!(edges.len(), index.number_of_edges());
        assert!(edges.iter().all(|e| e.v0() <= e.v1()));

        // Missing keys yield empty iterators / zero counts.
        assert_eq!(index.adjacent_cells(VertexKey::default()).count(), 0);
        assert_eq!(index.number_of_adjacent_cells(VertexKey::default()), 0);
        assert_eq!(index.incident_edges(VertexKey::default()).count(), 0);
        assert_eq!(index.number_of_incident_edges(VertexKey::default()), 0);
        assert_eq!(index.cell_neighbors(CellKey::default()).count(), 0);
        assert_eq!(index.number_of_cell_neighbors(CellKey::default()), 0);
    }
}
