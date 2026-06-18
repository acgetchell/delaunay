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

#![forbid(unsafe_code)]

use crate::core::collections::{
    FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToSimplicesMap,
};
use crate::core::edge::EdgeKey;
use crate::core::tds::{SimplexKey, VertexKey};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur while building an [`AdjacencyIndex`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::query::AdjacencyIndexBuildError;
/// use delaunay::prelude::tds::{SimplexKey, VertexKey};
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
/// let vertex_key = VertexKey::from(KeyData::from_ffi(2));
/// let err = AdjacencyIndexBuildError::MissingVertexKey { simplex_key, vertex_key };
/// std::assert_matches!(err, AdjacencyIndexBuildError::MissingVertexKey { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdjacencyIndexBuildError {
    /// A simplex references a vertex key that does not exist in vertex storage.
    #[error("Simplex {simplex_key:?} references missing vertex key {vertex_key:?}")]
    MissingVertexKey {
        /// The simplex that referenced the missing vertex.
        simplex_key: SimplexKey,
        /// The missing vertex key.
        vertex_key: VertexKey,
    },

    /// A simplex references a neighbor key that does not exist in simplex storage.
    #[error("Simplex {simplex_key:?} references missing neighbor simplex {neighbor_key:?}")]
    MissingNeighborSimplex {
        /// The simplex that referenced the missing neighbor.
        simplex_key: SimplexKey,
        /// The missing neighbor simplex key.
        neighbor_key: SimplexKey,
    },
}

/// Immutable adjacency maps for topology traversal.
///
/// This is an *opt-in* structure returned by
/// [`Triangulation::build_adjacency_index`](crate::prelude::triangulation::Triangulation::build_adjacency_index).
///
/// ## Notes
/// - No sorted-order guarantees are provided for the values.
/// - The collections are optimized for performance (FxHasher-backed).
/// - An index is tied to the triangulation snapshot that built it. Rebuild it
///   after mutating the triangulation, and do not share it across
///   triangulations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Two tetrahedra sharing a triangular facet.
/// let vertices: Vec<_> = vec![
///     // Shared triangle
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![2.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 2.0, 0.0]?,
///     // Two apices
///     delaunay::vertex![1.0, 0.7, 1.5]?,
///     delaunay::vertex![1.0, 0.7, -1.5]?,
/// ];
///
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let tri = dt.as_triangulation();
///
/// let index = tri.build_adjacency_index()?;
///
/// // Query incident simplices for some vertex key from the triangulation.
/// let Some((vk, _)) = tri.vertices().next() else { return Ok(()); };
/// assert!(index.number_of_adjacent_simplices(vk) > 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AdjacencyIndex {
    /// Runtime identity of the TDS snapshot this index was built from.
    pub(crate) tds_identity: Arc<Uuid>,

    /// Generation of the TDS snapshot this index was built from.
    pub(crate) tds_generation: u64,

    /// Vertex → incident edges.
    pub(crate) vertex_to_edges:
        FastHashMap<VertexKey, SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>>,

    /// Vertex → incident simplices.
    pub(crate) vertex_to_simplices: VertexToSimplicesMap,

    /// Simplex → neighboring simplices (boundary facets omitted).
    pub(crate) simplex_to_neighbors:
        FastHashMap<SimplexKey, SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
}

impl AdjacencyIndex {
    /// Returns an iterator over all simplices incident to `v`.
    ///
    /// If `v` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Two tetrahedra sharing a triangular facet.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![2.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 2.0, 0.0]?,
    ///     // Two apices
    ///     delaunay::vertex![1.0, 0.7, 1.5]?,
    ///     delaunay::vertex![1.0, 0.7, -1.5]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// let Some(shared_vertex_key) = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
    /// else {
    ///     return Ok(());
    /// };
    ///
    /// let index = tri.build_adjacency_index()?;
    /// assert_eq!(index.adjacent_simplices(shared_vertex_key).count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.vertex_to_simplices
            .get(&v)
            .into_iter()
            .flat_map(|simplices| simplices.iter().copied())
    }

    /// Returns the number of simplices incident to `v`.
    ///
    /// If `v` is not present in this index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![2.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 2.0, 0.0]?,
    /// #     // Two apices
    /// #     delaunay::vertex![1.0, 0.7, 1.5]?,
    /// #     delaunay::vertex![1.0, 0.7, -1.5]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let Some(shared_vertex_key) = tri
    /// #     .vertices()
    /// #     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
    /// # else {
    /// #     return Ok(());
    /// # };
    /// # let index = tri.build_adjacency_index()?;
    /// assert_eq!(index.number_of_adjacent_simplices(shared_vertex_key), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_adjacent_simplices(&self, v: VertexKey) -> usize {
        self.vertex_to_simplices.get(&v).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all unique edges incident to `v`.
    ///
    /// If `v` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 1.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((v0, _)) = tri.vertices().next() else { return Ok(()); };
    /// assert_eq!(index.incident_edges(v0).count(), 3);
    /// # Ok(())
    /// # }
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
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 1.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((v0, _)) = tri.vertices().next() else { return Ok(()); };
    /// assert_eq!(index.number_of_incident_edges(v0), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.vertex_to_edges.get(&v).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// If `c` is not present in this index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![2.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 2.0, 0.0]?,
    /// #     // Two apices
    /// #     delaunay::vertex![1.0, 0.7, 1.5]?,
    /// #     delaunay::vertex![1.0, 0.7, -1.5]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((simplex_key, _)) = tri.simplices().next() else { return Ok(()); };
    /// assert_eq!(index.simplex_neighbors(simplex_key).count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.simplex_to_neighbors
            .get(&c)
            .into_iter()
            .flat_map(|neighbors| neighbors.iter().copied())
    }

    /// Returns the number of neighbors of a simplex.
    ///
    /// If `c` is not present in this index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![2.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 2.0, 0.0]?,
    /// #     // Two apices
    /// #     delaunay::vertex![1.0, 0.7, 1.5]?,
    /// #     delaunay::vertex![1.0, 0.7, -1.5]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((simplex_key, _)) = tri.simplices().next() else { return Ok(()); };
    /// assert_eq!(index.number_of_simplex_neighbors(simplex_key), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_simplex_neighbors(&self, c: SimplexKey) -> usize {
        self.simplex_to_neighbors
            .get(&c)
            .map_or(0, SmallBuffer::len)
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
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 1.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let edges: std::collections::HashSet<_> = index.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// # Ok(())
    /// # }
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
    /// # use delaunay::prelude::*;
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::prelude::query::AdjacencyIndexBuildError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![1.0, 0.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 1.0, 0.0]?,
    /// #     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// assert_eq!(index.number_of_edges(), 6);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.edges().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use slotmap::SlotMap;
    use std::collections::HashSet;

    #[test]
    fn adjacency_index_build_error_display_is_informative() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let mut simplices: SlotMap<SimplexKey, ()> = SlotMap::with_key();

        let vk = vertices.insert(());
        let ck = simplices.insert(());

        let err = AdjacencyIndexBuildError::MissingVertexKey {
            simplex_key: ck,
            vertex_key: vk,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing vertex key"));

        let err = AdjacencyIndexBuildError::MissingNeighborSimplex {
            simplex_key: ck,
            neighbor_key: ck,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing neighbor simplex"));
    }

    #[test]
    fn adjacency_index_is_send_sync_unpin() {
        fn assert_auto_traits<T: Send + Sync + Unpin>() {}
        assert_auto_traits::<AdjacencyIndex>();
    }

    #[test]
    fn adjacency_index_query_helpers_are_consistent() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 0.0]).unwrap(),
            // Two apices
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, 1.5]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, -1.5]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        // Shared vertex is incident to both simplices.
        let shared_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();

        assert_eq!(
            index.adjacent_simplices(shared_vertex_key).count(),
            index.number_of_adjacent_simplices(shared_vertex_key)
        );

        assert_eq!(
            index.incident_edges(shared_vertex_key).count(),
            index.number_of_incident_edges(shared_vertex_key)
        );
        assert!(index.number_of_incident_edges(shared_vertex_key) > 0);

        // Each simplex has exactly one neighbor across the shared facet.
        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        for &ck in &simplex_keys {
            assert_eq!(
                index.simplex_neighbors(ck).count(),
                index.number_of_simplex_neighbors(ck)
            );
            assert_eq!(index.number_of_simplex_neighbors(ck), 1);
        }

        // Global edges view is consistent.
        let edges: HashSet<_> = index.edges().collect();
        assert_eq!(edges.len(), index.number_of_edges());
        assert!(edges.iter().all(|e| e.v0() <= e.v1()));

        // Missing keys yield empty iterators / zero counts.
        assert_eq!(index.adjacent_simplices(VertexKey::default()).count(), 0);
        assert_eq!(index.number_of_adjacent_simplices(VertexKey::default()), 0);
        assert_eq!(index.incident_edges(VertexKey::default()).count(), 0);
        assert_eq!(index.number_of_incident_edges(VertexKey::default()), 0);
        assert_eq!(index.simplex_neighbors(SimplexKey::default()).count(), 0);
        assert_eq!(index.number_of_simplex_neighbors(SimplexKey::default()), 0);
    }
}
