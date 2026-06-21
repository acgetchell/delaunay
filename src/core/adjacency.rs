//! Optional, opt-in topology indexes for fast repeated topology queries.
//!
//! The TDS maintains the canonical vertex→simplices incidence relation internally.
//! [`IncidenceView`] borrows that relation directly, while [`EdgeIndex`] and
//! [`SimplexNeighborIndex`] derive only the maps needed for repeated richer
//! queries (e.g. mesh analysis, FEM assembly, graph algorithms).
//!
//! This index is:
//! - immutable once built
//! - built as a read-only view over the current triangulation snapshot
//! - never stored inside the triangulation (no interior mutability)
//! - lifetime-bound to the borrowed TDS incidence index

#![forbid(unsafe_code)]

use crate::core::collections::{FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::edge::EdgeKey;
use crate::core::tds::incidence::VertexIncidenceIndex;
use crate::core::tds::{SimplexKey, TdsError, VertexKey};
use std::marker::PhantomData;
use thiserror::Error;

/// Errors that can occur while building optional topology indexes.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::query::TopologyIndexBuildError;
/// use delaunay::prelude::tds::{SimplexKey, VertexKey};
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
/// let vertex_key = VertexKey::from(KeyData::from_ffi(2));
/// let err = TopologyIndexBuildError::MissingVertexKey { simplex_key, vertex_key };
/// std::assert_matches!(err, TopologyIndexBuildError::MissingVertexKey { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TopologyIndexBuildError {
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

    /// The maintained vertex-to-simplices incidence relation is inconsistent.
    #[error("Vertex-to-simplices incidence index is inconsistent: {source}")]
    InvalidVertexIncidenceIndex {
        /// Underlying TDS invariant failure.
        #[source]
        source: TdsError,
    },
}

/// Borrowed view over the canonical vertex→simplices incidence relation.
///
/// This view does not derive edge or simplex-neighbor maps. Use it for repeated
/// vertex→simplex incidence queries when the canonical TDS relation is the only
/// data structure needed. Build it with [`Triangulation::incidence`].
///
/// [`Triangulation::incidence`]: crate::Triangulation::incidence
#[derive(Clone, Debug)]
#[non_exhaustive]
#[must_use]
pub struct IncidenceView<'tds> {
    /// Borrowed canonical vertex → incident simplices relation.
    pub(in crate::core) vertex_to_simplices: &'tds VertexIncidenceIndex,
}

/// Derived vertex→edge index for one triangulation snapshot.
///
/// This index owns the edge map it derives from simplex storage and remains
/// lifetime-bound to the triangulation snapshot that produced it. Build it with
/// [`Triangulation::build_edge_index`] when edge queries are needed without a
/// composite [`TriangulationAdjacency`] value.
///
/// [`Triangulation::build_edge_index`]: crate::Triangulation::build_edge_index
#[derive(Clone, Debug)]
#[non_exhaustive]
#[must_use]
pub struct EdgeIndex<'tds> {
    /// Vertex → incident edges.
    pub(in crate::core) vertex_to_edges:
        FastHashMap<VertexKey, SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>>,

    /// Number of unique edges in the triangulation snapshot.
    pub(in crate::core) edge_count: usize,

    /// Ties this derived index to the borrowed source TDS snapshot.
    pub(in crate::core) _tds: PhantomData<&'tds VertexIncidenceIndex>,
}

/// Derived simplex→neighbor index for one triangulation snapshot.
///
/// This index owns the neighbor map it derives from simplex neighbor slots and
/// remains lifetime-bound to the triangulation snapshot that produced it. Build
/// it with [`Triangulation::build_simplex_neighbor_index`] when simplex-neighbor
/// queries are needed without edge indexing.
///
/// [`Triangulation::build_simplex_neighbor_index`]: crate::Triangulation::build_simplex_neighbor_index
#[derive(Clone, Debug)]
#[non_exhaustive]
#[must_use]
pub struct SimplexNeighborIndex<'tds> {
    /// Simplex → neighboring simplices (boundary facets omitted).
    pub(in crate::core) simplex_to_neighbors:
        FastHashMap<SimplexKey, SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,

    /// Ties this derived index to the borrowed source TDS snapshot.
    pub(in crate::core) _tds: PhantomData<&'tds VertexIncidenceIndex>,
}

/// Borrowed adjacency view for one triangulation snapshot.
///
/// The view owns the derived edge and simplex-neighbor indexes, while its
/// [`IncidenceView`] borrows the canonical TDS vertex-to-simplices relation for
/// `'tds`. Holding this value therefore keeps the source triangulation
/// immutably borrowed, so Rust prevents mutation through the same owner while
/// callers are traversing adjacency.
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
/// #     Adjacency(#[from] delaunay::prelude::query::TopologyIndexBuildError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let adjacency = dt.as_triangulation().adjacency()?;
///
/// assert_eq!(adjacency.number_of_edges(), 6);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[non_exhaustive]
#[must_use]
pub struct TriangulationAdjacency<'tds> {
    /// Borrowed canonical vertex → incident simplices relation.
    incidence: IncidenceView<'tds>,
    /// Derived vertex → edge index.
    edges: EdgeIndex<'tds>,
    /// Derived simplex → neighbor index.
    simplex_neighbors: SimplexNeighborIndex<'tds>,
}

impl<'tds> TriangulationAdjacency<'tds> {
    /// Wraps the split topology views as the lifetime-bound composite API.
    #[inline]
    pub(in crate::core) const fn from_parts(
        incidence: IncidenceView<'tds>,
        edges: EdgeIndex<'tds>,
        simplex_neighbors: SimplexNeighborIndex<'tds>,
    ) -> Self {
        Self {
            incidence,
            edges,
            simplex_neighbors,
        }
    }

    /// Returns an iterator over all simplices incident to `v`.
    ///
    /// If `v` is not present in this view, the iterator is empty. Iteration
    /// order is not specified.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.incidence.adjacent_simplices(v)
    }

    /// Returns the number of simplices incident to `v`.
    ///
    /// If `v` is not present in this view, returns 0.
    #[must_use]
    #[inline]
    pub fn number_of_adjacent_simplices(&self, v: VertexKey) -> usize {
        self.incidence.number_of_adjacent_simplices(v)
    }

    /// Returns an iterator over all unique edges incident to `v`.
    ///
    /// If `v` is not present in this view, the iterator is empty. Iteration
    /// order is not specified.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.edges.incident_edges(v)
    }

    /// Returns the number of unique edges incident to `v`.
    ///
    /// If `v` is not present in this view, returns 0.
    #[must_use]
    #[inline]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.edges.number_of_incident_edges(v)
    }

    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// If `c` is not present in this view, the iterator is empty. Boundary
    /// facets are omitted because they have no neighboring simplex.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.simplex_neighbors.simplex_neighbors(c)
    }

    /// Returns the number of neighbors of a simplex.
    ///
    /// If `c` is not present in this view, returns 0.
    #[must_use]
    #[inline]
    pub fn number_of_simplex_neighbors(&self, c: SimplexKey) -> usize {
        self.simplex_neighbors.number_of_simplex_neighbors(c)
    }

    /// Returns an iterator over all unique edges in the triangulation snapshot.
    ///
    /// Iteration order is not specified.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.edges.edges()
    }

    /// Returns the number of unique edges in the triangulation snapshot.
    ///
    /// This is equivalent to `self.edges().count()`, but uses the count
    /// computed while building the immutable index.
    #[must_use]
    #[inline]
    pub const fn number_of_edges(&self) -> usize {
        self.edges.number_of_edges()
    }
}

impl IncidenceView<'_> {
    /// Returns an iterator over all simplices incident to `v`.
    ///
    /// If `v` is not present in this view, the iterator is empty.
    /// Iteration order is not specified.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.vertex_to_simplices.simplex_keys(v)
    }

    /// Returns the number of simplices incident to `v`.
    ///
    /// If `v` is not present in this view, returns 0.
    #[must_use]
    #[inline]
    pub fn number_of_adjacent_simplices(&self, v: VertexKey) -> usize {
        self.vertex_to_simplices.number_of_simplices(v)
    }
}

impl EdgeIndex<'_> {
    /// Returns an iterator over all unique edges incident to `v`.
    ///
    /// If `v` is not present in this index, the iterator is empty.
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
    #[must_use]
    #[inline]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.vertex_to_edges.get(&v).map_or(0, SmallBuffer::len)
    }

    /// Returns an iterator over all unique edges in the triangulation snapshot.
    ///
    /// This emits each edge exactly once by only yielding it from its canonical
    /// `v0()` endpoint. Iteration order is not specified.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.vertex_to_edges.iter().flat_map(|(vk, edges)| {
            let vk = *vk;
            edges.iter().copied().filter(move |edge| edge.v0() == vk)
        })
    }

    /// Returns the number of unique edges in the triangulation snapshot.
    #[must_use]
    pub const fn number_of_edges(&self) -> usize {
        self.edge_count
    }
}

impl SimplexNeighborIndex<'_> {
    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// If `c` is not present in this index, the iterator is empty.
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
    #[must_use]
    #[inline]
    pub fn number_of_simplex_neighbors(&self, c: SimplexKey) -> usize {
        self.simplex_to_neighbors
            .get(&c)
            .map_or(0, SmallBuffer::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::vertex::Vertex;
    use slotmap::SlotMap;
    use std::collections::HashSet;

    #[test]
    fn topology_index_build_error_display_is_informative() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let mut simplices: SlotMap<SimplexKey, ()> = SlotMap::with_key();

        let vk = vertices.insert(());
        let ck = simplices.insert(());

        let err = TopologyIndexBuildError::MissingVertexKey {
            simplex_key: ck,
            vertex_key: vk,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing vertex key"));

        let err = TopologyIndexBuildError::MissingNeighborSimplex {
            simplex_key: ck,
            neighbor_key: ck,
        };
        let msg = err.to_string();
        assert!(msg.contains("references missing neighbor simplex"));
    }

    #[test]
    fn topology_indexes_are_send_sync_unpin() {
        fn assert_auto_traits<T: Send + Sync + Unpin>() {}
        assert_auto_traits::<IncidenceView<'static>>();
        assert_auto_traits::<EdgeIndex<'static>>();
        assert_auto_traits::<SimplexNeighborIndex<'static>>();
        assert_auto_traits::<TriangulationAdjacency<'static>>();
    }

    #[test]
    fn split_topology_query_helpers_are_consistent() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 2.0, 0.0]).unwrap(),
            // Two apices
            Vertex::<(), _>::try_new([1.0, 0.7, 1.5]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.7, -1.5]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();

        // Shared vertex is incident to both simplices.
        let shared_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();

        assert_eq!(
            incidence.adjacent_simplices(shared_vertex_key).count(),
            incidence.number_of_adjacent_simplices(shared_vertex_key)
        );

        assert_eq!(
            edge_index.incident_edges(shared_vertex_key).count(),
            edge_index.number_of_incident_edges(shared_vertex_key)
        );
        assert!(edge_index.number_of_incident_edges(shared_vertex_key) > 0);

        // Each simplex has exactly one neighbor across the shared facet.
        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        for &ck in &simplex_keys {
            assert_eq!(
                neighbor_index.simplex_neighbors(ck).count(),
                neighbor_index.number_of_simplex_neighbors(ck)
            );
            assert_eq!(neighbor_index.number_of_simplex_neighbors(ck), 1);
        }

        // Global edges view is consistent.
        let edges: HashSet<_> = edge_index.edges().collect();
        assert_eq!(edges.len(), edge_index.number_of_edges());
        assert!(edges.iter().all(|e| e.v0() <= e.v1()));

        // Missing keys yield empty iterators / zero counts.
        assert_eq!(
            incidence.adjacent_simplices(VertexKey::default()).count(),
            0
        );
        assert_eq!(
            incidence.number_of_adjacent_simplices(VertexKey::default()),
            0
        );
        assert_eq!(edge_index.incident_edges(VertexKey::default()).count(), 0);
        assert_eq!(edge_index.number_of_incident_edges(VertexKey::default()), 0);
        assert_eq!(
            neighbor_index
                .simplex_neighbors(SimplexKey::default())
                .count(),
            0
        );
        assert_eq!(
            neighbor_index.number_of_simplex_neighbors(SimplexKey::default()),
            0
        );
    }
}
