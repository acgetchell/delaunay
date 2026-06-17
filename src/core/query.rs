//! Read-only queries for generic [`Triangulation`](crate::Triangulation).
//!
//! This module owns zero-mutation accessors and topology traversal helpers for
//! the generic triangulation layer. Mutation APIs stay with the construction and
//! editing modules; validation orchestration stays in [`crate::prelude::validation`].

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::collections::{
    FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToSimplicesMap,
    fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, TdsError, VertexKey};
use crate::core::triangulation::Triangulation;
use crate::core::vertex::Vertex;
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(debug_assertions)]
static VERTEX_TO_SIMPLICES_SPILL_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Errors returned by read-only triangulation queries.
///
/// These errors indicate that a read-only query could not safely use the
/// requested topology view. This can happen because the underlying
/// triangulation state is inconsistent, or because a precomputed
/// [`AdjacencyIndex`] no longer matches the triangulation snapshot being
/// queried.
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
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let boundary_count = dt
///     .as_triangulation()
///     .boundary_facets()?
///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
/// assert_eq!(boundary_count, 4);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum QueryError {
    /// The triangulation could not build a facet map for a read-only query.
    #[error("Triangulation data structure is corrupted: {source}")]
    TriangulationCorrupted {
        /// Typed TDS validation or bookkeeping error that prevented the query.
        #[from]
        source: TdsError,
    },
    /// A precomputed adjacency index was built from another triangulation snapshot.
    ///
    /// Indexed query helpers reject both cross-triangulation indexes and stale
    /// same-triangulation indexes after mutation. Rebuild the
    /// [`AdjacencyIndex`] from the triangulation snapshot being queried.
    #[error(
        "AdjacencyIndex snapshot mismatch: identity_matches={identity_matches}, expected generation {expected_generation}, got {actual_generation}"
    )]
    AdjacencyIndexSnapshotMismatch {
        /// Whether the index came from the same TDS runtime identity.
        identity_matches: bool,
        /// Current TDS generation.
        expected_generation: u64,
        /// Generation captured when the index was built.
        actual_generation: u64,
    },
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
    /// Returns an iterator over all simplices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over simplices
    /// for (_simplex_key, simplex) in tri.simplices() {
    ///     assert_eq!(simplex.number_of_vertices(), 3); // 2D triangle
    /// }
    /// assert_eq!(tri.simplices().count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<V, D>)> {
        self.tds.simplices()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over vertices
    /// for (_vertex_key, vertex) in tri.vertices() {
    ///     assert_eq!(vertex.dim(), 2); // 2D vertices
    /// }
    /// assert_eq!(tri.vertices().count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<U, D>)> {
        self.tds.vertices()
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.as_triangulation().number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tds.number_of_vertices()
    }

    /// Returns the number of simplices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.as_triangulation().number_of_simplices(), 1); // Single tetrahedron
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.tds.number_of_simplices()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::*;
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Empty triangulation has dimension -1
    /// let empty: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(empty.dim(), -1);
    ///
    /// // 3D tetrahedron has dimension 3
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.as_triangulation().dim(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tds.dim()
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// This provides efficient access to all facets without pre-allocating a vector.
    /// Each successful facet is a lightweight `FacetView` that references the
    /// underlying triangulation data.
    ///
    /// # Returns
    ///
    /// An iterator yielding `Result<FacetView, FacetError>` items for all facets
    /// in the triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::TriangulationCorrupted`] if the facet iterator cannot
    /// represent facet indices for this dimension. Individual iterator items
    /// return [`FacetError`](crate::prelude::tds::FacetError) if a facet view
    /// cannot be constructed from the current TDS state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Iterate over all facets
    /// let facet_count = dt
    ///     .as_triangulation()
    ///     .facets()?
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// # Ok(())
    /// # }
    /// ```
    pub fn facets(&self) -> Result<AllFacetsIter<'_, U, V, D>, QueryError> {
        self.tds
            .facets()
            .map_err(|source| QueryError::TriangulationCorrupted { source })
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one simplex. This method
    /// computes the facet-to-simplices map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `Result<FacetView, FacetError>` items for boundary
    /// facets only.
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
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let boundary_count = dt
    ///     .as_triangulation()
    ///     .boundary_facets()?
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::TriangulationCorrupted`] if facet-map construction
    /// detects invalid simplex or facet bookkeeping. The variant preserves the
    /// underlying [`TdsError`] so callers can inspect the structural failure.
    /// Individual iterator items return [`FacetError`](crate::prelude::tds::FacetError)
    /// if a boundary facet cannot be created or keyed from the simplices.
    pub fn boundary_facets(&self) -> Result<BoundaryFacetsIter<'_, U, V, D>, QueryError> {
        let facet_map = self
            .tds
            .build_facet_to_simplices_map()
            .map_err(|source| QueryError::TriangulationCorrupted { source })?;
        BoundaryFacetsIter::try_new(&self.tds, facet_map)
            .map_err(TdsError::from)
            .map_err(|source| QueryError::TriangulationCorrupted { source })
    }

    /// Verifies that a caller-supplied [`AdjacencyIndex`] belongs to this exact
    /// TDS snapshot before an indexed query reads from it.
    ///
    /// The identity check catches cross-triangulation reuse, while the generation
    /// check catches stale same-triangulation indexes after mutation.
    #[inline]
    fn validate_adjacency_index_matches(&self, index: &AdjacencyIndex) -> Result<(), QueryError> {
        let identity_matches = Arc::ptr_eq(&index.tds_identity, self.tds.identity());
        let expected_generation = self.tds.generation();
        if identity_matches && index.tds_generation == expected_generation {
            return Ok(());
        }

        Err(QueryError::AdjacencyIndexSnapshotMismatch {
            identity_matches,
            expected_generation,
            actual_generation: index.tds_generation,
        })
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// Edges are inferred from the vertex lists of each simplex; they are not stored explicitly.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method allocates an internal set to deduplicate edges. The iteration order is
    /// not specified.
    ///
    /// If you need fast repeated topology queries, consider building an
    /// [`AdjacencyIndex`] once via [`Triangulation::build_adjacency_index`](Self::build_adjacency_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// let edges: std::collections::HashSet<_> = tri.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_edges().into_iter()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index()?;
    /// let edges: std::collections::HashSet<_> = tri
    ///     .edges_with_index(&index)?
    ///     .collect();
    /// assert_eq!(edges.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> Result<impl Iterator<Item = EdgeKey> + 'a, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.edges())
    }

    /// Returns the number of unique edges in the triangulation.
    ///
    /// This is equivalent to `self.edges().count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // A single 2D triangle has 3 unique edges.
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tri = dt.as_triangulation();
    ///
    /// assert_eq!(tri.number_of_edges(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.collect_edges().len()
    }

    /// Returns the number of unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This is equivalent to `self.edges_with_index(index).count()`.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// assert_eq!(
    ///     tri.number_of_edges_with_index(&index)?,
    ///     6
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_edges_with_index(&self, index: &AdjacencyIndex) -> Result<usize, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.number_of_edges())
    }

    /// Returns an iterator over all simplices adjacent (incident) to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// Iteration order is not specified.
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.tds
            .find_simplices_containing_vertex_by_key(v)
            .into_iter()
    }

    /// Returns an iterator over all simplices adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.adjacent_simplices_with_index(&index, vertex_key)?.count(),
    ///     tri.adjacent_simplices(vertex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn adjacent_simplices_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> Result<impl Iterator<Item = SimplexKey> + 'a, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.adjacent_simplices(v))
    }

    /// Returns the number of simplices adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.number_of_adjacent_simplices_with_index(&index, vertex_key)?,
    ///     tri.adjacent_simplices(vertex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_adjacent_simplices_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> Result<usize, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.number_of_adjacent_simplices(v))
    }

    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// Iteration order is not specified.
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.tds
            .simplex(c)
            .and_then(Simplex::neighbors)
            .into_iter()
            .flat_map(IntoIterator::into_iter)
            .flatten()
            .filter(|&neighbor_key| self.tds.contains_simplex(neighbor_key))
    }

    /// Returns an iterator over all neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.simplex_neighbors_with_index(&index, simplex_key)?.count(),
    ///     tri.simplex_neighbors(simplex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: SimplexKey,
    ) -> Result<impl Iterator<Item = SimplexKey> + 'a, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.simplex_neighbors(c))
    }

    /// Returns the number of neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.number_of_simplex_neighbors_with_index(&index, simplex_key)?,
    ///     tri.simplex_neighbors(simplex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_simplex_neighbors_with_index(
        &self,
        index: &AdjacencyIndex,
        c: SimplexKey,
    ) -> Result<usize, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.number_of_simplex_neighbors(c))
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_incident_edges(v).into_iter()
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.incident_edges_with_index(&index, vertex_key)?.count(),
    ///     tri.incident_edges(vertex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> Result<impl Iterator<Item = EdgeKey> + 'a, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.incident_edges(v))
    }

    /// Returns the number of unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::AdjacencyIndexSnapshotMismatch`] if `index` was built
    /// from a different triangulation snapshot.
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
    /// #     Query(#[from] delaunay::prelude::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// # let vertices = vec![
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    /// #     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> =
    /// #     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index()?;
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(
    ///     tri.number_of_incident_edges_with_index(&index, vertex_key)?,
    ///     tri.incident_edges(vertex_key).count()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_incident_edges_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> Result<usize, QueryError> {
        self.validate_adjacency_index_matches(index)?;
        Ok(index.number_of_incident_edges(v))
    }

    /// Returns the number of unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, returns 0.
    #[must_use]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.collect_incident_edges(v).len()
    }

    /// Returns a slice view of a simplex's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    #[must_use]
    pub fn simplex_vertices(&self, c: SimplexKey) -> Option<&[VertexKey]> {
        self.tds.simplex(c).map(Simplex::vertices)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[f64]> {
        self.tds
            .vertex(v)
            .map(|vertex| &vertex.point().coords()[..])
    }

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This never stores any cache internally and does not mutate the triangulation.
    ///
    /// ## Notes
    ///
    /// - No sorted-order guarantees are provided for the values.
    /// - The returned collections are optimized for performance.
    /// - The maps include an entry for every vertex currently stored in the triangulation.
    ///   During the bootstrap phase (before the initial simplex is created), vertices have empty
    ///   adjacency lists because no simplices exist yet. This is expected and not an error condition.
    /// - Isolated vertices (present in the vertex store but not referenced by any simplex) are allowed at
    ///   the TDS structural layer, but violate the Level 3 manifold invariants checked by
    ///   [`Triangulation::is_valid`](Self::is_valid). When present, their adjacency lists are empty.
    /// - The returned index is tied to the current TDS identity and generation.
    ///   Indexed query helpers return [`QueryError::AdjacencyIndexSnapshotMismatch`]
    ///   if callers reuse it after mutation or with another triangulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the triangulation data structure is internally inconsistent
    /// (e.g., a simplex references a missing vertex key or a missing neighbor simplex key).
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        let vertex_cap = self.tds.number_of_vertices();
        let simplex_cap = self.tds.number_of_simplices();

        let mut vertex_to_simplices: VertexToSimplicesMap = fast_hash_map_with_capacity(vertex_cap);
        let mut simplex_to_neighbors: FastHashMap<
            SimplexKey,
            SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(simplex_cap);
        let mut vertex_to_edges: FastHashMap<
            VertexKey,
            SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(vertex_cap);

        let edges_per_simplex = (D + 1).saturating_mul(D) / 2;
        let mut seen_edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(simplex_cap.saturating_mul(edges_per_simplex));

        for (simplex_key, simplex) in self.tds.simplices() {
            let vertices = simplex.vertices();

            for &vk in vertices {
                if !self.tds.contains_vertex_key(vk) {
                    return Err(AdjacencyIndexBuildError::MissingVertexKey {
                        simplex_key,
                        vertex_key: vk,
                    });
                }
                let entry = vertex_to_simplices.entry(vk).or_default();
                #[cfg(debug_assertions)]
                let was_spilled = entry.spilled();
                entry.push(simplex_key);
                #[cfg(debug_assertions)]
                if !was_spilled && entry.spilled() {
                    let spill_count =
                        VERTEX_TO_SIMPLICES_SPILL_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
                    tracing::debug!(
                        "VertexToSimplicesMap spill #{spill_count}: vertex={vk:?} len={} cap={} (MAX_PRACTICAL_DIMENSION_SIZE={MAX_PRACTICAL_DIMENSION_SIZE})",
                        entry.len(),
                        entry.capacity()
                    );
                }
            }

            if let Some(neighbors) = simplex.neighbor_keys() {
                let mut neighs: SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::new();

                for n_opt in neighbors {
                    let Some(nk) = n_opt else {
                        continue;
                    };

                    if !self.tds.contains_simplex(nk) {
                        return Err(AdjacencyIndexBuildError::MissingNeighborSimplex {
                            simplex_key,
                            neighbor_key: nk,
                        });
                    }

                    neighs.push(nk);
                }

                if !neighs.is_empty() {
                    simplex_to_neighbors.insert(simplex_key, neighs);
                }
            }

            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    let edge = EdgeKey::from_validated_endpoints(vertices[i], vertices[j]);
                    if !seen_edges.insert(edge) {
                        continue;
                    }

                    let (a, b) = edge.endpoints();
                    vertex_to_edges.entry(a).or_default().push(edge);
                    vertex_to_edges.entry(b).or_default().push(edge);
                }
            }
        }

        for (vk, _) in self.tds.vertices() {
            vertex_to_simplices.entry(vk).or_default();
            vertex_to_edges.entry(vk).or_default();
        }

        Ok(AdjacencyIndex {
            tds_identity: Arc::clone(self.tds.identity()),
            tds_generation: self.tds.generation(),
            vertex_to_edges,
            vertex_to_simplices,
            simplex_to_neighbors,
        })
    }

    #[must_use]
    fn collect_edges(&self) -> FastHashSet<EdgeKey> {
        let simplex_cap = self.tds.number_of_simplices();
        let edges_per_simplex = (D + 1).saturating_mul(D) / 2;

        let mut edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(simplex_cap.saturating_mul(edges_per_simplex));

        for (_simplex_key, simplex) in self.tds.simplices() {
            let vertices = simplex.vertices();
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    edges.insert(EdgeKey::from_validated_endpoints(vertices[i], vertices[j]));
                }
            }
        }

        edges
    }

    #[must_use]
    fn collect_incident_edges(&self, v: VertexKey) -> FastHashSet<EdgeKey> {
        let mut edges: FastHashSet<EdgeKey> = FastHashSet::default();

        for simplex_key in self.adjacent_simplices(v) {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue;
            };

            for &other in simplex.vertices() {
                if other == v {
                    continue;
                }
                edges.insert(EdgeKey::from_validated_endpoints(v, other));
            }
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::FastKernel;
    use crate::triangulation::DelaunayTriangulation;

    use slotmap::KeyData;
    use std::assert_matches;
    use std::collections::HashSet;

    /// Basic accessor tests across dimensions.
    macro_rules! test_basic_accessors {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_basic_accessors_ $dim d>]() {
                    let empty: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());
                    assert_eq!(empty.number_of_vertices(), 0);
                    assert_eq!(empty.number_of_simplices(), 0);
                    assert_eq!(empty.dim(), -1);
                    assert_eq!(empty.simplices().count(), 0);
                    assert_eq!(empty.vertices().count(), 0);
                    assert_eq!(
                        empty
                            .facets()
                            .unwrap()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .unwrap(),
                        0
                    );
                    assert_eq!(
                        empty
                            .boundary_facets()
                            .unwrap()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .unwrap(),
                        0
                    );

                    let vertices = vec![
                        $(crate::core::vertex::Vertex::<(), _>::try_new($simplex_coords).unwrap()),+
                    ];
                    let expected_vertex_count = vertices.len();

                    let tds =
                        Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                            .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(
                        FastKernel::new(),
                        tds,
                    );

                    assert_eq!(tri.number_of_vertices(), expected_vertex_count);
                    assert_eq!(tri.number_of_simplices(), 1);
                    assert_eq!(tri.dim(), $dim as i32);
                    assert_eq!(tri.simplices().count(), 1);
                    assert_eq!(tri.vertices().count(), expected_vertex_count);
                    assert_eq!(
                        tri.facets()
                            .unwrap()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .unwrap(),
                        expected_vertex_count
                    );
                    assert_eq!(
                        tri.boundary_facets()
                            .unwrap()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .unwrap(),
                        expected_vertex_count
                    );
                }
            }
        };
    }

    test_basic_accessors!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_basic_accessors!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    #[test]
    fn test_boundary_facets_reports_corrupted_facet_map() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let (simplex_key, _) = tds.simplices().next().unwrap();
        let first_vertex = tds.simplex(simplex_key).unwrap().vertices()[0];

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            while simplex.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                simplex.push_vertex_key(first_vertex);
            }
        }

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        match tri.boundary_facets() {
            Ok(_) => panic!("corrupted facet map should return a query error"),
            Err(QueryError::TriangulationCorrupted {
                source: TdsError::IndexOutOfBounds { .. },
            }) => {}
            Err(err) => panic!("expected index-out-of-bounds query error, got {err:?}"),
        }
    }

    #[test]
    fn topology_edges_triangle_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 1);
        assert_eq!(tri.number_of_vertices(), 3);
        assert_eq!(tri.number_of_edges(), 3);

        let edges: HashSet<_> = tri.edges().collect();
        assert_eq!(edges.len(), 3);

        let index = tri.build_adjacency_index().unwrap();
        let edges_with_index: HashSet<_> = tri
            .edges_with_index(&index)
            .expect("fresh adjacency index")
            .collect();
        assert_eq!(edges_with_index, edges);
        assert_eq!(
            tri.number_of_edges_with_index(&index)
                .expect("fresh adjacency index"),
            3
        );

        assert!(edges.iter().all(|edge| {
            let (a, b) = edge.endpoints();
            a != b && tri.vertex_coords(a).is_some() && tri.vertex_coords(b).is_some()
        }));
    }

    #[test]
    fn topology_edges_and_incident_edges_double_tetrahedron_3d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, 1.5]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, -1.5]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 2);
        assert_eq!(tri.number_of_vertices(), 5);
        assert_eq!(tri.number_of_edges(), 9);

        let base_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);

        let index = tri.build_adjacency_index().unwrap();
        assert_eq!(
            tri.number_of_edges_with_index(&index)
                .expect("fresh adjacency index"),
            9
        );
        assert_eq!(tri.adjacent_simplices(base_vertex_key).count(), 2);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, base_vertex_key)
                .expect("fresh adjacency index")
                .count(),
            2
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, base_vertex_key)
                .expect("fresh adjacency index"),
            2
        );
        assert_eq!(
            tri.incident_edges_with_index(&index, base_vertex_key)
                .expect("fresh adjacency index")
                .count(),
            4
        );
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, base_vertex_key)
                .expect("fresh adjacency index"),
            4
        );

        let apex_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.7, 1.5]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(apex_vertex_key), 3);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, apex_vertex_key)
                .expect("fresh adjacency index")
                .count(),
            1
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, apex_vertex_key)
                .expect("fresh adjacency index"),
            1
        );

        for (simplex_key, _) in tri.simplices() {
            assert_eq!(
                tri.simplex_neighbors_with_index(&index, simplex_key)
                    .expect("fresh adjacency index")
                    .count(),
                1
            );
            assert_eq!(
                tri.number_of_simplex_neighbors_with_index(&index, simplex_key)
                    .expect("fresh adjacency index"),
                1
            );
        }
    }

    #[test]
    fn topology_queries_missing_keys_are_empty_or_none() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        let missing_vertex_key = VertexKey::default();
        assert_eq!(tri.adjacent_simplices(missing_vertex_key).count(), 0);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, missing_vertex_key)
                .expect("fresh adjacency index")
                .count(),
            0
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, missing_vertex_key)
                .expect("fresh adjacency index"),
            0
        );
        assert_eq!(tri.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(
            tri.incident_edges_with_index(&index, missing_vertex_key)
                .expect("fresh adjacency index")
                .count(),
            0
        );
        assert_eq!(tri.number_of_incident_edges(missing_vertex_key), 0);
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, missing_vertex_key)
                .expect("fresh adjacency index"),
            0
        );
        assert!(tri.vertex_coords(missing_vertex_key).is_none());

        let missing_simplex_key = SimplexKey::default();
        assert_eq!(tri.simplex_neighbors(missing_simplex_key).count(), 0);
        assert_eq!(
            tri.simplex_neighbors_with_index(&index, missing_simplex_key)
                .expect("fresh adjacency index")
                .count(),
            0
        );
        assert_eq!(
            tri.number_of_simplex_neighbors_with_index(&index, missing_simplex_key)
                .expect("fresh adjacency index"),
            0
        );
        assert!(tri.simplex_vertices(missing_simplex_key).is_none());
    }

    #[test]
    fn topology_geometry_accessors_round_trip() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.vertex_coords(vertex_key).unwrap(), [1.0, 0.0]);

        let simplex_key = tri.simplices().next().unwrap().0;
        let simplex_vertices = tri.simplex_vertices(simplex_key).unwrap();
        assert_eq!(simplex_vertices.len(), 3);
        assert!(simplex_vertices.contains(&vertex_key));
    }

    #[test]
    fn build_adjacency_index_basic_invariants() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, 1.5]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.7, -1.5]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        assert_eq!(simplex_keys.len(), 2);
        for &simplex_key in &simplex_keys {
            let neighbors = index.simplex_to_neighbors.get(&simplex_key).unwrap();
            assert_eq!(neighbors.len(), 1);
            assert!(simplex_keys.contains(&neighbors[0]));
            assert_ne!(neighbors[0], simplex_key);
        }

        for (vertex_key, _) in tri.vertices() {
            let simplices = index.vertex_to_simplices.get(&vertex_key).unwrap();
            assert!(!simplices.is_empty());

            let edges = index.vertex_to_edges.get(&vertex_key).unwrap();
            assert!(!edges.is_empty());
            assert!(edges.iter().all(
                |edge| matches!(edge.endpoints(), (a, b) if a == vertex_key || b == vertex_key)
            ));
        }
    }

    #[test]
    fn build_adjacency_index_empty_triangulation_is_empty() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let index = tri.build_adjacency_index().unwrap();
        assert!(index.vertex_to_simplices.is_empty());
        assert!(index.simplex_to_neighbors.is_empty());
        assert!(index.vertex_to_edges.is_empty());
    }

    #[test]
    fn build_adjacency_index_includes_isolated_vertex_entries() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let isolated_vertex = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 10.0]).unwrap(),
            )
            .unwrap();
        let index = tri.build_adjacency_index().unwrap();

        assert!(
            index
                .vertex_to_simplices
                .get(&isolated_vertex)
                .is_some_and(SmallBuffer::is_empty)
        );
        assert!(
            index
                .vertex_to_edges
                .get(&isolated_vertex)
                .is_some_and(SmallBuffer::is_empty)
        );
    }

    #[test]
    fn build_adjacency_index_errors_on_missing_neighbor_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();

        let mut missing_neighbor = SimplexKey::default();
        if tri.tds.contains_simplex(missing_neighbor) {
            missing_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        }
        assert!(!tri.tds.contains_simplex(missing_neighbor));

        tri.tds
            .simplex_mut(simplex_key)
            .unwrap()
            .set_neighbors_from_keys([Some(missing_neighbor), None, None])
            .unwrap();

        match tri.build_adjacency_index() {
            Err(AdjacencyIndexBuildError::MissingNeighborSimplex {
                simplex_key: err_simplex_key,
                neighbor_key,
            }) => {
                assert_eq!(err_simplex_key, simplex_key);
                assert_eq!(neighbor_key, missing_neighbor);
            }
            other => panic!("Expected MissingNeighborSimplex, got {other:?}"),
        }
    }

    #[test]
    fn simplex_neighbors_filters_missing_neighbor_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let missing_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        assert!(!tri.tds.contains_simplex(missing_neighbor));

        tri.tds
            .simplex_mut(simplex_key)
            .unwrap()
            .set_neighbors_from_keys([Some(missing_neighbor), None, None])
            .unwrap();

        assert_eq!(tri.simplex_neighbors(simplex_key).count(), 0);
    }

    #[test]
    fn build_adjacency_index_errors_on_missing_vertex_key() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let existing_vertices = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        let mut missing_vertex = VertexKey::default();
        if tri.tds.contains_vertex_key(missing_vertex) {
            missing_vertex = VertexKey::from(KeyData::from_ffi(u64::MAX));
        }
        assert!(!tri.tds.contains_vertex_key(missing_vertex));

        {
            let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(existing_vertices[0]);
            simplex.push_vertex_key(existing_vertices[1]);
            simplex.push_vertex_key(missing_vertex);
        }

        match tri.build_adjacency_index() {
            Err(AdjacencyIndexBuildError::MissingVertexKey {
                simplex_key: err_simplex_key,
                vertex_key,
            }) => {
                assert_eq!(err_simplex_key, simplex_key);
                assert_eq!(vertex_key, missing_vertex);
            }
            other => panic!("Expected MissingVertexKey, got {other:?}"),
        }
    }

    #[test]
    fn topology_queries_on_two_tet_triangulation() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let edge_count = tri.number_of_edges();
        let edges_collected: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_collected.len(), edge_count);
        assert!(edge_count >= 6);

        assert!(tri.facets().unwrap().next().transpose().unwrap().is_some());
        assert!(
            tri.boundary_facets()
                .unwrap()
                .next()
                .transpose()
                .unwrap()
                .is_some()
        );

        let (simplex_key, _) = tri.simplices().next().unwrap();
        let simplex_vertices = tri.simplex_vertices(simplex_key).unwrap();
        assert_eq!(simplex_vertices.len(), 4);
        for &vertex_key in simplex_vertices {
            let coords = tri.vertex_coords(vertex_key).unwrap();
            assert_eq!(coords.len(), 3);
        }

        assert!(
            tri.simplex_vertices(SimplexKey::from(KeyData::from_ffi(0xDEAD)))
                .is_none()
        );
        assert!(
            tri.vertex_coords(VertexKey::from(KeyData::from_ffi(0xBEEF)))
                .is_none()
        );

        let vertex_key = tri.vertices().next().unwrap().0;
        assert!(tri.adjacent_simplices(vertex_key).next().is_some());
        assert!(tri.simplex_neighbors(simplex_key).next().is_some());

        let incident_edges: Vec<_> = tri.incident_edges(vertex_key).collect();
        assert!(!incident_edges.is_empty());
        assert_eq!(
            tri.number_of_incident_edges(vertex_key),
            incident_edges.len()
        );
    }

    #[test]
    fn adjacency_index_with_index_methods() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        let idx_edges: HashSet<_> = tri
            .edges_with_index(&index)
            .expect("fresh adjacency index")
            .collect();
        let direct_edges: HashSet<_> = tri.edges().collect();
        assert_eq!(idx_edges, direct_edges);
        assert_eq!(
            tri.number_of_edges_with_index(&index)
                .expect("fresh adjacency index"),
            tri.number_of_edges()
        );

        let vertex_key = tri.vertices().next().unwrap().0;
        let idx_adj: HashSet<_> = tri
            .adjacent_simplices_with_index(&index, vertex_key)
            .expect("fresh adjacency index")
            .collect();
        let direct_adj: HashSet<_> = tri.adjacent_simplices(vertex_key).collect();
        assert_eq!(idx_adj, direct_adj);
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, vertex_key)
                .expect("fresh adjacency index"),
            direct_adj.len()
        );

        let simplex_key = tri.simplices().next().unwrap().0;
        let direct_neighbors: Vec<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(
            tri.simplex_neighbors_with_index(&index, simplex_key)
                .expect("fresh adjacency index")
                .count(),
            direct_neighbors.len()
        );
        assert_eq!(
            tri.number_of_simplex_neighbors_with_index(&index, simplex_key)
                .expect("fresh adjacency index"),
            direct_neighbors.len()
        );

        let idx_incident: HashSet<_> = tri
            .incident_edges_with_index(&index, vertex_key)
            .expect("fresh adjacency index")
            .collect();
        let direct_incident: HashSet<_> = tri.incident_edges(vertex_key).collect();
        assert_eq!(idx_incident, direct_incident);
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, vertex_key)
                .expect("fresh adjacency index"),
            direct_incident.len()
        );
    }

    #[test]
    fn adjacency_index_rejects_stale_generation() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tri = dt.as_triangulation().clone();
        let index = tri.build_adjacency_index().unwrap();

        tri.tds.clear_all_neighbors();

        assert_matches!(
            tri.number_of_edges_with_index(&index),
            Err(QueryError::AdjacencyIndexSnapshotMismatch {
                identity_matches: true,
                ..
            })
        );
    }

    #[test]
    fn adjacency_index_rejects_cross_triangulation_identity() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let dt_b: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri_a = dt_a.as_triangulation();
        let tri_b = dt_b.as_triangulation();
        let index_a = tri_a.build_adjacency_index().unwrap();

        let Err(err) = tri_b.edges_with_index(&index_a).map(drop) else {
            panic!("cross-triangulation adjacency index should be rejected");
        };
        assert_matches!(
            err,
            QueryError::AdjacencyIndexSnapshotMismatch {
                identity_matches: false,
                ..
            }
        );
    }
}
