//! Read-only queries for generic [`Triangulation`](crate::Triangulation).
//!
//! This module owns zero-mutation accessors and topology traversal helpers for
//! the generic triangulation layer. Mutation APIs stay with the construction and
//! editing modules; validation orchestration stays in [`crate::prelude::validation`].

use crate::core::adjacency::{
    EdgeIndex, IncidenceView, SimplexNeighborIndex, TopologyIndexBuildError, TriangulationAdjacency,
};
use crate::core::algorithms::flips::{FlipError, RidgeHandle};
use crate::core::algorithms::locate::{
    ConflictError, LocateError, LocateResult, LocateStats,
    find_conflict_region as find_conflict_region_in_tds, locate as locate_in_tds,
    locate_with_stats as locate_with_stats_in_tds,
};
use crate::core::collections::{
    FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SmallBuffer, Uuid,
    VertexKeyBuffer, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::edge::{EdgeKey, EdgeKeyError, EdgeView};
use crate::core::facet::{
    AllFacetsIter, BoundaryFacetsIter, FacetError, FacetHandle, FacetToSimplicesIndex, FacetView,
    SimplexFacetsIter,
};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, TdsError, VertexKey};
use crate::core::triangulation::Triangulation;
use crate::core::util::usize_to_u8;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::topology::manifold::{ManifoldError, boundary_facet_handles_from_index};
use crate::topology::ridge::{
    RidgeCandidate, RidgeCandidateError, RidgeQuery, RidgeView,
    ridge_star_simplices as ridge_star_simplices_in_tds,
};

/// Errors returned by read-only triangulation queries.
///
/// These errors indicate that a read-only query could not safely use the
/// requested topology view. This can happen because the underlying
/// triangulation state is inconsistent.
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
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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
    /// The triangulation could not build a facet index for a read-only query.
    #[error("Triangulation data structure is corrupted: {source}")]
    TriangulationCorrupted {
        /// Typed TDS validation or bookkeeping error that prevented the query.
        #[source]
        source: Box<TdsError>,
    },

    /// The triangulation's topology metadata rejects the requested boundary query.
    #[error("Triangulation topology is invalid for this query: {source}")]
    TopologyInvalid {
        /// Typed topology validation error that prevented the query.
        #[source]
        source: Box<ManifoldError>,
    },

    /// A simplex-local ridge handle cannot represent the omitted vertex index.
    #[error(
        "ridge index {original_index} for simplex {simplex_key:?} cannot fit in u8 handle storage; simplex has {vertex_count} vertices"
    )]
    RidgeIndexCapacityExceeded {
        /// Simplex whose local ridge index overflowed handle storage.
        simplex_key: SimplexKey,
        /// Local omitted-vertex index that could not fit in `u8`.
        original_index: usize,
        /// Number of vertices in the simplex.
        vertex_count: usize,
    },

    /// A simplex produced invalid ridge vertices while building a topology ridge query.
    #[error("simplex {simplex_key:?} produced an invalid topology ridge: {source}")]
    InvalidRidgeCandidate {
        /// Simplex whose vertex set produced the invalid ridge candidate.
        simplex_key: SimplexKey,
        /// Typed ridge-candidate validation failure.
        #[source]
        source: RidgeCandidateError,
    },
}

impl From<TdsError> for QueryError {
    fn from(source: TdsError) -> Self {
        Self::TriangulationCorrupted {
            source: Box::new(source),
        }
    }
}

impl From<ManifoldError> for QueryError {
    fn from(source: ManifoldError) -> Self {
        match source {
            ManifoldError::Tds(source) => Self::from(source),
            source => Self::TopologyInvalid {
                source: Box::new(source),
            },
        }
    }
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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

    /// Returns simplex keys paired with their stable UUIDs.
    ///
    /// This is a zero-allocation identity iterator for diagnostics, snapshots,
    /// and downstream bookkeeping that should not borrow the storage owner.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, simplex_uuid)) = tri.simplex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.simplex_uuid_from_key(simplex_key), Some(simplex_uuid));
    /// assert_eq!(tri.simplex_key_from_uuid(&simplex_uuid), Some(simplex_key));
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_uuids(&self) -> impl Iterator<Item = (SimplexKey, Uuid)> + '_ {
        self.tds
            .simplices()
            .map(|(simplex_key, simplex)| (simplex_key, simplex.uuid()))
    }

    /// Returns a read-only simplex view by key.
    ///
    /// This is the keyed counterpart to [`Triangulation::simplices`](Self::simplices).
    /// It lends only the requested element instead of exposing the underlying
    /// topology owner.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::SimplexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(tri.simplex(simplex_key).is_some());
    /// assert!(tri.simplex(SimplexKey::default()).is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn simplex(&self, key: SimplexKey) -> Option<&Simplex<V, D>> {
        self.tds.simplex(key)
    }

    /// Returns a simplex key for a stable simplex UUID.
    ///
    /// This is the owner-bound counterpart to
    /// [`Tds::simplex_key_from_uuid`](crate::prelude::tds::Tds::simplex_key_from_uuid).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use uuid::Uuid;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, simplex_uuid)) = tri.simplex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.simplex_key_from_uuid(&simplex_uuid), Some(simplex_key));
    /// assert_eq!(tri.simplex_key_from_uuid(&Uuid::nil()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn simplex_key_from_uuid(&self, simplex_uuid: &Uuid) -> Option<SimplexKey> {
        self.tds.simplex_key_from_uuid(simplex_uuid)
    }

    /// Returns the stable UUID for a live simplex key.
    ///
    /// This is the owner-bound counterpart to
    /// [`Tds::simplex_uuid_from_key`](crate::prelude::tds::Tds::simplex_uuid_from_key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::SimplexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, simplex_uuid)) = tri.simplex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.simplex_uuid_from_key(simplex_key), Some(simplex_uuid));
    /// assert_eq!(tri.simplex_uuid_from_key(SimplexKey::default()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn simplex_uuid_from_key(&self, simplex_key: SimplexKey) -> Option<Uuid> {
        self.tds.simplex_uuid_from_key(simplex_key)
    }

    /// Returns `true` when `key` identifies a live simplex in this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::SimplexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(tri.contains_simplex(simplex_key));
    /// assert!(!tri.contains_simplex(SimplexKey::default()));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn contains_simplex(&self, key: SimplexKey) -> bool {
        self.tds.contains_simplex(key)
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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

    /// Returns vertex keys paired with their stable UUIDs.
    ///
    /// This is a zero-allocation identity iterator for diagnostics, snapshots,
    /// and downstream bookkeeping that should not borrow the storage owner.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((vertex_key, vertex_uuid)) = tri.vertex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.vertex_uuid_from_key(vertex_key), Some(vertex_uuid));
    /// assert_eq!(tri.vertex_key_from_uuid(&vertex_uuid), Some(vertex_key));
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertex_uuids(&self) -> impl Iterator<Item = (VertexKey, Uuid)> + '_ {
        self.tds
            .vertices()
            .map(|(vertex_key, vertex)| (vertex_key, vertex.uuid()))
    }

    /// Returns a read-only vertex view by key.
    ///
    /// This is the keyed counterpart to [`Triangulation::vertices`](Self::vertices).
    /// It lends only the requested element instead of exposing the underlying
    /// topology owner.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::VertexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(tri.vertex(vertex_key).is_some());
    /// assert!(tri.vertex(VertexKey::default()).is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vertex(&self, key: VertexKey) -> Option<&Vertex<U, D>> {
        self.tds.vertex(key)
    }

    /// Returns a vertex key for a stable vertex UUID.
    ///
    /// This is the owner-bound counterpart to
    /// [`Tds::vertex_key_from_uuid`](crate::prelude::tds::Tds::vertex_key_from_uuid).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use uuid::Uuid;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((vertex_key, vertex_uuid)) = tri.vertex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.vertex_key_from_uuid(&vertex_uuid), Some(vertex_key));
    /// assert_eq!(tri.vertex_key_from_uuid(&Uuid::nil()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vertex_key_from_uuid(&self, vertex_uuid: &Uuid) -> Option<VertexKey> {
        self.tds.vertex_key_from_uuid(vertex_uuid)
    }

    /// Returns the stable UUID for a live vertex key.
    ///
    /// This is the owner-bound counterpart to
    /// [`Tds::vertex_uuid_from_key`](crate::prelude::tds::Tds::vertex_uuid_from_key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::VertexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((vertex_key, vertex_uuid)) = tri.vertex_uuids().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.vertex_uuid_from_key(vertex_key), Some(vertex_uuid));
    /// assert_eq!(tri.vertex_uuid_from_key(VertexKey::default()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vertex_uuid_from_key(&self, vertex_key: VertexKey) -> Option<Uuid> {
        self.tds.vertex_uuid_from_key(vertex_key)
    }

    /// Returns `true` when `key` identifies a live vertex in this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::VertexKey;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((vertex_key, _)) = tri.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(tri.contains_vertex_key(vertex_key));
    /// assert!(!tri.contains_vertex_key(VertexKey::default()));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn contains_vertex_key(&self, key: VertexKey) -> bool {
        self.tds.contains_vertex_key(key)
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// assert_eq!(dt.as_triangulation().number_of_simplices(), 1); // Single tetrahedron
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.tds.number_of_simplices()
    }

    /// Returns the topology generation counter for this triangulation.
    ///
    /// The value changes after topology mutations and is useful for detecting
    /// stale detached handles in tests and diagnostics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    ///
    /// assert_eq!(tri.topology_generation(), tri.topology_generation());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn topology_generation(&self) -> u64 {
        self.tds.generation()
    }

    /// Returns whether adjacent simplices have coherent opposite facet orientations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// assert!(dt.as_triangulation().is_coherently_oriented());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        self.tds.is_coherently_oriented()
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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
    /// Individual iterator items return [`FacetError`]
    /// if a facet view cannot be constructed from the current TDS state.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// // Iterate over all facets
    /// let facet_count = dt
    ///     .as_triangulation()
    ///     .facets()
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn facets(&self) -> AllFacetsIter<'_, U, V, D> {
        self.tds.facets()
    }

    /// Returns an iterator over all facets of one simplex.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError`] if `simplex_key` is missing or this dimension
    /// cannot be represented by public facet-index storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(tri.simplex_facets(simplex_key)?.count(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_facets(
        &self,
        simplex_key: SimplexKey,
    ) -> Result<SimplexFacetsIter<'_, U, V, D>, FacetError> {
        self.tds.try_simplex_facets(simplex_key)
    }

    /// Validates and returns a simplex-local facet handle.
    ///
    /// This is the owner-bound counterpart to [`FacetHandle::try_new`]. It lets
    /// callers validate a runtime simplex key and facet index without borrowing
    /// the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError`] if `simplex_key` is missing or `facet_index` is
    /// outside the simplex's local facet range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = tri.facet_handle(simplex_key, 0)?;
    /// assert_eq!(facet.simplex_key(), simplex_key);
    /// # Ok(())
    /// # }
    /// ```
    pub fn facet_handle(
        &self,
        simplex_key: SimplexKey,
        facet_index: u8,
    ) -> Result<FacetHandle, FacetError> {
        FacetHandle::try_new(&self.tds, simplex_key, facet_index)
    }

    /// Revalidates a facet handle and returns a borrowed facet view.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError`] if the handle is stale or no longer identifies a
    /// live simplex-local facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let facet = tri.facet_handle(simplex_key, 0)?;
    ///
    /// let view = tri.facet_view(facet)?;
    /// assert_eq!(view.handle(), facet);
    /// # Ok(())
    /// # }
    /// ```
    pub fn facet_view(&self, facet: FacetHandle) -> Result<FacetView<'_, U, V, D>, FacetError> {
        facet.view(&self.tds)
    }

    /// Builds the owner-bound facet-to-simplices incidence index.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if the triangulation's facet incidence is
    /// structurally inconsistent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.facet_incidence_index()?;
    /// assert_eq!(index.iter().filter(|facet| facet.is_one_sided()).count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn facet_incidence_index(&self) -> Result<FacetToSimplicesIndex<'_, U, V, D>, TdsError> {
        self.tds.build_facet_to_simplices_index()
    }

    /// Returns unique topology ridges in the triangulation.
    ///
    /// A ridge is a codimension-two face. In 2D, ridges are vertices; in 3D,
    /// ridges are edges. Dimensions below 2 have no topological ridges and
    /// yield an empty iterator.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method streams ridge candidates while maintaining an internal
    /// deduplication set for ridges shared by multiple simplices. The iteration
    /// order follows the current simplex storage order and is not stable across
    /// topology edits.
    ///
    /// # Errors
    ///
    /// Individual iterator items return [`QueryError::InvalidRidgeCandidate`]
    /// if stored simplex vertices cannot form a valid codimension-two ridge.
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge_count = dt
    ///     .as_triangulation()
    ///     .ridges()
    ///     .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))?;
    /// assert_eq!(ridge_count, 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridges(&self) -> impl Iterator<Item = Result<RidgeCandidate<D>, QueryError>> + '_ {
        let simplex_cap = self.tds.number_of_simplices();
        let ridges_per_simplex = (D + 1).saturating_mul(D) / 2;
        let mut seen = fast_hash_set_with_capacity(simplex_cap.saturating_mul(ridges_per_simplex));

        self.tds
            .simplices()
            .filter(move |_| D >= 2)
            .flat_map(|(simplex_key, simplex)| {
                let vertices = simplex.vertices();
                (0..vertices.len()).flat_map(move |omit_a| {
                    ((omit_a + 1)..vertices.len()).map(move |omit_b| {
                        ridge_candidate_from_simplex_vertices::<D>(
                            simplex_key,
                            vertices,
                            omit_a,
                            omit_b,
                        )
                    })
                })
            })
            .filter_map(move |result| match result {
                Ok(ridge) => {
                    if seen.insert(ridge.clone()) {
                        Some(Ok(ridge))
                    } else {
                        None
                    }
                }
                Err(error) => Some(Err(error)),
            })
    }

    /// Returns simplex-local ridge handles for `K3` Pachner moves.
    ///
    /// These handles identify codimension-two faces by a containing simplex and
    /// two omitted vertex indices, matching the representation required by
    /// `K3` Pachner moves. Dimensions below 3 have no `K3` Pachner ridge
    /// candidates and yield an empty iterator.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method streams handles in the current simplex storage order. The
    /// order is not stable across topology edits.
    ///
    /// # Errors
    ///
    /// Individual iterator items return [`QueryError::RidgeIndexCapacityExceeded`]
    /// if a simplex-local omitted vertex index cannot fit in the public
    /// [`RidgeHandle`] index storage.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge_count = dt
    ///     .as_triangulation()
    ///     .ridge_handles()
    ///     .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))?;
    /// assert_eq!(ridge_count, 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridge_handles(&self) -> impl Iterator<Item = Result<RidgeHandle, QueryError>> + '_ {
        self.tds
            .simplices()
            .filter(move |_| D >= 3)
            .flat_map(|(simplex_key, simplex)| {
                ridge_handles_for_simplex(simplex_key, simplex.number_of_vertices())
            })
    }

    /// Validates and returns a simplex-local ridge handle.
    ///
    /// This is the owner-bound counterpart to [`RidgeHandle::try_new`]. It lets
    /// callers validate runtime omitted-vertex indices without borrowing the
    /// underlying storage.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the dimension does not support ridge flips, the
    /// simplex key is missing, or either omitted vertex index is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((simplex_key, _)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let ridge = tri.ridge_handle(simplex_key, 0, 1)?;
    /// assert_eq!(ridge.simplex_key(), simplex_key);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridge_handle(
        &self,
        simplex_key: SimplexKey,
        omit_a: u8,
        omit_b: u8,
    ) -> Result<RidgeHandle, FlipError> {
        RidgeHandle::try_new(&self.tds, simplex_key, omit_a, omit_b)
    }

    /// Returns the simplex star incident to a ridge candidate.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError`] if any ridge vertex is stale or incidence
    /// bookkeeping cannot be traversed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::query::RidgeCandidate;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Ok(ridge) = RidgeCandidate::<2>::try_from_vertices(
    ///     tri.vertices().map(|(key, _)| key).take(1),
    /// ) else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(!tri.ridge_star_simplices(&ridge)?.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridge_star_simplices(
        &self,
        ridge_candidate: &RidgeCandidate<D>,
    ) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
        ridge_star_simplices_in_tds(&self.tds, ridge_candidate)
    }

    /// Revalidates a ridge candidate and returns a borrowed ridge query.
    ///
    /// Unlike [`Triangulation::ridge_view`], the query permits an empty
    /// incident star.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError`] if any ridge vertex is stale or incidence
    /// bookkeeping cannot be traversed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::query::RidgeCandidate;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Ok(ridge) = RidgeCandidate::<2>::try_from_vertices(
    ///     tri.vertices().map(|(key, _)| key).take(1),
    /// ) else {
    ///     return Ok(());
    /// };
    ///
    /// let query = tri.ridge_query(&ridge)?;
    /// assert_eq!(query.ridge_candidate(), &ridge);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridge_query(
        &self,
        ridge_candidate: &RidgeCandidate<D>,
    ) -> Result<RidgeQuery<'_, U, V, D>, ManifoldError> {
        ridge_candidate.query(&self.tds)
    }

    /// Revalidates a ridge candidate and returns a borrowed ridge view.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError`] if any ridge vertex is stale, incidence
    /// bookkeeping cannot be traversed, or the ridge has an empty star.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::query::RidgeCandidate;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Ok(ridge) = RidgeCandidate::<2>::try_from_vertices(
    ///     tri.vertices().map(|(key, _)| key).take(1),
    /// ) else {
    ///     return Ok(());
    /// };
    ///
    /// let view = tri.ridge_view(&ridge)?;
    /// assert_eq!(view.ridge_candidate(), &ridge);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ridge_view(
        &self,
        ridge_candidate: &RidgeCandidate<D>,
    ) -> Result<RidgeView<'_, U, V, D>, ManifoldError> {
        ridge_candidate.view(&self.tds)
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are one-sided facets not identified by closed periodic
    /// topology. This method computes the facet-to-simplices index internally
    /// for convenience.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
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
    /// Returns [`QueryError::TriangulationCorrupted`] if facet-incidence index
    /// construction detects invalid simplex or facet bookkeeping. The variant
    /// preserves the underlying [`TdsError`] so callers can inspect the
    /// structural failure. Returns [`QueryError::TopologyInvalid`] if
    /// topology-aware boundary classification detects a closed topology that
    /// cannot contain open boundary facets, or another manifold-boundary
    /// inconsistency.
    /// Individual iterator items return [`FacetError`]
    /// if a boundary facet handle cannot be reborrowed as a view.
    pub fn boundary_facets(&self) -> Result<BoundaryFacetsIter<'_, U, V, D>, QueryError> {
        let facet_index = self
            .tds
            .build_facet_to_simplices_index()
            .map_err(QueryError::from)?;
        let boundary_facet_handles =
            boundary_facet_handles_from_index(&facet_index, self.global_topology)
                .map_err(QueryError::from)?;
        BoundaryFacetsIter::try_new(&facet_index, boundary_facet_handles)
            .map_err(TdsError::from)
            .map_err(QueryError::from)
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
    /// If you need fast repeated topology queries against the same triangulation,
    /// prefer [`Triangulation::build_edge_index`](Self::build_edge_index) for
    /// edge-only traversal, or the lifetime-bound [`TriangulationAdjacency`]
    /// view returned by [`Triangulation::adjacency`](Self::adjacency) when a
    /// caller needs all topology query families.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
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

    /// Validates and returns an edge key for two live vertices.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if the vertices are stale, duplicated, or do not
    /// share a stored simplex edge.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((_simplex_key, simplex)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let edge = tri.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// assert!(tri.edges().any(|candidate| candidate == edge));
    /// # Ok(())
    /// # }
    /// ```
    pub fn edge_key(&self, a: VertexKey, b: VertexKey) -> Result<EdgeKey, EdgeKeyError> {
        EdgeKey::try_new(&self.tds, a, b)
    }

    /// Revalidates an edge key and returns a borrowed edge view.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if the key is stale or the maintained incidence
    /// relation cannot prove a live edge star.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((_simplex_key, simplex)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let key = tri.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// let view = tri.edge_view(key)?;
    /// assert_eq!(view.key(), key);
    /// # Ok(())
    /// # }
    /// ```
    pub fn edge_view(&self, edge: EdgeKey) -> Result<EdgeView<'_, U, V, D>, EdgeKeyError> {
        edge.view(&self.tds)
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
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

    /// Returns an iterator over all simplices adjacent (incident) to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// Iteration order is not specified.
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.tds.simplex_keys_containing_vertex(v)
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

    /// Locates a point in this triangulation using the owner's kernel.
    ///
    /// This is the owner-bound counterpart to the low-level
    /// [`locate_in_tds`](crate::prelude::algorithms::locate) function. It keeps
    /// callers on the `Triangulation` API while preserving the same typed
    /// location result.
    ///
    /// # Errors
    ///
    /// Returns [`LocateError`] if structural data or predicates fail during the
    /// facet-walking query.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::algorithms::{LocateError, LocateResult};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Locate(#[from] LocateError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let query = Point::try_from([0.2, 0.2])?;
    ///
    /// let location = dt.as_triangulation().locate(&query, None)?;
    /// std::assert_matches!(location, LocateResult::InsideSimplex(_));
    /// # Ok(())
    /// # }
    /// ```
    pub fn locate(
        &self,
        point: &Point<D>,
        hint: Option<SimplexKey>,
    ) -> Result<LocateResult, LocateError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        locate_in_tds(&self.tds, &self.kernel, point, hint)
    }

    /// Locates a point and returns facet-walk traversal statistics.
    ///
    /// # Errors
    ///
    /// Returns [`LocateError`] if structural data or predicates fail during the
    /// facet-walking query.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::algorithms::LocateError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Locate(#[from] LocateError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let query = Point::try_from([0.2, 0.2])?;
    ///
    /// let (_location, stats) = dt.as_triangulation().locate_with_stats(&query, None)?;
    /// assert!(!stats.fell_back_to_scan());
    /// # Ok(())
    /// # }
    /// ```
    pub fn locate_with_stats(
        &self,
        point: &Point<D>,
        hint: Option<SimplexKey>,
    ) -> Result<(LocateResult, LocateStats), LocateError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        locate_with_stats_in_tds(&self.tds, &self.kernel, point, hint)
    }

    /// Finds the conflict region for inserting `point` from a known start simplex.
    ///
    /// This is the owner-bound counterpart to the low-level
    /// [`find_conflict_region_in_tds`](crate::prelude::algorithms::find_conflict_region)
    /// function.
    ///
    /// # Errors
    ///
    /// Returns [`ConflictError`] if the start simplex is stale, structural data is
    /// inconsistent, or the conflict boundary cannot be classified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::algorithms::{ConflictError, LocateError, LocateResult};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Locate(#[from] LocateError),
    /// #     #[error(transparent)]
    /// #     Conflict(#[from] ConflictError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let query = Point::try_from([0.2, 0.2])?;
    ///
    /// if let LocateResult::InsideSimplex(start) = tri.locate(&query, None)? {
    ///     let conflict = tri.find_conflict_region(&query, start)?;
    ///     assert!(!conflict.is_empty());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn find_conflict_region(
        &self,
        point: &Point<D>,
        start_simplex: SimplexKey,
    ) -> Result<SimplexKeyBuffer, ConflictError>
    where
        K: Kernel<D, Scalar = f64>,
    {
        find_conflict_region_in_tds(&self.tds, &self.kernel, point, start_simplex)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_incident_edges(v).into_iter()
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
    /// This is a zero-allocation accessor that validates the simplex key and
    /// referenced vertex keys before lending the canonical slice.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if `c` does not identify a simplex in this
    /// triangulation, or if the simplex references a missing vertex key.
    pub fn simplex_vertices(&self, c: SimplexKey) -> Result<&[VertexKey], TdsError> {
        self.tds.simplex_vertices(c)
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

    /// Builds a lifetime-bound adjacency view for fast repeated topology queries.
    ///
    /// The returned view owns the derived maps and keeps the source TDS
    /// immutably borrowed for its lifetime. Use this when all indexed queries
    /// are against the triangulation that built the view.
    ///
    /// # Errors
    ///
    /// Returns an error if the triangulation data structure is internally inconsistent
    /// (e.g., a simplex references a missing vertex key or a missing neighbor simplex key).
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
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let adjacency = dt.as_triangulation().adjacency()?;
    ///
    /// assert_eq!(adjacency.number_of_edges(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn adjacency(&self) -> Result<TriangulationAdjacency<'_>, TopologyIndexBuildError> {
        let incidence = self.incidence()?;
        let edges = self.build_edge_index()?;
        let simplex_neighbors = self.build_simplex_neighbor_index()?;
        Ok(TriangulationAdjacency::from_parts(
            incidence,
            edges,
            simplex_neighbors,
        ))
    }

    /// Borrows the canonical vertex→simplices incidence relation.
    ///
    /// Use this for repeated adjacent-simplex queries without building derived
    /// edge or simplex-neighbor maps.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyIndexBuildError::InvalidVertexIncidenceIndex`] if the
    /// maintained incidence relation is internally inconsistent.
    pub fn incidence(&self) -> Result<IncidenceView<'_>, TopologyIndexBuildError> {
        self.tds
            .validate_vertex_to_simplices_index()
            .map_err(|source| TopologyIndexBuildError::InvalidVertexIncidenceIndex { source })?;

        Ok(IncidenceView {
            vertex_to_simplices: self.tds.vertex_to_simplices_index(),
        })
    }

    /// Builds only the derived vertex→edge index for this triangulation snapshot.
    ///
    /// Use this for repeated edge and incident-edge queries without deriving a
    /// simplex-neighbor map or validating canonical vertex→simplices incidence.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyIndexBuildError::MissingVertexKey`] if a simplex
    /// references a vertex key that is absent from vertex storage.
    pub fn build_edge_index(&self) -> Result<EdgeIndex<'_>, TopologyIndexBuildError> {
        let vertex_cap = self.tds.number_of_vertices();
        let simplex_cap = self.tds.number_of_simplices();
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
                    return Err(TopologyIndexBuildError::MissingVertexKey {
                        simplex_key,
                        vertex_key: vk,
                    });
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
            vertex_to_edges.entry(vk).or_default();
        }

        Ok(EdgeIndex {
            edge_count: seen_edges.len(),
            vertex_to_edges,
            _source_incidence: self.tds.vertex_to_simplices_index(),
        })
    }

    /// Builds only the derived simplex→neighbor index for this triangulation snapshot.
    ///
    /// Use this for repeated simplex-neighbor queries without deriving edge maps
    /// or validating canonical vertex→simplices incidence.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyIndexBuildError::MissingNeighborSimplex`] if a simplex
    /// references a neighbor key that is absent from simplex storage.
    pub fn build_simplex_neighbor_index(
        &self,
    ) -> Result<SimplexNeighborIndex<'_>, TopologyIndexBuildError> {
        let simplex_cap = self.tds.number_of_simplices();
        let mut simplex_to_neighbors: FastHashMap<
            SimplexKey,
            SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(simplex_cap);

        for (simplex_key, simplex) in self.tds.simplices() {
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            let mut neighs: SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();

            for n_opt in neighbors {
                let Some(nk) = n_opt else {
                    continue;
                };

                if !self.tds.contains_simplex(nk) {
                    return Err(TopologyIndexBuildError::MissingNeighborSimplex {
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

        Ok(SimplexNeighborIndex {
            simplex_to_neighbors,
            _source_incidence: self.tds.vertex_to_simplices_index(),
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

/// Builds one topology ridge candidate from a simplex after omitting two vertices.
fn ridge_candidate_from_simplex_vertices<const D: usize>(
    simplex_key: SimplexKey,
    vertices: &[VertexKey],
    omit_a: usize,
    omit_b: usize,
) -> Result<RidgeCandidate<D>, QueryError> {
    let mut ridge_vertices = VertexKeyBuffer::with_capacity(D.saturating_sub(1));
    for (index, &vertex_key) in vertices.iter().enumerate() {
        if index != omit_a && index != omit_b {
            ridge_vertices.push(vertex_key);
        }
    }

    RidgeCandidate::try_from_vertices(ridge_vertices).map_err(|source| {
        QueryError::InvalidRidgeCandidate {
            simplex_key,
            source,
        }
    })
}

/// Streams simplex-local Pachner ridge handles for one simplex.
fn ridge_handles_for_simplex(
    simplex_key: SimplexKey,
    vertex_count: usize,
) -> impl Iterator<Item = Result<RidgeHandle, QueryError>> {
    (0..vertex_count).flat_map(move |omit_a| {
        ((omit_a + 1)..vertex_count).map(move |omit_b| {
            let omit_a =
                u8::try_from(omit_a).map_err(|_| QueryError::RidgeIndexCapacityExceeded {
                    simplex_key,
                    original_index: omit_a,
                    vertex_count,
                })?;
            let omit_b =
                u8::try_from(omit_b).map_err(|_| QueryError::RidgeIndexCapacityExceeded {
                    simplex_key,
                    original_index: omit_b,
                    vertex_count,
                })?;
            Ok(RidgeHandle::from_validated(simplex_key, omit_a, omit_b))
        })
    })
}

impl<K, U, V> Triangulation<K, U, V, 2> {
    /// Returns one simplex-local facet handle for an interior 2D edge.
    ///
    /// In 2D, a cell is a triangle and each cell facet is an edge. This query
    /// maps a live interior [`EdgeKey`] to one of the two incident
    /// [`FacetHandle`] values that can be passed to 2D local-edit APIs such as
    /// Pachner k=2 flips.
    ///
    /// Edges that do not have exactly two incident 2D facets return `Ok(None)`.
    /// In a valid 2D PL-manifold this means a boundary edge, but callers working
    /// with deliberately invalid low-level topology can inspect the exact
    /// multiplicity with [`Self::try_incident_facets_to_edge_2d`]. Stale edge
    /// keys and keys that no longer identify a live edge return the
    /// [`EdgeKeyError`] produced while parsing the edge. Use
    /// [`Self::try_incident_facets_to_edge_2d`] when callers need to distinguish
    /// boundary edges from interior edges by multiplicity.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if `edge` is stale, no longer identifies a live
    /// edge, or exposes inconsistent maintained incidence metadata in this
    /// triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::EdgeKey;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Edge(#[from] delaunay::prelude::tds::EdgeKeyError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((_simplex_key, simplex)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let boundary_edge = tri.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// assert!(tri.try_interior_facet_for_edge_2d(boundary_edge)?.is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_interior_facet_for_edge_2d(
        &self,
        edge: EdgeKey,
    ) -> Result<Option<FacetHandle>, EdgeKeyError> {
        let edge_view = edge.view(&self.tds)?;
        let facets = Self::collect_incident_facets_to_edge_2d(&edge_view);
        Ok((facets.len() == 2).then(|| facets[0]))
    }

    /// Returns the simplex-local facet handles incident to a 2D edge.
    ///
    /// In a valid 2D triangulation, this yields one handle for a boundary edge
    /// and two handles for an interior edge. Stale edge keys and keys that no
    /// longer identify a live edge return the [`EdgeKeyError`] produced while
    /// parsing the edge.
    ///
    /// The query is computed from the current TDS vertex→simplices incidence
    /// relation, so it reflects local topology mutations without requiring a
    /// public mutable cache.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if `edge` is stale, no longer identifies a live
    /// edge, or exposes inconsistent maintained incidence metadata in this
    /// triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::EdgeKey;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Edge(#[from] delaunay::prelude::tds::EdgeKeyError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tri = dt.as_triangulation();
    /// let Some((_simplex_key, simplex)) = tri.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let boundary_edge = tri.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// assert_eq!(tri.try_incident_facets_to_edge_2d(boundary_edge)?.count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_incident_facets_to_edge_2d(
        &self,
        edge: EdgeKey,
    ) -> Result<impl Iterator<Item = FacetHandle>, EdgeKeyError> {
        let edge_view = edge.view(&self.tds)?;
        Ok(Self::collect_incident_facets_to_edge_2d(&edge_view).into_iter())
    }

    /// Builds the edge-to-facet answer from a parsed live edge star without storing a cache.
    ///
    /// Public callers reach this helper only after [`EdgeKey::view`] has proven
    /// that the detached key is a live edge in this TDS, so the collector stays
    /// infallible and preserves parse failures at the public query boundary.
    fn collect_incident_facets_to_edge_2d(
        edge_view: &EdgeView<'_, U, V, 2>,
    ) -> SmallBuffer<FacetHandle, 2> {
        let (v0, v1) = edge_view.endpoint_keys();
        let mut facets = SmallBuffer::new();

        for &simplex_key in edge_view.incident_simplices() {
            let simplex = edge_view
                .tds()
                .simplex(simplex_key)
                .expect("validated EdgeView contains only live incident simplices");
            let facet_index = Self::facet_index_for_edge_2d(simplex, v0, v1);
            facets.push(FacetHandle::from_validated(simplex_key, facet_index));
        }

        facets
    }

    /// Converts a validated triangle edge into the facet slot opposite the third vertex.
    ///
    /// Callers only pass simplices from a parsed [`EdgeView`], so each simplex
    /// is already proven to contain both edge endpoints.
    fn facet_index_for_edge_2d(simplex: &Simplex<V, 2>, v0: VertexKey, v1: VertexKey) -> u8 {
        let vertices = simplex.vertices();
        let facet_index = vertices
            .iter()
            .position(|&vertex_key| vertex_key != v0 && vertex_key != v1)
            .expect("validated 2D EdgeView incident simplex contains both edge endpoints");
        usize_to_u8(facet_index, vertices.len()).expect("2D simplex facet index fits in u8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::Tds;
    use crate::geometry::kernel::FastKernel;
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;

    use slotmap::KeyData;
    use std::assert_matches;
    use std::collections::HashSet;

    /// Builds two D-simplices sharing one facet so split topology indexes can be
    /// tested without depending on Delaunay construction tie-breaking.
    fn split_topology_fixture<const D: usize>() -> Triangulation<FastKernel<f64>, (), (), D> {
        let mut tds = Tds::empty();
        let mut shared_vertex_keys = Vec::with_capacity(D);
        for point_idx in 0..D {
            let mut coords = [0.0; D];
            if point_idx > 0 {
                coords[point_idx - 1] = 1.0;
            }
            let vertex_key = tds
                .insert_vertex_with_mapping(vertex!(coords).unwrap())
                .unwrap();
            shared_vertex_keys.push(vertex_key);
        }

        let mut positive_apex = [0.2; D];
        positive_apex[D - 1] = 1.0;
        let positive_apex_key = tds
            .insert_vertex_with_mapping(vertex!(positive_apex).unwrap())
            .unwrap();

        let mut negative_apex = [0.2; D];
        negative_apex[D - 1] = -1.0;
        let negative_apex_key = tds
            .insert_vertex_with_mapping(vertex!(negative_apex).unwrap())
            .unwrap();

        let mut positive_simplex_vertices = shared_vertex_keys.clone();
        positive_simplex_vertices.push(positive_apex_key);
        let positive_simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new(positive_simplex_vertices).unwrap())
            .unwrap();

        let mut negative_simplex_vertices = shared_vertex_keys;
        negative_simplex_vertices.push(negative_apex_key);
        let negative_simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new(negative_simplex_vertices).unwrap())
            .unwrap();

        let mut neighbors = vec![None; D + 1];
        neighbors[D] = Some(negative_simplex_key);
        tds.set_neighbors_by_key(positive_simplex_key, &neighbors)
            .unwrap();
        let mut neighbors = vec![None; D + 1];
        neighbors[D] = Some(positive_simplex_key);
        tds.set_neighbors_by_key(negative_simplex_key, &neighbors)
            .unwrap();

        Triangulation::new_with_tds(FastKernel::new(), tds)
    }

    /// Builds an invalid 2D complex with three triangles sharing the same edge.
    fn overshared_edge_fixture_2d() -> (
        Triangulation<FastKernel<f64>, (), (), 2>,
        VertexKey,
        VertexKey,
    ) {
        let mut tds = Tds::empty();
        let left_endpoint = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let right_endpoint = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let upper_apex = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let lower_apex = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]).unwrap())
            .unwrap();
        let extra_apex = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5]).unwrap())
            .unwrap();

        for apex in [upper_apex, lower_apex, extra_apex] {
            tds.insert_simplex_with_mapping_prechecked_topology(
                Simplex::try_new(vec![left_endpoint, right_endpoint, apex]).unwrap(),
            )
            .unwrap();
        }

        (
            Triangulation::new_with_tds(FastKernel::new(), tds),
            left_endpoint,
            right_endpoint,
        )
    }

    /// Returns the edge count for two D-simplices sharing one facet.
    const fn expected_split_topology_fixture_edges<const D: usize>() -> usize {
        D * (D + 3) / 2
    }

    /// Finds one shared-facet vertex in the split topology fixture.
    fn shared_fixture_vertex<const D: usize>(
        tri: &Triangulation<FastKernel<f64>, (), (), D>,
    ) -> VertexKey {
        fixture_vertex_by_coords(tri, [0.0; D])
    }

    /// Finds one fixture vertex by exact coordinates chosen by the deterministic helper.
    fn fixture_vertex_by_coords<const D: usize>(
        tri: &Triangulation<FastKernel<f64>, (), (), D>,
        coords: [f64; D],
    ) -> VertexKey {
        tri.vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == coords).then_some(vk))
            .unwrap()
    }

    /// Checks that each returned 2D facet handle resolves to the queried edge.
    fn assert_facet_handles_match_edge_2d(
        tri: &Triangulation<FastKernel<f64>, (), (), 2>,
        edge: EdgeKey,
        facets: impl IntoIterator<Item = FacetHandle>,
    ) {
        let (v0, v1) = edge.endpoints();

        for facet in facets {
            let view = facet
                .view(&tri.tds)
                .expect("edge-to-facet query should return live facet handles");
            let facet_index = usize::from(facet.facet_index());
            let mut facet_vertices = view
                .simplex()
                .vertices()
                .iter()
                .enumerate()
                .filter_map(|(index, &vertex_key)| (index != facet_index).then_some(vertex_key));

            assert_matches!(
                (
                    facet_vertices.next(),
                    facet_vertices.next(),
                    facet_vertices.next()
                ),
                (Some(a), Some(b), None) if (a == v0 && b == v1) || (a == v1 && b == v0),
                "facet {facet:?} should describe queried edge {edge:?}"
            );
        }
    }

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
                    assert_eq!(
                        empty
                            .ridges()
                            .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))
                            .unwrap(),
                        0
                    );
                    assert_eq!(
                        empty
                            .ridge_handles()
                            .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))
                            .unwrap(),
                        0
                    );

                    let vertices = vec![
                        $(vertex!($simplex_coords).unwrap()),+
                    ];
                    let expected_vertex_count = vertices.len();
                    let expected_ridge_count = if $dim >= 2 {
                        expected_vertex_count * (expected_vertex_count - 1) / 2
                    } else {
                        0
                    };
                    let expected_ridge_handle_count = if $dim >= 3 {
                        expected_ridge_count
                    } else {
                        0
                    };

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
                    assert_eq!(
                        tri.ridges()
                            .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))
                            .unwrap(),
                        expected_ridge_count
                    );
                    assert_eq!(
                        tri.ridge_handles()
                            .try_fold(0_usize, |count, ridge| ridge.map(|_| count + 1))
                            .unwrap(),
                        expected_ridge_handle_count
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
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
            Err(QueryError::TriangulationCorrupted { source })
                if matches!(*source, TdsError::IndexOutOfBounds { .. }) => {}
            Err(err) => panic!("expected index-out-of-bounds query error, got {err:?}"),
        }
    }

    #[test]
    fn query_error_preserves_tds_provenance_from_manifold_error() {
        let source = TdsError::InconsistentDataStructure {
            message: "facet incidence".to_string(),
        };

        assert_matches!(
            QueryError::from(ManifoldError::Tds(source)),
            QueryError::TriangulationCorrupted { source }
                if matches!(
                    source.as_ref(),
                    TdsError::InconsistentDataStructure { message }
                        if message == "facet incidence"
                )
        );
    }

    #[test]
    fn topology_edges_triangle_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 1);
        assert_eq!(tri.number_of_vertices(), 3);
        assert_eq!(tri.number_of_edges(), 3);

        let edges: HashSet<_> = tri.edges().collect();
        assert_eq!(edges.len(), 3);

        let edge_index = tri.build_edge_index().unwrap();
        let indexed_edges: HashSet<_> = edge_index.edges().collect();
        assert_eq!(indexed_edges, edges);
        assert_eq!(edge_index.number_of_edges(), 3);

        assert!(edges.iter().all(|edge| {
            let (a, b) = edge.endpoints();
            a != b && tri.vertex_coords(a).is_some() && tri.vertex_coords(b).is_some()
        }));
    }

    #[test]
    fn edge_to_facet_queries_classify_2d_boundary_and_interior_edges() {
        let tri = split_topology_fixture::<2>();
        let shared_left = fixture_vertex_by_coords(&tri, [0.0, 0.0]);
        let shared_right = fixture_vertex_by_coords(&tri, [1.0, 0.0]);
        let positive_apex = fixture_vertex_by_coords(&tri, [0.2, 1.0]);

        let interior_edge = EdgeKey::try_new(&tri.tds, shared_left, shared_right).unwrap();
        let incident_facets: HashSet<_> = tri
            .try_incident_facets_to_edge_2d(interior_edge)
            .unwrap()
            .collect();
        let interior_facet = tri
            .try_interior_facet_for_edge_2d(interior_edge)
            .unwrap()
            .expect("shared edge should be interior");

        assert_eq!(incident_facets.len(), 2);
        assert!(incident_facets.contains(&interior_facet));
        assert_facet_handles_match_edge_2d(&tri, interior_edge, incident_facets.iter().copied());
        assert_facet_handles_match_edge_2d(&tri, interior_edge, [interior_facet]);

        let boundary_edge = EdgeKey::try_new(&tri.tds, shared_left, positive_apex).unwrap();
        let boundary_facets: Vec<_> = tri
            .try_incident_facets_to_edge_2d(boundary_edge)
            .unwrap()
            .collect();
        assert_eq!(boundary_facets.len(), 1);
        assert_facet_handles_match_edge_2d(&tri, boundary_edge, boundary_facets);
        assert!(
            tri.try_interior_facet_for_edge_2d(boundary_edge)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn edge_to_facet_queries_expose_non_manifold_2d_multiplicity() {
        let (tri, a, b) = overshared_edge_fixture_2d();
        let overshared_edge = EdgeKey::try_new(&tri.tds, a, b).unwrap();

        let incident_facets: Vec<_> = tri
            .try_incident_facets_to_edge_2d(overshared_edge)
            .unwrap()
            .collect();

        assert_eq!(incident_facets.len(), 3);
        assert_facet_handles_match_edge_2d(&tri, overshared_edge, incident_facets);
        assert!(
            tri.try_interior_facet_for_edge_2d(overshared_edge)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn edge_to_facet_query_preserves_incidence_parse_errors() {
        let mut tri = split_topology_fixture::<2>();
        let shared_left = fixture_vertex_by_coords(&tri, [0.0, 0.0]);
        let shared_right = fixture_vertex_by_coords(&tri, [1.0, 0.0]);
        let edge = EdgeKey::try_new(&tri.tds, shared_left, shared_right).unwrap();
        let (first, _) = edge.endpoints();

        tri.tds.clear_vertex_incidence_for_test(first);

        match tri.try_incident_facets_to_edge_2d(edge) {
            Err(EdgeKeyError::MissingVertexIncidence { vertex_key, .. }) if vertex_key == first => {
            }
            Err(err) => panic!("expected missing vertex incidence for {first:?}, got {err:?}"),
            Ok(_) => panic!("expected missing vertex incidence for {first:?}, got facets"),
        }
        assert_matches!(
            tri.try_interior_facet_for_edge_2d(edge),
            Err(EdgeKeyError::MissingVertexIncidence { vertex_key, .. })
                if vertex_key == first
        );
    }

    #[test]
    fn topology_edges_and_incident_edges_double_tetrahedron_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 2.0, 0.0]).unwrap(),
            vertex!([1.0, 0.7, 1.5]).unwrap(),
            vertex!([1.0, 0.7, -1.5]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 2);
        assert_eq!(tri.number_of_vertices(), 5);
        assert_eq!(tri.number_of_edges(), 9);

        let base_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);

        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();
        assert_eq!(edge_index.number_of_edges(), 9);
        assert_eq!(tri.adjacent_simplices(base_vertex_key).count(), 2);
        assert_eq!(incidence.adjacent_simplices(base_vertex_key).count(), 2);
        assert_eq!(incidence.number_of_adjacent_simplices(base_vertex_key), 2);
        assert_eq!(edge_index.incident_edges(base_vertex_key).count(), 4);
        assert_eq!(edge_index.number_of_incident_edges(base_vertex_key), 4);

        let apex_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.7, 1.5]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(apex_vertex_key), 3);
        assert_eq!(incidence.adjacent_simplices(apex_vertex_key).count(), 1);
        assert_eq!(incidence.number_of_adjacent_simplices(apex_vertex_key), 1);

        for (simplex_key, _) in tri.simplices() {
            assert_eq!(neighbor_index.simplex_neighbors(simplex_key).count(), 1);
            assert_eq!(neighbor_index.number_of_simplex_neighbors(simplex_key), 1);
        }
    }

    fn assert_split_topology_indexes_cover_independent_query_families<const D: usize>() {
        let tri = split_topology_fixture::<D>();
        let base_vertex_key = shared_fixture_vertex(&tri);
        let expected_edges = expected_split_topology_fixture_edges::<D>();

        let incidence = tri.incidence().unwrap();
        assert_eq!(incidence.number_of_adjacent_simplices(base_vertex_key), 2);
        assert_eq!(incidence.adjacent_simplices(base_vertex_key).count(), 2);

        let edge_index = tri.build_edge_index().unwrap();
        assert_eq!(edge_index.number_of_edges(), expected_edges);
        assert_eq!(edge_index.number_of_incident_edges(base_vertex_key), D + 1);
        assert_eq!(
            edge_index.edges().collect::<HashSet<_>>().len(),
            expected_edges
        );

        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();
        for (simplex_key, _) in tri.simplices() {
            assert_eq!(neighbor_index.number_of_simplex_neighbors(simplex_key), 1);
            assert_eq!(neighbor_index.simplex_neighbors(simplex_key).count(), 1);
        }
    }

    #[test]
    fn topology_queries_missing_keys_are_empty_or_none() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tri = dt.as_triangulation();
        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();

        let missing_vertex_key = VertexKey::default();
        assert_eq!(tri.adjacent_simplices(missing_vertex_key).count(), 0);
        assert_eq!(incidence.adjacent_simplices(missing_vertex_key).count(), 0);
        assert_eq!(
            incidence.number_of_adjacent_simplices(missing_vertex_key),
            0
        );
        assert_eq!(tri.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(edge_index.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(tri.number_of_incident_edges(missing_vertex_key), 0);
        assert_eq!(edge_index.number_of_incident_edges(missing_vertex_key), 0);
        assert!(tri.vertex_coords(missing_vertex_key).is_none());

        let missing_simplex_key = SimplexKey::default();
        assert_eq!(tri.simplex_neighbors(missing_simplex_key).count(), 0);
        assert_eq!(
            neighbor_index
                .simplex_neighbors(missing_simplex_key)
                .count(),
            0
        );
        assert_eq!(
            neighbor_index.number_of_simplex_neighbors(missing_simplex_key),
            0
        );
        assert_matches!(
            tri.simplex_vertices(missing_simplex_key),
            Err(TdsError::SimplexNotFound { .. })
        );
    }

    #[test]
    fn topology_geometry_accessors_round_trip() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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

    fn assert_split_topology_indexes_basic_invariants<const D: usize>() {
        let tri = split_topology_fixture::<D>();
        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();

        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        assert_eq!(simplex_keys.len(), 2);
        for &simplex_key in &simplex_keys {
            let neighbors = neighbor_index
                .simplex_to_neighbors
                .get(&simplex_key)
                .unwrap();
            assert_eq!(neighbors.len(), 1);
            assert!(simplex_keys.contains(&neighbors[0]));
            assert_ne!(neighbors[0], simplex_key);
        }

        for (vertex_key, _) in tri.vertices() {
            assert!(incidence.vertex_to_simplices.contains_vertex(vertex_key));
            assert!(
                incidence
                    .vertex_to_simplices
                    .number_of_simplices(vertex_key)
                    > 0
            );

            let edges = edge_index.vertex_to_edges.get(&vertex_key).unwrap();
            assert!(!edges.is_empty());
            assert!(edges.iter().all(
                |edge| matches!(edge.endpoints(), (a, b) if a == vertex_key || b == vertex_key)
            ));
        }
    }

    fn assert_split_topology_indexes_match_direct_queries<const D: usize>() {
        let tri = split_topology_fixture::<D>();
        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();

        let idx_edges: HashSet<_> = edge_index.edges().collect();
        let direct_edges: HashSet<_> = tri.edges().collect();
        assert_eq!(idx_edges, direct_edges);
        assert_eq!(edge_index.number_of_edges(), tri.number_of_edges());

        let vertex_key = tri.vertices().next().unwrap().0;
        let idx_adj: HashSet<_> = incidence.adjacent_simplices(vertex_key).collect();
        let direct_adj: HashSet<_> = tri.adjacent_simplices(vertex_key).collect();
        assert_eq!(idx_adj, direct_adj);
        assert_eq!(
            incidence.number_of_adjacent_simplices(vertex_key),
            direct_adj.len()
        );

        let simplex_key = tri.simplices().next().unwrap().0;
        let direct_neighbors: Vec<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(
            neighbor_index.simplex_neighbors(simplex_key).count(),
            direct_neighbors.len()
        );
        assert_eq!(
            neighbor_index.number_of_simplex_neighbors(simplex_key),
            direct_neighbors.len()
        );

        let idx_incident: HashSet<_> = edge_index.incident_edges(vertex_key).collect();
        let direct_incident: HashSet<_> = tri.incident_edges(vertex_key).collect();
        assert_eq!(idx_incident, direct_incident);
        assert_eq!(
            edge_index.number_of_incident_edges(vertex_key),
            direct_incident.len()
        );
    }

    fn assert_triangulation_adjacency_view_methods_match_direct_queries<const D: usize>() {
        let tri = split_topology_fixture::<D>();
        let adjacency = tri.adjacency().unwrap();

        let view_edges: HashSet<_> = adjacency.edges().collect();
        let direct_edges: HashSet<_> = tri.edges().collect();
        assert_eq!(view_edges, direct_edges);
        assert_eq!(adjacency.number_of_edges(), direct_edges.len());

        let vertex_key = tri.vertices().next().unwrap().0;
        let view_adjacent: HashSet<_> = adjacency.adjacent_simplices(vertex_key).collect();
        let direct_adjacent: HashSet<_> = tri.adjacent_simplices(vertex_key).collect();
        assert_eq!(view_adjacent, direct_adjacent);
        assert_eq!(
            adjacency.number_of_adjacent_simplices(vertex_key),
            direct_adjacent.len()
        );

        let view_incident: HashSet<_> = adjacency.incident_edges(vertex_key).collect();
        let direct_incident: HashSet<_> = tri.incident_edges(vertex_key).collect();
        assert_eq!(view_incident, direct_incident);
        assert_eq!(
            adjacency.number_of_incident_edges(vertex_key),
            direct_incident.len()
        );

        let simplex_key = tri.simplices().next().unwrap().0;
        let view_neighbors: HashSet<_> = adjacency.simplex_neighbors(simplex_key).collect();
        let direct_neighbors: HashSet<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(view_neighbors, direct_neighbors);
        assert_eq!(
            adjacency.number_of_simplex_neighbors(simplex_key),
            direct_neighbors.len()
        );
    }

    macro_rules! gen_split_topology_index_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<split_topology_indexes_cover_independent_query_families_ $dim d>]() {
                    assert_split_topology_indexes_cover_independent_query_families::<$dim>();
                }

                #[test]
                fn [<split_topology_indexes_basic_invariants_ $dim d>]() {
                    assert_split_topology_indexes_basic_invariants::<$dim>();
                }

                #[test]
                fn [<split_topology_indexes_match_direct_queries_ $dim d>]() {
                    assert_split_topology_indexes_match_direct_queries::<$dim>();
                }

                #[test]
                fn [<triangulation_adjacency_view_methods_match_direct_queries_ $dim d>]() {
                    assert_triangulation_adjacency_view_methods_match_direct_queries::<$dim>();
                }
            }
        };
    }

    gen_split_topology_index_tests!(2);
    gen_split_topology_index_tests!(3);
    gen_split_topology_index_tests!(4);
    gen_split_topology_index_tests!(5);

    #[test]
    fn split_topology_indexes_empty_triangulation_is_empty() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();
        let neighbor_index = tri.build_simplex_neighbor_index().unwrap();
        assert!(incidence.vertex_to_simplices.is_empty());
        assert!(neighbor_index.simplex_to_neighbors.is_empty());
        assert!(edge_index.vertex_to_edges.is_empty());
    }

    #[test]
    fn split_topology_indexes_include_isolated_vertex_entries() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let isolated_vertex = tri
            .tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]).unwrap())
            .unwrap();
        let incidence = tri.incidence().unwrap();
        let edge_index = tri.build_edge_index().unwrap();

        assert!(
            incidence
                .vertex_to_simplices
                .contains_vertex(isolated_vertex)
        );
        assert_eq!(
            incidence
                .vertex_to_simplices
                .number_of_simplices(isolated_vertex),
            0
        );
        assert!(
            edge_index
                .vertex_to_edges
                .get(&isolated_vertex)
                .is_some_and(SmallBuffer::is_empty)
        );
    }

    #[test]
    fn incidence_errors_on_stale_vertex_incidence_index() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let vertex_key = tri.tds.vertex_keys().next().unwrap();

        tri.tds.clear_vertex_incidence_for_test(vertex_key);

        match tri.incidence() {
            Err(TopologyIndexBuildError::InvalidVertexIncidenceIndex { source }) => {
                assert_matches!(source, TdsError::InconsistentDataStructure { .. });
            }
            other => panic!("Expected InvalidVertexIncidenceIndex, got {other:?}"),
        }
    }

    #[test]
    fn simplex_neighbor_index_errors_on_missing_neighbor_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
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

        match tri.build_simplex_neighbor_index() {
            Err(TopologyIndexBuildError::MissingNeighborSimplex {
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
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
    fn edge_index_errors_on_missing_vertex_key() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
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

        match tri.build_edge_index() {
            Err(TopologyIndexBuildError::MissingVertexKey {
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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tri = dt.as_triangulation();

        let edge_count = tri.number_of_edges();
        let edges_collected: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_collected.len(), edge_count);
        assert!(edge_count >= 6);

        assert!(tri.facets().next().transpose().unwrap().is_some());
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

        assert_matches!(
            tri.simplex_vertices(SimplexKey::from(KeyData::from_ffi(0xDEAD))),
            Err(TdsError::SimplexNotFound { .. })
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
}
