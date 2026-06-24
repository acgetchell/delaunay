//! Read-only Delaunay triangulation query, traversal, and accessor methods.
//!
//! This module owns the high-level forwarding surface for inspecting a
//! `DelaunayTriangulation`: counts, iterator access, TDS views, topology
//! metadata accessors, and adjacency traversal helpers. It also keeps the small
//! cache invalidation helpers next to the accessors they protect.

#![forbid(unsafe_code)]

use crate::core::adjacency::{
    EdgeIndex, IncidenceView, SimplexNeighborIndex, TopologyIndexBuildError, TriangulationAdjacency,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::query::QueryError;
use crate::core::simplex::Simplex;
use crate::core::tds::{InvariantError, SimplexKey, Tds, TdsError, TdsMutationError, VertexKey};
use crate::core::triangulation::Triangulation;
use crate::core::validation::{TopologyGuarantee, ValidationConfigurationError, ValidationPolicy};
use crate::core::vertex::Vertex;
use crate::repair::{DelaunayCheckPolicy, DelaunayRepairPolicy};
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::DelaunayTriangulation;
use crate::validation::DelaunayTriangulationValidationError;

// =============================================================================
// QUERY, ACCESSORS, AND CONFIGURATION (Minimal Bounds)
// =============================================================================
//
impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D> {
    // -------------------------------------------------------------------------
    // QUERY / ACCESSORS
    // -------------------------------------------------------------------------

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![0.2, 0.2, 0.2, 0.2]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tri.number_of_vertices()
    }

    /// Returns the number of simplices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// // One 4-simplex in 4D
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.tri.number_of_simplices()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// Returns the dimension `D` as an `i32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.dim(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tri.dim()
    }

    /// Returns an iterator over all simplices in the triangulation.
    ///
    /// This method provides access to the simplices stored in the underlying
    /// triangulation data structure. The iterator yields `(SimplexKey, &Simplex)`
    /// pairs for each simplex in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(SimplexKey, &Simplex<V, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// for (simplex_key, simplex) in dt.simplices() {
    ///     println!("Simplex {:?} has {} vertices", simplex_key, simplex.number_of_vertices());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<V, D>)> {
        self.tri.tds.simplices()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// This method provides access to the vertices stored in the underlying
    /// triangulation data structure. The iterator yields `(VertexKey, &Vertex)`
    /// pairs for each vertex in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(VertexKey, &Vertex<U, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<U, D>)> {
        self.tri.vertices()
    }

    /// Sets the auxiliary data on a vertex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants, so
    /// no caches are invalidated.
    ///
    /// # Returns
    ///
    /// The old `Option<U>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a vertex in the
    /// underlying TDS.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: [Vertex<i32, 2>; 3] = [
    ///     delaunay::vertex![0.0, 0.0; data = 10i32]?,
    ///     delaunay::vertex![1.0, 0.0; data = 20]?,
    ///     delaunay::vertex![0.0, 1.0; data = 30]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((key, _)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let prev = dt.set_vertex_data(key, Some(99))?;
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None)?;
    /// assert_eq!(prev, Some(99));
    /// assert_eq!(dt.tds().vertex(key).map(|v| v.data()), Some(None));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_vertex_data(
        &mut self,
        key: VertexKey,
        data: Option<U>,
    ) -> Result<Option<U>, TdsMutationError> {
        self.tri.tds.set_vertex_data(key, data)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants, so
    /// no caches are invalidated.
    ///
    /// # Returns
    ///
    /// The old `Option<V>` value when the key exists.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`] if `key` does not identify a simplex in
    /// the underlying TDS.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<i32>()?;
    /// let Some((key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let prev = dt.set_simplex_data(key, Some(42))?;
    /// assert_eq!(prev, None);
    ///
    /// // Clear data
    /// let prev = dt.set_simplex_data(key, None)?;
    /// assert_eq!(prev, Some(42));
    /// assert_eq!(dt.tds().simplex(key).map(|s| s.data()), Some(None));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_simplex_data(
        &mut self,
        key: SimplexKey,
        data: Option<V>,
    ) -> Result<Option<V>, TdsMutationError> {
        self.tri.tds.set_simplex_data(key, data)
    }

    /// Returns a reference to the underlying triangulation data structure.
    ///
    /// This provides access to the purely combinatorial Tds layer for
    /// advanced operations and performance testing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
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
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// assert_eq!(tds.number_of_vertices(), 5);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn tds(&self) -> &Tds<U, V, D> {
        &self.tri.tds
    }

    /// Returns a mutable reference to the underlying triangulation data structure.
    ///
    /// This provides mutable access to the purely combinatorial Tds layer for
    /// advanced operations and testing of internal algorithms.
    ///
    /// # Safety
    ///
    /// Modifying the Tds directly can break Delaunay invariants. Use this only
    /// when you know what you're doing (typically in tests or specialized algorithms).
    #[cfg(test)]
    pub(crate) fn tds_mut(&mut self) -> &mut Tds<U, V, D> {
        // Direct mutable access can invalidate performance caches.
        self.invalidate_repair_caches();
        &mut self.tri.tds
    }

    pub(crate) const fn invalidate_locate_hint_cache(&mut self) {
        self.insertion_state.last_inserted_simplex = None;
    }

    pub(crate) fn invalidate_repair_caches(&mut self) {
        self.invalidate_locate_hint_cache();
        self.spatial_index = None;
    }

    /// Returns mutable TDS access for crate-internal repair algorithms.
    ///
    /// Repair passes may rewrite topology and invalidate locate hints, so this
    /// deliberately clears the ephemeral caches before handing out the borrow.
    pub(crate) fn tds_mut_for_repair(&mut self) -> &mut Tds<U, V, D> {
        self.invalidate_repair_caches();
        &mut self.tri.tds
    }

    /// Returns a reference to the underlying `Triangulation` (kernel + tds).
    ///
    /// This is useful when you need to pass the triangulation to methods that
    /// expect a `&Triangulation`, such as
    /// [`ConvexHull::try_from_triangulation`](crate::geometry::algorithms::convex_hull::ConvexHull::try_from_triangulation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::ConvexHull;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Hull(#[from] delaunay::query::ConvexHullConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: Vec<_> = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())?;
    /// assert_eq!(hull.number_of_facets(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn as_triangulation(&self) -> &Triangulation<K, U, V, D> {
        &self.tri
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
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
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
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let boundary_count = dt
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
    /// preserves the lower-level [`TdsError`] for diagnostics. Returns
    /// [`QueryError::TopologyInvalid`] when topology-aware boundary
    /// classification rejects the declared global topology or detects another
    /// manifold-boundary inconsistency.
    /// Individual iterator items return [`FacetError`](crate::prelude::tds::FacetError)
    /// if a boundary facet handle cannot be reborrowed as a view.
    pub fn boundary_facets(&self) -> Result<BoundaryFacetsIter<'_, U, V, D>, QueryError> {
        self.tri.boundary_facets()
    }

    /// Returns the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This policy controls when Level 3 (`Triangulation::is_valid()`) is run automatically
    /// during incremental insertion (as part of the topology safety net).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::validation::ValidationPolicy;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.tri.validation_policy
    }

    /// Tries to set the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This affects subsequent incremental insertions. (Construction-time behavior is determined
    /// by the policy active during `new()` / `with_kernel()`.)
    ///
    /// # Errors
    ///
    /// Returns [`ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy`] when the
    /// requested policy is incompatible with the current topology guarantee.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulation;
    /// use delaunay::prelude::validation::{
    ///     ValidationConfigurationError, ValidationPolicy,
    /// };
    ///
    /// # fn main() -> Result<(), ValidationConfigurationError> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
    ///
    /// dt.try_set_validation_policy(ValidationPolicy::Always)?;
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     ValidationPolicy::Always
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_set_validation_policy(
        &mut self,
        policy: ValidationPolicy,
    ) -> Result<(), ValidationConfigurationError> {
        self.tri.try_set_validation_policy(policy)
    }

    /// Sets the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// Prefer [`try_set_validation_policy`](Self::try_set_validation_policy) when callers need
    /// typed feedback for rejected combinations. This compatibility setter leaves the existing
    /// policy unchanged and emits a warning if the requested combination is incoherent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulation;
    /// use delaunay::prelude::validation::ValidationPolicy;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
    ///
    /// dt.set_validation_policy(ValidationPolicy::Always);
    /// assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        self.tri.set_validation_policy(policy);
    }

    /// Tries to set the topology guarantee used for Level 3 topology validation.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy`] when the
    /// requested guarantee cannot be represented with the current validation policy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulation;
    /// use delaunay::prelude::validation::{
    ///     TopologyGuarantee, ValidationConfigurationError,
    /// };
    ///
    /// # fn main() -> Result<(), ValidationConfigurationError> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// dt.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)?;
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_set_topology_guarantee(
        &mut self,
        guarantee: TopologyGuarantee,
    ) -> Result<(), ValidationConfigurationError> {
        self.tri.try_set_topology_guarantee(guarantee)
    }

    /// Returns the automatic Delaunay repair policy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::repair::DelaunayRepairPolicy;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert_eq!(dt.delaunay_repair_policy(), DelaunayRepairPolicy::EveryInsertion);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn delaunay_repair_policy(&self) -> DelaunayRepairPolicy {
        self.insertion_state.delaunay_repair_policy
    }

    /// Sets the automatic Delaunay repair policy.
    ///
    /// This affects future incremental insertions; it does not rewrite already
    /// stored topology.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::repair::DelaunayRepairPolicy;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
    /// assert_eq!(dt.delaunay_repair_policy(), DelaunayRepairPolicy::Never);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn set_delaunay_repair_policy(&mut self, policy: DelaunayRepairPolicy) {
        self.insertion_state.delaunay_repair_policy = policy;
    }

    /// Returns the automatic global Delaunay validation policy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::repair::DelaunayCheckPolicy;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert_eq!(dt.delaunay_check_policy(), DelaunayCheckPolicy::EndOnly);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn delaunay_check_policy(&self) -> DelaunayCheckPolicy {
        self.insertion_state.delaunay_check_policy
    }

    /// Sets the automatic global Delaunay validation policy.
    ///
    /// This affects future incremental insertions; it does not perform an
    /// immediate global Delaunay check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::repair::DelaunayCheckPolicy;
    /// use std::num::NonZeroUsize;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some(every_two) = NonZeroUsize::new(2) else {
    ///     return Ok(());
    /// };
    ///
    /// dt.set_delaunay_check_policy(DelaunayCheckPolicy::EveryN(every_two));
    /// assert_eq!(dt.delaunay_check_policy(), DelaunayCheckPolicy::EveryN(every_two));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn set_delaunay_check_policy(&mut self, policy: DelaunayCheckPolicy) {
        self.insertion_state.delaunay_check_policy = policy;
    }
}

// =============================================================================
// CONFIGURATION & TRAVERSAL (Minimal Bounds, continued)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D> {
    // -------------------------------------------------------------------------
    // CONFIGURATION
    // -------------------------------------------------------------------------

    /// Returns the topology guarantee used for Level 3 topology validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.tri.topology_guarantee()
    }

    /// Returns runtime global topology metadata associated with this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, GlobalTopology,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert!(dt.global_topology().is_euclidean());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        self.tri.global_topology()
    }

    /// Returns the high-level topology kind (`Euclidean`, `Toroidal`, etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyKind,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![delaunay::vertex![0.0, 0.0]?, delaunay::vertex![1.0, 0.0]?, delaunay::vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.topology_kind(), TopologyKind::Euclidean);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        self.tri.topology_kind()
    }

    /// Sets runtime global topology metadata after validating it against current topology.
    ///
    /// The update is atomic: if the current triangulation does not satisfy the
    /// requested global topology, the previous metadata is restored before the
    /// error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationValidationError::Tds`] if lower-level
    /// structure is invalid while checking topology, or
    /// [`DelaunayTriangulationValidationError::Triangulation`] when Level 3
    /// topology violates the requested metadata, for example when Euclidean
    /// boundary facets are relabeled as closed spherical or toroidal topology.
    /// The previous topology metadata is restored before the error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, GlobalTopology,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// dt.try_set_global_topology(GlobalTopology::Euclidean)?;
    /// assert!(dt.global_topology().is_euclidean());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_set_global_topology(
        &mut self,
        global_topology: GlobalTopology<D>,
    ) -> Result<(), DelaunayTriangulationValidationError> {
        match self.tri.try_set_global_topology(global_topology) {
            Ok(()) => Ok(()),
            Err(InvariantError::Tds(err)) => Err(err.into()),
            Err(InvariantError::Triangulation(err)) => Err(err.into()),
            Err(InvariantError::Delaunay(err)) => Err(err),
        }
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
    ///
    /// Prefer [`try_set_topology_guarantee`](Self::try_set_topology_guarantee) when callers need
    /// typed feedback for rejected combinations. This compatibility setter leaves the existing
    /// guarantee unchanged and emits a warning if the requested combination is incoherent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulation, TopologyGuarantee,
    /// };
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[inline]
    pub fn set_topology_guarantee(&mut self, guarantee: TopologyGuarantee) {
        self.tri.set_topology_guarantee(guarantee);
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// Delegates to the underlying `Triangulation` layer. This provides
    /// efficient access to all facets without pre-allocating a vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding `Result<FacetView, FacetError>` items for all facets.
    ///
    /// Individual iterator items return
    /// [`FacetError`](crate::prelude::tds::FacetError) if a facet view cannot be
    /// constructed from the current TDS state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
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
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let facet_count = dt
    ///     .facets()
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn facets(&self) -> AllFacetsIter<'_, U, V, D> {
        self.tri.facets()
    }

    /// Builds a lifetime-bound adjacency view for fast repeated topology queries.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::adjacency`](crate::Triangulation::adjacency). Prefer
    /// the narrower [`DelaunayTriangulation::incidence`](Self::incidence),
    /// [`DelaunayTriangulation::build_edge_index`](Self::build_edge_index), or
    /// [`DelaunayTriangulation::build_simplex_neighbor_index`](Self::build_simplex_neighbor_index)
    /// when a caller only needs one topology query family.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying triangulation data structure is internally inconsistent
    /// (e.g., a simplex references a missing vertex key or a missing neighbor simplex key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Adjacency(#[from] delaunay::query::TopologyIndexBuildError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let adjacency = dt.adjacency()?;
    ///
    /// assert_eq!(adjacency.number_of_edges(), 6);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn adjacency(&self) -> Result<TriangulationAdjacency<'_>, TopologyIndexBuildError> {
        self.as_triangulation().adjacency()
    }

    /// Borrows the canonical vertex→simplices incidence relation.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incidence`](crate::Triangulation::incidence).
    ///
    /// # Errors
    ///
    /// Returns an error if the maintained incidence relation is internally inconsistent.
    #[inline]
    pub fn incidence(&self) -> Result<IncidenceView<'_>, TopologyIndexBuildError> {
        self.as_triangulation().incidence()
    }

    /// Builds only the derived vertex→edge index for this triangulation snapshot.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::build_edge_index`](crate::Triangulation::build_edge_index).
    ///
    /// # Errors
    ///
    /// Returns an error if a simplex references a missing vertex key.
    #[inline]
    pub fn build_edge_index(&self) -> Result<EdgeIndex<'_>, TopologyIndexBuildError> {
        self.as_triangulation().build_edge_index()
    }

    /// Builds only the derived simplex→neighbor index for this triangulation snapshot.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::build_simplex_neighbor_index`](crate::Triangulation::build_simplex_neighbor_index).
    ///
    /// # Errors
    ///
    /// Returns an error if a simplex references a missing neighbor key.
    #[inline]
    pub fn build_simplex_neighbor_index(
        &self,
    ) -> Result<SimplexNeighborIndex<'_>, TopologyIndexBuildError> {
        self.as_triangulation().build_simplex_neighbor_index()
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges`](crate::Triangulation::edges).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let edges: std::collections::HashSet<_> = dt.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// # Ok(())
    /// # }
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().edges()
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges`](crate::Triangulation::incident_edges).
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((v0, _)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // In a tetrahedron, each vertex has degree 3.
    /// assert_eq!(dt.incident_edges(v0).count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().incident_edges(v)
    }

    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::simplex_neighbors`](crate::Triangulation::simplex_neighbors).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// // A single tetrahedron has no simplex neighbors.
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
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
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(dt.simplex_neighbors(simplex_key).count(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.as_triangulation().simplex_neighbors(c)
    }

    /// Returns a slice view of a simplex's vertex keys.
    ///
    /// This is a zero-allocation accessor that validates the simplex key and
    /// referenced vertex keys before lending the canonical slice.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::simplex_vertices`](crate::Triangulation::simplex_vertices).
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if `c` does not identify a simplex in this
    /// triangulation, or if the simplex references a missing vertex key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let simplex_vertices = dt.simplex_vertices(simplex_key)?;
    /// assert_eq!(simplex_vertices.len(), 3); // D+1 for a 2D simplex
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_vertices(&self, c: SimplexKey) -> Result<&[VertexKey], TdsError> {
        self.as_triangulation().simplex_vertices(c)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::vertex_coords`](crate::Triangulation::vertex_coords).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::query::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let Some(v_key) = dt
    ///     .vertices()
    ///     .find_map(|(vk, _)| (dt.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    /// else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(dt.vertex_coords(v_key), Some([1.0, 0.0].as_slice()));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[f64]> {
        self.as_triangulation().vertex_coords(v)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::DelaunayInsertionState;
    use crate::core::tds::TdsError;
    use crate::core::validation::TriangulationValidationError;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::vertex;
    use std::{assert_matches, collections::HashSet, num::NonZeroUsize, sync::Once};

    struct Payload;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    #[test]
    fn test_delaunay_constructors_default_to_pl_manifold_mode() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        assert_eq!(dt_new.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();

        assert_eq!(
            dt_with_kernel.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );

        let dt_from_tds: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_from_tds(dt_new.tds().clone(), FastKernel::new()).unwrap();
        assert_eq!(
            dt_from_tds.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
    }

    #[test]
    fn test_set_topology_guarantee_updates_underlying_triangulation() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
        assert_eq!(dt.tri.topology_guarantee, TopologyGuarantee::PLManifold);

        dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
        assert_eq!(dt.tri.topology_guarantee, TopologyGuarantee::Pseudomanifold);
    }

    #[test]
    fn test_try_global_topology_setter_rejects_closed_metadata_for_euclidean_boundary() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let err = dt
            .try_set_global_topology(GlobalTopology::Spherical)
            .unwrap_err();

        assert_matches!(
            err,
            DelaunayTriangulationValidationError::Triangulation(source)
                if matches!(
                    &*source,
                    TriangulationValidationError::BoundaryFacetInClosedTopology {
                        topology: TopologyKind::Spherical,
                        ..
                    }
                )
        );
        assert_eq!(dt.global_topology(), GlobalTopology::Euclidean);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_set_delaunay_check_policy_updates_state() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.delaunay_check_policy(), DelaunayCheckPolicy::EndOnly);

        let policy = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(3).unwrap());
        dt.set_delaunay_check_policy(policy);
        assert_eq!(dt.delaunay_check_policy(), policy);
    }

    #[test]
    fn test_boundary_facets_propagates_core_query_error() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let (simplex_key, _) = dt.tri.tds.simplices().next().unwrap();
        let first_vertex = dt.tri.tds.simplex(simplex_key).unwrap().vertices()[0];

        {
            let simplex = dt.tri.tds.simplex_mut(simplex_key).unwrap();
            while simplex.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                simplex.push_vertex_key(first_vertex);
            }
        }

        match dt.boundary_facets() {
            Ok(_) => panic!("corrupted facet map should return a query error"),
            Err(QueryError::TriangulationCorrupted { source })
                if matches!(*source, TdsError::IndexOutOfBounds { .. }) => {}
            Err(err) => panic!("expected index-out-of-bounds query error, got {err:?}"),
        }
    }

    #[test]
    fn test_boundary_facets_accepts_non_datatype_payloads() {
        let dt: DelaunayTriangulation<FastKernel<f64>, Payload, Payload, 2> =
            DelaunayTriangulation {
                tri: Triangulation::new_empty(FastKernel::new()),
                insertion_state: DelaunayInsertionState::new(),
                spatial_index: None,
            };

        assert_eq!(
            dt.boundary_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_validation_policy_defaults_to_topology_guarantee_policy() {
        init_tracing();
        // empty() -> Triangulation::new_empty() -> TopologyGuarantee::DEFAULT policy.
        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.validation_policy(), ValidationPolicy::ExplicitOnly);

        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        // new() -> with_kernel() -> explicit validation_policy initialization
        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        assert_eq!(dt_new.validation_policy(), ValidationPolicy::ExplicitOnly);

        // with_kernel() constructor path should also use the default policy
        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();
        assert_eq!(
            dt_with_kernel.validation_policy(),
            ValidationPolicy::ExplicitOnly
        );

        // try_from_tds() is a separate reconstruction path and should also
        // default to the topology guarantee policy after validation succeeds.
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let dt_from_tds: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_from_tds(tds, FastKernel::new()).unwrap();
        assert_eq!(
            dt_from_tds.validation_policy(),
            ValidationPolicy::ExplicitOnly
        );
    }

    #[test]
    fn test_validation_policy_setter_and_getter_roundtrip() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Getter reflects the underlying Triangulation policy.
        assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::ExplicitOnly);

        dt.set_validation_policy(ValidationPolicy::Always);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Always);

        dt.try_set_validation_policy(ValidationPolicy::ExplicitOnly)
            .unwrap();
        assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::ExplicitOnly);

        dt.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::ExplicitOnly);

        assert_eq!(
            dt.try_set_validation_policy(ValidationPolicy::Never),
            Err(
                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                    topology_guarantee: TopologyGuarantee::PLManifold,
                    validation_policy: ValidationPolicy::Never,
                }
            )
        );

        dt.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .unwrap();
        dt.try_set_validation_policy(ValidationPolicy::Never)
            .unwrap();
        assert_eq!(dt.validation_policy(), ValidationPolicy::Never);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Never);

        dt.set_validation_policy(ValidationPolicy::OnSuspicion);
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);
    }

    #[test]
    fn test_number_of_vertices_minimal_simplex() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
    }

    #[test]
    fn test_number_of_simplices_minimal_simplex() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Minimal 3D simplex has exactly 1 tetrahedron
        assert_eq!(dt.number_of_simplices(), 1);
    }

    #[test]
    fn test_number_of_simplices_after_insertion() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        assert_eq!(dt.number_of_simplices(), 1);

        // Insert interior point - should create 3 triangles
        dt.insert_vertex(vertex![0.3, 0.3].unwrap()).unwrap();
        assert_eq!(dt.number_of_simplices(), 3);
    }

    #[test]
    fn test_dim_returns_correct_dimension() {
        init_tracing();
        let vertices_2d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.dim(), 2);

        let vertices_3d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.dim(), 3);

        let vertices_4d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt_4d: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::try_new(&vertices_4d).unwrap();
        assert_eq!(dt_4d.dim(), 4);
    }

    #[test]
    fn test_new_with_exact_minimum_vertices() {
        init_tracing();
        // 2D: exactly 3 vertices (minimum for 2D simplex)
        let vertices_2d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.number_of_vertices(), 3);
        assert_eq!(dt_2d.number_of_simplices(), 1);

        // 3D: exactly 4 vertices (minimum for 3D simplex)
        let vertices_3d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.number_of_vertices(), 4);
        assert_eq!(dt_3d.number_of_simplices(), 1);
    }

    #[test]
    fn test_tds_accessor_provides_readonly_access() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Access TDS via immutable reference
        let tds = dt.tds();
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_simplices(), 1);

        // Verify we can call other TDS methods
        assert!(tds.is_valid().is_ok());
        assert!(tds.simplex_keys().next().is_some());
    }

    #[test]
    fn test_internal_tds_access() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);

        // Internal code can access TDS directly for mutations
        let tds = &mut dt.tri.tds;
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_simplices(), 1);

        // Can call mutating methods like remove_duplicate_simplices
        let result = tds.remove_duplicate_simplices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tds_accessor_reflects_insertions() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Before insertion
        assert_eq!(dt.tds().number_of_vertices(), 3);

        // Insert a new vertex
        dt.insert_vertex(vertex![0.3, 0.3].unwrap()).unwrap();

        // After insertion, TDS accessor reflects the change
        assert_eq!(dt.tds().number_of_vertices(), 4);
        assert!(dt.tds().number_of_simplices() > 1);
    }

    #[test]
    fn test_tds_accessors_maintain_validation_invariants() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Verify TDS is valid through accessor
        assert!(dt.tds().is_valid().is_ok());

        // Insert additional vertex
        dt.insert_vertex(vertex![0.2, 0.2, 0.2, 0.2].unwrap())
            .unwrap();

        // TDS should still be valid after mutation
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate().is_ok());
    }

    #[test]
    fn test_topology_traversal_methods_are_forwarded() {
        init_tracing();
        // Single tetrahedron: 4 vertices, 1 simplex, 6 unique edges.
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let edges_dt: HashSet<_> = dt.edges().collect();
        let edges_tri: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_dt, edges_tri);
        assert_eq!(edges_dt.len(), 6);

        let edge_index_dt = dt.build_edge_index().unwrap();
        let edge_index_tri = tri.build_edge_index().unwrap();
        let edges_dt_index: HashSet<_> = edge_index_dt.edges().collect();
        let edges_tri_index: HashSet<_> = edge_index_tri.edges().collect();
        assert_eq!(edges_dt_index, edges_tri_index);
        assert_eq!(edges_dt_index, edges_dt);

        let adjacency_dt = dt.adjacency().unwrap();
        let adjacency_tri = tri.adjacency().unwrap();
        let edges_dt_view: HashSet<_> = adjacency_dt.edges().collect();
        let edges_tri_view: HashSet<_> = adjacency_tri.edges().collect();
        assert_eq!(edges_dt_view, edges_tri_view);
        assert_eq!(edges_dt_view, edges_dt);
        assert_eq!(adjacency_dt.number_of_edges(), edges_dt.len());

        let v0 = dt.vertices().next().unwrap().0;
        let incident_dt: HashSet<_> = dt.incident_edges(v0).collect();
        let incident_tri: HashSet<_> = tri.incident_edges(v0).collect();
        assert_eq!(incident_dt, incident_tri);
        assert_eq!(incident_dt.len(), 3);

        let incident_dt_index: HashSet<_> = edge_index_dt.incident_edges(v0).collect();
        let incident_tri_index: HashSet<_> = edge_index_tri.incident_edges(v0).collect();
        assert_eq!(incident_dt_index, incident_tri_index);
        assert_eq!(incident_dt_index, incident_dt);
        assert_eq!(
            adjacency_dt.incident_edges(v0).collect::<HashSet<_>>(),
            incident_dt
        );
        assert_eq!(adjacency_dt.number_of_incident_edges(v0), incident_dt.len());
        assert_eq!(
            adjacency_dt.adjacent_simplices(v0).collect::<HashSet<_>>(),
            tri.adjacent_simplices(v0).collect::<HashSet<_>>()
        );
        assert_eq!(
            adjacency_dt.number_of_adjacent_simplices(v0),
            tri.adjacent_simplices(v0).count()
        );

        let simplex_key = dt.simplices().next().unwrap().0;
        let neighbors_dt: Vec<_> = dt.simplex_neighbors(simplex_key).collect();
        let neighbors_tri: Vec<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(neighbors_dt, neighbors_tri);
        assert!(neighbors_dt.is_empty());

        let neighbor_index_dt = dt.build_simplex_neighbor_index().unwrap();
        let neighbor_index_tri = tri.build_simplex_neighbor_index().unwrap();
        let neighbors_dt_index: Vec<_> = neighbor_index_dt.simplex_neighbors(simplex_key).collect();
        let neighbors_tri_index: Vec<_> =
            neighbor_index_tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(neighbors_dt_index, neighbors_tri_index);
        assert_eq!(neighbors_dt_index, neighbors_dt);
        assert_eq!(
            adjacency_dt
                .simplex_neighbors(simplex_key)
                .collect::<Vec<_>>(),
            neighbors_dt
        );
        assert_eq!(
            adjacency_dt.number_of_simplex_neighbors(simplex_key),
            neighbors_dt.len()
        );

        // Geometry/topology accessors should be forwarded as well.
        let simplex_vertices_dt = dt.simplex_vertices(simplex_key).unwrap();
        let simplex_vertices_tri = tri.simplex_vertices(simplex_key).unwrap();
        assert_eq!(simplex_vertices_dt, simplex_vertices_tri);
        assert_eq!(simplex_vertices_dt.len(), 4);

        let coords_dt = dt.vertex_coords(v0).unwrap();
        let coords_tri = tri.vertex_coords(v0).unwrap();
        assert_eq!(coords_dt, coords_tri);

        // Missing keys should behave the same as on `Triangulation`.
        assert!(dt.vertex_coords(VertexKey::default()).is_none());
        assert_matches!(
            dt.simplex_vertices(SimplexKey::default()),
            Err(TdsError::SimplexNotFound { .. })
        );
    }
}
