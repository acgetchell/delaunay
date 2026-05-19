//! Read-only Delaunay triangulation query, traversal, and accessor methods.
//!
//! This module owns the high-level forwarding surface for inspecting a
//! `DelaunayTriangulation`: counts, iterator access, TDS views, topology
//! metadata accessors, and adjacency traversal helpers. It also keeps the small
//! cache invalidation helpers next to the accessors they protect.

#![forbid(unsafe_code)]

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::validation::{TopologyGuarantee, ValidationPolicy};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::repair::{DelaunayCheckPolicy, DelaunayRepairPolicy};
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::DelaunayTriangulation;

// =============================================================================
// QUERY, ACCESSORS, AND CONFIGURATION (Minimal Bounds)
// =============================================================================
//
// Methods that only need `K: Kernel<D>` — no scalar arithmetic.  Downstream
// generic code (e.g. `delaunayize_by_flips`) does not need to carry
// `CoordinateScalar + NumCast` bounds when calling these methods.
//
// Follows the precedent of the existing PURE STRUCT ASSEMBLY impl block.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    // -------------------------------------------------------------------------
    // QUERY / ACCESSORS
    // -------------------------------------------------------------------------

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    ///     vertex!([0.2, 0.2, 0.2, 0.2]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 6);
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
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// // One 4-simplex in 4D
    /// assert_eq!(dt.number_of_simplices(), 1);
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
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.dim(), 4);
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
    /// An iterator over `(SimplexKey, &Simplex<K::Scalar, U, V, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (simplex_key, simplex) in dt.simplices() {
    ///     println!("Simplex {:?} has {} vertices", simplex_key, simplex.number_of_vertices());
    /// }
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<K::Scalar, U, V, D>)> {
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
    /// An iterator over `(VertexKey, &Vertex<K::Scalar, U, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
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
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<U>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, Vertex, vertex,
    /// };
    ///
    /// let vertices: [Vertex<f64, i32, 2>; 3] = [
    ///     vertex!([0.0, 0.0], 10i32),
    ///     vertex!([1.0, 0.0], 20),
    ///     vertex!([0.0, 1.0], 30),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    /// let key = dt.vertices().next().unwrap().0;
    ///
    /// let prev = dt.set_vertex_data(key, Some(99));
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None);
    /// assert_eq!(prev, Some(Some(99)));
    /// assert_eq!(dt.tds().vertex(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_vertex_data(&mut self, key: VertexKey, data: Option<U>) -> Option<Option<U>> {
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
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<V>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<i32>()
    ///     .unwrap();
    /// let key = dt.simplices().next().unwrap().0;
    ///
    /// let prev = dt.set_simplex_data(key, Some(42));
    /// assert_eq!(prev, Some(None));
    ///
    /// // Clear data
    /// let prev = dt.set_simplex_data(key, None);
    /// assert_eq!(prev, Some(Some(42)));
    /// assert_eq!(dt.tds().simplex(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_simplex_data(&mut self, key: SimplexKey, data: Option<V>) -> Option<Option<V>> {
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
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// assert_eq!(tds.number_of_vertices(), 5);
    /// ```
    #[must_use]
    pub const fn tds(&self) -> &Tds<K::Scalar, U, V, D> {
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
    pub(crate) fn tds_mut(&mut self) -> &mut Tds<K::Scalar, U, V, D> {
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
    pub(crate) fn tds_mut_for_repair(&mut self) -> &mut Tds<K::Scalar, U, V, D> {
        self.invalidate_repair_caches();
        &mut self.tri.tds
    }

    /// Returns a reference to the underlying `Triangulation` (kernel + tds).
    ///
    /// This is useful when you need to pass the triangulation to methods that
    /// expect a `&Triangulation`, such as `ConvexHull::from_triangulation()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::ConvexHull;
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull = ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
    /// assert_eq!(hull.number_of_facets(), 4);
    /// ```
    #[must_use]
    pub const fn as_triangulation(&self) -> &Triangulation<K, U, V, D> {
        &self.tri
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
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     delaunay::prelude::validation::ValidationPolicy::OnSuspicion
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.tri.validation_policy
    }

    /// Sets the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This affects subsequent incremental insertions. (Construction-time behavior is determined
    /// by the policy active during `new()` / `with_kernel()`.)
    ///
    /// If the requested policy is incompatible with the current topology guarantee (for example,
    /// `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](crate::Triangulation::validate_at_completion)
    /// to provide immediate feedback and emits a warning. Call `validate_at_completion()` after
    /// batch construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    /// use delaunay::prelude::validation::ValidationPolicy;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// dt.set_validation_policy(ValidationPolicy::Always);
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     ValidationPolicy::Always
    /// );
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        self.tri.set_validation_policy(policy);
    }
    /// Returns the automatic Delaunay repair policy.
    #[inline]
    #[must_use]
    pub const fn delaunay_repair_policy(&self) -> DelaunayRepairPolicy {
        self.insertion_state.delaunay_repair_policy
    }

    /// Sets the automatic Delaunay repair policy.
    #[inline]
    pub const fn set_delaunay_repair_policy(&mut self, policy: DelaunayRepairPolicy) {
        self.insertion_state.delaunay_repair_policy = policy;
    }

    /// Returns the automatic global Delaunay validation policy.
    #[inline]
    #[must_use]
    pub const fn delaunay_check_policy(&self) -> DelaunayCheckPolicy {
        self.insertion_state.delaunay_check_policy
    }

    /// Sets the automatic global Delaunay validation policy.
    #[inline]
    pub const fn set_delaunay_check_policy(&mut self, policy: DelaunayCheckPolicy) {
        self.insertion_state.delaunay_check_policy = policy;
    }
}

// =============================================================================
// CONFIGURATION & TRAVERSAL (Minimal Bounds, continued)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // CONFIGURATION
    // -------------------------------------------------------------------------

    /// Returns the topology guarantee used for Level 3 topology validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
    /// };
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
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
    ///     DelaunayTriangulationBuilder, GlobalTopology, vertex,
    /// };
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert!(dt.global_topology().is_euclidean());
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
    ///     DelaunayTriangulationBuilder, TopologyKind, vertex,
    /// };
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert_eq!(dt.topology_kind(), TopologyKind::Euclidean);
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        self.tri.topology_kind()
    }

    /// Sets runtime global topology metadata on this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, GlobalTopology, vertex,
    /// };
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// dt.set_global_topology(GlobalTopology::Euclidean);
    /// assert!(dt.global_topology().is_euclidean());
    /// ```
    #[inline]
    pub const fn set_global_topology(&mut self, global_topology: GlobalTopology<D>) {
        self.tri.set_global_topology(global_topology);
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
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
    /// An iterator yielding `FacetView` objects for all facets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let facet_count = dt.facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.facets()
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one simplex. This method
    /// computes the facet-to-simplices map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.boundary_facets()
    }

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::build_adjacency_index`](crate::Triangulation::build_adjacency_index).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying triangulation data structure is internally inconsistent
    /// (e.g., a simplex references a missing vertex key or a missing neighbor simplex key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// assert_eq!(index.number_of_edges(), 6);
    /// ```
    #[inline]
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        self.as_triangulation().build_adjacency_index()
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges`](crate::Triangulation::edges).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let edges: std::collections::HashSet<_> = dt.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().edges()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges_with_index`](crate::Triangulation::edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// let edges: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.as_triangulation().edges_with_index(index)
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
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// // In a tetrahedron, each vertex has degree 3.
    /// assert_eq!(dt.incident_edges(v0).count(), 3);
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().incident_edges(v)
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges_with_index`](crate::Triangulation::incident_edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// assert_eq!(dt.incident_edges_with_index(&index, v0).count(), 3);
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.as_triangulation().incident_edges_with_index(index, v)
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
    /// use delaunay::prelude::query::*;
    ///
    /// // A single tetrahedron has no simplex neighbors.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let simplex_key = dt.simplices().next().unwrap().0;
    /// assert_eq!(dt.simplex_neighbors(simplex_key).count(), 0);
    /// ```
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.as_triangulation().simplex_neighbors(c)
    }

    /// Returns an iterator over all neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::simplex_neighbors_with_index`](crate::Triangulation::simplex_neighbors_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // Two tetrahedra sharing a triangular facet => each tetra has exactly one neighbor.
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
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// let simplex_key = dt.simplices().next().unwrap().0;
    /// assert_eq!(dt.simplex_neighbors_with_index(&index, simplex_key).count(), 1);
    /// ```
    pub fn simplex_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: SimplexKey,
    ) -> impl Iterator<Item = SimplexKey> + 'a {
        self.as_triangulation()
            .simplex_neighbors_with_index(index, c)
    }

    /// Returns a slice view of a simplex's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::simplex_vertices`](crate::Triangulation::simplex_vertices).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let simplex_key = dt.simplices().next().unwrap().0;
    /// let simplex_vertices = dt.simplex_vertices(simplex_key).unwrap();
    /// assert_eq!(simplex_vertices.len(), 3); // D+1 for a 2D simplex
    /// ```
    #[must_use]
    pub fn simplex_vertices(&self, c: SimplexKey) -> Option<&[VertexKey]> {
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
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let v_key = dt
    ///     .vertices()
    ///     .find_map(|(vk, _)| (dt.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// assert_eq!(dt.vertex_coords(v_key).unwrap(), [1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]> {
        self.as_triangulation().vertex_coords(v)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::vertex;
    use std::{collections::HashSet, num::NonZeroUsize, sync::Once};

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
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();

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
    fn test_set_delaunay_check_policy_updates_state() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.delaunay_check_policy(), DelaunayCheckPolicy::EndOnly);

        let policy = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(3).unwrap());
        dt.set_delaunay_check_policy(policy);
        assert_eq!(dt.delaunay_check_policy(), policy);
    }

    #[test]
    fn test_validation_policy_defaults_to_on_suspicion() {
        init_tracing();
        // empty() -> Triangulation::new_empty() -> ValidationPolicy::default()
        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.validation_policy(), ValidationPolicy::OnSuspicion);

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        // new() -> with_kernel() -> explicit validation_policy initialization
        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.validation_policy(), ValidationPolicy::OnSuspicion);

        // with_kernel() constructor path should also use the default policy
        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();
        assert_eq!(
            dt_with_kernel.validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        // try_from_tds() is a separate reconstruction path and should also
        // default to OnSuspicion after validation succeeds.
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let dt_from_tds: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_from_tds(tds, FastKernel::new()).unwrap();
        assert_eq!(
            dt_from_tds.validation_policy(),
            ValidationPolicy::OnSuspicion
        );
    }

    #[test]
    fn test_validation_policy_setter_and_getter_roundtrip() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Getter reflects the underlying Triangulation policy.
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);

        dt.set_validation_policy(ValidationPolicy::Always);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Always);

        dt.set_validation_policy(ValidationPolicy::Never);
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
    }

    #[test]
    fn test_number_of_simplices_minimal_simplex() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Minimal 3D simplex has exactly 1 tetrahedron
        assert_eq!(dt.number_of_simplices(), 1);
    }

    #[test]
    fn test_number_of_simplices_after_insertion() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_simplices(), 1);

        // Insert interior point - should create 3 triangles
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert_eq!(dt.number_of_simplices(), 3);
    }

    #[test]
    fn test_dim_returns_correct_dimension() {
        init_tracing();
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.dim(), 2);

        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.dim(), 3);

        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices_4d).unwrap();
        assert_eq!(dt_4d.dim(), 4);
    }

    #[test]
    fn test_new_with_exact_minimum_vertices() {
        init_tracing();
        // 2D: exactly 3 vertices (minimum for 2D simplex)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.number_of_vertices(), 3);
        assert_eq!(dt_2d.number_of_simplices(), 1);

        // 3D: exactly 4 vertices (minimum for 3D simplex)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.number_of_vertices(), 4);
        assert_eq!(dt_3d.number_of_simplices(), 1);
    }

    #[test]
    fn test_tds_accessor_provides_readonly_access() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Before insertion
        assert_eq!(dt.tds().number_of_vertices(), 3);

        // Insert a new vertex
        dt.insert(vertex!([0.3, 0.3])).unwrap();

        // After insertion, TDS accessor reflects the change
        assert_eq!(dt.tds().number_of_vertices(), 4);
        assert!(dt.tds().number_of_simplices() > 1);
    }

    #[test]
    fn test_tds_accessors_maintain_validation_invariants() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Verify TDS is valid through accessor
        assert!(dt.tds().is_valid().is_ok());

        // Insert additional vertex
        dt.insert(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();

        // TDS should still be valid after mutation
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate().is_ok());
    }

    #[test]
    fn test_topology_traversal_methods_are_forwarded() {
        init_tracing();
        // Single tetrahedron: 4 vertices, 1 simplex, 6 unique edges.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let edges_dt: HashSet<_> = dt.edges().collect();
        let edges_tri: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_dt, edges_tri);
        assert_eq!(edges_dt.len(), 6);

        let index = dt.build_adjacency_index().unwrap();
        let edges_dt_index: HashSet<_> = dt.edges_with_index(&index).collect();
        let edges_tri_index: HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_dt_index, edges_tri_index);
        assert_eq!(edges_dt_index, edges_dt);

        let v0 = dt.vertices().next().unwrap().0;
        let incident_dt: HashSet<_> = dt.incident_edges(v0).collect();
        let incident_tri: HashSet<_> = tri.incident_edges(v0).collect();
        assert_eq!(incident_dt, incident_tri);
        assert_eq!(incident_dt.len(), 3);

        let incident_dt_index: HashSet<_> = dt.incident_edges_with_index(&index, v0).collect();
        let incident_tri_index: HashSet<_> = tri.incident_edges_with_index(&index, v0).collect();
        assert_eq!(incident_dt_index, incident_tri_index);
        assert_eq!(incident_dt_index, incident_dt);

        let simplex_key = dt.simplices().next().unwrap().0;
        let neighbors_dt: Vec<_> = dt.simplex_neighbors(simplex_key).collect();
        let neighbors_tri: Vec<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(neighbors_dt, neighbors_tri);
        assert!(neighbors_dt.is_empty());

        let neighbors_dt_index: Vec<_> = dt
            .simplex_neighbors_with_index(&index, simplex_key)
            .collect();
        let neighbors_tri_index: Vec<_> = tri
            .simplex_neighbors_with_index(&index, simplex_key)
            .collect();
        assert_eq!(neighbors_dt_index, neighbors_tri_index);
        assert_eq!(neighbors_dt_index, neighbors_dt);

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
        assert!(dt.simplex_vertices(SimplexKey::default()).is_none());
    }
}
