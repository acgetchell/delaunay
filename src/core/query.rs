//! Read-only queries for generic [`Triangulation`](crate::Triangulation).
//!
//! This module owns zero-mutation accessors and topology traversal helpers for
//! the generic triangulation layer. Mutation APIs stay with the construction and
//! editing modules; validation orchestration stays in [`crate::core::validation`].

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::collections::{
    FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToSimplicesMap,
    fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, VertexKey};
use crate::core::triangulation::Triangulation;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(debug_assertions)]
static VERTEX_TO_SIMPLICES_SPILL_EVENTS: AtomicU64 = AtomicU64::new(0);

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Returns an iterator over all simplices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over simplices
    /// for (_simplex_key, simplex) in tri.simplices() {
    ///     assert_eq!(simplex.number_of_vertices(), 3); // 2D triangle
    /// }
    /// assert_eq!(tri.simplices().count(), 1);
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<K::Scalar, U, V, D>)> {
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
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over vertices
    /// for (_vertex_key, vertex) in tri.vertices() {
    ///     assert_eq!(vertex.dim(), 2); // 2D vertices
    /// }
    /// assert_eq!(tri.vertices().count(), 3);
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tds.vertices()
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_vertices(), 4);
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
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_simplices(), 1); // Single tetrahedron
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
    /// // Empty triangulation has dimension -1
    /// let empty: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(empty.dim(), -1);
    ///
    /// // 3D tetrahedron has dimension 3
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().dim(), 3);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tds.dim()
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// This provides efficient access to all facets without pre-allocating a vector.
    /// Each facet is represented as a lightweight `FacetView` that references the
    /// underlying triangulation data.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Iterate over all facets
    /// let facet_count = dt.as_triangulation().facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        AllFacetsIter::new(&self.tds)
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
    /// # Panics
    ///
    /// Panics if the triangulation data structure is corrupted (simplices have invalid
    /// neighbor relationships or facet information). This indicates a bug in the
    /// library and should never happen with a properly constructed triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.as_triangulation().boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        // build_facet_to_simplices_map only fails if simplices have invalid structure,
        // which should never happen in a valid triangulation
        let facet_map = self
            .tds
            .build_facet_to_simplices_map()
            .expect("Failed to build facet map - triangulation structure is corrupted");
        BoundaryFacetsIter::new(&self.tds, facet_map)
    }

    #[inline]
    fn debug_assert_adjacency_index_matches(&self, index: &AdjacencyIndex) {
        // AdjacencyIndex is built from a snapshot of a triangulation. We cannot enforce at
        // compile-time that an index belongs to this triangulation, but we can cheaply catch
        // obvious mix-ups in debug builds.
        debug_assert_eq!(
            index.vertex_to_simplices.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_simplices size does not match triangulation vertex count"
        );
        debug_assert_eq!(
            index.vertex_to_edges.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_edges size does not match triangulation vertex count"
        );
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
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let edges: std::collections::HashSet<_> = tri.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_edges().into_iter()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
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
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    /// let edges: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.edges()
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
    /// // A single 2D triangle has 3 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// assert_eq!(tri.number_of_edges(), 3);
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.collect_edges().len()
    }

    /// Returns the number of unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This is equivalent to `self.edges_with_index(index).count()`.
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
    /// assert_eq!(tri.number_of_edges_with_index(&index), 6);
    /// ```
    #[must_use]
    pub fn number_of_edges_with_index(&self, index: &AdjacencyIndex) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_edges()
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
    pub fn adjacent_simplices_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = SimplexKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.adjacent_simplices(v)
    }

    /// Returns the number of simplices adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    #[must_use]
    pub fn number_of_adjacent_simplices_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_adjacent_simplices(v)
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
    pub fn simplex_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: SimplexKey,
    ) -> impl Iterator<Item = SimplexKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.simplex_neighbors(c)
    }

    /// Returns the number of neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    #[must_use]
    pub fn number_of_simplex_neighbors_with_index(
        &self,
        index: &AdjacencyIndex,
        c: SimplexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_simplex_neighbors(c)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_incident_edges(v).into_iter()
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.incident_edges(v)
    }

    /// Returns the number of unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    #[must_use]
    pub fn number_of_incident_edges_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_incident_edges(v)
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
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]> {
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
                    let edge = EdgeKey::new(vertices[i], vertices[j]);
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
                    edges.insert(EdgeKey::new(vertices[i], vertices[j]));
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
                edges.insert(EdgeKey::new(v, other));
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
    use crate::vertex;

    use slotmap::KeyData;
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
                    assert_eq!(empty.facets().count(), 0);
                    assert_eq!(empty.boundary_facets().count(), 0);

                    let vertices = vec![
                        $(vertex!($simplex_coords)),+
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
                    assert_eq!(tri.facets().count(), expected_vertex_count);
                    assert_eq!(tri.boundary_facets().count(), expected_vertex_count);
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
    fn topology_edges_triangle_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 1);
        assert_eq!(tri.number_of_vertices(), 3);
        assert_eq!(tri.number_of_edges(), 3);

        let edges: HashSet<_> = tri.edges().collect();
        assert_eq!(edges.len(), 3);

        let index = tri.build_adjacency_index().unwrap();
        let edges_with_index: HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_with_index, edges);
        assert_eq!(tri.number_of_edges_with_index(&index), 3);

        assert!(edges.iter().all(|edge| {
            let (a, b) = edge.endpoints();
            a != b && tri.vertex_coords(a).is_some() && tri.vertex_coords(b).is_some()
        }));
    }

    #[test]
    fn topology_edges_and_incident_edges_double_tetrahedron_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
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
        assert_eq!(tri.number_of_edges_with_index(&index), 9);
        assert_eq!(tri.adjacent_simplices(base_vertex_key).count(), 2);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, base_vertex_key)
                .count(),
            2
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, base_vertex_key),
            2
        );
        assert_eq!(
            tri.incident_edges_with_index(&index, base_vertex_key)
                .count(),
            4
        );
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, base_vertex_key),
            4
        );

        let apex_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.7, 1.5]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(apex_vertex_key), 3);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, apex_vertex_key)
                .count(),
            1
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, apex_vertex_key),
            1
        );

        for (simplex_key, _) in tri.simplices() {
            assert_eq!(
                tri.simplex_neighbors_with_index(&index, simplex_key)
                    .count(),
                1
            );
            assert_eq!(
                tri.number_of_simplex_neighbors_with_index(&index, simplex_key),
                1
            );
        }
    }

    #[test]
    fn topology_queries_missing_keys_are_empty_or_none() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        let missing_vertex_key = VertexKey::default();
        assert_eq!(tri.adjacent_simplices(missing_vertex_key).count(), 0);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, missing_vertex_key),
            0
        );
        assert_eq!(tri.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(
            tri.incident_edges_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(tri.number_of_incident_edges(missing_vertex_key), 0);
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, missing_vertex_key),
            0
        );
        assert!(tri.vertex_coords(missing_vertex_key).is_none());

        let missing_simplex_key = SimplexKey::default();
        assert_eq!(tri.simplex_neighbors(missing_simplex_key).count(), 0);
        assert_eq!(
            tri.simplex_neighbors_with_index(&index, missing_simplex_key)
                .count(),
            0
        );
        assert_eq!(
            tri.number_of_simplex_neighbors_with_index(&index, missing_simplex_key),
            0
        );
        assert!(tri.simplex_vertices(missing_simplex_key).is_none());
    }

    #[test]
    fn topology_geometry_accessors_round_trip() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let isolated_vertex = tri
            .tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let edge_count = tri.number_of_edges();
        let edges_collected: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_collected.len(), edge_count);
        assert!(edge_count >= 6);

        assert!(tri.facets().next().is_some());
        assert!(tri.boundary_facets().next().is_some());

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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        let idx_edges: HashSet<_> = tri.edges_with_index(&index).collect();
        let direct_edges: HashSet<_> = tri.edges().collect();
        assert_eq!(idx_edges, direct_edges);
        assert_eq!(
            tri.number_of_edges_with_index(&index),
            tri.number_of_edges()
        );

        let vertex_key = tri.vertices().next().unwrap().0;
        let idx_adj: HashSet<_> = tri
            .adjacent_simplices_with_index(&index, vertex_key)
            .collect();
        let direct_adj: HashSet<_> = tri.adjacent_simplices(vertex_key).collect();
        assert_eq!(idx_adj, direct_adj);
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, vertex_key),
            direct_adj.len()
        );

        let simplex_key = tri.simplices().next().unwrap().0;
        let direct_neighbors: Vec<_> = tri.simplex_neighbors(simplex_key).collect();
        assert_eq!(
            tri.simplex_neighbors_with_index(&index, simplex_key)
                .count(),
            direct_neighbors.len()
        );
        assert_eq!(
            tri.number_of_simplex_neighbors_with_index(&index, simplex_key),
            direct_neighbors.len()
        );

        let idx_incident: HashSet<_> = tri.incident_edges_with_index(&index, vertex_key).collect();
        let direct_incident: HashSet<_> = tri.incident_edges(vertex_key).collect();
        assert_eq!(idx_incident, direct_incident);
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, vertex_key),
            direct_incident.len()
        );
    }
}
