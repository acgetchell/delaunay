//! Canonical edge identifiers for public topology traversal.
//!
//! Downstream code frequently needs a stable, comparable identifier for an *edge* in the
//! triangulation topology. Since edges are not stored explicitly (they are inferred from
//! maximal simplices), we expose a lightweight `EdgeKey` that:
//!
//! - identifies an edge by two live endpoint [`VertexKey`]s and a lattice displacement
//! - canonicalizes endpoint ordering so `(a, b)` and `(b, a)` map to the same edge
//! - is `Copy`/`Hash`/`Ord` for fast use in sets and maps
//! - can be revalidated into an [`EdgeView`] for borrowed access to live topology
//!
//! ## Closed-star convention
//!
//! Unqualified **star** means the **closed star** of [Lickorish, §2]:
//! `st(A, K) = A ⋆ lk(A, K)`, where `A` includes all its faces. It consists
//! of simplices containing `A` and **all their faces**.
//! [`EdgeView::incident_simplices`] returns only its incident maximal
//! D-simplices; taking their faces recovers the closed star.
//!
//! For a vertex `a` in a single triangle `abc`, the closed star contains the
//! triangle and all edges and vertices, including the opposite edge `bc` and
//! its endpoints. Its **open star** in the realization contains `a`, the
//! relative interiors of `ab` and `ac`, and the triangle's relative interior;
//! it excludes `bc`, `b`, and `c`. It is not the interior of the closed star
//! relative to the whole triangle (which is the whole triangle).
//! No open-star set is returned by these APIs. See the
//! [terminology mapping] for joins, links, closure, and ambient spaces.
//!
//! [Lickorish, §2]: https://arxiv.org/html/math/9911256v1
//! [terminology mapping]: https://github.com/acgetchell/delaunay/blob/main/docs/topology.md#simplicial-terminology-and-notation
//!
//! ## Determinism
//!
//! `EdgeKey` ordering is **not** guaranteed to be deterministic across processes or
//! serialization round-trips, because [`VertexKey`] ordering is derived from internal
//! slotmap keys. If you need a deterministic order, sort using stable vertex UUIDs.

#![forbid(unsafe_code)]

use crate::core::collections::{FastHashSet, SimplexKeyBuffer, fast_hash_set_with_capacity};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use crate::topology::spaces::toroidal::{LiftedVertexId, lifted_vertex_id};
use slotmap::Key;
use std::{array::from_fn, collections::hash_set::IntoIter as HashSetIntoIter, fmt};
use thiserror::Error;

/// Unique edges collected from one immutable TDS snapshot.
///
/// Endpoint-only storage is sound only after checking that *every* simplex lacks
/// periodic offsets. Keep that proof and the compact representation private;
/// callers always receive complete, dimension-bearing edge keys.
pub(crate) struct EdgeSet<const D: usize>(EdgeStorage<D>);

enum EdgeStorage<const D: usize> {
    Endpoints(FastHashSet<(VertexKey, VertexKey)>),
    Lifted(FastHashSet<EdgeKey<D>>),
}

impl<const D: usize> EdgeSet<D> {
    pub(crate) fn from_tds<U, V>(tds: &Tds<U, V, D>, capacity: usize) -> Self {
        if tds
            .simplices()
            .all(|(_, simplex)| simplex.periodic_vertex_offsets().is_none())
        {
            let mut edges = fast_hash_set_with_capacity(capacity);
            for (_, simplex) in tds.simplices() {
                let vertices = simplex.vertices();
                for (i, &a) in vertices.iter().enumerate() {
                    for &b in &vertices[i + 1..] {
                        let pair = if a.data().as_ffi() <= b.data().as_ffi() {
                            (a, b)
                        } else {
                            (b, a)
                        };
                        edges.insert(pair);
                    }
                }
            }
            Self(EdgeStorage::Endpoints(edges))
        } else {
            let mut edges = fast_hash_set_with_capacity(capacity);
            for (_, simplex) in tds.simplices() {
                for i in 0..simplex.number_of_vertices() {
                    for j in i + 1..simplex.number_of_vertices() {
                        edges.insert(EdgeKey::from_simplex_slots(simplex, i, j));
                    }
                }
            }
            Self(EdgeStorage::Lifted(edges))
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.0 {
            EdgeStorage::Endpoints(edges) => edges.len(),
            EdgeStorage::Lifted(edges) => edges.len(),
        }
    }
}

impl<const D: usize> IntoIterator for EdgeSet<D> {
    type Item = EdgeKey<D>;
    type IntoIter = EdgeSetIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        EdgeSetIter(match self.0 {
            EdgeStorage::Endpoints(edges) => EdgeIterStorage::Endpoints(edges.into_iter()),
            EdgeStorage::Lifted(edges) => EdgeIterStorage::Lifted(edges.into_iter()),
        })
    }
}

/// Expands compact keys lazily while retaining the exact remaining item count.
pub(crate) struct EdgeSetIter<const D: usize>(EdgeIterStorage<D>);

enum EdgeIterStorage<const D: usize> {
    Endpoints(HashSetIntoIter<(VertexKey, VertexKey)>),
    Lifted(HashSetIntoIter<EdgeKey<D>>),
}

impl<const D: usize> Iterator for EdgeSetIter<D> {
    type Item = EdgeKey<D>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            EdgeIterStorage::Endpoints(edges) => edges.next().map(|(v0, v1)| EdgeKey {
                v0,
                v1,
                displacement: [0; D],
            }),
            EdgeIterStorage::Lifted(edges) => edges.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<const D: usize> ExactSizeIterator for EdgeSetIter<D> {
    fn len(&self) -> usize {
        match &self.0 {
            EdgeIterStorage::Endpoints(edges) => edges.len(),
            EdgeIterStorage::Lifted(edges) => edges.len(),
        }
    }
}

/// Error returned when constructing an [`EdgeKey`] from invalid endpoints.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum EdgeKeyError {
    /// More than one lattice class joins these quotient endpoints.
    #[error("Vertices {v0:?} and {v1:?} have multiple periodic edges; select a simplex occurrence")]
    AmbiguousEndpoints {
        /// First endpoint.
        v0: VertexKey,
        /// Second endpoint.
        v1: VertexKey,
    },
    /// The requested simplex-local edge occurrence does not exist.
    #[error("Invalid edge occurrence {vertices:?} in simplex {simplex_key:?}")]
    InvalidOccurrence {
        /// Simplex containing the requested occurrence.
        simplex_key: SimplexKey,
        /// Requested local vertex indices.
        vertices: [usize; 2],
    },
    /// Both endpoints refer to the same vertex.
    #[error("Edge endpoints must be distinct, got {endpoint:?} twice")]
    DuplicateEndpoint {
        /// The repeated vertex key.
        endpoint: VertexKey,
    },
    /// One endpoint is not present in the TDS.
    #[error("Edge endpoint {endpoint:?} is not present in the TDS")]
    MissingEndpoint {
        /// The missing endpoint key.
        endpoint: VertexKey,
    },
    /// The endpoints are live vertices but do not form an edge in any simplex.
    #[error("Vertices {v0:?} and {v1:?} are not joined by an edge in the TDS")]
    EdgeNotFound {
        /// First endpoint.
        v0: VertexKey,
        /// Second endpoint.
        v1: VertexKey,
    },
    /// The maintained incidence index does not list any simplex for this edge.
    #[error("Vertex incidence index does not list any simplex containing edge {v0:?}-{v1:?}")]
    MissingEdgeIncidence {
        /// First endpoint.
        v0: VertexKey,
        /// Second endpoint.
        v1: VertexKey,
    },
    /// The vertex incidence index references a simplex that is no longer present.
    #[error("Vertex incidence index for {vertex_key:?} references missing simplex {simplex_key:?}")]
    DanglingVertexIncidence {
        /// Vertex whose incidence list contains the dangling simplex key.
        vertex_key: VertexKey,
        /// Missing simplex key referenced by the incidence index.
        simplex_key: SimplexKey,
    },
    /// A simplex contains an endpoint, but that endpoint's incidence index does not list it.
    #[error("Vertex incidence index for {vertex_key:?} is missing simplex {simplex_key:?}")]
    MissingVertexIncidence {
        /// Vertex whose incidence list is missing the simplex key.
        vertex_key: VertexKey,
        /// Simplex expected in the vertex's incidence list.
        simplex_key: SimplexKey,
    },
    /// A vertex incidence entry points at a simplex that does not contain that vertex.
    #[error(
        "Vertex incidence index for {vertex_key:?} references simplex {simplex_key:?}, but the simplex does not contain that vertex"
    )]
    VertexIncidenceMismatch {
        /// Vertex whose incidence list contains the inconsistent simplex key.
        vertex_key: VertexKey,
        /// Simplex expected to contain the vertex.
        simplex_key: SimplexKey,
    },
}

/// Canonical identifier for an (undirected) edge.
///
/// `EdgeKey` is a runtime topology key, not durable identity. It stores canonicalized
/// storage-local [`VertexKey`] endpoints and their relative periodic offset, so it is valid only for the live [`Tds`] whose
/// endpoints were checked by [`EdgeKey::try_new`]. Persist or compare edges across I/O
/// boundaries with stable vertex UUIDs instead.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::{construction::*, tds::*};
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
/// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
///     return Ok(());
/// };
/// let a = simplex.vertices()[0];
/// let b = simplex.vertices()[1];
/// let edge = dt.edge_key(a, b)?;
/// assert_eq!(edge.endpoints(), (edge.v0(), edge.v1()));
/// # Ok(())
/// # }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeKey<const D: usize> {
    v0: VertexKey,
    v1: VertexKey,
    displacement: [i16; D],
}

impl<const D: usize> EdgeKey<D> {
    /// Creates a new canonical edge key.
    ///
    /// The endpoints must identify exactly one edge class in `tds`. Endpoints
    /// are reordered so that `v0 <= v1` under the internal key order. Periodic
    /// loops are allowed when their lattice displacement is nonzero. Use
    /// [`Self::try_from_simplex`] to select one of several parallel classes.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError::DuplicateEndpoint`] if equal endpoints have no
    /// periodic loop, [`EdgeKeyError::AmbiguousEndpoints`] if several classes
    /// exist, [`EdgeKeyError::MissingEndpoint`] if either endpoint is not
    /// live in `tds`, [`EdgeKeyError::EdgeNotFound`] if no simplex contains both
    /// endpoints, or another [`EdgeKeyError`] if the maintained vertex-incidence
    /// metadata needed to prove the edge is inconsistent with simplex storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e1 = dt.edge_key(a, b)?;
    /// let e2 = dt.edge_key(b, a)?;
    /// assert_eq!(e1, e2);
    /// assert!(e1.v0() <= e1.v1());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new<U, V>(
        tds: &Tds<U, V, D>,
        a: VertexKey,
        b: VertexKey,
    ) -> Result<Self, EdgeKeyError> {
        if tds.vertex(a).is_none() {
            return Err(EdgeKeyError::MissingEndpoint { endpoint: a });
        }
        if tds.vertex(b).is_none() {
            return Err(EdgeKeyError::MissingEndpoint { endpoint: b });
        }
        let mut candidate = None;
        for endpoint in [a, b] {
            for simplex_key in tds.simplex_keys_containing_vertex(endpoint) {
                let simplex =
                    tds.simplex(simplex_key)
                        .ok_or(EdgeKeyError::DanglingVertexIncidence {
                            vertex_key: endpoint,
                            simplex_key,
                        })?;
                if !simplex.contains_vertex(endpoint) {
                    return Err(EdgeKeyError::VertexIncidenceMismatch {
                        vertex_key: endpoint,
                        simplex_key,
                    });
                }
                Self::merge_endpoint_edge_class(simplex, a, b, &mut candidate)?;
            }
        }
        if candidate.is_none() {
            for (_, simplex) in tds.simplices() {
                Self::merge_endpoint_edge_class(simplex, a, b, &mut candidate)?;
            }
        }
        let key = candidate.ok_or(if a == b {
            EdgeKeyError::DuplicateEndpoint { endpoint: a }
        } else {
            EdgeKeyError::EdgeNotFound { v0: a, v1: b }
        })?;
        let _validated_edge_star = Self::parse_edge_star(tds, key)?;

        Ok(key)
    }

    /// Selects one class without letting unrelated incidence entries hide a
    /// missing edge relation at the other endpoint.
    fn merge_endpoint_edge_class<V>(
        simplex: &Simplex<V, D>,
        a: VertexKey,
        b: VertexKey,
        candidate: &mut Option<Self>,
    ) -> Result<(), EdgeKeyError> {
        for i in 0..simplex.number_of_vertices() {
            for j in i + 1..simplex.number_of_vertices() {
                let key = Self::from_simplex_slots(simplex, i, j);
                if (key.v0 == a && key.v1 == b) || (key.v0 == b && key.v1 == a) {
                    if candidate.is_some_and(|previous| previous != key) {
                        return Err(EdgeKeyError::AmbiguousEndpoints { v0: a, v1: b });
                    }
                    *candidate = Some(key);
                }
            }
        }
        Ok(())
    }

    /// Parses the maximal support of the closed edge star from valid endpoint incidence.
    ///
    /// Faces are implicit: the returned keys identify only incident D-simplices.
    /// The success path inspects endpoint incidence instead of scanning every
    /// simplex. Both endpoint incidence lists are checked for every simplex containing the
    /// opposite endpoint, so partially stale reverse incidence is rejected before
    /// returning a key. A full simplex-storage scan is reserved for the cold path
    /// where neither endpoint incidence list reports the edge, so callers still get a
    /// precise `MissingEdgeIncidence` diagnostic when storage and incidence
    /// diverge.
    fn parse_edge_star<U, V>(
        tds: &Tds<U, V, D>,
        key: Self,
    ) -> Result<SimplexKeyBuffer, EdgeKeyError> {
        let (v0, v1) = key.endpoints();
        let v0_degree = tds.vertex_to_simplices_index().number_of_simplices(v0);
        let v1_degree = tds.vertex_to_simplices_index().number_of_simplices(v1);
        let (primary, secondary) = if v0_degree <= v1_degree {
            (v0, v1)
        } else {
            (v1, v0)
        };

        let primary_edge_simplices = key.edge_simplices_in_star(tds, primary)?;
        let secondary_edge_simplices = key.edge_simplices_in_star(tds, secondary)?;

        for &simplex_key in primary_edge_simplices.as_slice() {
            if !secondary_edge_simplices.contains(&simplex_key) {
                return Err(EdgeKeyError::MissingVertexIncidence {
                    vertex_key: secondary,
                    simplex_key,
                });
            }
        }

        for &simplex_key in secondary_edge_simplices.as_slice() {
            if !primary_edge_simplices.contains(&simplex_key) {
                return Err(EdgeKeyError::MissingVertexIncidence {
                    vertex_key: primary,
                    simplex_key,
                });
            }
        }

        if !primary_edge_simplices.is_empty() {
            return Ok(primary_edge_simplices);
        }
        if tds
            .simplices()
            .any(|(_, simplex)| key.slots_in(simplex).next().is_some())
        {
            return Err(EdgeKeyError::MissingEdgeIncidence { v0, v1 });
        }
        Err(EdgeKeyError::EdgeNotFound { v0, v1 })
    }

    /// Finds incident D-simplices at `source` containing this lifted edge class.
    fn edge_simplices_in_star<U, V>(
        self,
        tds: &Tds<U, V, D>,
        source: VertexKey,
    ) -> Result<SimplexKeyBuffer, EdgeKeyError> {
        let mut edge_simplices = SimplexKeyBuffer::new();
        for simplex_key in tds.simplex_keys_containing_vertex(source) {
            let simplex =
                tds.simplex(simplex_key)
                    .ok_or(EdgeKeyError::DanglingVertexIncidence {
                        vertex_key: source,
                        simplex_key,
                    })?;
            if !simplex.contains_vertex(source) {
                return Err(EdgeKeyError::VertexIncidenceMismatch {
                    vertex_key: source,
                    simplex_key,
                });
            }
            if self.slots_in(simplex).next().is_some() {
                edge_simplices.push(simplex_key);
            }
        }
        Ok(edge_simplices)
    }

    /// Selects an edge by its occurrence in a stored simplex, including parallel
    /// quotient edges and periodic loops.
    ///
    /// # Errors
    /// Returns an [`EdgeKeyError`] for invalid slots, missing endpoints, or
    /// inconsistent vertex incidence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::EdgeKey};
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let tds = dt.into_triangulation().into_tds();
    /// for (simplex_key, _) in tds.simplices() {
    ///     let edge = EdgeKey::try_from_simplex(&tds, simplex_key, [0, 1])?;
    ///     assert_eq!(edge.displacement(), [0, 0]);
    ///     let view = edge.view(&tds)?;
    ///     assert_eq!(view.incident_occurrences().collect::<Vec<_>>(), [(simplex_key, [0, 1])]);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_simplex<U, V>(
        tds: &Tds<U, V, D>,
        simplex_key: SimplexKey,
        vertices: [usize; 2],
    ) -> Result<Self, EdgeKeyError> {
        let simplex = tds
            .simplex(simplex_key)
            .filter(|simplex| {
                vertices[0] != vertices[1]
                    && vertices.iter().all(|&i| i < simplex.number_of_vertices())
            })
            .ok_or(EdgeKeyError::InvalidOccurrence {
                simplex_key,
                vertices,
            })?;
        let key = Self::from_simplex_slots(simplex, vertices[0], vertices[1]);
        let _validated_view = key.view(tds)?;
        Ok(key)
    }

    /// Lattice displacement from the first canonical endpoint to the second.
    #[must_use]
    pub const fn displacement(self) -> [i16; D] {
        self.displacement
    }

    pub(crate) fn lifted_endpoints(self) -> [LiftedVertexId; 2] {
        [
            LiftedVertexId::base(self.v0),
            lifted_vertex_id(self.v1, self.displacement),
        ]
    }

    pub(crate) fn from_simplex_slots<V>(simplex: &Simplex<V, D>, i: usize, j: usize) -> Self {
        let a = simplex.vertices()[i];
        let b = simplex.vertices()[j];
        let delta = simplex.periodic_vertex_offsets().map_or([0; D], |offsets| {
            from_fn(|axis| i16::from(offsets[j][axis]) - i16::from(offsets[i][axis]))
        });
        let reverse = a.data().as_ffi() > b.data().as_ffi() || (a == b && delta < [0; D]);
        if reverse {
            Self {
                v0: b,
                v1: a,
                displacement: delta.map(|value| -value),
            }
        } else {
            Self {
                v0: a,
                v1: b,
                displacement: delta,
            }
        }
    }

    pub(crate) fn slots_in<V>(
        self,
        simplex: &Simplex<V, D>,
    ) -> impl Iterator<Item = [usize; 2]> + '_ {
        (0..simplex.number_of_vertices()).flat_map(move |i| {
            (i + 1..simplex.number_of_vertices()).filter_map(move |j| {
                (Self::from_simplex_slots(simplex, i, j) == self).then_some([i, j])
            })
        })
    }

    /// Returns the first (canonical) endpoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[1];
    /// let b = simplex.vertices()[0];
    ///
    /// let e = dt.edge_key(a, b)?;
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn v0(self) -> VertexKey {
        self.v0
    }

    /// Returns the second (canonical) endpoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e = dt.edge_key(a, b)?;
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn v1(self) -> VertexKey {
        self.v1
    }

    /// Returns the two endpoints as a tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e = dt.edge_key(a, b)?;
    /// let (v0, v1) = e.endpoints();
    /// assert_eq!(v0, e.v0());
    /// assert_eq!(v1, e.v1());
    /// assert!(v0 <= v1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn endpoints(self) -> (VertexKey, VertexKey) {
        (self.v0, self.v1)
    }

    /// Revalidates this runtime edge handle against a live TDS and returns a borrowed view.
    ///
    /// `EdgeKey` stores only storage-local endpoint keys. This method checks those
    /// endpoints against `tds` before lending access to endpoint vertices and the
    /// maximal D-simplex support of the edge's closed star (see [`EdgeView`]).
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if either endpoint is stale, the endpoints no
    /// longer share a stored simplex, the edge has no live incidence entry, or
    /// the maintained incidence index is inconsistent with simplex storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let edge = dt.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// let view = dt.edge_view(edge)?;
    /// assert_eq!(view.endpoint_keys(), edge.endpoints());
    /// # Ok(())
    /// # }
    /// ```
    pub fn view<U, V>(self, tds: &Tds<U, V, D>) -> Result<EdgeView<'_, U, V, D>, EdgeKeyError> {
        EdgeView::try_new(tds, self)
    }
}

/// Borrowed live-TDS view over an [`EdgeKey`].
///
/// `EdgeView` is a non-durable topology view. It borrows one in-memory [`Tds`]
/// and revalidates a copyable [`EdgeKey`] before exposing endpoint vertices and
/// the edge's incident D-simplices. Persist stable vertex UUIDs or a full TDS
/// snapshot instead of serializing edge views.
///
/// # Closed-star convention
///
/// Unqualified **star** means the **closed star** of
/// [Lickorish, §2](https://arxiv.org/html/math/9911256v1):
/// `st(A, K) = A ⋆ lk(A, K)`, with all faces of `A` included. The incident
/// D-simplices generate this subcomplex by taking all their faces.
///
/// For vertex `a` in triangle `abc`, the closed star contains the triangle,
/// all edges and all vertices, including opposite edge `bc` and its endpoints.
/// The **open star** in the realization contains `a`, the relative interiors
/// of `ab` and `ac`, and the triangle's relative interior, but excludes `bc`,
/// `b`, and `c`. It is not the interior of the closed star relative to the
/// whole triangle, which is the whole triangle. See the
/// [terminology mapping](https://github.com/acgetchell/delaunay/blob/main/docs/topology.md#simplicial-terminology-and-notation).
#[must_use]
pub struct EdgeView<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    key: EdgeKey<D>,
    vertices: (&'tds Vertex<U, D>, &'tds Vertex<U, D>),
    incident_simplices: SimplexKeyBuffer,
}

impl<'tds, U, V, const D: usize> EdgeView<'tds, U, V, D> {
    /// Creates a borrowed edge view after validating `key` against `tds`.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError`] if the key has stale endpoints, no longer
    /// identifies a stored edge, does not have a live incidence entry, or the
    /// maintained incidence index is inconsistent with simplex storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{construction::*, tds::*};
    /// use delaunay::prelude::tds::EdgeKeyError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Edge(#[from] EdgeKeyError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let key = dt.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    /// let view = dt.edge_view(key)?;
    /// assert_eq!(view.key(), key);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(tds: &'tds Tds<U, V, D>, key: EdgeKey<D>) -> Result<Self, EdgeKeyError> {
        let (v0, v1) = key.endpoints();
        if v0 == v1 && key.displacement == [0; D] {
            return Err(EdgeKeyError::DuplicateEndpoint { endpoint: v0 });
        }
        let first = tds
            .vertex(v0)
            .ok_or(EdgeKeyError::MissingEndpoint { endpoint: v0 })?;
        let second = tds
            .vertex(v1)
            .ok_or(EdgeKeyError::MissingEndpoint { endpoint: v1 })?;

        let incident_simplices = EdgeKey::parse_edge_star(tds, key)?;

        Ok(Self {
            tds,
            key,
            vertices: (first, second),
            incident_simplices,
        })
    }

    /// Returns the copyable runtime key represented by this view.
    #[inline]
    #[must_use]
    pub const fn key(&self) -> EdgeKey<D> {
        self.key
    }

    /// Returns the borrowed TDS backing this view.
    #[inline]
    #[must_use]
    pub(crate) const fn tds(&self) -> &'tds Tds<U, V, D> {
        self.tds
    }

    /// Returns the endpoint keys in canonical order.
    #[inline]
    #[must_use]
    pub const fn endpoint_keys(&self) -> (VertexKey, VertexKey) {
        self.key.endpoints()
    }

    /// Returns borrowed endpoint vertices in canonical key order.
    #[inline]
    #[must_use]
    pub const fn vertices(&self) -> (&'tds Vertex<U, D>, &'tds Vertex<U, D>) {
        self.vertices
    }

    /// Returns all D-simplices incident to this edge.
    ///
    /// These are the maximal simplices generating the **closed star** under
    /// taking faces, following [Lickorish, §2](https://arxiv.org/html/math/9911256v1).
    /// Lower-dimensional faces (even those disjoint from the edge) are implicit.
    /// The incidence was validated during [`Self::try_new`]; see the
    /// [closed/open-star example](EdgeView#closed-star-convention).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// for (simplex_key, simplex) in dt.simplices() {
    ///     let edge = dt.edge_key(simplex.vertices()[0], simplex.vertices()[1])?;
    ///     let view = dt.edge_view(edge)?;
    ///     assert_eq!(view.incident_simplices(), &[simplex_key]);
    ///     // Taking all faces of this triangle includes the opposite vertex,
    ///     // which does not contain the edge. Those faces are implicit here.
    ///     assert_eq!(simplex.vertices().len(), 3);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn incident_simplices(&self) -> &[SimplexKey] {
        self.incident_simplices.as_slice()
    }

    /// Returns each simplex-local occurrence of this edge. Unlike the maximal
    /// simplex support, this retains multiplicity when a quotient simplex
    /// contains several translates of the same edge.
    #[must_use = "this iterator is lazy and does nothing unless consumed"]
    pub fn incident_occurrences(&self) -> impl Iterator<Item = (SimplexKey, [usize; 2])> + '_ {
        self.incident_simplices
            .iter()
            .flat_map(move |&simplex_key| {
                // The parsed view and its borrow guarantee that every support
                // simplex remains live throughout this iterator.
                self.tds
                    .simplex(simplex_key)
                    .into_iter()
                    .flat_map(move |simplex| {
                        self.key
                            .slots_in(simplex)
                            .map(move |slots| (simplex_key, slots))
                    })
            })
    }
}

impl<U, V, const D: usize> fmt::Debug for EdgeView<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EdgeView")
            .field("key", &self.key)
            .field("vertices", &self.key.endpoints())
            .field("incident_simplices", &self.incident_simplices)
            .field("dimension", &D)
            .finish()
    }
}

impl<U, V, const D: usize> Clone for EdgeView<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            key: self.key,
            vertices: self.vertices,
            incident_simplices: self.incident_simplices.clone(),
        }
    }
}

impl<U, V, const D: usize> PartialEq for EdgeView<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.tds, other.tds) && self.key == other.key
    }
}

impl<U, V, const D: usize> Eq for EdgeView<'_, U, V, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TdsBuilder;
    use crate::vertex;
    use std::{
        collections::{BTreeSet, HashSet},
        ptr,
    };

    impl<const D: usize> EdgeSet<D> {
        pub(crate) fn capacity(&self) -> usize {
            match &self.0 {
                EdgeStorage::Endpoints(edges) => edges.capacity(),
                EdgeStorage::Lifted(edges) => edges.capacity(),
            }
        }
    }

    impl<const D: usize> EdgeKey<D> {
        /// Constructs the zero-displacement edge class used by ordinary fixtures.
        pub(crate) fn from_validated_endpoints(a: VertexKey, b: VertexKey) -> Self {
            let (v0, v1) = if a.data().as_ffi() <= b.data().as_ffi() {
                (a, b)
            } else {
                (b, a)
            };
            Self {
                v0,
                v1,
                displacement: [0; D],
            }
        }
    }

    fn with_triangle_tds(test: impl FnOnce(&Tds<(), (), 2>, [VertexKey; 3])) {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let simplex = tds.simplices().next().unwrap().1;
        let vertices = simplex.vertices();
        let simplex_vertices = [vertices[0], vertices[1], vertices[2]];
        test(&tds, simplex_vertices);
    }

    #[test]
    fn edge_set_keeps_explicit_zero_offsets_in_lifted_storage() {
        with_triangle_tds(|tds, vertices| {
            // This is a combinatorial TDS test, not a Euclidean realization
            // carrying periodic metadata. Explicit offsets retain the lifted
            // representation even when their current values are all zero.
            let mut tds = tds.clone();
            let simplex_key = tds.simplex_keys().next().unwrap();
            tds.simplex_mut(simplex_key)
                .unwrap()
                .set_periodic_vertex_offsets(vec![[0; 2]; 3])
                .unwrap();
            tds.validate().unwrap();
            let edges = EdgeSet::from_tds(&tds, 3);
            assert!(matches!(&edges.0, EdgeStorage::Lifted(_)));
            let expected: HashSet<_> = [[0, 1], [1, 2], [2, 0]]
                .map(|[i, j]| EdgeKey::from_validated_endpoints(vertices[i], vertices[j]))
                .into_iter()
                .collect();
            let iter = edges.into_iter();
            assert_eq!(iter.size_hint(), (3, Some(3)));
            let actual: HashSet<_> = iter.collect();
            assert_eq!(actual, expected);
            for edge in actual {
                assert_eq!(edge.view(&tds).unwrap().incident_simplices().len(), 1);
            }
        });
    }

    #[test]
    fn edge_set_count_tracks_partial_consumption() {
        with_triangle_tds(|tds, _| {
            let mut tds = tds.clone();
            for explicit_offsets in [false, true] {
                if explicit_offsets {
                    let simplex_key = tds.simplex_keys().next().unwrap();
                    tds.simplex_mut(simplex_key)
                        .unwrap()
                        .set_periodic_vertex_offsets(vec![[0; 2]; 3])
                        .unwrap();
                }
                assert_eq!(EdgeSet::from_tds(&tds, 3).into_iter().count(), 3);
                let mut iter = EdgeSet::from_tds(&tds, 3).into_iter();
                assert!(iter.next().is_some());
                assert_eq!(iter.size_hint(), (2, Some(2)));
                assert_eq!(iter.count(), 2);
                let mut empty = EdgeSet::from_tds(&tds, 3).into_iter();
                assert_eq!(empty.by_ref().take(3).count(), 3);
                assert_eq!(empty.count(), 0);
            }
        });
    }

    #[test]
    fn edge_set_preserves_one_vertex_t2_self_loops() {
        let (tds, vertex) = Tds::periodic_one_vertex_t2_for_test();
        let edges: HashSet<_> = EdgeSet::from_tds(&tds, 1).into_iter().collect();
        // The square's horizontal, vertical and diagonal edge classes survive
        // the quotient, although every endpoint is the same stored vertex.
        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges
                .iter()
                .map(|edge| edge.displacement())
                .collect::<HashSet<_>>(),
            HashSet::from([[1, 0], [0, 1], [1, 1]])
        );
        for edge in edges {
            assert_eq!(edge.endpoints(), (vertex, vertex));
            assert_eq!(edge.view(&tds).unwrap().incident_occurrences().count(), 2);
        }
    }

    #[test]
    fn periodic_traversal_self_loops_retain_classes_and_occurrences() {
        let (tds, vertex) = Tds::periodic_one_vertex_t2_for_test();
        let mut edges = HashSet::new();
        for (simplex_key, _) in tds.simplices() {
            for slots in [[0, 1], [1, 2], [2, 0]] {
                let edge = EdgeKey::try_from_simplex(&tds, simplex_key, slots).unwrap();
                assert_eq!(edge.endpoints(), (vertex, vertex));
                edges.insert(edge);
            }
        }
        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges
                .iter()
                .map(|edge| edge.displacement())
                .collect::<HashSet<_>>(),
            [[1, 0], [0, 1], [1, 1]].into_iter().collect()
        );
        for edge in edges {
            let view = edge.view(&tds).unwrap();
            assert_eq!(view.incident_simplices().len(), 2);
            assert_eq!(view.incident_occurrences().count(), 2);
        }
        assert!(matches!(
            EdgeKey::try_new(&tds, vertex, vertex),
            Err(EdgeKeyError::AmbiguousEndpoints { .. })
        ));
    }

    #[test]
    fn periodic_traversal_endpoint_lookup_preserves_missing_reverse_incidence() {
        let (mut tds, vertices) = Tds::periodic_grid_t2_for_test(false);
        let a = vertices[0];
        let b = vertices[1];
        let edge = EdgeKey::try_new(&tds, a, b).unwrap();
        let missing = edge.view(&tds).unwrap().incident_simplices().to_vec();
        let unrelated: Vec<_> = tds
            .simplex_keys_containing_vertex(a)
            .filter(|&key| !tds.simplex(key).unwrap().contains_vertex(b))
            .collect();
        assert_ne!(unrelated, []);
        tds.clear_vertex_incidence_for_test(a);
        for key in unrelated {
            tds.add_simplex_to_vertex_incidence_for_test(a, key);
        }
        assert!(matches!(
            EdgeKey::try_new(&tds, a, b),
            Err(EdgeKeyError::MissingVertexIncidence { vertex_key, simplex_key })
                if vertex_key == a && missing.contains(&simplex_key)
        ));
    }

    #[test]
    fn edge_key_is_canonical() {
        with_triangle_tds(|tds, [a, b, _c]| {
            let e1 = EdgeKey::try_new(tds, a, b).unwrap();
            let e2 = EdgeKey::try_new(tds, b, a).unwrap();

            assert_eq!(e1, e2);

            // Ensure the ordering invariant holds.
            assert!(e1.v0().data().as_ffi() <= e1.v1().data().as_ffi());
        });
    }

    #[test]
    fn edge_key_endpoints_roundtrip() {
        with_triangle_tds(|tds, [a, b, _c]| {
            let e = EdgeKey::try_new(tds, b, a).unwrap();
            let (v0, v1) = e.endpoints();

            assert_eq!(v0, e.v0());
            assert_eq!(v1, e.v1());
            assert!(v0.data().as_ffi() <= v1.data().as_ffi());
            assert_eq!(EdgeKey::try_new(tds, a, b).unwrap(), e);
        });
    }

    #[test]
    fn edge_key_rejects_duplicate_endpoint() {
        with_triangle_tds(|tds, [a, _b, _c]| {
            assert_eq!(
                EdgeKey::try_new(tds, a, a),
                Err(EdgeKeyError::DuplicateEndpoint { endpoint: a })
            );
        });
    }

    #[test]
    fn edge_key_rejects_missing_endpoint() {
        with_triangle_tds(|tds, [a, _b, _c]| {
            let missing = VertexKey::default();
            assert_eq!(
                EdgeKey::try_new(tds, a, missing),
                Err(EdgeKeyError::MissingEndpoint { endpoint: missing })
            );
        });
    }

    #[test]
    fn edge_key_rejects_missing_first_endpoint() {
        with_triangle_tds(|tds, [_a, b, _c]| {
            let missing = VertexKey::default();
            assert_eq!(
                EdgeKey::try_new(tds, missing, b),
                Err(EdgeKeyError::MissingEndpoint { endpoint: missing })
            );
        });
    }

    #[test]
    fn edge_key_rejects_live_vertices_without_edge() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let simplices = [vec![0, 1, 2], vec![0, 2, 3]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let keys: Vec<VertexKey> = tds.vertex_keys().collect();
        let missing_edge = keys.iter().enumerate().find_map(|(i, &a)| {
            keys.iter().skip(i + 1).copied().find_map(|b| {
                matches!(
                    EdgeKey::try_new(&tds, a, b),
                    Err(EdgeKeyError::EdgeNotFound { .. })
                )
                .then_some((a, b))
            })
        });
        let (a, b) = missing_edge.expect("square triangulation should have one missing diagonal");
        assert_eq!(
            EdgeKey::try_new(&tds, a, b),
            Err(EdgeKeyError::EdgeNotFound { v0: a, v1: b })
        );
    }

    #[test]
    fn edge_key_is_hashable_and_orderable() {
        with_triangle_tds(|tds, [a, b, c]| {
            let mut hash_set: HashSet<EdgeKey<2>> = HashSet::new();
            hash_set.insert(EdgeKey::try_new(tds, a, b).unwrap());
            hash_set.insert(EdgeKey::try_new(tds, b, a).unwrap());
            hash_set.insert(EdgeKey::try_new(tds, a, c).unwrap());
            assert_eq!(hash_set.len(), 2);

            let mut btree_set: BTreeSet<EdgeKey<2>> = BTreeSet::new();
            btree_set.insert(EdgeKey::try_new(tds, a, b).unwrap());
            btree_set.insert(EdgeKey::try_new(tds, b, a).unwrap());
            btree_set.insert(EdgeKey::try_new(tds, a, c).unwrap());
            assert_eq!(btree_set.len(), 2);
        });
    }

    #[test]
    fn edge_view_exposes_endpoint_vertices_and_key() {
        with_triangle_tds(|tds, [a, b, _c]| {
            let key = EdgeKey::try_new(tds, a, b).unwrap();
            let view = key.view(tds).unwrap();
            let (first, second) = view.vertices();

            assert_eq!(view.key(), key);
            assert!(ptr::eq(view.tds(), tds));
            assert_eq!(view.endpoint_keys(), key.endpoints());
            assert_eq!(first.uuid(), tds.vertex(view.key().v0()).unwrap().uuid());
            assert_eq!(second.uuid(), tds.vertex(view.key().v1()).unwrap().uuid());
        });
    }

    #[test]
    fn edge_view_clone_debug_and_equality_use_owner_and_key() {
        with_triangle_tds(|tds, [a, b, c]| {
            let first = EdgeKey::try_new(tds, a, b).unwrap().view(tds).unwrap();
            let first_clone = first.clone();
            let second = EdgeKey::try_new(tds, a, c).unwrap().view(tds).unwrap();

            assert_eq!(first, first_clone);
            assert_ne!(first, second);

            let debug = format!("{first:?}");
            assert!(debug.contains("EdgeView"));
            assert!(debug.contains("incident_simplices"));
        });
    }

    #[test]
    fn edge_view_enumerates_incident_simplices_from_incidence_index() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let c0 = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v1, v2]).unwrap())
            .unwrap();
        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v1, v0, v3]).unwrap())
            .unwrap();
        let _only_v0 = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v2, v3]).unwrap())
            .unwrap();

        let edge = EdgeKey::try_new(&tds, v1, v0).unwrap();
        let incident: HashSet<_> = edge
            .view(&tds)
            .unwrap()
            .incident_simplices()
            .iter()
            .copied()
            .collect();

        assert_eq!(incident, HashSet::from([c0, c1]));
    }

    #[test]
    fn edge_key_rejects_partial_missing_reverse_vertex_incidence() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let retained_simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v1, v2]).unwrap())
            .unwrap();
        let missing_simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v1, v0, v3]).unwrap())
            .unwrap();
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let (_first, second) = edge.endpoints();

        tds.clear_vertex_incidence_for_test(second);
        tds.add_simplex_to_vertex_incidence_for_test(second, retained_simplex);

        let expected_error = EdgeKeyError::MissingVertexIncidence {
            vertex_key: second,
            simplex_key: missing_simplex,
        };
        assert_eq!(EdgeKey::try_new(&tds, v0, v1), Err(expected_error));
        assert_eq!(edge.view(&tds), Err(expected_error));
    }

    #[test]
    fn edge_view_rejects_stale_vertex_incidence() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let stale_simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v1, v2, v3]).unwrap())
            .unwrap();
        tds.remove_simplex_storage_only_for_test(stale_simplex);

        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        assert_eq!(
            EdgeKey::try_new(&tds, v0, v1),
            Err(EdgeKeyError::DanglingVertexIncidence {
                vertex_key: edge.v0(),
                simplex_key: stale_simplex
            })
        );
        assert_eq!(
            edge.view(&tds),
            Err(EdgeKeyError::DanglingVertexIncidence {
                vertex_key: edge.v0(),
                simplex_key: stale_simplex
            })
        );
    }

    #[test]
    fn edge_view_rejects_missing_reverse_vertex_incidence() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v1, v2]).unwrap())
            .unwrap();
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let (_first, second) = edge.endpoints();

        tds.clear_vertex_incidence_for_test(second);

        assert_eq!(
            EdgeKey::try_new(&tds, v0, v1),
            Err(EdgeKeyError::MissingVertexIncidence {
                vertex_key: second,
                simplex_key
            })
        );
        assert_eq!(
            edge.view(&tds),
            Err(EdgeKeyError::MissingVertexIncidence {
                vertex_key: second,
                simplex_key
            })
        );
    }

    #[test]
    fn edge_view_rejects_missing_forward_vertex_incidence() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new(vec![v0, v1, v2]).unwrap())
            .unwrap();
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let (first, _second) = edge.endpoints();

        tds.clear_vertex_incidence_for_test(first);

        assert_eq!(
            EdgeKey::try_new(&tds, v0, v1),
            Err(EdgeKeyError::MissingVertexIncidence {
                vertex_key: first,
                simplex_key
            })
        );
        assert_eq!(
            edge.view(&tds),
            Err(EdgeKeyError::MissingVertexIncidence {
                vertex_key: first,
                simplex_key
            })
        );
    }

    #[test]
    fn edge_view_rejects_live_endpoints_without_stored_edge() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();

        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let (v0, v1) = edge.endpoints();

        assert_eq!(edge.view(&tds), Err(EdgeKeyError::EdgeNotFound { v0, v1 }));
    }

    #[test]
    fn edge_lookup_rejects_incidence_to_a_simplex_without_its_endpoint() {
        with_triangle_tds(|tds, [a, _, _]| {
            let mut tds = tds.clone();
            let unrelated = tds
                .insert_vertex_with_mapping(vertex![2.0, 2.0].unwrap())
                .unwrap();
            let simplex_key = tds.simplex_keys().next().unwrap();
            tds.add_simplex_to_vertex_incidence_for_test(unrelated, simplex_key);
            for (first, second) in [(a, unrelated), (unrelated, a)] {
                assert_eq!(
                    EdgeKey::try_new(&tds, first, second),
                    Err(EdgeKeyError::VertexIncidenceMismatch {
                        vertex_key: unrelated,
                        simplex_key
                    })
                );
            }
        });
    }

    #[test]
    fn edge_view_rejects_stale_endpoint_handles() {
        with_triangle_tds(|_tds, [a, b, _c]| {
            let stale = EdgeKey::from_validated_endpoints(a, b);
            let empty: Tds<(), (), 2> = Tds::empty();

            assert_eq!(
                stale.view(&empty),
                Err(EdgeKeyError::MissingEndpoint {
                    endpoint: stale.v0()
                })
            );
        });
    }
}
