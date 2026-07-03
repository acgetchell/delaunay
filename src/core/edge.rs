//! Canonical edge identifiers for public topology traversal.
//!
//! Downstream code frequently needs a stable, comparable identifier for an *edge* in the
//! triangulation topology. Since edges are not stored explicitly (they are inferred from
//! maximal simplices), we expose a lightweight `EdgeKey` that:
//!
//! - identifies an edge by two live endpoint [`VertexKey`]s that share a simplex
//! - canonicalizes endpoint ordering so `(a, b)` and `(b, a)` map to the same edge
//! - is `Copy`/`Hash`/`Ord` for fast use in sets and maps
//! - can be revalidated into an [`EdgeView`] for borrowed access to live topology
//!
//! ## Determinism
//!
//! `EdgeKey` ordering is **not** guaranteed to be deterministic across processes or
//! serialization round-trips, because [`VertexKey`] ordering is derived from internal
//! slotmap keys. If you need a deterministic order, sort using stable vertex UUIDs.

#![forbid(unsafe_code)]

use crate::core::collections::SimplexKeyBuffer;
use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use slotmap::Key;
use std::fmt;
use thiserror::Error;

/// Error returned when constructing an [`EdgeKey`] from invalid endpoints.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum EdgeKeyError {
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
/// storage-local [`VertexKey`] endpoints, so it is valid only for the live [`Tds`] whose
/// endpoints were checked by [`EdgeKey::try_new`]. Persist or compare edges across I/O
/// boundaries with stable vertex UUIDs instead.
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
/// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
///     return Ok(());
/// };
/// let a = simplex.vertices()[0];
/// let b = simplex.vertices()[1];
/// let edge = EdgeKey::try_new(dt.tds(), a, b)?;
/// assert_eq!(edge.endpoints(), (edge.v0(), edge.v1()));
/// # Ok(())
/// # }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeKey {
    v0: VertexKey,
    v1: VertexKey,
}

impl EdgeKey {
    /// Creates a new canonical edge key.
    ///
    /// The endpoints must be distinct, present in `tds`, and co-incident in at
    /// least one simplex. Endpoints are reordered so that `v0 <= v1` under the
    /// internal key order.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError::DuplicateEndpoint`] if both endpoints are the
    /// same vertex, [`EdgeKeyError::MissingEndpoint`] if either endpoint is not
    /// live in `tds`, [`EdgeKeyError::EdgeNotFound`] if no simplex contains both
    /// endpoints, or another [`EdgeKeyError`] if the maintained vertex-incidence
    /// metadata needed to prove the edge is inconsistent with simplex storage.
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e1 = EdgeKey::try_new(dt.tds(), a, b)?;
    /// let e2 = EdgeKey::try_new(dt.tds(), b, a)?;
    /// assert_eq!(e1, e2);
    /// assert!(e1.v0() <= e1.v1());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        a: VertexKey,
        b: VertexKey,
    ) -> Result<Self, EdgeKeyError> {
        if a == b {
            return Err(EdgeKeyError::DuplicateEndpoint { endpoint: a });
        }
        if tds.vertex(a).is_none() {
            return Err(EdgeKeyError::MissingEndpoint { endpoint: a });
        }
        if tds.vertex(b).is_none() {
            return Err(EdgeKeyError::MissingEndpoint { endpoint: b });
        }
        let key = Self::from_validated_endpoints(a, b);
        let _validated_edge_star = Self::parse_edge_star(tds, key)?;

        Ok(key)
    }

    /// Parses the edge star from mutually complete, internally valid endpoint incidence.
    ///
    /// The success path inspects endpoint stars instead of scanning every
    /// simplex. Both endpoint stars are checked for every simplex containing the
    /// opposite endpoint, so partially stale reverse incidence is rejected before
    /// returning a key. A full simplex-storage scan is reserved for the cold path
    /// where neither endpoint star reports the edge, so callers still get a
    /// precise `MissingEdgeIncidence` diagnostic when storage and incidence
    /// diverge.
    fn parse_edge_star<U, V, const D: usize>(
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

        let primary_edge_simplices = Self::edge_simplices_in_star(tds, primary, secondary)?;
        let secondary_edge_simplices = Self::edge_simplices_in_star(tds, secondary, primary)?;

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
        if EdgeView::endpoints_share_stored_simplex(tds, v0, v1) {
            return Err(EdgeKeyError::MissingEdgeIncidence { v0, v1 });
        }
        Err(EdgeKeyError::EdgeNotFound { v0, v1 })
    }

    /// Finds all simplices in `source`'s validated star that also contain `target`.
    fn edge_simplices_in_star<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        source: VertexKey,
        target: VertexKey,
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
            if simplex.contains_vertex(target) {
                edge_simplices.push(simplex_key);
            }
        }
        Ok(edge_simplices)
    }

    /// Creates a canonical edge key from endpoints already known to be distinct.
    #[inline]
    #[must_use]
    pub(crate) fn from_validated_endpoints(a: VertexKey, b: VertexKey) -> Self {
        // Use the raw slotmap key representation for ordering to avoid relying on
        // any higher-level semantic ordering.
        let a_raw = a.data().as_ffi();
        let b_raw = b.data().as_ffi();

        if a_raw <= b_raw {
            Self { v0: a, v1: b }
        } else {
            Self { v0: b, v1: a }
        }
    }

    /// Returns the first (canonical) endpoint.
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[1];
    /// let b = simplex.vertices()[0];
    ///
    /// let e = EdgeKey::try_new(dt.tds(), a, b)?;
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e = EdgeKey::try_new(dt.tds(), a, b)?;
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let a = simplex.vertices()[0];
    /// let b = simplex.vertices()[1];
    ///
    /// let e = EdgeKey::try_new(dt.tds(), a, b)?;
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
    /// edge's incident simplex star.
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
    /// let Some((_simplex_key, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let edge = EdgeKey::try_new(dt.tds(), simplex.vertices()[0], simplex.vertices()[1])?;
    /// let view = edge.view(dt.tds())?;
    /// assert_eq!(view.endpoint_keys(), edge.endpoints());
    /// # Ok(())
    /// # }
    /// ```
    pub fn view<U, V, const D: usize>(
        self,
        tds: &Tds<U, V, D>,
    ) -> Result<EdgeView<'_, U, V, D>, EdgeKeyError> {
        EdgeView::try_new(tds, self)
    }
}

/// Borrowed live-TDS view over an [`EdgeKey`].
///
/// `EdgeView` is a non-durable topology view. It borrows one in-memory [`Tds`]
/// and revalidates a copyable [`EdgeKey`] before exposing endpoint vertices and
/// the edge's incident D-simplices. Persist stable vertex UUIDs or a full TDS
/// snapshot instead of serializing edge views.
#[must_use]
pub struct EdgeView<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    key: EdgeKey,
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
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::{EdgeKey, EdgeKeyError, EdgeView};
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
    /// let key = EdgeKey::try_new(dt.tds(), simplex.vertices()[0], simplex.vertices()[1])?;
    /// let view = EdgeView::try_new(dt.tds(), key)?;
    /// assert_eq!(view.key(), key);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(tds: &'tds Tds<U, V, D>, key: EdgeKey) -> Result<Self, EdgeKeyError> {
        let (v0, v1) = key.endpoints();
        if v0 == v1 {
            return Err(EdgeKeyError::DuplicateEndpoint { endpoint: v0 });
        }
        let first = tds
            .vertex(v0)
            .ok_or(EdgeKeyError::MissingEndpoint { endpoint: v0 })?;
        let second = tds
            .vertex(v1)
            .ok_or(EdgeKeyError::MissingEndpoint { endpoint: v1 })?;

        let key = EdgeKey::from_validated_endpoints(v0, v1);
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
    pub const fn key(&self) -> EdgeKey {
        self.key
    }

    /// Returns the borrowed TDS backing this view.
    #[inline]
    #[must_use]
    pub const fn tds(&self) -> &'tds Tds<U, V, D> {
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
    /// The star was parsed and validated during [`Self::try_new`].
    #[must_use]
    pub fn incident_simplices(&self) -> &[SimplexKey] {
        self.incident_simplices.as_slice()
    }

    /// Scans canonical simplex storage to classify a missing edge-incidence entry precisely.
    ///
    /// Returning `true` means the edge exists in simplex storage and the
    /// incidence index is missing it; returning `false` means the live endpoints
    /// do not currently form a stored edge.
    fn endpoints_share_stored_simplex(tds: &Tds<U, V, D>, v0: VertexKey, v1: VertexKey) -> bool {
        tds.simplices().any(|(_simplex_key, simplex)| {
            simplex.contains_vertex(v0) && simplex.contains_vertex(v1)
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
    use crate::core::simplex::Simplex;
    use crate::prelude::DelaunayTriangulationBuilder;
    use crate::vertex;
    use std::{
        collections::{BTreeSet, HashSet},
        ptr,
    };

    fn with_triangle_tds(test: impl FnOnce(&Tds<(), (), 2>, [VertexKey; 3])) {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let simplex = dt.simplices().next().unwrap().1;
        let vertices = simplex.vertices();
        let simplex_vertices = [vertices[0], vertices[1], vertices[2]];
        test(dt.tds(), simplex_vertices);
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
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let keys: Vec<VertexKey> = dt.tds().vertex_keys().collect();
        let missing_edge = keys.iter().enumerate().find_map(|(i, &a)| {
            keys.iter().skip(i + 1).copied().find_map(|b| {
                matches!(
                    EdgeKey::try_new(dt.tds(), a, b),
                    Err(EdgeKeyError::EdgeNotFound { .. })
                )
                .then_some((a, b))
            })
        });
        let (a, b) = missing_edge.expect("square triangulation should have one missing diagonal");
        assert_eq!(
            EdgeKey::try_new(dt.tds(), a, b),
            Err(EdgeKeyError::EdgeNotFound { v0: a, v1: b })
        );
    }

    #[test]
    fn edge_key_is_hashable_and_orderable() {
        with_triangle_tds(|tds, [a, b, c]| {
            let mut hash_set: HashSet<EdgeKey> = HashSet::new();
            hash_set.insert(EdgeKey::try_new(tds, a, b).unwrap());
            hash_set.insert(EdgeKey::try_new(tds, b, a).unwrap());
            hash_set.insert(EdgeKey::try_new(tds, a, c).unwrap());
            assert_eq!(hash_set.len(), 2);

            let mut btree_set: BTreeSet<EdgeKey> = BTreeSet::new();
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
