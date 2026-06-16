//! Canonical edge identifiers for public topology traversal.
//!
//! Downstream code frequently needs a stable, comparable identifier for an *edge* in the
//! triangulation topology. Since edges are not stored explicitly (they are inferred from
//! maximal simplices), we expose a lightweight `EdgeKey` that:
//!
//! - identifies an edge by two live endpoint [`VertexKey`]s that share a simplex
//! - canonicalizes endpoint ordering so `(a, b)` and `(b, a)` map to the same edge
//! - is `Copy`/`Hash`/`Ord` for fast use in sets and maps
//!
//! ## Determinism
//!
//! `EdgeKey` ordering is **not** guaranteed to be deterministic across processes or
//! serialization round-trips, because [`VertexKey`] ordering is derived from internal
//! slotmap keys. If you need a deterministic order, sort using stable vertex UUIDs.

#![forbid(unsafe_code)]

use crate::core::tds::{Tds, VertexKey};
use slotmap::Key;
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
///     Vertex::<(), _>::try_new([0.0, 0.0])?,
///     Vertex::<(), _>::try_new([1.0, 0.0])?,
///     Vertex::<(), _>::try_new([0.0, 1.0])?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
    /// live in `tds`, or [`EdgeKeyError::EdgeNotFound`] if no simplex contains
    /// both endpoints.
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
    ///     Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
        if !tds
            .simplices()
            .any(|(_simplex_key, simplex)| simplex.contains_vertex(a) && simplex.contains_vertex(b))
        {
            return Err(EdgeKeyError::EdgeNotFound { v0: a, v1: b });
        }

        Ok(Self::from_validated_endpoints(a, b))
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
    ///     Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
    ///     Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
    ///     Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{DelaunayTriangulationBuilder, Vertex};
    use std::collections::{BTreeSet, HashSet};

    fn with_triangle_tds(test: impl FnOnce(&Tds<(), (), 2>, [VertexKey; 3])) {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
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
    fn edge_key_rejects_live_vertices_without_edge() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
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
}
