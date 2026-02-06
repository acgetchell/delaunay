//! Canonical edge identifiers for public topology traversal.
//!
//! Downstream code frequently needs a stable, comparable identifier for an *edge* in the
//! triangulation topology. Since edges are not stored explicitly (they are inferred from
//! maximal cells), we expose a lightweight `EdgeKey` that:
//!
//! - identifies an edge purely by its two endpoint [`VertexKey`]s
//! - canonicalizes endpoint ordering so `(a, b)` and `(b, a)` map to the same edge
//! - is `Copy`/`Hash`/`Ord` for fast use in sets and maps
//!
//! ## Determinism
//!
//! `EdgeKey` ordering is **not** guaranteed to be deterministic across processes or
//! serialization round-trips, because [`VertexKey`] ordering is derived from internal
//! slotmap keys. If you need a deterministic order, sort using stable vertex UUIDs.

use crate::core::triangulation_data_structure::VertexKey;
use slotmap::Key;

/// Canonical identifier for an (undirected) edge.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::edge::EdgeKey;
/// use delaunay::core::triangulation_data_structure::VertexKey;
/// use slotmap::KeyData;
///
/// let a = VertexKey::from(KeyData::from_ffi(1));
/// let b = VertexKey::from(KeyData::from_ffi(2));
/// let edge = EdgeKey::new(a, b);
/// assert_eq!(edge.endpoints(), (edge.v0(), edge.v1()));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeKey {
    v0: VertexKey,
    v1: VertexKey,
}

impl EdgeKey {
    /// Creates a new canonical edge key.
    ///
    /// The endpoints are reordered so that `v0 <= v1` under the internal key order.
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
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let mut it = tri.vertices();
    /// let a = it.next().unwrap().0;
    /// let b = it.next().unwrap().0;
    ///
    /// let e1 = EdgeKey::new(a, b);
    /// let e2 = EdgeKey::new(b, a);
    /// assert_eq!(e1, e2);
    /// assert!(e1.v0() <= e1.v1());
    /// ```
    #[must_use]
    pub fn new(a: VertexKey, b: VertexKey) -> Self {
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
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let mut it = tri.vertices();
    /// let a = it.next().unwrap().0;
    /// let b = it.next().unwrap().0;
    ///
    /// let e = EdgeKey::new(a, b);
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
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
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let mut it = tri.vertices();
    /// let a = it.next().unwrap().0;
    /// let b = it.next().unwrap().0;
    ///
    /// let e = EdgeKey::new(a, b);
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
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
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let mut it = tri.vertices();
    /// let a = it.next().unwrap().0;
    /// let b = it.next().unwrap().0;
    ///
    /// let e = EdgeKey::new(a, b);
    /// let (v0, v1) = e.endpoints();
    /// assert_eq!(v0, e.v0());
    /// assert_eq!(v1, e.v1());
    /// assert!(v0 <= v1);
    /// ```
    #[inline]
    #[must_use]
    pub const fn endpoints(self) -> (VertexKey, VertexKey) {
        (self.v0, self.v1)
    }
}

impl From<(VertexKey, VertexKey)> for EdgeKey {
    #[inline]
    fn from((a, b): (VertexKey, VertexKey)) -> Self {
        Self::new(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    #[test]
    fn edge_key_is_canonical() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());

        let e1 = EdgeKey::new(a, b);
        let e2 = EdgeKey::new(b, a);

        assert_eq!(e1, e2);

        // Ensure the ordering invariant holds.
        assert!(e1.v0().data().as_ffi() <= e1.v1().data().as_ffi());
    }

    #[test]
    fn edge_key_endpoints_roundtrip() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());

        let e = EdgeKey::new(b, a);
        let (v0, v1) = e.endpoints();

        assert_eq!(v0, e.v0());
        assert_eq!(v1, e.v1());
        assert!(v0.data().as_ffi() <= v1.data().as_ffi());

        assert_eq!(EdgeKey::from((a, b)), EdgeKey::new(a, b));
    }

    #[test]
    fn edge_key_is_hashable_and_orderable() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());
        let c = vertices.insert(());

        let mut hash_set: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
        hash_set.insert(EdgeKey::new(a, b));
        hash_set.insert(EdgeKey::new(b, a));
        hash_set.insert(EdgeKey::new(a, c));
        assert_eq!(hash_set.len(), 2);

        let mut btree_set: std::collections::BTreeSet<EdgeKey> = std::collections::BTreeSet::new();
        btree_set.insert(EdgeKey::new(a, b));
        btree_set.insert(EdgeKey::new(b, a));
        btree_set.insert(EdgeKey::new(a, c));
        assert_eq!(btree_set.len(), 2);
    }
}
