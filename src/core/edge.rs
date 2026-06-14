//! Canonical edge identifiers for public topology traversal.
//!
//! Downstream code frequently needs a stable, comparable identifier for an *edge* in the
//! triangulation topology. Since edges are not stored explicitly (they are inferred from
//! maximal simplices), we expose a lightweight `EdgeKey` that:
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

#![forbid(unsafe_code)]

use crate::core::tds::VertexKey;
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
}

/// Canonical identifier for an (undirected) edge.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::EdgeKey;
/// use delaunay::prelude::tds::VertexKey;
/// use slotmap::KeyData;
///
/// let a = VertexKey::from(KeyData::from_ffi(1));
/// let b = VertexKey::from(KeyData::from_ffi(2));
/// let edge = EdgeKey::try_new(a, b)?;
/// assert_eq!(edge.endpoints(), (edge.v0(), edge.v1()));
/// # Ok::<(), delaunay::prelude::tds::EdgeKeyError>(())
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeKey {
    v0: VertexKey,
    v1: VertexKey,
}

impl EdgeKey {
    /// Creates a new canonical edge key.
    ///
    /// The endpoints must be distinct and are reordered so that `v0 <= v1` under
    /// the internal key order.
    ///
    /// # Errors
    ///
    /// Returns [`EdgeKeyError::DuplicateEndpoint`] if both endpoints are the same vertex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::{EdgeKey, VertexKey};
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(1));
    /// let b = VertexKey::from(KeyData::from_ffi(2));
    ///
    /// let e1 = EdgeKey::try_new(a, b)?;
    /// let e2 = EdgeKey::try_new(b, a)?;
    /// assert_eq!(e1, e2);
    /// assert!(e1.v0() <= e1.v1());
    /// # Ok::<(), delaunay::prelude::tds::EdgeKeyError>(())
    /// ```
    pub fn try_new(a: VertexKey, b: VertexKey) -> Result<Self, EdgeKeyError> {
        if a == b {
            return Err(EdgeKeyError::DuplicateEndpoint { endpoint: a });
        }

        Ok(Self::from_validated_endpoints(a, b))
    }

    /// Creates a canonical edge key from endpoints already known to be distinct.
    #[inline]
    #[must_use]
    pub(crate) fn from_validated_endpoints(a: VertexKey, b: VertexKey) -> Self {
        debug_assert_ne!(a, b, "validated edge endpoints must be distinct");

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
    /// use delaunay::prelude::tds::{EdgeKey, VertexKey};
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(2));
    /// let b = VertexKey::from(KeyData::from_ffi(1));
    ///
    /// let e = EdgeKey::try_new(a, b)?;
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
    /// # Ok::<(), delaunay::prelude::tds::EdgeKeyError>(())
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
    /// use delaunay::prelude::tds::{EdgeKey, VertexKey};
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(1));
    /// let b = VertexKey::from(KeyData::from_ffi(2));
    ///
    /// let e = EdgeKey::try_new(a, b)?;
    /// let v0 = e.v0();
    /// let v1 = e.v1();
    /// assert!(v0 <= v1);
    /// # Ok::<(), delaunay::prelude::tds::EdgeKeyError>(())
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
    /// use delaunay::prelude::tds::{EdgeKey, VertexKey};
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(1));
    /// let b = VertexKey::from(KeyData::from_ffi(2));
    ///
    /// let e = EdgeKey::try_new(a, b)?;
    /// let (v0, v1) = e.endpoints();
    /// assert_eq!(v0, e.v0());
    /// assert_eq!(v1, e.v1());
    /// assert!(v0 <= v1);
    /// # Ok::<(), delaunay::prelude::tds::EdgeKeyError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn endpoints(self) -> (VertexKey, VertexKey) {
        (self.v0, self.v1)
    }
}

impl TryFrom<(VertexKey, VertexKey)> for EdgeKey {
    type Error = EdgeKeyError;

    #[inline]
    fn try_from((a, b): (VertexKey, VertexKey)) -> Result<Self, Self::Error> {
        Self::try_new(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn edge_key_is_canonical() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());

        let e1 = EdgeKey::try_new(a, b).unwrap();
        let e2 = EdgeKey::try_new(b, a).unwrap();

        assert_eq!(e1, e2);

        // Ensure the ordering invariant holds.
        assert!(e1.v0().data().as_ffi() <= e1.v1().data().as_ffi());
    }

    #[test]
    fn edge_key_endpoints_roundtrip() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());

        let e = EdgeKey::try_new(b, a).unwrap();
        let (v0, v1) = e.endpoints();

        assert_eq!(v0, e.v0());
        assert_eq!(v1, e.v1());
        assert!(v0.data().as_ffi() <= v1.data().as_ffi());

        assert_eq!(
            EdgeKey::try_from((a, b)).unwrap(),
            EdgeKey::try_new(a, b).unwrap()
        );
    }

    #[test]
    fn edge_key_rejects_duplicate_endpoint() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());

        assert_eq!(
            EdgeKey::try_new(a, a),
            Err(EdgeKeyError::DuplicateEndpoint { endpoint: a })
        );
        assert_eq!(
            EdgeKey::try_from((a, a)),
            Err(EdgeKeyError::DuplicateEndpoint { endpoint: a })
        );
    }

    #[test]
    fn edge_key_is_hashable_and_orderable() {
        let mut vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let a = vertices.insert(());
        let b = vertices.insert(());
        let c = vertices.insert(());

        let mut hash_set: HashSet<EdgeKey> = HashSet::new();
        hash_set.insert(EdgeKey::try_new(a, b).unwrap());
        hash_set.insert(EdgeKey::try_new(b, a).unwrap());
        hash_set.insert(EdgeKey::try_new(a, c).unwrap());
        assert_eq!(hash_set.len(), 2);

        let mut btree_set: BTreeSet<EdgeKey> = BTreeSet::new();
        btree_set.insert(EdgeKey::try_new(a, b).unwrap());
        btree_set.insert(EdgeKey::try_new(b, a).unwrap());
        btree_set.insert(EdgeKey::try_new(a, c).unwrap());
        assert_eq!(btree_set.len(), 2);
    }
}
