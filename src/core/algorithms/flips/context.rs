//! Flip evidence, feasibility reports, and topology handles.

#![forbid(unsafe_code)]

use super::{
    BistellarFlipKind, FlipDirection, FlipError, Hash, Key, MAX_PRACTICAL_DIMENSION_SIZE,
    SimplexKey, SimplexKeyBuffer, SmallBuffer, Tds, TriangleHandleError, VertexKey,
};

/// Information about a successful flip.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarFlipKind, FlipDirection, FlipInfo};
/// use delaunay::prelude::collections::{SimplexKeyBuffer, SmallBuffer, MAX_PRACTICAL_DIMENSION_SIZE};
/// use delaunay::prelude::tds::{SimplexKey, VertexKey};
/// use slotmap::KeyData;
///
/// # fn main() -> Result<(), delaunay::flips::BistellarFlipKindError> {
/// let mut removed_simplices = SimplexKeyBuffer::new();
/// removed_simplices.push(SimplexKey::from(KeyData::from_ffi(1)));
/// let mut new_simplices = SimplexKeyBuffer::new();
/// new_simplices.push(SimplexKey::from(KeyData::from_ffi(2)));
///
/// let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(3)));
/// let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
///     SmallBuffer::new();
/// inserted_face_vertices.push(VertexKey::from(KeyData::from_ffi(4)));
///
/// let info: FlipInfo<3> = FlipInfo {
///     kind: BistellarFlipKind::try_k2(3)?,
///     direction: FlipDirection::Forward,
///     removed_simplices,
///     new_simplices,
///     removed_face_vertices,
///     inserted_face_vertices,
/// };
/// assert_eq!(info.kind.k(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FlipInfo<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Simplices removed by the flip.
    pub removed_simplices: SimplexKeyBuffer,
    /// Newly created simplices.
    pub new_simplices: SimplexKeyBuffer,
    /// The removed-face simplex (shared by removed simplices).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// The inserted-face simplex (complementary simplex).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
}

/// Metadata returned by immutable bistellar flip feasibility checks.
///
/// A feasibility report describes the local move that passed deterministic
/// pre-mutation validation. It intentionally omits `new_simplices`, because
/// those runtime keys are allocated only by the mutating flip executor.
/// For forward k=1 insertion, [`inserted_face_vertices`](Self::inserted_face_vertices)
/// is `None` for the same reason: the inserted vertex key does not exist until
/// the mutation commits. Other move kinds report the already-live inserted face.
/// Feasibility checks build only local replacement buffers and do not clone the
/// full triangulation. They report deterministic failures on the inspected
/// state; allocation failure, process termination, and other environmental
/// failures remain outside the feasibility guarantee.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{BistellarFlips, FlipDirection};
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let vertex = delaunay::vertex![0.25, 0.25]?;
/// let feasibility = dt.can_flip_k1_insert(simplex_key, &vertex)?;
/// assert_eq!((feasibility.kind.k(), feasibility.kind.d()), (1, 2));
/// assert_eq!(feasibility.direction, FlipDirection::Forward);
/// assert!(feasibility.inserted_face_vertices.is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct FlipFeasibility<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Simplices that the corresponding mutating flip would remove.
    pub removed_simplices: SimplexKeyBuffer,
    /// The removed-face simplex shared by the removed simplices.
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// The inserted complementary face when all of its vertices already exist.
    ///
    /// This is `None` for forward k=1 insertion because the inserted vertex key
    /// is allocated by the mutating executor.
    pub inserted_face_vertices: Option<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
}
/// Const-generic flip context for a k-move (forward or inverse).
#[derive(Debug, Clone)]
pub(crate) struct FlipContext<const D: usize, const K: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−K).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension K−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Simplices removed by the flip (count = K).
    pub removed_simplices: SimplexKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Runtime-k flip context for moves where k depends on D.
#[derive(Debug, Clone)]
pub(crate) struct FlipContextDyn<const D: usize> {
    /// Vertices of the removed-face simplex (dimension D+1−k).
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted-face simplex (dimension k−1).
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Simplices removed by the flip (count = k).
    pub removed_simplices: SimplexKeyBuffer,
    /// Flip direction (forward/inverse).
    pub direction: FlipDirection,
}

/// Canonical handle to a triangle (three vertices).
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::TriangleHandle;
/// use delaunay::prelude::tds::VertexKey;
/// use slotmap::KeyData;
///
/// let a = VertexKey::from(KeyData::from_ffi(1));
/// let b = VertexKey::from(KeyData::from_ffi(2));
/// let c = VertexKey::from(KeyData::from_ffi(3));
///
/// let handle = TriangleHandle::try_new(b, a, c)?;
/// assert_eq!(handle.vertices().len(), 3);
/// # Ok::<(), delaunay::flips::TriangleHandleError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleHandle {
    v0: VertexKey,
    v1: VertexKey,
    v2: VertexKey,
}

impl TriangleHandle {
    /// Create a canonical triangle handle with ordered vertex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::TriangleHandle;
    /// use delaunay::prelude::tds::VertexKey;
    /// use slotmap::KeyData;
    ///
    /// let a = VertexKey::from(KeyData::from_ffi(10));
    /// let b = VertexKey::from(KeyData::from_ffi(20));
    /// let c = VertexKey::from(KeyData::from_ffi(30));
    ///
    /// let handle = TriangleHandle::try_new(a, b, c)?;
    /// assert_eq!(handle.vertices(), [a, b, c]);
    /// # Ok::<(), delaunay::flips::TriangleHandleError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TriangleHandleError::DuplicateVertices`] if any two supplied
    /// vertices are equal.
    pub fn try_new(a: VertexKey, b: VertexKey, c: VertexKey) -> Result<Self, TriangleHandleError> {
        if a == b || a == c || b == c {
            return Err(TriangleHandleError::DuplicateVertices {
                vertices: [a, b, c],
            });
        }

        Ok(Self::from_validated_vertices(a, b, c))
    }

    /// Creates a canonical triangle handle from vertices already known to be distinct.
    #[must_use]
    pub(crate) fn from_validated_vertices(a: VertexKey, b: VertexKey, c: VertexKey) -> Self {
        let mut verts = [a, b, c];
        verts.sort_unstable_by_key(|v| v.data().as_ffi());
        Self {
            v0: verts[0],
            v1: verts[1],
            v2: verts[2],
        }
    }

    /// Return the triangle vertices.
    #[must_use]
    pub const fn vertices(self) -> [VertexKey; 3] {
        [self.v0, self.v1, self.v2]
    }
}

impl TryFrom<[VertexKey; 3]> for TriangleHandle {
    type Error = TriangleHandleError;

    fn try_from([a, b, c]: [VertexKey; 3]) -> Result<Self, Self::Error> {
        Self::try_new(a, b, c)
    }
}

/// Lightweight handle to a ridge (codimension-2 face) within a simplex.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::flips::RidgeHandle;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Flip(#[from] delaunay::flips::FlipError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
/// let handle: RidgeHandle = dt.ridge_handle(simplex_key, 2, 0)?;
/// assert_eq!(handle.omit_a(), 0);
/// assert_eq!(handle.omit_b(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RidgeHandle {
    simplex_key: SimplexKey,
    omit_a: u8,
    omit_b: u8,
}

impl RidgeHandle {
    /// Creates a new ridge handle by validating the omitted vertex indices
    /// against a live TDS simplex.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError::UnsupportedDimension`] for dimensions below 3,
    /// [`FlipError::MissingSimplex`] if `simplex_key` is not present in `tds`,
    /// or [`FlipError::InvalidRidgeIndex`] if either omitted index is out of
    /// bounds or both indices are the same.
    pub fn try_new<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        simplex_key: SimplexKey,
        omit_a: u8,
        omit_b: u8,
    ) -> Result<Self, FlipError> {
        if D < 3 {
            return Err(FlipError::UnsupportedDimension { dimension: D });
        }

        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        let vertex_count = simplex.number_of_vertices();
        let first_omit_index = usize::from(omit_a);
        let second_omit_index = usize::from(omit_b);
        if first_omit_index >= vertex_count || second_omit_index >= vertex_count || omit_a == omit_b
        {
            return Err(FlipError::InvalidRidgeIndex {
                simplex_key,
                omit_a,
                omit_b,
                vertex_count,
            });
        }

        Ok(Self::from_validated(simplex_key, omit_a, omit_b))
    }

    /// Creates a ridge handle from omitted vertex indices already proven valid
    /// by the caller.
    #[inline]
    pub(crate) const fn from_validated(simplex_key: SimplexKey, omit_a: u8, omit_b: u8) -> Self {
        if omit_a <= omit_b {
            Self {
                simplex_key,
                omit_a,
                omit_b,
            }
        } else {
            Self {
                simplex_key,
                omit_a: omit_b,
                omit_b: omit_a,
            }
        }
    }

    /// Returns the simplex key.
    #[must_use]
    pub const fn simplex_key(&self) -> SimplexKey {
        self.simplex_key
    }

    /// Returns the first omitted index.
    #[must_use]
    pub const fn omit_a(&self) -> u8 {
        self.omit_a
    }

    /// Returns the second omitted index.
    #[must_use]
    pub const fn omit_b(&self) -> u8 {
        self.omit_b
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use slotmap::KeyData;
    use std::assert_matches;

    #[test]
    fn triangle_handle_rejects_duplicate_vertices() {
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));

        assert_matches!(
            TriangleHandle::try_new(a, b, a),
            Err(TriangleHandleError::DuplicateVertices { vertices })
                if vertices == [a, b, a]
        );
    }

    #[test]
    fn triangle_handle_try_from_canonicalizes_vertex_order() {
        let a = VertexKey::from(KeyData::from_ffi(10));
        let b = VertexKey::from(KeyData::from_ffi(20));
        let c = VertexKey::from(KeyData::from_ffi(30));

        let handle = TriangleHandle::try_from([c, a, b]).unwrap();

        assert_eq!(handle.vertices(), [a, b, c]);
    }

    #[test]
    fn ridge_handle_rejects_dimensions_below_three_before_simplex_lookup() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(40));

        assert_matches!(
            RidgeHandle::try_new(&tds, missing_simplex, 0, 1),
            Err(FlipError::UnsupportedDimension { dimension: 2 })
        );
    }

    #[test]
    fn ridge_handle_canonicalizes_omitted_indices() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(50));
        let canonical = RidgeHandle::from_validated(simplex_key, 1, 3);
        let reversed = RidgeHandle::from_validated(simplex_key, 3, 1);

        assert_eq!(reversed, canonical);
        assert_eq!(reversed.simplex_key(), simplex_key);
        assert_eq!((reversed.omit_a(), reversed.omit_b()), (1, 3));
    }
}
