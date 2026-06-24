//! Unified Pachner move workflow API.
//!
//! This module is the Monte-Carlo-oriented layer above the explicit bistellar
//! flip primitives in [`crate::flips`]. It lets callers choose a local
//! [`PachnerMove`](crate::pachner::PachnerMove) value first and attempt it
//! through one dispatch method while preserving the primitive flip APIs for
//! deterministic editing workflows. Move requests that contain topology handles
//! are detached proposals:
//! [`PachnerMoves::attempt_pachner`](crate::pachner::PachnerMoves::attempt_pachner)
//! revalidates them against the target triangulation before mutating storage.
//! Use borrowed topology views for immediate observation; collapse them to
//! [`PachnerMove`](crate::pachner::PachnerMove) only when a queued or randomized
//! workflow needs a storable proposal.

#![forbid(unsafe_code)]

use crate::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SmallBuffer};
use crate::core::vertex::Vertex;
use crate::flips::{
    BistellarFlipKind, BistellarFlips, FlipDirection, FlipError, FlipInfo, RidgeHandle,
    TriangleHandle,
};
use crate::tds::{EdgeKey, FacetHandle, SimplexKey, VertexKey};

/// Unified Pachner move request for explicit triangulation editing.
///
/// The enum stores the vertex payload or detached topology handle needed by
/// the corresponding low-level flip primitive. Forward k=1 moves need an owned
/// [`Vertex`] because they create new geometry; the other moves refer to local
/// topology through runtime handles that are revalidated against the target
/// triangulation when [`PachnerMoves::attempt_pachner`] runs.
/// A `PachnerMove` is therefore not proof that the referenced topology is still
/// live; it is a request that the mutation boundary parses into a live move or
/// rejects with a typed [`FlipError`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
/// use delaunay::prelude::pachner::{
///     BistellarFlipKind, FlipDirection, PachnerMove, PachnerMoves, vertex,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     vertex![0.0, 0.0, 0.0]?,
///     vertex![1.0, 0.0, 0.0]?,
///     vertex![0.0, 1.0, 0.0]?,
///     vertex![0.0, 0.0, 1.0]?,
/// ];
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let result = dt.attempt_pachner(PachnerMove::K1Insert {
///     simplex_key,
///     vertex: vertex![0.2, 0.2, 0.2]?,
/// })?;
/// assert_eq!(result.kind, BistellarFlipKind::k1(3));
/// assert_eq!(result.direction, FlipDirection::Forward);
/// assert_eq!(result.inserted_face_vertices.len(), 1);
/// assert_eq!(result.new_simplices.len(), 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub enum PachnerMove<U, const D: usize> {
    /// Split one simplex by inserting a new vertex into it.
    K1Insert {
        /// Simplex to split.
        simplex_key: SimplexKey,
        /// Vertex inserted into the simplex.
        vertex: Vertex<U, D>,
    },
    /// Collapse a vertex whose star is a simplex.
    K1Remove {
        /// Vertex to remove.
        vertex_key: VertexKey,
    },
    /// Apply a forward k=2 move across an interior facet.
    K2 {
        /// Interior facet whose two incident simplices form the move support.
        facet: FacetHandle,
    },
    /// Apply the inverse k=2 move from an edge star.
    K2Inverse {
        /// Edge whose star is collapsed by the inverse move.
        edge: EdgeKey,
    },
    /// Apply a forward k=3 move around a ridge.
    K3 {
        /// Ridge whose three-simplex star forms the move support.
        ridge: RidgeHandle,
    },
    /// Apply the inverse k=3 move from a triangle star.
    K3Inverse {
        /// Triangle whose star is collapsed by the inverse move.
        triangle: TriangleHandle,
    },
}

/// Result returned by [`PachnerMoves::attempt_pachner`].
///
/// The fields mirror [`FlipInfo`] so callers can inspect the exact topology
/// change without allocating conversion vectors. Use [`From`] to convert
/// between the unified result and the lower-level flip result when sharing code
/// with existing flip workflows.
/// The result owns detached keys because the move has already ended the mutable
/// borrow and may have deleted some referenced simplices. Revalidate any key
/// against the current triangulation before using it for later topology lookup,
/// especially after another mutation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
/// use delaunay::prelude::pachner::{
///     BistellarFlipKind, FlipDirection, PachnerMove, PachnerMoves, vertex,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     vertex![0.0, 0.0, 0.0]?,
///     vertex![1.0, 0.0, 0.0]?,
///     vertex![0.0, 1.0, 0.0]?,
///     vertex![0.0, 0.0, 1.0]?,
/// ];
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let result = dt.attempt_pachner(PachnerMove::K1Insert {
///     simplex_key,
///     vertex: vertex![0.2, 0.2, 0.2]?,
/// })?;
/// assert_eq!(result.kind, BistellarFlipKind::k1(3));
/// assert_eq!(result.direction, FlipDirection::Forward);
/// assert_eq!(result.inserted_face_vertices.len(), 1);
/// assert_eq!(result.new_simplices.len(), 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct PachnerMoveResult<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Simplex keys removed by the move.
    ///
    /// These keys are historical after a successful move and are intended for
    /// diagnostics, accounting, or inverse-candidate construction, not live
    /// simplex lookup.
    pub removed_simplices: SimplexKeyBuffer,
    /// Newly created simplex keys that are live immediately after a successful move.
    ///
    /// Later triangulation mutations may stale these runtime-local keys.
    pub new_simplices: SimplexKeyBuffer,
    /// Vertices of the removed face shared by removed simplices.
    ///
    /// These are detached vertex keys describing the post-move report, not
    /// borrowed access to live vertex storage.
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted complementary face.
    ///
    /// These are detached vertex keys describing the post-move report, not
    /// borrowed access to live vertex storage.
    pub inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
}

impl<const D: usize> From<FlipInfo<D>> for PachnerMoveResult<D> {
    fn from(info: FlipInfo<D>) -> Self {
        Self {
            kind: info.kind,
            direction: info.direction,
            removed_simplices: info.removed_simplices,
            new_simplices: info.new_simplices,
            removed_face_vertices: info.removed_face_vertices,
            inserted_face_vertices: info.inserted_face_vertices,
        }
    }
}

impl<const D: usize> From<PachnerMoveResult<D>> for FlipInfo<D> {
    fn from(result: PachnerMoveResult<D>) -> Self {
        Self {
            kind: result.kind,
            direction: result.direction,
            removed_simplices: result.removed_simplices,
            new_simplices: result.new_simplices,
            removed_face_vertices: result.removed_face_vertices,
            inserted_face_vertices: result.inserted_face_vertices,
        }
    }
}

/// Unified Pachner move dispatch for Monte-Carlo-style workflows.
///
/// This extension trait is implemented for every type that implements
/// [`BistellarFlips`]. Successful moves therefore preserve the same cache
/// invalidation and validation behavior as calling the primitive flip method
/// directly.
pub trait PachnerMoves<const D: usize> {
    /// User data type stored on vertices inserted through k=1 moves.
    type VertexData;

    /// Attempt one unified Pachner move request.
    ///
    /// The request is a detached proposal. Implementations revalidate its
    /// handles or keys against `self`, build the short-lived flip context from a
    /// shared borrow, drop that read-only context borrow, and then mutate through
    /// the method's `&mut self` borrow.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] from the selected flip primitive when the requested
    /// local configuration is stale, belongs to a different triangulation, is
    /// non-admissible, is geometrically degenerate, or would violate topology
    /// invariants. On error, the selected flip primitive preserves its usual
    /// failure-atomic mutation contract.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{
    ///     BistellarFlipKind, FlipDirection, PachnerMove, PachnerMoves, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0, 0.0]?,
    ///     vertex![1.0, 0.0, 0.0]?,
    ///     vertex![0.0, 1.0, 0.0]?,
    ///     vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt.attempt_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.2, 0.2, 0.2]?,
    /// })?;
    /// assert_eq!(result.kind, BistellarFlipKind::k1(3));
    /// assert_eq!(result.direction, FlipDirection::Forward);
    /// assert_eq!(result.inserted_face_vertices.len(), 1);
    /// assert_eq!(result.new_simplices.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    fn attempt_pachner(
        &mut self,
        pachner_move: PachnerMove<Self::VertexData, D>,
    ) -> Result<PachnerMoveResult<D>, FlipError>;
}

impl<const D: usize, T> PachnerMoves<D> for T
where
    T: BistellarFlips<D> + ?Sized,
{
    type VertexData = T::VertexData;

    #[inline]
    fn attempt_pachner(
        &mut self,
        pachner_move: PachnerMove<Self::VertexData, D>,
    ) -> Result<PachnerMoveResult<D>, FlipError> {
        let info = match pachner_move {
            PachnerMove::K1Insert {
                simplex_key,
                vertex,
            } => self.flip_k1_insert(simplex_key, vertex)?,
            PachnerMove::K1Remove { vertex_key } => self.flip_k1_remove(vertex_key)?,
            PachnerMove::K2 { facet } => self.flip_k2(facet)?,
            PachnerMove::K2Inverse { edge } => self.flip_k2_inverse_from_edge(edge)?,
            PachnerMove::K3 { ridge } => self.flip_k3(ridge)?,
            PachnerMove::K3Inverse { triangle } => self.flip_k3_inverse_from_triangle(triangle)?,
        };
        Ok(info.into())
    }
}
