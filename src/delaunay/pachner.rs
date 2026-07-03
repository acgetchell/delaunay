//! Unified Pachner move workflow API.
//!
//! This module is the Monte-Carlo-oriented layer above the explicit bistellar
//! flip primitives in [`crate::flips`]. It lets callers choose a local
//! [`PachnerMove`](crate::pachner::PachnerMove) request first, parse it into a
//! provenanced [`PachnerProposal`](crate::pachner::PachnerProposal), and then
//! attempt the proposal through a fluent terminal method while preserving the
//! primitive flip APIs for deterministic editing workflows. Move requests that
//! contain topology handles are raw detached input: use borrowed topology views
//! for immediate observation, and collapse them to
//! [`PachnerMove`](crate::pachner::PachnerMove) only when a queued or randomized
//! workflow needs a storable request.

#![forbid(unsafe_code)]

use crate::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SmallBuffer};
use crate::core::vertex::Vertex;
use crate::flips::{
    BistellarFlipKind, BistellarFlips, FlipDirection, FlipError, FlipFeasibility, FlipInfo,
    RidgeHandle, TriangleHandle,
};
use crate::tds::{EdgeKey, FacetHandle, SimplexKey, TopologyOwner, TopologyOwnerId, VertexKey};

/// Raw Pachner move request for explicit triangulation editing.
///
/// The enum stores the vertex payload or detached topology handle needed by
/// the corresponding low-level flip primitive. Forward k=1 moves need an owned
/// [`Vertex`] because they create new geometry; the other moves refer to local
/// topology through runtime handles. A `PachnerMove` is not proof that the
/// referenced topology is still live or that it came from a particular
/// triangulation. Parse it through [`PachnerMoves::propose_pachner`] before
/// attempting a unified Pachner move.
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
///     .build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let result = dt
///     .propose_pachner(PachnerMove::K1Insert {
///         simplex_key,
///         vertex: vertex![0.2, 0.2, 0.2]?,
///     })?
///     .attempt_on(&mut dt)?;
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

/// Provenanced Pachner move proposal parsed from a raw [`PachnerMove`].
///
/// A proposal carries the topology owner identity and structural generation of
/// the triangulation that parsed it. This lets mutation and dry-run APIs reject
/// proposals from another owner or an older topology version before
/// interpreting runtime-local keys. It also carries the immutable feasibility
/// report proven when the raw request was parsed. The raw request is still
/// revalidated at commit time, so owner/generation checks complement rather
/// than replace mutating local topology validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
/// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, TopologyOwner, vertex};
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
/// let dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
///     simplex_key,
///     vertex: vertex![0.25, 0.25]?,
/// })?;
/// assert_eq!(proposal.topology_generation(), dt.topology_generation());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct PachnerProposal<U, const D: usize> {
    owner_id: TopologyOwnerId,
    topology_generation: u64,
    pachner_move: PachnerMove<U, D>,
    feasibility: PachnerMoveFeasibility<D>,
}

impl<U, const D: usize> PachnerProposal<U, D> {
    /// Builds a proposal after the owning triangulation has validated the raw request.
    const fn from_validated(
        owner_id: TopologyOwnerId,
        topology_generation: u64,
        pachner_move: PachnerMove<U, D>,
        feasibility: PachnerMoveFeasibility<D>,
    ) -> Self {
        Self {
            owner_id,
            topology_generation,
            pachner_move,
            feasibility,
        }
    }

    /// Returns the topology owner identity that parsed this proposal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, TopologyOwner, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    ///
    /// assert_eq!(proposal.owner_id(), &dt.topology_owner_id());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn owner_id(&self) -> &TopologyOwnerId {
        &self.owner_id
    }

    /// Returns the structural topology generation observed when this proposal was parsed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, TopologyOwner, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    ///
    /// assert_eq!(proposal.topology_generation(), dt.topology_generation());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_generation(&self) -> u64 {
        self.topology_generation
    }

    /// Returns the raw request stored in this proposal without discarding
    /// provenance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    ///
    /// let PachnerMove::K1Insert {
    ///     simplex_key: stored_key,
    ///     ..
    /// } = proposal.request()
    /// else {
    ///     unreachable!("proposal should store the original request");
    /// };
    /// assert_eq!(*stored_key, simplex_key);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn request(&self) -> &PachnerMove<U, D> {
        &self.pachner_move
    }

    /// Consumes this proposal and returns its raw request.
    ///
    /// This discards the owner and generation proof. Parse the returned request
    /// again with [`PachnerMoves::propose_pachner`] before attempting it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    ///
    /// let request = proposal.into_request();
    /// let PachnerMove::K1Insert {
    ///     simplex_key: stored_key,
    ///     ..
    /// } = request
    /// else {
    ///     unreachable!("proposal should store the original request");
    /// };
    /// assert_eq!(stored_key, simplex_key);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn into_request(self) -> PachnerMove<U, D> {
        self.pachner_move
    }

    /// Validates this provenanced proposal against a topology owner without mutating it.
    ///
    /// This checks owner identity and topology generation before returning the
    /// deterministic pre-mutation flip conditions proven when the proposal was
    /// parsed, while preserving the proposal for a later mutation attempt.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError::WrongTopologyOwner`] when this proposal was parsed
    /// by another topology owner, or [`FlipError::StaleTopologyProposal`] when
    /// it was parsed before the current structural generation. Raw-request
    /// validation failures are returned by [`PachnerMoves::propose_pachner`],
    /// before a proposal exists.
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
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    /// let feasibility = proposal.can_attempt_on(&dt)?;
    /// assert_eq!(feasibility.kind, BistellarFlipKind::k1(2));
    /// assert_eq!(feasibility.direction, FlipDirection::Forward);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn can_attempt_on<T>(&self, moves: &T) -> Result<PachnerMoveFeasibility<D>, FlipError>
    where
        T: BistellarFlips<D, VertexData = U> + TopologyOwner + ?Sized,
    {
        can_attempt_pachner_proposal(moves, self)
    }

    /// Attempts this provenanced proposal on a topology owner.
    ///
    /// Proposal provenance is checked before any runtime-local key is
    /// interpreted, so wrong-owner and stale-generation errors return before
    /// mutation.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError::WrongTopologyOwner`] when this proposal was parsed
    /// by another topology owner, [`FlipError::StaleTopologyProposal`] when it
    /// was parsed before the current structural generation, or another
    /// [`FlipError`] from the selected flip primitive when the local
    /// configuration cannot be mutated. On error, the selected primitive
    /// preserves its usual failure-atomic mutation contract.
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
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt
    ///     .propose_pachner(PachnerMove::K1Insert {
    ///         simplex_key,
    ///         vertex: vertex![0.25, 0.25]?,
    ///     })?
    ///     .attempt_on(&mut dt)?;
    /// assert_eq!(result.kind, BistellarFlipKind::k1(2));
    /// assert_eq!(result.direction, FlipDirection::Forward);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn attempt_on<T>(self, moves: &mut T) -> Result<PachnerMoveResult<D>, FlipError>
    where
        T: BistellarFlips<D, VertexData = U> + TopologyOwner + ?Sized,
    {
        attempt_pachner_proposal(moves, self)
    }
}

/// Result returned by [`PachnerProposal::attempt_on`].
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
///     .build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let result = dt
///     .propose_pachner(PachnerMove::K1Insert {
///         simplex_key,
///         vertex: vertex![0.2, 0.2, 0.2]?,
///     })?
///     .attempt_on(&mut dt)?;
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

/// Result returned by [`PachnerProposal::can_attempt_on`].
///
/// This report describes the local move that passed immutable feasibility
/// validation. It intentionally omits post-mutation simplex keys because those
/// keys are allocated only by [`PachnerProposal::attempt_on`]. For forward
/// k=1 insertion, [`inserted_face_vertices`](Self::inserted_face_vertices) is
/// `None` because the inserted vertex key is likewise allocated by the
/// mutating operation.
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
/// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
/// let dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
///     simplex_key,
///     vertex: vertex![0.25, 0.25]?,
/// })?;
/// let feasibility = proposal.can_attempt_on(&dt)?;
/// assert_eq!(feasibility.kind, BistellarFlipKind::k1(2));
/// assert_eq!(feasibility.direction, FlipDirection::Forward);
/// assert!(feasibility.inserted_face_vertices.is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub struct PachnerMoveFeasibility<const D: usize> {
    /// Flip kind (k, d).
    pub kind: BistellarFlipKind,
    /// Flip direction.
    pub direction: FlipDirection,
    /// Simplex keys the corresponding mutating move would remove.
    pub removed_simplices: SimplexKeyBuffer,
    /// Vertices of the removed face shared by removed simplices.
    pub removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Vertices of the inserted complementary face when all are already live.
    ///
    /// This is `None` for forward k=1 insertion because the inserted vertex key
    /// is allocated only by the mutating operation.
    pub inserted_face_vertices: Option<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
}

impl<const D: usize> From<FlipFeasibility<D>> for PachnerMoveFeasibility<D> {
    fn from(feasibility: FlipFeasibility<D>) -> Self {
        Self {
            kind: feasibility.kind,
            direction: feasibility.direction,
            removed_simplices: feasibility.removed_simplices,
            removed_face_vertices: feasibility.removed_face_vertices,
            inserted_face_vertices: feasibility.inserted_face_vertices,
        }
    }
}

/// Unified Pachner move proposal parser for Monte-Carlo-style workflows.
///
/// This extension trait is implemented for every type that implements
/// [`BistellarFlips`] and [`TopologyOwner`]. It parses raw detached
/// [`PachnerMove`] requests into provenanced [`PachnerProposal`] values. The
/// proposal exposes the fluent dry-run and mutation terminal methods, so
/// mutation still consumes a proof-bearing value and rejects cross-owner or
/// stale proposals before runtime-local keys are interpreted.
pub trait PachnerMoves<const D: usize>: BistellarFlips<D> + TopologyOwner {
    /// Parse a raw Pachner move request into a provenanced proposal.
    ///
    /// This validates deterministic pre-mutation conditions, then stamps the
    /// proposal with the current topology owner and generation. Mutating APIs
    /// consume only proposals, not raw requests, so queued moves cannot silently
    /// cross triangulation owners or topology versions.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] from the selected flip primitive when the raw
    /// request is stale, non-admissible, geometrically degenerate, or would
    /// violate deterministic local topology checks. No proposal is minted on
    /// error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{PachnerMove, PachnerMoves, TopologyOwner, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let proposal = dt.propose_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.25, 0.25]?,
    /// })?;
    /// assert_eq!(proposal.topology_generation(), dt.topology_generation());
    /// # Ok(())
    /// # }
    /// ```
    fn propose_pachner(
        &self,
        pachner_move: PachnerMove<Self::VertexData, D>,
    ) -> Result<PachnerProposal<Self::VertexData, D>, FlipError>;
}

/// Validates raw Pachner request topology before it is stamped as a proposal.
///
/// This keeps proposal parsing and mutation dispatch on the same primitive flip
/// predicates so public Pachner APIs agree on accepted and rejected local
/// configurations.
fn validate_pachner_request<const D: usize, T>(
    moves: &T,
    pachner_move: &PachnerMove<T::VertexData, D>,
) -> Result<FlipFeasibility<D>, FlipError>
where
    T: BistellarFlips<D> + ?Sized,
{
    match pachner_move {
        PachnerMove::K1Insert {
            simplex_key,
            vertex,
        } => moves.can_flip_k1_insert(*simplex_key, vertex),
        PachnerMove::K1Remove { vertex_key } => moves.can_flip_k1_remove(*vertex_key),
        PachnerMove::K2 { facet } => moves.can_flip_k2(*facet),
        PachnerMove::K2Inverse { edge } => moves.can_flip_k2_inverse_from_edge(*edge),
        PachnerMove::K3 { ridge } => moves.can_flip_k3(*ridge),
        PachnerMove::K3Inverse { triangle } => moves.can_flip_k3_inverse_from_triangle(*triangle),
    }
}

/// Rejects proposals whose owner or generation no longer matches the target.
///
/// This protects public Pachner APIs from interpreting runtime-local keys after
/// they cross topology owners or outlive a structural mutation.
fn validate_pachner_proposal_provenance<const D: usize, T, U>(
    owner: &T,
    proposal: &PachnerProposal<U, D>,
) -> Result<(), FlipError>
where
    T: TopologyOwner + ?Sized,
{
    let expected = owner.topology_owner_id();
    if proposal.owner_id() != &expected {
        return Err(FlipError::WrongTopologyOwner {
            expected,
            found: proposal.owner_id().clone(),
        });
    }

    let current_generation = owner.topology_generation();
    if proposal.topology_generation() != current_generation {
        return Err(FlipError::StaleTopologyProposal {
            proposal_generation: proposal.topology_generation(),
            current_generation,
        });
    }

    Ok(())
}

/// Attempts a provenanced proposal after checking it still belongs to `moves`.
fn attempt_pachner_proposal<const D: usize, T>(
    moves: &mut T,
    proposal: PachnerProposal<T::VertexData, D>,
) -> Result<PachnerMoveResult<D>, FlipError>
where
    T: BistellarFlips<D> + TopologyOwner + ?Sized,
{
    validate_pachner_proposal_provenance(moves, &proposal)?;
    let info = match proposal.into_request() {
        PachnerMove::K1Insert {
            simplex_key,
            vertex,
        } => moves.flip_k1_insert(simplex_key, vertex)?,
        PachnerMove::K1Remove { vertex_key } => moves.flip_k1_remove(vertex_key)?,
        PachnerMove::K2 { facet } => moves.flip_k2(facet)?,
        PachnerMove::K2Inverse { edge } => moves.flip_k2_inverse_from_edge(edge)?,
        PachnerMove::K3 { ridge } => moves.flip_k3(ridge)?,
        PachnerMove::K3Inverse { triangle } => moves.flip_k3_inverse_from_triangle(triangle)?,
    };
    Ok(info.into())
}

/// Returns the stored feasibility proof after checking it still belongs to `moves`.
fn can_attempt_pachner_proposal<const D: usize, T>(
    moves: &T,
    proposal: &PachnerProposal<T::VertexData, D>,
) -> Result<PachnerMoveFeasibility<D>, FlipError>
where
    T: BistellarFlips<D> + TopologyOwner + ?Sized,
{
    validate_pachner_proposal_provenance(moves, proposal)?;
    Ok(proposal.feasibility.clone())
}

impl<const D: usize, T> PachnerMoves<D> for T
where
    T: BistellarFlips<D> + TopologyOwner + ?Sized,
{
    #[inline]
    fn propose_pachner(
        &self,
        pachner_move: PachnerMove<Self::VertexData, D>,
    ) -> Result<PachnerProposal<Self::VertexData, D>, FlipError> {
        let feasibility = validate_pachner_request(self, &pachner_move)?.into();
        Ok(PachnerProposal::from_validated(
            self.topology_owner_id(),
            self.topology_generation(),
            pachner_move,
            feasibility,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::KeyData;

    #[test]
    fn pachner_move_result_roundtrips_through_flip_info() {
        let mut removed_simplices = SimplexKeyBuffer::new();
        removed_simplices.push(SimplexKey::from(KeyData::from_ffi(11)));
        removed_simplices.push(SimplexKey::from(KeyData::from_ffi(12)));

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(SimplexKey::from(KeyData::from_ffi(21)));
        new_simplices.push(SimplexKey::from(KeyData::from_ffi(22)));
        new_simplices.push(SimplexKey::from(KeyData::from_ffi(23)));

        let mut removed_face_vertices = SmallBuffer::new();
        removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(31)));
        removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(32)));
        removed_face_vertices.push(VertexKey::from(KeyData::from_ffi(33)));

        let mut inserted_face_vertices = SmallBuffer::new();
        inserted_face_vertices.push(VertexKey::from(KeyData::from_ffi(41)));
        inserted_face_vertices.push(VertexKey::from(KeyData::from_ffi(42)));

        let result = PachnerMoveResult::<3> {
            kind: BistellarFlipKind::k2(3),
            direction: FlipDirection::Forward,
            removed_simplices,
            new_simplices,
            removed_face_vertices,
            inserted_face_vertices,
        };

        let info: FlipInfo<3> = result.clone().into();

        assert_eq!(info.kind, result.kind);
        assert_eq!(info.direction, result.direction);
        assert_eq!(info.removed_simplices, result.removed_simplices);
        assert_eq!(info.new_simplices, result.new_simplices);
        assert_eq!(info.removed_face_vertices, result.removed_face_vertices);
        assert_eq!(info.inserted_face_vertices, result.inserted_face_vertices);
        assert_eq!(PachnerMoveResult::from(info), result);
    }
}
