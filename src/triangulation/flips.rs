//! Triangulation editing operations (bistellar flips).
//!
//! This module exposes direct bistellar-flip methods for explicit triangulation editing.
//! These operations do **not** automatically restore the Delaunay property.
//! For queued, randomized, or Monte-Carlo-style local edits, prefer the staged
//! [`PachnerMoves`](crate::pachner::PachnerMoves) workflow:
//! `propose_pachner(...)?.attempt_on(...)`.
//! For Delaunay construction/deletion, use
//! [`crate::DelaunayTriangulation::insert_vertex`] and
//! [`crate::DelaunayTriangulation::delete_vertex`].

#![forbid(unsafe_code)]

pub use crate::core::algorithms::flips::{
    BistellarFlipKind, BistellarFlipKindError, BistellarMove, ConstK, DelaunayRepairDiagnostics,
    DelaunayRepairError, DelaunayRepairHeuristicRebuildFailure,
    DelaunayRepairHeuristicRebuildFailureKind, DelaunayRepairHeuristicVertexContext,
    DelaunayRepairOrientationCanonicalizationFailure,
    DelaunayRepairOrientationCanonicalizationFailureKind, DelaunayRepairPostconditionFailure,
    DelaunayRepairStats, DelaunayRepairVerificationContext, FlipContextError, FlipDirection,
    FlipEdgeAdjacencyError, FlipError, FlipFailureKind, FlipFeasibility, FlipInfo,
    FlipMutationError, FlipNeighborCavityFailureKind, FlipNeighborDelaunayValidationFailureKind,
    FlipNeighborHullExtensionFailureKind, FlipNeighborRepairDiagnostics, FlipNeighborRepairFailure,
    FlipNeighborWiringError, FlipOrientationCheckStage, FlipPredicateError, FlipPredicateOperation,
    FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder, RidgeHandle,
    TriangleHandle, TriangleHandleError,
};
pub use crate::tds::{EdgeKey, FacetHandle, SimplexKey, VertexKey};

use crate::core::algorithms::flips::{
    apply_bistellar_flip_dynamic_raw, apply_bistellar_flip_k1_inverse_raw,
    apply_bistellar_flip_k1_raw, apply_bistellar_flip_raw, build_k2_flip_context,
    build_k2_flip_context_from_edge, build_k3_flip_context, build_k3_flip_context_from_triangle,
    validate_bistellar_flip_dynamic, validate_bistellar_flip_k1_insert,
    validate_bistellar_flip_k1_inverse, validate_bistellar_flip_k2, validate_bistellar_flip_k3,
};
use crate::core::operations::SuspicionFlags;
use crate::core::operations::TopologicalOperation;
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::triangulation::Triangulation;
use crate::triangulation::rollback::TriangulationRollbackTransaction;

/// Applies a high-level flip transaction and preserves topology/realization invariants.
fn apply_realized_flip<K, U, V, const D: usize>(
    tri: &mut Triangulation<K, U, V, D>,
    operation: TopologicalOperation,
    apply: impl FnOnce(&mut Triangulation<K, U, V, D>) -> Result<FlipInfo<D>, FlipError>,
) -> Result<FlipInfo<D>, FlipError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    validate_flip_topology(tri, operation)?;
    let mut transaction = TriangulationRollbackTransaction::begin(tri);
    let result = apply(transaction.triangulation_mut());
    let info = match result {
        Ok(info) => info,
        Err(error) => {
            transaction.rollback();
            return Err(error);
        }
    };

    if let Err(error) = transaction
        .triangulation_mut()
        .normalize_and_promote_positive_orientation()
    {
        transaction.rollback();
        return Err(FlipError::PostconditionRepair {
            source: Box::new(error),
        });
    }
    if info.new_simplices.is_empty() {
        if let Err(source) = transaction.triangulation_mut().validate() {
            transaction.rollback();
            return Err(FlipError::InvariantValidation {
                source: Box::new(source),
            });
        }
        if let Err(source) = transaction.triangulation_mut().is_valid_realization() {
            transaction.rollback();
            return Err(FlipError::RealizationValidation {
                source: Box::new(source),
            });
        }
    } else {
        if let Err(source) = transaction
            .triangulation_mut()
            .validate_mandatory_mutation_postconditions_for_simplices(&info.new_simplices)
        {
            transaction.rollback();
            return Err(FlipError::InvariantValidation {
                source: Box::new(source),
            });
        }
        if let Err(source) = transaction
            .triangulation_mut()
            .validate_realization_for_simplices(&info.new_simplices)
        {
            transaction.rollback();
            return Err(FlipError::RealizationValidation {
                source: Box::new(source),
            });
        }

        let run_global_audit = transaction
            .triangulation_mut()
            .validation_policy()
            .should_validate(SuspicionFlags::default());
        if run_global_audit {
            if let Err(source) = transaction.triangulation_mut().validate() {
                transaction.rollback();
                return Err(FlipError::InvariantValidation {
                    source: Box::new(source),
                });
            }
            if let Err(source) = transaction.triangulation_mut().is_valid_realization() {
                transaction.rollback();
                return Err(FlipError::RealizationValidation {
                    source: Box::new(source),
                });
            }
        }
    }

    transaction.commit();
    Ok(info)
}

/// Confirms that a triangulation carries the topology proof required by a flip.
pub(crate) fn validate_flip_topology<K, U, V, const D: usize>(
    tri: &Triangulation<K, U, V, D>,
    operation: TopologicalOperation,
) -> Result<(), FlipError> {
    let found = tri.topology_guarantee();
    if operation.is_admissible_under(found) {
        Ok(())
    } else {
        Err(FlipError::FlipTopologyNotAdmissible {
            required: operation.required_topology(),
            found,
        })
    }
}

/// Direct triangulation editing operations via bistellar flips.
///
/// This trait is the primitive/expert editing layer. Public workflows that
/// store, randomize, or queue moves should normally parse a raw
/// [`PachnerMove`](crate::pachner::PachnerMove) into a provenanced
/// [`PachnerProposal`](crate::pachner::PachnerProposal), then call
/// [`PachnerProposal::attempt_on`](crate::pachner::PachnerProposal::attempt_on)
/// as the mutating terminal step.
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
/// };
/// use delaunay::flips::BistellarFlips;
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build()?
///     .into_triangulation();
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// // Split a simplex by inserting a vertex (k=1 move).
/// let _info = dt.flip_k1_insert(simplex_key, delaunay::vertex![0.1, 0.1, 0.1]?)?;
/// # Ok(())
/// # }
/// ```
///
/// A [`DelaunayTriangulation`](crate::DelaunayTriangulation) does not implement
/// this trait because a topology edit can invalidate Level 5. Demote explicitly
/// with [`DelaunayTriangulation::into_triangulation`](crate::DelaunayTriangulation::into_triangulation)
/// before editing, then recertify or delaunayize the result.
///
/// ```compile_fail
/// use delaunay::prelude::construction::DelaunayTriangulation;
/// use delaunay::prelude::geometry::AdaptiveKernel;
/// use delaunay::prelude::pachner::BistellarFlips;
///
/// fn requires_topology_edits<T: BistellarFlips<2>>() {}
///
/// requires_topology_edits::<
///     DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>,
/// >();
/// ```
pub trait BistellarFlips<const D: usize> {
    /// User data type stored on vertices inserted through k=1 flips.
    type VertexData;

    /// Apply a forward k=1 move (simplex split) by inserting a vertex into a simplex.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the simplex is missing, the vertex cannot be inserted,
    /// the point is not strictly inside the selected simplex's active realization
    /// chart, or the flip would create invalid topology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::flips::BistellarFlips;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Insert a vertex into the simplex
    /// let info = dt.flip_k1_insert(simplex_key, delaunay::vertex![0.25, 0.25, 0.25]?)?;
    /// assert!(!info.new_simplices.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    fn flip_k1_insert(
        &mut self,
        simplex_key: SimplexKey,
        vertex: Vertex<Self::VertexData, D>,
    ) -> Result<FlipInfo<D>, FlipError>;

    /// Validate a forward k=1 move (simplex split) without mutating topology.
    ///
    /// This checks the same deterministic structural, topological, and
    /// orientation-realization conditions as [`Self::flip_k1_insert`] on the
    /// same unchanged triangulation state. These include simplex liveness,
    /// duplicate inserted-vertex UUIDs, exact replacement-simplex degeneracy,
    /// and strict containment in the selected simplex's active affine chart.
    /// Periodic containment is evaluated in the simplex's lifted local frame.
    /// The returned feasibility report omits the inserted vertex key because
    /// that key is allocated only by the mutating executor.
    ///
    /// Success predicts deterministic execution on the unchanged state; it
    /// cannot guarantee against allocation failure, process termination, or
    /// other environmental failures outside the triangulation contract.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic structural, topological, or
    /// orientation-realization validation.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let vertex = delaunay::vertex![0.25, 0.25, 0.25]?;
    ///
    /// let feasibility = dt.can_flip_k1_insert(simplex_key, &vertex)?;
    /// assert_eq!((feasibility.kind.k(), feasibility.kind.d()), (1, 3));
    /// assert_eq!(feasibility.direction, FlipDirection::Forward);
    /// assert!(feasibility.inserted_face_vertices.is_none());
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k1_insert(
        &self,
        simplex_key: SimplexKey,
        vertex: &Vertex<Self::VertexData, D>,
    ) -> Result<FlipFeasibility<D>, FlipError>;

    /// Apply an inverse k=1 move (vertex collapse).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the vertex star is not collapsible or the flip
    /// would create invalid topology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::flips::BistellarFlips;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let inserted = dt.flip_k1_insert(simplex_key, delaunay::vertex![0.25, 0.25, 0.25]?)?;
    /// let inserted_vertex = inserted.inserted_face_vertices[0];
    ///
    /// // Remove the inserted vertex
    /// let info = dt.flip_k1_remove(inserted_vertex)?;
    /// assert!(!info.removed_simplices.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError>;

    /// Validate an inverse k=1 move (vertex collapse) without mutating topology.
    ///
    /// This checks the same deterministic pre-mutation conditions as
    /// [`Self::flip_k1_remove`] on the same triangulation state.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic pre-mutation validation.
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let inserted = dt.flip_k1_insert(simplex_key, delaunay::vertex![0.25, 0.25, 0.25]?)?;
    /// let [inserted_vertex] = inserted.inserted_face_vertices.as_slice() else {
    ///     return Ok(());
    /// };
    ///
    /// let feasibility = dt.can_flip_k1_remove(*inserted_vertex)?;
    /// assert_eq!((feasibility.kind.k(), feasibility.kind.d()), (4, 3));
    /// assert_eq!(feasibility.direction, FlipDirection::Inverse);
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k1_remove(&self, vertex_key: VertexKey) -> Result<FlipFeasibility<D>, FlipError>;

    /// Apply a k=2 facet flip (forward).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the facet is invalid, the flip would be degenerate,
    /// or the resulting topology would be non-manifold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::flips::BistellarFlips;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![0.5, 0.5, 0.3]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build()?
    ///     .into_triangulation();
    ///
    /// // Find an interior facet and attempt a k=2 flip. k=2 flips require
    /// // specific geometric conditions, so this may still fail.
    /// let mut interior_facet = None;
    /// for facet in dt.facets() {
    ///     let facet = facet?;
    ///     if facet
    ///         .simplex()
    ///         .neighbor_key(usize::from(facet.facet_index()))
    ///         .flatten()
    ///         .is_some()
    ///     {
    ///         interior_facet = Some(facet.handle());
    ///         break;
    ///     }
    /// }
    /// if let Some(facet) = interior_facet {
    ///     let _ = dt.flip_k2(facet);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError>;

    /// Validate a k=2 facet flip without mutating topology.
    ///
    /// This checks the same deterministic pre-mutation conditions as
    /// [`Self::flip_k2`] on the same triangulation state.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic pre-mutation validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::BistellarFlips;
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    ///     DelaunayTriangulationConstructionError,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![1.0, 1.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    /// let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(
    ///     &vertices,
    ///     &simplices,
    /// )
    /// .map_err(DelaunayTriangulationConstructionError::from)?
    /// .build()?
    /// .into_triangulation();
    ///
    /// let mut accepted = None;
    /// for facet in dt.facets() {
    ///     let Ok(facet) = facet else {
    ///         continue;
    ///     };
    ///     if facet
    ///         .simplex()
    ///         .neighbor_key(usize::from(facet.facet_index()))
    ///         .flatten()
    ///         .is_none()
    ///     {
    ///         continue;
    ///     }
    ///     if let Ok(feasibility) = dt.can_flip_k2(facet.handle()) {
    ///         accepted = Some(feasibility);
    ///         break;
    ///     }
    /// }
    ///
    /// assert_eq!(
    ///     accepted.map(|feasibility| (feasibility.kind.k(), feasibility.kind.d())),
    ///     Some((2, 2)),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k2(&self, facet: FacetHandle) -> Result<FlipFeasibility<D>, FlipError>;

    /// Apply a k=3 ridge flip (forward).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the ridge is invalid, the flip would be degenerate,
    /// or the resulting topology would be non-manifold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::flips::{BistellarFlips, RidgeHandle};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![1.0, 1.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build()?
    ///     .into_triangulation();
    ///
    /// // k=3 flips require specific ridge configurations in 3D and above
    /// // This is an illustrative example; actual ridge selection depends on topology
    /// let _ = dt;  // Use dt to prevent unused variable warning
    /// # Ok(())
    /// # }
    /// ```
    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError>;

    /// Validate a k=3 ridge flip without mutating topology.
    ///
    /// This checks the same deterministic pre-mutation conditions as
    /// [`Self::flip_k3`] on the same triangulation state.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic pre-mutation validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::{BistellarFlips, FlipError};
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Ok(ridge) = dt.ridge_handle(simplex_key, 0, 1) else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt.can_flip_k3(ridge);
    /// std::assert_matches!(result, Err(FlipError::InvalidRidgeMultiplicity { .. }));
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k3(&self, ridge: RidgeHandle) -> Result<FlipFeasibility<D>, FlipError>;

    /// Apply an inverse k=2 flip from an edge star (D >= 3).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the edge star is invalid or the inverse flip
    /// would create invalid topology.
    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError>;

    /// Validate an inverse k=2 edge-star flip without mutating topology.
    ///
    /// This checks the same deterministic pre-mutation conditions as
    /// [`Self::flip_k2_inverse_from_edge`] on the same triangulation state.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic pre-mutation validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::{BistellarFlips, FlipError};
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some(edge) = dt.edges().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt.can_flip_k2_inverse_from_edge(edge);
    /// std::assert_matches!(result, Err(FlipError::InvalidEdgeMultiplicity { .. }));
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k2_inverse_from_edge(&self, edge: EdgeKey)
    -> Result<FlipFeasibility<D>, FlipError>;

    /// Apply an inverse k=3 flip from a triangle star (D >= 4).
    ///
    /// If `D < 4`, this returns [`FlipError::UnsupportedDimension`].
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the triangle star is invalid or the inverse flip
    /// would create invalid topology.
    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError>;

    /// Validate an inverse k=3 triangle-star flip without mutating topology.
    ///
    /// If `D < 4`, this returns [`FlipError::UnsupportedDimension`].
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] when the corresponding mutating operation would
    /// fail during deterministic pre-mutation validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::flips::{BistellarFlips, FlipError, TriangleHandle};
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build()?
    ///     .into_triangulation();
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let [a, b, c, ..] = simplex.vertices() else {
    ///     return Ok(());
    /// };
    /// let Ok(triangle) = TriangleHandle::try_new(*a, *b, *c) else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt.can_flip_k3_inverse_from_triangle(triangle);
    /// std::assert_matches!(
    ///     result,
    ///     Err(FlipError::InvalidTriangleMultiplicity { .. })
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn can_flip_k3_inverse_from_triangle(
        &self,
        triangle: TriangleHandle,
    ) -> Result<FlipFeasibility<D>, FlipError>;
}

impl<K, U, V, const D: usize> BistellarFlips<D> for Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    type VertexData = U;

    fn flip_k1_insert(
        &mut self,
        simplex_key: SimplexKey,
        vertex: Vertex<U, D>,
    ) -> Result<FlipInfo<D>, FlipError> {
        let _ = self.can_flip_k1_insert(simplex_key, &vertex)?;
        apply_realized_flip(self, TopologicalOperation::InsertVertex, |tri| {
            apply_bistellar_flip_k1_raw(&mut tri.tds, simplex_key, vertex)
        })
    }

    fn can_flip_k1_insert(
        &self,
        simplex_key: SimplexKey,
        vertex: &Vertex<U, D>,
    ) -> Result<FlipFeasibility<D>, FlipError> {
        validate_flip_topology(self, TopologicalOperation::InsertVertex)?;
        let topology_model = self.global_topology.model();
        validate_bistellar_flip_k1_insert(&self.tds, &topology_model, simplex_key, vertex)
    }

    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError> {
        apply_realized_flip(self, TopologicalOperation::DeleteVertex, |tri| {
            apply_bistellar_flip_k1_inverse_raw(&mut tri.tds, vertex_key)
        })
    }

    fn can_flip_k1_remove(&self, vertex_key: VertexKey) -> Result<FlipFeasibility<D>, FlipError> {
        validate_flip_topology(self, TopologicalOperation::DeleteVertex)?;
        validate_bistellar_flip_k1_inverse(&self.tds, vertex_key)
    }

    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError> {
        apply_realized_flip(self, TopologicalOperation::FacetFlip, |tri| {
            let context = build_k2_flip_context(&tri.tds, facet)?;
            apply_bistellar_flip_raw::<U, V, D, 2>(&mut tri.tds, &context)
        })
    }

    fn can_flip_k2(&self, facet: FacetHandle) -> Result<FlipFeasibility<D>, FlipError> {
        validate_flip_topology(self, TopologicalOperation::FacetFlip)?;
        let context = build_k2_flip_context(&self.tds, facet)?;
        validate_bistellar_flip_k2(&self.tds, &context)
    }

    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError> {
        apply_realized_flip(self, TopologicalOperation::CavityFlip, |tri| {
            let context = build_k3_flip_context(&tri.tds, ridge)?;
            apply_bistellar_flip_raw::<U, V, D, 3>(&mut tri.tds, &context)
        })
    }

    fn can_flip_k3(&self, ridge: RidgeHandle) -> Result<FlipFeasibility<D>, FlipError> {
        validate_flip_topology(self, TopologicalOperation::CavityFlip)?;
        let context = build_k3_flip_context(&self.tds, ridge)?;
        validate_bistellar_flip_k3(&self.tds, &context)
    }

    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError> {
        apply_realized_flip(self, TopologicalOperation::CavityFlip, |tri| {
            let context = build_k2_flip_context_from_edge(&tri.tds, edge)?;
            apply_bistellar_flip_dynamic_raw(&mut tri.tds, D, &context)
        })
    }

    fn can_flip_k2_inverse_from_edge(
        &self,
        edge: EdgeKey,
    ) -> Result<FlipFeasibility<D>, FlipError> {
        validate_flip_topology(self, TopologicalOperation::CavityFlip)?;
        let context = build_k2_flip_context_from_edge(&self.tds, edge)?;
        validate_bistellar_flip_dynamic(&self.tds, D, &context)
    }

    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError> {
        if D < 4 {
            return Err(FlipError::UnsupportedDimension { dimension: D });
        }

        apply_realized_flip(self, TopologicalOperation::CavityFlip, |tri| {
            let context = build_k3_flip_context_from_triangle(&tri.tds, triangle)?;

            // Avoid const-eval underflow for invalid instantiations (e.g. D=0), even though
            // the public contract for this method requires D>=4.
            let k_move = D
                .checked_sub(1)
                .ok_or(FlipError::UnsupportedDimension { dimension: D })?;

            apply_bistellar_flip_dynamic_raw(&mut tri.tds, k_move, &context)
        })
    }

    fn can_flip_k3_inverse_from_triangle(
        &self,
        triangle: TriangleHandle,
    ) -> Result<FlipFeasibility<D>, FlipError> {
        if D < 4 {
            return Err(FlipError::UnsupportedDimension { dimension: D });
        }
        validate_flip_topology(self, TopologicalOperation::CavityFlip)?;

        let context = build_k3_flip_context_from_triangle(&self.tds, triangle)?;

        // Avoid const-eval underflow for invalid instantiations (e.g. D=0), even though
        // the public contract for this method requires D>=4.
        let k_move = D
            .checked_sub(1)
            .ok_or(FlipError::UnsupportedDimension { dimension: D })?;

        validate_bistellar_flip_dynamic(&self.tds, k_move, &context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::facet::FacetError;
    use crate::core::tds::InvariantError;
    use crate::vertex;
    use std::assert_matches;

    use crate::TopologyGuarantee;
    use crate::core::collections::{SimplexKeyBuffer, SmallBuffer};
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use slotmap::KeyData;

    #[test]
    fn triangulation_flip_k1_insert_and_remove_roundtrip() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .build()
            .unwrap();
        let mut tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;
        let candidate = vertex!([0.25, 0.25, 0.25]).unwrap();
        let feasibility = tri
            .can_flip_k1_insert(simplex_key, &candidate)
            .expect("interior k=1 insertion should pass preflight");

        let inserted = tri.flip_k1_insert(simplex_key, candidate).unwrap();
        let inserted_vertex = inserted.inserted_face_vertices[0];
        assert_eq!(feasibility.kind, inserted.kind);
        assert_eq!(feasibility.direction, inserted.direction);
        assert_eq!(feasibility.removed_simplices, inserted.removed_simplices);
        assert_eq!(
            feasibility.removed_face_vertices,
            inserted.removed_face_vertices
        );
        assert!(!inserted.new_simplices.is_empty());
        assert!(tri.validate().is_ok());

        let removed = tri.flip_k1_remove(inserted_vertex).unwrap();
        assert!(!removed.removed_simplices.is_empty());
        assert!(tri.validate().is_ok());
    }

    #[test]
    fn triangulation_flip_k1_insert_rolls_back_degenerate_insert() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .build()
            .unwrap();
        let mut tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;
        let before_vertices = tri.tds.number_of_vertices();
        let before_simplices = tri.tds.number_of_simplices();
        let inserted = vertex!([0.5, 0.0]).unwrap();
        let inserted_uuid = inserted.uuid();

        let preflight_err = tri.can_flip_k1_insert(simplex_key, &inserted).unwrap_err();
        assert_matches!(preflight_err, FlipError::DegenerateSimplex);
        assert_eq!(tri.tds.number_of_vertices(), before_vertices);
        assert_eq!(tri.tds.number_of_simplices(), before_simplices);

        let err = tri.flip_k1_insert(simplex_key, inserted).unwrap_err();

        assert_matches!(err, FlipError::DegenerateSimplex);
        assert_eq!(tri.tds.number_of_vertices(), before_vertices);
        assert_eq!(tri.tds.number_of_simplices(), before_simplices);
        assert!(tri.vertex_key_from_uuid(&inserted_uuid).is_none());
        assert!(tri.validate().is_ok());
        assert!(tri.is_valid_realization().is_ok());
    }

    #[test]
    fn triangulation_flip_k1_insert_rejects_exterior_point_before_mutation() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .build()
            .unwrap();
        let mut tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;
        let expected_opposite_index = tri
            .simplex(simplex_key)
            .unwrap()
            .vertices()
            .iter()
            .position(|&vertex_key| *tri.vertex(vertex_key).unwrap().point().coords() == [0.0, 0.0])
            .expect("fixture simplex should contain the origin");
        let expected_opposite_vertex =
            tri.simplex(simplex_key).unwrap().vertices()[expected_opposite_index];
        let inserted = vertex!([0.75, 0.75]).unwrap();
        let inserted_uuid = inserted.uuid();
        let before = tri.tds.clone();

        let preflight_err = tri.can_flip_k1_insert(simplex_key, &inserted).unwrap_err();
        assert_eq!(
            FlipFailureKind::from(&preflight_err),
            FlipFailureKind::K1InsertionOutsideSimplex
        );
        assert_matches!(
            preflight_err,
            FlipError::K1InsertionOutsideSimplex {
                simplex_key: rejected,
                opposite_vertex,
                opposite_vertex_index,
            } if rejected == simplex_key
                && opposite_vertex == expected_opposite_vertex
                && opposite_vertex_index == expected_opposite_index
        );
        assert_eq!(tri.tds, before);

        let commit_err = tri.flip_k1_insert(simplex_key, inserted).unwrap_err();
        assert_matches!(
            commit_err,
            FlipError::K1InsertionOutsideSimplex {
                simplex_key: rejected,
                opposite_vertex,
                opposite_vertex_index,
            } if rejected == simplex_key
                && opposite_vertex == expected_opposite_vertex
                && opposite_vertex_index == expected_opposite_index
        );
        assert_eq!(tri.tds, before);
        assert!(tri.vertex_key_from_uuid(&inserted_uuid).is_none());
        assert!(tri.validate().is_ok());
        assert!(tri.is_valid_realization().is_ok());
    }

    #[test]
    fn realized_flip_transaction_rolls_back_topology_validation_failure() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .build()
            .unwrap();
        let mut tri = dt.into_triangulation();
        let before_vertices = tri.tds.number_of_vertices();
        let before_simplices = tri.tds.number_of_simplices();

        let err = apply_realized_flip(&mut tri, TopologicalOperation::InsertVertex, |tri| {
            tri.tds
                .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
                .unwrap();
            Ok(FlipInfo {
                kind: BistellarFlipKind::try_k1(2).unwrap(),
                direction: FlipDirection::Forward,
                removed_simplices: SimplexKeyBuffer::default(),
                new_simplices: SimplexKeyBuffer::default(),
                removed_face_vertices: SmallBuffer::default(),
                inserted_face_vertices: SmallBuffer::default(),
            })
        })
        .unwrap_err();

        assert_matches!(
            err,
            FlipError::InvariantValidation { source }
                if matches!(*source, InvariantError::Triangulation { source: _ })
        );
        assert_eq!(tri.tds.number_of_vertices(), before_vertices);
        assert_eq!(tri.tds.number_of_simplices(), before_simplices);
        assert!(tri.validate().is_ok());
        assert!(tri.validate_realization().is_ok());
    }

    #[test]
    fn flip_k1_insert_requires_explicit_delaunay_demotion() {
        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;

        tri.flip_k1_insert(simplex_key, vertex!([0.2, 0.2, 0.2]).unwrap())
            .unwrap();

        assert!(tri.validate().is_ok());
        assert!(tri.validate_realization().is_ok());
    }

    #[test]
    fn facet_flips_require_a_pl_manifold_proof_before_mutation() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .build()
            .unwrap();
        let mut tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;
        let facet = tri.facet_handle(simplex_key, 0).unwrap();
        let before = tri.tds.clone();

        let preflight_error = tri.can_flip_k2(facet).unwrap_err();
        assert_matches!(
            preflight_error,
            FlipError::FlipTopologyNotAdmissible {
                required: TopologyGuarantee::PLManifold,
                found: TopologyGuarantee::Pseudomanifold,
            }
        );

        let mutation_error = tri.flip_k2(facet).unwrap_err();
        assert_matches!(
            mutation_error,
            FlipError::FlipTopologyNotAdmissible {
                required: TopologyGuarantee::PLManifold,
                found: TopologyGuarantee::Pseudomanifold,
            }
        );
        assert_eq!(tri.tds, before);
    }

    #[test]
    fn triangulation_flip_k2_rejects_invalid_facet_index() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .build()
            .unwrap();
        let tri = dt.into_triangulation();
        let simplex_key = tri.simplices().next().unwrap().0;

        let err = tri.facet_handle(simplex_key, u8::MAX).unwrap_err();

        assert_matches!(
            err,
            FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count: 4,
            }
        );
    }

    #[test]
    fn triangulation_flip_k3_inverse_rejects_unsupported_dimension() {
        let mut tri: Triangulation<AdaptiveKernel<f64>, (), (), 3> =
            Triangulation::new_empty(AdaptiveKernel::new());
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let err = tri
            .flip_k3_inverse_from_triangle(TriangleHandle::try_new(a, b, c).unwrap())
            .unwrap_err();

        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 3 });
    }

    #[test]
    fn triangulation_can_flip_k3_inverse_prioritizes_unsupported_dimension() {
        let tri: Triangulation<AdaptiveKernel<f64>, (), (), 3> =
            Triangulation::new_empty(AdaptiveKernel::new());
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let error = tri
            .can_flip_k3_inverse_from_triangle(TriangleHandle::try_new(a, b, c).unwrap())
            .expect_err("3D inverse k=3 feasibility must reject the dimension first");

        assert_matches!(error, FlipError::UnsupportedDimension { dimension: 3 });
    }

    #[test]
    fn triangulation_flip_k3_inverse_rejects_zero_dimension_without_underflow() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 0> =
            Triangulation::new_empty(FastKernel::new());
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let err = tri
            .flip_k3_inverse_from_triangle(TriangleHandle::try_new(a, b, c).unwrap())
            .unwrap_err();

        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 0 });
    }
}
