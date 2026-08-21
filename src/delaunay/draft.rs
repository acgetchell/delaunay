//! Unpublished incremental Delaunay construction state.

#![forbid(unsafe_code)]

use crate::construction::default_duplicate_tolerance;
use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::{DelaunayInsertionState, InsertionOutcome, InsertionStatistics};
use crate::core::tds::{TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::delaunay_model::{DelaunayTriangulation, EuclideanDelaunayReportDomain};
use crate::geometry::kernel::{AdaptiveKernel, Kernel};
use crate::repair::DelaunayCheckPolicy;
use crate::topology::traits::{GlobalTopology, TopologyKind};
use crate::triangulation::builder::TriangulationBuilderError;
use crate::triangulation::draft::TriangulationDraft;
use crate::triangulation::validation::{
    TopologyGuarantee, ValidationConfigurationError, ValidationPolicy,
};
use crate::validation::{DelaunayRefinementCandidate, DelaunayTriangulationValidationError};
use thiserror::Error;

/// Typed failures while mutating or publishing a Delaunay construction draft.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayTriangulationDraftError {
    /// Incremental insertion rejected the requested vertex.
    #[error("incremental draft insertion failed: {source}")]
    Insertion {
        /// Typed insertion failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// The unpublished lower-layer workspace could not establish Levels 3–4.
    #[error("triangulation draft publication failed: {source}")]
    TriangulationPublication {
        /// Typed Levels 3–4 publication failure.
        #[source]
        source: TriangulationBuilderError,
    },
    /// The Levels 1–4 candidate could not establish Level 5.
    #[error("Delaunay draft certification failed: {source}")]
    DelaunayCertification {
        /// Typed Level 5 certification failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
    /// The verified draft failed its cumulative final audit.
    #[error("Delaunay draft final validation failed: {source}")]
    FinalValidation {
        /// Typed cumulative Levels 1–5 validation failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
    /// A nonempty bootstrap does not yet span a full-dimensional simplex.
    #[error(
        "cannot publish a {dimension}D Delaunay draft with {vertex_count} bootstrap vertices and no maximal simplex"
    )]
    IncompleteBootstrap {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Number of vertices accumulated before publication was attempted.
        vertex_count: usize,
    },
}

impl From<InsertionError> for DelaunayTriangulationDraftError {
    fn from(source: InsertionError) -> Self {
        Self::Insertion {
            source: Box::new(source),
        }
    }
}

/// Unpublished bootstrap state before the first full-dimensional simplex exists.
#[derive(Clone, Debug)]
struct DelaunayBootstrap<K, U, V, const D: usize> {
    triangulation: TriangulationDraft<K, U, V, D>,
    insertion_state: DelaunayInsertionState,
    spatial_index: Option<HashGridIndex<D>>,
    euclidean_report_domain: EuclideanDelaunayReportDomain,
}

impl<K, U, V, const D: usize> DelaunayBootstrap<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Creates empty unpublished lower-layer state with Delaunay-owned caches.
    fn with_topology_context(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            triangulation: TriangulationDraft::empty_unpublished_with_topology_context(
                kernel,
                topology_guarantee,
                global_topology,
            ),
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: HashGridIndex::try_new(default_duplicate_tolerance()).ok(),
            euclidean_report_domain: if global_topology.is_euclidean() {
                EuclideanDelaunayReportDomain::CompletePointSet
            } else {
                EuclideanDelaunayReportDomain::Unproven
            },
        }
    }
}

impl<K, U, V, const D: usize> DelaunayBootstrap<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Copies the small bootstrap workspace while preserving owner identity for rollback.
    fn clone_for_rollback(&self) -> Self {
        Self {
            triangulation: self.triangulation.clone_for_rollback(),
            insertion_state: self.insertion_state,
            spatial_index: self.spatial_index.clone(),
            euclidean_report_domain: self.euclidean_report_domain,
        }
    }

    /// Returns whether the lower-layer insertion has formed a maximal simplex.
    fn is_publishable(&self) -> bool {
        self.triangulation.number_of_simplices() > 0
    }

    /// Publishes Levels 3–4 and then crosses the single Level 5 candidate boundary.
    fn publish(self) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationDraftError> {
        let triangulation = self
            .triangulation
            .finish_canonicalizing()
            .map_err(
                |failure| DelaunayTriangulationDraftError::TriangulationPublication {
                    source: failure.into_reason(),
                },
            )?;
        DelaunayRefinementCandidate::from_parts(
            triangulation,
            self.insertion_state,
            self.spatial_index,
            self.euclidean_report_domain,
        )
        .try_into_delaunay()
        .map_err(
            |failure| DelaunayTriangulationDraftError::DelaunayCertification {
                source: failure.into_reason(),
            },
        )
    }
}

/// Internal state split between incomplete bootstrap and a fully verified owner.
#[derive(Clone, Debug)]
enum DelaunayDraftState<K, U, V, const D: usize> {
    Bootstrap(DelaunayBootstrap<K, U, V, D>),
    Verified(DelaunayTriangulation<K, U, V, D>),
}

/// Mutable, unpublished construction state for a [`DelaunayTriangulation`].
///
/// The draft owns bootstrap vertices before they span a full-dimensional
/// simplex. The insertion that first creates one failure-atomically establishes
/// Levels 3–5 and changes the private state to an already verified owner; a
/// failed transition restores the exact pre-insertion bootstrap. Later
/// insertions use that owner's transactional mutation path.
/// [`finish`](Self::finish) is the only operation that returns the Levels 1–5
/// owner to the caller, after a cumulative final audit.
#[derive(Clone, Debug)]
pub struct DelaunayTriangulationDraft<K, U, V, const D: usize> {
    state: DelaunayDraftState<K, U, V, D>,
}

impl<const D: usize> DelaunayTriangulationDraft<AdaptiveKernel<f64>, (), (), D> {
    /// Starts an empty draft using the default adaptive kernel and topology.
    #[must_use]
    pub fn new() -> Self {
        Self::with_kernel(AdaptiveKernel::new())
    }

    /// Starts an empty draft with an explicit topology guarantee.
    #[must_use]
    pub fn with_topology_guarantee(topology_guarantee: TopologyGuarantee) -> Self {
        Self::with_kernel_and_topology_guarantee(AdaptiveKernel::new(), topology_guarantee)
    }
}

impl<const D: usize> Default for DelaunayTriangulationDraft<AdaptiveKernel<f64>, (), (), D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulationDraft<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Starts an empty unpublished draft with a caller-selected kernel.
    #[must_use]
    pub fn with_kernel(kernel: K) -> Self {
        Self::with_kernel_and_topology_context(
            kernel,
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
    }

    /// Starts an empty unpublished draft with a kernel and topology guarantee.
    #[must_use]
    pub fn with_kernel_and_topology_guarantee(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        Self::with_kernel_and_topology_context(kernel, topology_guarantee, GlobalTopology::DEFAULT)
    }

    /// Starts an empty unpublished draft with complete topology context.
    #[must_use]
    pub(crate) fn with_kernel_and_topology_context(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            state: DelaunayDraftState::Bootstrap(DelaunayBootstrap::with_topology_context(
                kernel,
                topology_guarantee,
                global_topology,
            )),
        }
    }

    /// Returns the number of vertices currently accumulated by the draft.
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.number_of_vertices(),
            DelaunayDraftState::Verified(triangulation) => triangulation.number_of_vertices(),
        }
    }

    /// Returns the number of maximal simplices currently accumulated by the draft.
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.number_of_simplices(),
            DelaunayDraftState::Verified(triangulation) => triangulation.number_of_simplices(),
        }
    }

    /// Returns the current dimensionality reported by the draft storage.
    #[must_use]
    pub fn dim(&self) -> i32 {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.dim(),
            DelaunayDraftState::Verified(triangulation) => triangulation.dim(),
        }
    }

    /// Returns the topology guarantee selected for eventual publication.
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.topology_guarantee(),
            DelaunayDraftState::Verified(triangulation) => triangulation.topology_guarantee(),
        }
    }

    /// Returns the global topology selected for eventual publication.
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.global_topology(),
            DelaunayDraftState::Verified(triangulation) => triangulation.global_topology(),
        }
    }

    /// Returns the high-level topology kind selected for the draft.
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.topology_kind(),
            DelaunayDraftState::Verified(triangulation) => triangulation.topology_kind(),
        }
    }

    /// Returns the insertion-time validation policy selected for the draft.
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => {
                state.triangulation.retained_validation_policy()
            }
            DelaunayDraftState::Verified(triangulation) => triangulation.validation_policy(),
        }
    }

    /// Changes the draft's insertion-time validation policy when compatible.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationConfigurationError`] when the policy cannot support
    /// the selected topology guarantee.
    pub fn try_set_validation_policy(
        &mut self,
        policy: ValidationPolicy,
    ) -> Result<(), ValidationConfigurationError> {
        match &mut self.state {
            DelaunayDraftState::Bootstrap(state) => {
                state.triangulation.try_set_validation_policy(policy)
            }
            DelaunayDraftState::Verified(triangulation) => {
                triangulation.try_set_validation_policy(policy)
            }
        }
    }

    /// Returns the Level 5 checking cadence used while inserting into the draft.
    #[must_use]
    pub const fn delaunay_check_policy(&self) -> DelaunayCheckPolicy {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.insertion_state.delaunay_check_policy,
            DelaunayDraftState::Verified(triangulation) => triangulation.delaunay_check_policy(),
        }
    }

    /// Sets the Level 5 checking cadence used while inserting into the draft.
    pub const fn set_delaunay_check_policy(&mut self, policy: DelaunayCheckPolicy) {
        match &mut self.state {
            DelaunayDraftState::Bootstrap(state) => {
                state.insertion_state.delaunay_check_policy = policy;
            }
            DelaunayDraftState::Verified(triangulation) => {
                triangulation.set_delaunay_check_policy(policy);
            }
        }
    }

    /// Returns whether the draft's currently assembled simplices are coherently oriented.
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.is_coherently_oriented(),
            DelaunayDraftState::Verified(triangulation) => triangulation.is_coherently_oriented(),
        }
    }

    /// Validates the currently assembled Levels 1–2 state without publishing it.
    ///
    /// # Errors
    ///
    /// Returns the first structural error in the draft. A successful result does
    /// not imply that enough vertices exist to publish a full-dimensional owner.
    pub fn validate_structure(&self) -> Result<(), TdsError> {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.validate_structure(),
            DelaunayDraftState::Verified(triangulation) => triangulation.validate_structure(),
        }
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulationDraft<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Incrementally inserts a vertex into the unpublished draft.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationDraftError`] when bootstrap insertion or
    /// first-simplex publication fails. Transactional failures leave the draft
    /// reusable.
    pub fn insert_vertex(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<VertexKey, DelaunayTriangulationDraftError> {
        match self.insert_best_effort_with_statistics(vertex)? {
            (InsertionOutcome::Inserted { vertex_key, .. }, _) => Ok(vertex_key),
            (InsertionOutcome::Skipped { error }, _) => Err(error.into()),
        }
    }

    /// Returns a staged vertex by key without exposing the unpublished owner.
    #[must_use]
    pub fn vertex(&self, key: VertexKey) -> Option<&Vertex<U, D>> {
        match &self.state {
            DelaunayDraftState::Bootstrap(state) => state.triangulation.vertex(key),
            DelaunayDraftState::Verified(triangulation) => triangulation.vertex(key),
        }
    }

    /// Inserts a vertex and returns the public insertion outcome and statistics.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationDraftError`] when insertion or the first
    /// Levels 3–5 publication checkpoint fails.
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), DelaunayTriangulationDraftError> {
        match self.insert_best_effort_with_statistics(vertex)? {
            (outcome @ InsertionOutcome::Inserted { .. }, statistics) => Ok((outcome, statistics)),
            (InsertionOutcome::Skipped { error }, _) => Err(error.into()),
        }
    }

    /// Inserts a vertex and reports retryable skips in the returned outcome.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationDraftError`] when insertion or the first
    /// Levels 3–5 publication checkpoint encounters a non-retryable failure.
    pub fn insert_best_effort_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), DelaunayTriangulationDraftError> {
        let mut publication = None;
        let result = match &mut self.state {
            DelaunayDraftState::Bootstrap(state) => {
                let publication_snapshot = (state.triangulation.number_of_vertices() + 1 > D)
                    .then(|| state.clone_for_rollback());
                let hint = state.insertion_state.last_inserted_simplex;
                let detail = state
                    .triangulation
                    .insert_with_statistics_seeded_indexed_detailed(
                        vertex,
                        hint,
                        0,
                        state.spatial_index.as_mut(),
                    )?;
                let outcome = detail.outcome;
                if let InsertionOutcome::Inserted { hint, .. } = outcome {
                    state.insertion_state.last_inserted_simplex = hint;
                    state.insertion_state.delaunay_repair_insertion_count = state
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    if state.is_publishable()
                        && let Some(snapshot) = publication_snapshot
                    {
                        publication = Some(std::mem::replace(state, snapshot));
                    }
                }
                (outcome, detail.stats)
            }
            DelaunayDraftState::Verified(triangulation) => {
                return triangulation
                    .insert_best_effort_with_statistics(vertex)
                    .map_err(Into::into);
            }
        };

        if let Some(state) = publication {
            let triangulation = state.publish()?;
            self.state = DelaunayDraftState::Verified(triangulation);
        }

        Ok(result)
    }

    /// Consumes the draft and publishes it only after Levels 1–5 validation.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationDraftError`] when the draft is incomplete
    /// or violates any cumulative structural, topology, realization, or
    /// Delaunay invariant.
    pub fn finish(
        self,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationDraftError> {
        match self.state {
            DelaunayDraftState::Bootstrap(state) => {
                let vertex_count = state.triangulation.number_of_vertices();
                if vertex_count != 0 {
                    return Err(DelaunayTriangulationDraftError::IncompleteBootstrap {
                        dimension: D,
                        vertex_count,
                    });
                }
                state.publish()
            }
            DelaunayDraftState::Verified(triangulation) => {
                triangulation.validate().map_err(|source| {
                    DelaunayTriangulationDraftError::FinalValidation { source }
                })?;
                Ok(triangulation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::RobustKernel;
    use crate::vertex;

    macro_rules! test_incremental_draft {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<incremental_draft_bootstraps_and_continues_in_ $dim d>]() {
                    let mut draft: DelaunayTriangulationDraft<_, (), (), $dim> =
                        DelaunayTriangulationDraft::new();
                    let initial_vertices = vec![$(vertex!($simplex_coords).unwrap()),+];
                    assert_eq!(initial_vertices.len(), $dim + 1);

                    for (index, vertex) in initial_vertices.iter().take($dim).enumerate() {
                        draft.insert_vertex(*vertex).unwrap();
                        assert_eq!(draft.number_of_vertices(), index + 1);
                        assert_eq!(draft.number_of_simplices(), 0);
                        assert!(matches!(draft.state, DelaunayDraftState::Bootstrap(_)));
                    }

                    draft.insert_vertex(*initial_vertices.last().unwrap()).unwrap();
                    assert_eq!(draft.number_of_vertices(), $dim + 1);
                    assert_eq!(draft.number_of_simplices(), 1);
                    assert!(matches!(draft.state, DelaunayDraftState::Verified(_)));

                    draft.insert_vertex(vertex!($interior_point).unwrap()).unwrap();
                    let triangulation = draft.finish().unwrap();
                    assert_eq!(triangulation.number_of_vertices(), $dim + 2);
                    assert!(triangulation.number_of_simplices() > 1);
                    assert!(triangulation.validate().is_ok());
                }
            }
        };
    }

    test_incremental_draft!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.2, 0.2]);

    test_incremental_draft!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2]
    );

    test_incremental_draft!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.1, 0.1, 0.1, 0.1]
    );

    test_incremental_draft!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.1, 0.1, 0.1, 0.1, 0.1]
    );

    #[test]
    fn draft_bootstraps_with_custom_kernel() {
        let mut draft: DelaunayTriangulationDraft<RobustKernel<f64>, (), (), 3> =
            DelaunayTriangulationDraft::with_kernel(RobustKernel::new());

        draft
            .insert_vertex(vertex![0.0, 0.0, 0.0].unwrap())
            .unwrap();
        draft
            .insert_vertex(vertex![1.0, 0.0, 0.0].unwrap())
            .unwrap();
        draft
            .insert_vertex(vertex![0.0, 1.0, 0.0].unwrap())
            .unwrap();
        assert_eq!(draft.number_of_simplices(), 0);

        draft
            .insert_vertex(vertex![0.0, 0.0, 1.0].unwrap())
            .unwrap();
        let triangulation = draft.finish().unwrap();
        assert_eq!(triangulation.number_of_simplices(), 1);
        assert!(triangulation.validate().is_ok());
    }

    #[test]
    fn empty_draft_publishes_as_the_verified_empty_delaunay_triangulation() {
        let draft: DelaunayTriangulationDraft<_, (), (), 2> = DelaunayTriangulationDraft::new();

        let triangulation = draft.finish().unwrap();
        assert_eq!(triangulation.number_of_vertices(), 0);
        assert!(triangulation.validate().is_ok());
    }

    #[test]
    fn partial_bootstrap_draft_cannot_publish() {
        let mut draft: DelaunayTriangulationDraft<_, (), (), 2> = DelaunayTriangulationDraft::new();
        draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();

        assert_eq!(
            draft.finish().unwrap_err(),
            DelaunayTriangulationDraftError::IncompleteBootstrap {
                dimension: 2,
                vertex_count: 1,
            }
        );
    }

    #[test]
    fn failed_first_publication_restores_the_exact_bootstrap_owner() {
        let mut draft: DelaunayTriangulationDraft<_, (), (), 2> =
            DelaunayTriangulationDraft::with_kernel_and_topology_context(
                AdaptiveKernel::new(),
                TopologyGuarantee::PLManifold,
                GlobalTopology::Euclidean,
            );
        let first_key = draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        let second_key = draft.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();

        let DelaunayDraftState::Bootstrap(state) = &mut draft.state else {
            panic!("two 2D vertices cannot publish");
        };
        state
            .triangulation
            .select_validation_policy_unchecked_for_test(ValidationPolicy::Never);

        let (owner_before, generation_before, insertion_count_before) = match &draft.state {
            DelaunayDraftState::Bootstrap(state) => (
                state.triangulation.topology_owner_id(),
                state.triangulation.topology_generation(),
                state.insertion_state.delaunay_repair_insertion_count,
            ),
            DelaunayDraftState::Verified(_) => panic!("two 2D vertices cannot publish"),
        };

        let failure = draft.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap_err();
        assert!(
            matches!(
                &failure,
                DelaunayTriangulationDraftError::TriangulationPublication { .. }
            ),
            "unexpected publication failure: {failure:?}"
        );

        let DelaunayDraftState::Bootstrap(state) = &draft.state else {
            panic!("failed publication must restore bootstrap state");
        };
        assert_eq!(state.triangulation.topology_owner_id(), owner_before);
        assert_eq!(state.triangulation.topology_generation(), generation_before);
        assert_eq!(
            state.insertion_state.delaunay_repair_insertion_count,
            insertion_count_before
        );
        assert_eq!(state.triangulation.number_of_vertices(), 2);
        assert_eq!(state.triangulation.number_of_simplices(), 0);
        assert!(state.triangulation.vertex(first_key).is_some());
        assert!(state.triangulation.vertex(second_key).is_some());

        let retry = draft.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap_err();
        assert!(
            matches!(
                &retry,
                DelaunayTriangulationDraftError::TriangulationPublication { .. }
            ),
            "unexpected retry failure: {retry:?}"
        );
    }
}
