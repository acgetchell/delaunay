//! Incremental Delaunay construction from an empty bootstrap workspace.
//!
//! Use [`DelaunayIncrementalBuilder`] when vertices arrive one at a time before
//! a full-dimensional simplex exists. The builder keeps partial connectivity
//! unpublished, crosses validation Levels 1–5 when the first maximal simplex
//! is formed, and uses transactional owner insertion thereafter. Calling
//! [`DelaunayIncrementalBuilder::finish`] on an empty builder publishes the
//! valid empty complex; finishing a nonempty partial bootstrap returns
//! [`DelaunayIncrementalBuilderError::IncompleteBootstrap`].
//!
//! For batch point-set construction, use
//! [`DelaunayTriangulationBuilder`](crate::DelaunayTriangulationBuilder).
//! Once an owner already contains a maximal simplex, insert through
//! [`DelaunayTriangulation::insert_vertex`] instead.

#![forbid(unsafe_code)]

use thiserror::Error;

use crate::construction::default_duplicate_tolerance;
use crate::core::algorithms::insertion::InsertionError;
use crate::core::collections::{SimplexVertexKeyBuffer, spatial_hash_grid::HashGridIndex};
use crate::core::operations::{
    DelaunayInsertionState, InsertionOutcome, InsertionResult, InsertionStatistics,
};
use crate::core::tds::{
    TdsDraft, TdsDraftError, TdsDraftInsertionError, TdsError, TdsRollbackSavepoint,
    TriangulationValidationReport, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::delaunay_model::{DelaunayTriangulation, EuclideanDelaunayReportDomain};
use crate::draft::DelaunayTriangulationDraft;
use crate::geometry::kernel::{AdaptiveKernel, ExactPredicates};
use crate::geometry::traits::coordinate::CoordinateValues;
use crate::refinement::RefinementError;
use crate::repair::DelaunayCheckPolicy;
use crate::topology::traits::{GlobalTopology, TopologyKind};
use crate::triangulation::builder::TriangulationBuilderError;
use crate::triangulation::draft::TriangulationDraft;
use crate::triangulation::insertion::duplicate_coordinate_tolerance_from_references;
use crate::triangulation::validation::{
    TopologyConstructionProvenance, TopologyGuarantee, ValidationConfigurationError,
    ValidationPolicy,
};
use crate::validation::DelaunayTriangulationValidationError;

/// Typed failures while operating or finishing an incremental Delaunay builder.
///
/// Each variant identifies the proof or mutation stage that rejected the
/// operation, so callers can distinguish an insertion failure from publication
/// through Levels 1–2, Levels 3–4, or Level 5.
///
/// # Examples
///
/// A nonempty bootstrap that does not yet span a maximal simplex cannot be
/// published:
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayIncrementalBuilder, DelaunayIncrementalBuilderError, DelaunayResult, vertex,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
///     DelaunayIncrementalBuilder::new();
/// builder.insert_vertex(vertex![0.0, 0.0]?)?;
///
/// std::assert_matches!(
///     builder.finish(),
///     Err(DelaunayIncrementalBuilderError::IncompleteBootstrap {
///         dimension: 2,
///         vertex_count: 1,
///     })
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayIncrementalBuilderError {
    /// Incremental insertion rejected the requested vertex.
    #[error("incremental builder insertion failed: {source}")]
    Insertion {
        /// Typed insertion failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Explicit bootstrap connectivity could not be staged.
    #[error("Delaunay bootstrap assembly failed: {source}")]
    BootstrapAssembly {
        /// Typed TDS draft insertion failure.
        #[source]
        source: TdsDraftInsertionError,
    },
    /// The bootstrap TDS could not establish Levels 1–2.
    #[error("Delaunay bootstrap TDS publication failed: {source}")]
    TdsPublication {
        /// Typed Levels 1–2 publication failure.
        #[source]
        source: TdsDraftError,
    },
    /// The unpublished lower-layer workspace could not establish Levels 3–4.
    #[error("triangulation draft publication failed: {source}")]
    TriangulationPublication {
        /// Typed Levels 3–4 publication failure.
        #[source]
        source: TriangulationBuilderError,
    },
    /// The Levels 1–4 draft could not establish Level 5.
    #[error("Delaunay incremental certification failed: {source}")]
    DelaunayCertification {
        /// Typed Level 5 certification failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
    /// The verified builder state failed its cumulative final audit.
    #[error("Delaunay incremental final validation failed: {source}")]
    FinalValidation {
        /// Typed cumulative Levels 1–5 validation failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
    /// A nonempty bootstrap does not yet span a full-dimensional simplex.
    #[error(
        "cannot finish a {dimension}D Delaunay incremental builder with {vertex_count} bootstrap vertices and no maximal simplex"
    )]
    IncompleteBootstrap {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Number of vertices accumulated before publication was attempted.
        vertex_count: usize,
    },
}

impl From<InsertionError> for DelaunayIncrementalBuilderError {
    fn from(source: InsertionError) -> Self {
        Self::Insertion {
            source: Box::new(source),
        }
    }
}

/// Unpublished bootstrap state before the first full-dimensional simplex exists.
#[derive(Clone, Debug)]
struct DelaunayBootstrapWorkspace<K, U, V, const D: usize> {
    tds: TdsDraft<U, V, D>,
    vertex_keys: SimplexVertexKeyBuffer,
    kernel: K,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    validation_policy: ValidationPolicy,
    insertion_state: DelaunayInsertionState,
    spatial_index: Option<HashGridIndex<D>>,
    euclidean_report_domain: EuclideanDelaunayReportDomain,
}

impl<K, U, V, const D: usize> DelaunayBootstrapWorkspace<K, U, V, D> {
    /// Creates empty unpublished lower-layer state with Delaunay-owned caches.
    fn with_topology_context(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            tds: TdsDraft::new(),
            vertex_keys: SimplexVertexKeyBuffer::new(),
            kernel,
            topology_guarantee,
            global_topology,
            validation_policy: topology_guarantee.default_validation_policy(),
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: HashGridIndex::try_new(default_duplicate_tolerance()).ok(),
            euclidean_report_domain: if global_topology.is_euclidean() {
                EuclideanDelaunayReportDomain::CompletePointSet
            } else {
                EuclideanDelaunayReportDomain::Unproven
            },
        }
    }
    /// Returns whether the lower-layer insertion has formed a maximal simplex.
    fn is_publishable(&self) -> bool {
        self.tds.number_of_simplices() > 0
    }

    /// Returns a scale-aware duplicate-coordinate failure for bootstrap input.
    fn duplicate_coordinates_error(&self, coords: &[f64; D]) -> Option<InsertionError> {
        let mut minimum_distance_squared: Option<f64> = None;

        for (_, existing) in self.tds.vertices() {
            let mut distance_squared = 0.0;
            for (coordinate, existing_coordinate) in coords.iter().zip(existing.point().coords()) {
                let difference = coordinate - existing_coordinate;
                distance_squared = difference.mul_add(difference, distance_squared);
            }
            minimum_distance_squared = Some(
                minimum_distance_squared
                    .map_or(distance_squared, |minimum| minimum.min(distance_squared)),
            );
        }

        let tolerance = duplicate_coordinate_tolerance_from_references(
            coords,
            self.tds
                .vertices()
                .map(|(_, vertex)| vertex.point().coords()),
        );
        let tolerance_squared = tolerance * tolerance;
        minimum_distance_squared
            .is_some_and(|distance_squared| {
                if tolerance_squared.is_finite() {
                    distance_squared <= tolerance_squared
                } else {
                    distance_squared.sqrt() <= tolerance
                }
            })
            .then(|| InsertionError::DuplicateCoordinates {
                coordinates: CoordinateValues::from_numeric_slice(coords),
            })
    }

    /// Inserts one bootstrap vertex without constructing a higher-layer owner.
    fn insert_vertex(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), DelaunayIncrementalBuilderError> {
        let coords = *vertex.point().coords();
        if let Some(error) = self.duplicate_coordinates_error(&coords) {
            return Ok((
                InsertionOutcome::Skipped { error },
                InsertionStatistics {
                    attempts: 1,
                    simplices_removed_during_repair: 0,
                    result: InsertionResult::SkippedDuplicate,
                },
            ));
        }

        let vertex_key = self
            .tds
            .insert_vertex(vertex)
            .map_err(InsertionError::from)?;

        let hint = if self.vertex_keys.len() + 1 == D + 1 {
            Some(
                self.tds
                    .insert_simplex(
                        self.vertex_keys
                            .iter()
                            .copied()
                            .chain(std::iter::once(vertex_key)),
                    )
                    .map_err(
                        |source| DelaunayIncrementalBuilderError::BootstrapAssembly { source },
                    )?,
            )
        } else {
            None
        };

        // Publish auxiliary state only after every fallible TDS mutation has
        // succeeded. The caller-owned TDS savepoint handles a rejected first
        // simplex without requiring a workspace clone.
        self.vertex_keys.push(vertex_key);
        if let Some(index) = self.spatial_index.as_mut() {
            index.insert_vertex(vertex_key, &coords);
        }

        Ok((
            InsertionOutcome::Inserted { vertex_key, hint },
            InsertionStatistics {
                attempts: 1,
                simplices_removed_during_repair: 0,
                result: InsertionResult::Inserted,
            },
        ))
    }
}

impl<K, U, V, const D: usize> DelaunayBootstrapWorkspace<K, U, V, D>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Consumes the workspace, publishing Levels 1–4 before crossing the Level 5 draft boundary.
    fn try_into_owner(
        self,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayIncrementalBuilderError> {
        let tds = self
            .tds
            .finish()
            .map_err(|source| DelaunayIncrementalBuilderError::TdsPublication { source })?;
        let triangulation = TriangulationDraft::with_topology_context(
            tds,
            self.kernel,
            self.topology_guarantee,
            self.global_topology,
        )
        .construction_provenance(TopologyConstructionProvenance::EuclideanDelaunayInsertion)
        .validation_policy(self.validation_policy)
        .finish_strict()
        .map_err(
            |failure| DelaunayIncrementalBuilderError::TriangulationPublication {
                source: failure.into_reason(),
            },
        )?;
        DelaunayTriangulationDraft::from_parts(
            triangulation,
            self.insertion_state,
            self.spatial_index,
            self.euclidean_report_domain,
        )
        .try_into_delaunay()
        .map_err(
            |failure| DelaunayIncrementalBuilderError::DelaunayCertification {
                source: failure.into_reason(),
            },
        )
    }

    /// Publishes the first simplex while carrying its TDS journal through every proof layer.
    fn try_into_owner_after_insertion(
        mut self,
        savepoint: TdsRollbackSavepoint,
        inserted_vertex_key: VertexKey,
        inserted_coords: [f64; D],
        insertion_state_before: DelaunayInsertionState,
    ) -> Result<
        DelaunayTriangulation<K, U, V, D>,
        RefinementError<Self, DelaunayIncrementalBuilderError>,
    > {
        let tds_draft = std::mem::take(&mut self.tds);
        let tds = match tds_draft.finish_recoverable() {
            Ok(tds) => tds,
            Err(failure) => {
                let (mut tds_draft, source) = failure.into_parts();
                tds_draft.rollback_savepoint(savepoint);
                self.tds = tds_draft;
                self.restore_auxiliary_after_failed_insertion(
                    inserted_vertex_key,
                    &inserted_coords,
                    insertion_state_before,
                );
                return Err(RefinementError::new(
                    self,
                    DelaunayIncrementalBuilderError::TdsPublication { source },
                ));
            }
        };

        let triangulation = match TriangulationDraft::with_topology_context(
            tds,
            self.kernel.clone(),
            self.topology_guarantee,
            self.global_topology,
        )
        .construction_provenance(TopologyConstructionProvenance::EuclideanDelaunayInsertion)
        .validation_policy(self.validation_policy)
        .finish_canonicalizing_in_transaction()
        {
            Ok(triangulation) => triangulation,
            Err(failure) => {
                let (mut tds, source) = failure.into_parts();
                tds.rollback_savepoint(savepoint);
                self.tds = TdsDraft::from_rolled_back_storage(tds);
                self.restore_auxiliary_after_failed_insertion(
                    inserted_vertex_key,
                    &inserted_coords,
                    insertion_state_before,
                );
                return Err(RefinementError::new(
                    self,
                    DelaunayIncrementalBuilderError::TriangulationPublication { source },
                ));
            }
        };

        let candidate = DelaunayTriangulationDraft::from_parts(
            triangulation,
            self.insertion_state,
            self.spatial_index.clone(),
            self.euclidean_report_domain,
        );
        match candidate.try_into_delaunay() {
            Ok(mut triangulation) => {
                triangulation.commit_tds_savepoint(savepoint);
                Ok(triangulation)
            }
            Err(failure) => {
                let (triangulation, source) = failure.into_parts();
                let mut tds = triangulation.into_tds();
                tds.rollback_savepoint(savepoint);
                self.tds = TdsDraft::from_rolled_back_storage(tds);
                self.restore_auxiliary_after_failed_insertion(
                    inserted_vertex_key,
                    &inserted_coords,
                    insertion_state_before,
                );
                Err(RefinementError::new(
                    self,
                    DelaunayIncrementalBuilderError::DelaunayCertification { source },
                ))
            }
        }
    }

    /// Restores non-TDS fields after the TDS savepoint has removed the candidate vertex.
    fn restore_auxiliary_after_failed_insertion(
        &mut self,
        inserted_vertex_key: VertexKey,
        inserted_coords: &[f64; D],
        insertion_state_before: DelaunayInsertionState,
    ) {
        assert_eq!(
            self.vertex_keys.pop(),
            Some(inserted_vertex_key),
            "bootstrap publication must roll back its most recent vertex"
        );
        if let Some(index) = self.spatial_index.as_mut() {
            index.remove_vertex(&inserted_vertex_key, inserted_coords);
        }
        self.insertion_state = insertion_state_before;
    }
}

/// Internal builder state split between incomplete bootstrap and a Levels 1–5 owner.
#[derive(Clone, Debug)]
enum DelaunayIncrementalBuilderState<K, U, V, const D: usize> {
    Bootstrap(DelaunayBootstrapWorkspace<K, U, V, D>),
    Owner(DelaunayTriangulation<K, U, V, D>),
}

/// Public stateful workflow for incremental [`DelaunayTriangulation`] construction.
///
/// The builder owns bootstrap vertices before they span a full-dimensional
/// simplex. The insertion that first creates one failure-atomically establishes
/// Levels 3–5 and changes the private state to an owner; a
/// failed transition restores the exact pre-insertion bootstrap. Later
/// insertions use that owner's transactional mutation path.
/// [`finish`](Self::finish) is the only operation that returns the Levels 1–5
/// owner to the caller, after a cumulative final audit.
///
/// # Lifecycle
///
/// - An empty builder can be finished as the valid empty complex.
/// - A nonempty builder remains unpublished until its vertices span a maximal
///   simplex.
/// - The first maximal simplex crosses Levels 1–5 failure-atomically.
/// - Later insertions use the transactional mutation path of the private
///   proof-bearing owner.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayIncrementalBuilder, DelaunayResult, vertex,
/// };
///
/// # fn main() -> DelaunayResult<()> {
/// let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
///     DelaunayIncrementalBuilder::new();
/// for vertex in [
///     vertex![0.0, 0.0]?,
///     vertex![1.0, 0.0]?,
///     vertex![0.0, 1.0]?,
/// ] {
///     builder.insert_vertex(vertex)?;
/// }
///
/// let triangulation = builder.finish()?;
/// assert_eq!(triangulation.number_of_vertices(), 3);
/// triangulation.validate()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct DelaunayIncrementalBuilder<K, U, V, const D: usize> {
    state: DelaunayIncrementalBuilderState<K, U, V, D>,
}

impl<const D: usize> DelaunayIncrementalBuilder<AdaptiveKernel<f64>, (), (), D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    /// Starts an empty incremental builder using the default adaptive kernel and topology.
    #[must_use]
    pub fn new() -> Self {
        Self::with_kernel(AdaptiveKernel::new())
    }

    /// Starts an empty incremental builder with an explicit topology guarantee.
    #[must_use]
    pub fn with_topology_guarantee(topology_guarantee: TopologyGuarantee) -> Self {
        Self::with_kernel_and_topology_guarantee(AdaptiveKernel::new(), topology_guarantee)
    }
}

impl<const D: usize> Default for DelaunayIncrementalBuilder<AdaptiveKernel<f64>, (), (), D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, U, V, const D: usize> DelaunayIncrementalBuilder<K, U, V, D>
where
    K: ExactPredicates<D, Scalar = f64>,
{
    /// Starts an empty incremental builder with a caller-selected kernel.
    #[must_use]
    pub fn with_kernel(kernel: K) -> Self {
        Self::with_kernel_and_topology_context(
            kernel,
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
    }

    /// Starts an empty incremental builder with a kernel and topology guarantee.
    #[must_use]
    pub fn with_kernel_and_topology_guarantee(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        Self::with_kernel_and_topology_context(kernel, topology_guarantee, GlobalTopology::DEFAULT)
    }

    /// Starts an empty incremental builder with complete topology context.
    #[must_use]
    pub(crate) fn with_kernel_and_topology_context(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            state: DelaunayIncrementalBuilderState::Bootstrap(
                DelaunayBootstrapWorkspace::with_topology_context(
                    kernel,
                    topology_guarantee,
                    global_topology,
                ),
            ),
        }
    }
}

impl<K, U, V, const D: usize> DelaunayIncrementalBuilder<K, U, V, D> {
    /// Returns the number of vertices currently accumulated by the builder.
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.number_of_vertices(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.number_of_vertices()
            }
        }
    }

    /// Returns the number of maximal simplices currently accumulated by the builder.
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.number_of_simplices(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.number_of_simplices()
            }
        }
    }

    /// Returns the current dimensionality reported by the builder state.
    #[must_use]
    pub fn dim(&self) -> i32 {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.dim(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => triangulation.dim(),
        }
    }

    /// Returns the topology guarantee selected for eventual publication.
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.topology_guarantee,
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.topology_guarantee()
            }
        }
    }

    /// Returns the global topology selected for eventual publication.
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.global_topology,
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.global_topology()
            }
        }
    }

    /// Returns the high-level topology kind selected for the builder.
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.global_topology.kind(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => triangulation.topology_kind(),
        }
    }

    /// Returns the insertion-time validation policy selected for the builder.
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.validation_policy,
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.validation_policy()
            }
        }
    }

    /// Changes the builder's insertion-time validation policy when compatible.
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
            DelaunayIncrementalBuilderState::Bootstrap(state) => {
                if state.topology_guarantee.is_compatible_with_policy(policy) {
                    state.validation_policy = policy;
                    Ok(())
                } else {
                    Err(
                        ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                            topology_guarantee: state.topology_guarantee,
                            validation_policy: policy,
                        },
                    )
                }
            }
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.try_set_validation_policy(policy)
            }
        }
    }

    /// Returns the Level 5 checking cadence used by the builder.
    #[must_use]
    pub const fn delaunay_check_policy(&self) -> DelaunayCheckPolicy {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => {
                state.insertion_state.delaunay_check_policy
            }
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.delaunay_check_policy()
            }
        }
    }

    /// Sets the Level 5 checking cadence used by the builder.
    pub const fn set_delaunay_check_policy(&mut self, policy: DelaunayCheckPolicy) {
        match &mut self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => {
                state.insertion_state.delaunay_check_policy = policy;
            }
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.set_delaunay_check_policy(policy);
            }
        }
    }

    /// Returns whether the builder's currently assembled simplices are coherently oriented.
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.is_coherently_oriented(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.is_coherently_oriented()
            }
        }
    }

    /// Validates the currently assembled Levels 1–2 state without publishing it.
    ///
    /// # Errors
    ///
    /// Returns the first structural error in the builder state. A successful result does
    /// not imply that enough vertices exist to publish a full-dimensional owner.
    pub fn validate_structure(&self) -> Result<(), TdsError> {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.validate_structure(),
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.validate_structure()
            }
        }
    }

    /// Returns the cumulative Levels 1–3 report for the published owner state.
    ///
    /// Before the first maximal simplex is published, the builder contains only
    /// a bootstrap workspace and therefore has no owner-level topology to audit.
    /// Call [`validate_structure`](Self::validate_structure) independently when
    /// checking that partial state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayIncrementalBuilder, DelaunayIncrementalBuilderError,
    /// };
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Incremental(#[from] DelaunayIncrementalBuilderError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let mut builder = DelaunayIncrementalBuilder::<_, (), (), 2>::new();
    /// assert!(builder.owner_topology_report().is_none());
    ///
    /// for vertex in [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ] {
    ///     let _ = builder.insert_vertex(vertex)?;
    /// }
    /// std::assert_matches!(builder.owner_topology_report(), Some(Ok(())));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn owner_topology_report(&self) -> Option<Result<(), TriangulationValidationReport>>
    where
        K: ExactPredicates<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(_) => None,
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                Some(triangulation.as_triangulation().validation_report())
            }
        }
    }

    /// Returns a staged vertex by key without exposing the unpublished owner.
    #[must_use]
    pub fn vertex(&self, key: VertexKey) -> Option<&Vertex<U, D>> {
        match &self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => state.tds.vertex(key),
            DelaunayIncrementalBuilderState::Owner(triangulation) => triangulation.vertex(key),
        }
    }
}

impl<K, U, V, const D: usize> DelaunayIncrementalBuilder<K, U, V, D>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Incrementally inserts a vertex through the builder.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`Insertion`](DelaunayIncrementalBuilderError::Insertion) when the
    ///   vertex is rejected, including a retryable skip that this strict
    ///   terminal promotes to an error;
    /// - [`BootstrapAssembly`](DelaunayIncrementalBuilderError::BootstrapAssembly)
    ///   when the first maximal simplex cannot be staged;
    /// - [`TdsPublication`](DelaunayIncrementalBuilderError::TdsPublication),
    ///   [`TriangulationPublication`](DelaunayIncrementalBuilderError::TriangulationPublication),
    ///   or [`DelaunayCertification`](DelaunayIncrementalBuilderError::DelaunayCertification)
    ///   when that first simplex cannot cross its Levels 1–5 publication
    ///   boundaries.
    ///
    /// Publication failures restore the pre-insertion bootstrap, leaving the
    /// builder reusable.
    pub fn insert_vertex(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<VertexKey, DelaunayIncrementalBuilderError> {
        match self.insert_best_effort_with_statistics(vertex)? {
            (InsertionOutcome::Inserted { vertex_key, .. }, _) => Ok(vertex_key),
            (InsertionOutcome::Skipped { error }, _) => Err(error.into()),
        }
    }

    /// Inserts a vertex and returns the public insertion outcome and statistics.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`Insertion`](DelaunayIncrementalBuilderError::Insertion) when the
    ///   vertex is rejected, including a retryable skip that this strict
    ///   terminal promotes to an error;
    /// - [`BootstrapAssembly`](DelaunayIncrementalBuilderError::BootstrapAssembly)
    ///   when the first maximal simplex cannot be staged;
    /// - [`TdsPublication`](DelaunayIncrementalBuilderError::TdsPublication),
    ///   [`TriangulationPublication`](DelaunayIncrementalBuilderError::TriangulationPublication),
    ///   or [`DelaunayCertification`](DelaunayIncrementalBuilderError::DelaunayCertification)
    ///   when the first maximal simplex cannot be published through Levels
    ///   1–5.
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), DelaunayIncrementalBuilderError> {
        match self.insert_best_effort_with_statistics(vertex)? {
            (outcome @ InsertionOutcome::Inserted { .. }, statistics) => Ok((outcome, statistics)),
            (InsertionOutcome::Skipped { error }, _) => Err(error.into()),
        }
    }

    /// Inserts a vertex and reports retryable skips in the returned outcome.
    ///
    /// # Errors
    ///
    /// Retryable insertion failures are returned as
    /// [`InsertionOutcome::Skipped`]. This method returns:
    ///
    /// - [`Insertion`](DelaunayIncrementalBuilderError::Insertion) for a
    ///   non-retryable insertion failure;
    /// - [`BootstrapAssembly`](DelaunayIncrementalBuilderError::BootstrapAssembly)
    ///   when the first maximal simplex cannot be staged;
    /// - [`TdsPublication`](DelaunayIncrementalBuilderError::TdsPublication),
    ///   [`TriangulationPublication`](DelaunayIncrementalBuilderError::TriangulationPublication),
    ///   or [`DelaunayCertification`](DelaunayIncrementalBuilderError::DelaunayCertification)
    ///   when the first maximal simplex cannot be published through Levels
    ///   1–5.
    pub fn insert_best_effort_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), DelaunayIncrementalBuilderError> {
        let mut publication = None;
        let result = match &mut self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => {
                let publication_savepoint = state.tds.begin_rollback_savepoint();
                let inserted_coords = *vertex.point().coords();
                let insertion_state_before = state.insertion_state;
                let (outcome, statistics) = match state.insert_vertex(vertex) {
                    Ok(result) => result,
                    Err(error) => {
                        state.tds.rollback_savepoint(publication_savepoint);
                        return Err(error);
                    }
                };
                if let InsertionOutcome::Inserted { vertex_key, hint } = outcome {
                    state.insertion_state.last_inserted_simplex = hint;
                    state.insertion_state.delaunay_repair_insertion_count = state
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    if state.is_publishable() {
                        let replacement = DelaunayBootstrapWorkspace::with_topology_context(
                            state.kernel.clone(),
                            state.topology_guarantee,
                            state.global_topology,
                        );
                        let candidate = std::mem::replace(state, replacement);
                        match candidate.try_into_owner_after_insertion(
                            publication_savepoint,
                            vertex_key,
                            inserted_coords,
                            insertion_state_before,
                        ) {
                            Ok(triangulation) => publication = Some(triangulation),
                            Err(failure) => {
                                let (restored, error) = failure.into_parts();
                                *state = restored;
                                return Err(error);
                            }
                        }
                    } else {
                        state.tds.commit_savepoint(publication_savepoint);
                    }
                } else {
                    state.tds.commit_savepoint(publication_savepoint);
                }
                (outcome, statistics)
            }
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                return triangulation
                    .insert_best_effort_with_statistics(vertex)
                    .map_err(Into::into);
            }
        };

        if let Some(triangulation) = publication {
            self.state = DelaunayIncrementalBuilderState::Owner(triangulation);
        }

        Ok(result)
    }

    /// Consumes the builder and returns its owner only after Levels 1–5 validation.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`IncompleteBootstrap`](DelaunayIncrementalBuilderError::IncompleteBootstrap)
    /// when the builder contains between one and `D` vertices and therefore no
    /// maximal simplex. An empty builder is complete and publishes the valid
    /// empty complex.
    ///
    /// Empty-bootstrap publication can also return
    /// [`TdsPublication`](DelaunayIncrementalBuilderError::TdsPublication),
    /// [`TriangulationPublication`](DelaunayIncrementalBuilderError::TriangulationPublication),
    /// or [`DelaunayCertification`](DelaunayIncrementalBuilderError::DelaunayCertification)
    /// if the corresponding proof boundary rejects the empty complex. A
    /// builder that already owns a triangulation returns
    /// [`FinalValidation`](DelaunayIncrementalBuilderError::FinalValidation)
    /// if its cumulative Levels 1–5 audit fails.
    pub fn finish(
        self,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayIncrementalBuilderError> {
        match self.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => {
                let vertex_count = state.tds.number_of_vertices();
                if vertex_count != 0 {
                    return Err(DelaunayIncrementalBuilderError::IncompleteBootstrap {
                        dimension: D,
                        vertex_count,
                    });
                }
                state.try_into_owner()
            }
            DelaunayIncrementalBuilderState::Owner(triangulation) => {
                triangulation.validate().map_err(|source| {
                    DelaunayIncrementalBuilderError::FinalValidation { source }
                })?;
                Ok(triangulation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::core::simplex::SimplexValidationError;
    use crate::geometry::kernel::RobustKernel;
    use crate::vertex;

    macro_rules! test_incremental_builder {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<incremental_builder_bootstraps_and_continues_in_ $dim d>]() {
                    let mut builder: DelaunayIncrementalBuilder<_, (), (), $dim> =
                        DelaunayIncrementalBuilder::new();
                    let initial_vertices = vec![$(vertex!($simplex_coords).unwrap()),+];
                    assert_eq!(initial_vertices.len(), $dim + 1);

                    for (index, vertex) in initial_vertices.iter().take($dim).enumerate() {
                        builder.insert_vertex(*vertex).unwrap();
                        assert_eq!(builder.number_of_vertices(), index + 1);
                        assert_eq!(builder.number_of_simplices(), 0);
                        assert!(matches!(
                            builder.state,
                            DelaunayIncrementalBuilderState::Bootstrap(_)
                        ));
                    }

                    builder.insert_vertex(*initial_vertices.last().unwrap()).unwrap();
                    assert_eq!(builder.number_of_vertices(), $dim + 1);
                    assert_eq!(builder.number_of_simplices(), 1);
                    assert!(matches!(
                        builder.state,
                        DelaunayIncrementalBuilderState::Owner(_)
                    ));

                    builder.insert_vertex(vertex!($interior_point).unwrap()).unwrap();
                    let triangulation = builder.finish().unwrap();
                    assert_eq!(triangulation.number_of_vertices(), $dim + 2);
                    assert!(triangulation.number_of_simplices() > 1);
                    assert!(triangulation.validate().is_ok());
                }
            }
        };
    }

    test_incremental_builder!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.2, 0.2]);

    test_incremental_builder!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2]
    );

    test_incremental_builder!(
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

    test_incremental_builder!(
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
    fn builder_bootstraps_with_custom_kernel() {
        let mut builder: DelaunayIncrementalBuilder<RobustKernel<f64>, (), (), 3> =
            DelaunayIncrementalBuilder::with_kernel(RobustKernel::new());

        builder
            .insert_vertex(vertex![0.0, 0.0, 0.0].unwrap())
            .unwrap();
        builder
            .insert_vertex(vertex![1.0, 0.0, 0.0].unwrap())
            .unwrap();
        builder
            .insert_vertex(vertex![0.0, 1.0, 0.0].unwrap())
            .unwrap();
        assert_eq!(builder.number_of_simplices(), 0);

        builder
            .insert_vertex(vertex![0.0, 0.0, 1.0].unwrap())
            .unwrap();
        let triangulation = builder.finish().unwrap();
        assert_eq!(triangulation.number_of_simplices(), 1);
        assert!(triangulation.validate().is_ok());
    }

    #[test]
    fn first_publication_preserves_all_bootstrap_vertex_keys() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::new();
        let keys = [
            builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap(),
            builder.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap(),
            builder.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap(),
        ];

        assert!(matches!(
            builder.state,
            DelaunayIncrementalBuilderState::Owner(_)
        ));
        assert!(keys.into_iter().all(|key| builder.vertex(key).is_some()));
    }

    #[test]
    fn empty_builder_publishes_as_the_verified_empty_delaunay_triangulation() {
        let builder: DelaunayIncrementalBuilder<_, (), (), 2> = DelaunayIncrementalBuilder::new();

        let triangulation = builder.finish().unwrap();
        assert_eq!(triangulation.number_of_vertices(), 0);
        assert!(triangulation.validate().is_ok());
    }

    #[test]
    fn bootstrap_configuration_and_structure_are_observable_before_publication() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::default();

        assert_eq!(builder.number_of_vertices(), 0);
        assert_eq!(builder.number_of_simplices(), 0);
        assert_eq!(builder.dim(), -1);
        assert_eq!(builder.topology_guarantee(), TopologyGuarantee::PLManifold);
        assert_eq!(builder.global_topology(), GlobalTopology::Euclidean);
        assert_eq!(builder.topology_kind(), TopologyKind::Euclidean);
        assert_eq!(builder.validation_policy(), ValidationPolicy::ExplicitOnly);
        assert_eq!(
            builder.delaunay_check_policy(),
            DelaunayCheckPolicy::EndOnly
        );
        assert!(builder.is_coherently_oriented());
        assert!(builder.validate_structure().is_ok());

        builder
            .try_set_validation_policy(ValidationPolicy::Always)
            .unwrap();
        let every_two = NonZeroUsize::new(2).unwrap();
        builder.set_delaunay_check_policy(DelaunayCheckPolicy::EveryN(every_two));
        assert_eq!(builder.validation_policy(), ValidationPolicy::Always);
        assert_eq!(
            builder.delaunay_check_policy(),
            DelaunayCheckPolicy::EveryN(every_two)
        );

        let vertex_key = builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        assert!(builder.vertex(vertex_key).is_some());
        assert_eq!(builder.number_of_vertices(), 1);
    }

    #[test]
    fn bootstrap_rejects_an_incompatible_validation_policy_without_mutation() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);
        let policy_before = builder.validation_policy();

        let error = builder
            .try_set_validation_policy(ValidationPolicy::Never)
            .unwrap_err();

        assert_eq!(
            error,
            ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                topology_guarantee: TopologyGuarantee::PLManifold,
                validation_policy: ValidationPolicy::Never,
            }
        );
        assert_eq!(builder.validation_policy(), policy_before);
    }

    #[test]
    fn bootstrap_duplicate_check_handles_an_overflowing_squared_tolerance() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::new();
        builder
            .insert_vertex(vertex![1.0e308, -1.0e308].unwrap())
            .unwrap();

        let (outcome, statistics) = builder
            .insert_best_effort_with_statistics(vertex![1.0e308, -1.0e308].unwrap())
            .unwrap();

        assert!(matches!(
            outcome,
            InsertionOutcome::Skipped {
                error: InsertionError::DuplicateCoordinates { .. }
            }
        ));
        assert_eq!(statistics.result, InsertionResult::SkippedDuplicate);
        assert_eq!(builder.number_of_vertices(), 1);
        assert_eq!(builder.number_of_simplices(), 0);
    }

    #[test]
    fn verified_builder_forwards_configuration_and_survives_a_skipped_insertion() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::new();
        let first_key = builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        builder.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();
        builder.insert_vertex(vertex![0.0, 1.0].unwrap()).unwrap();
        assert!(matches!(
            builder.state,
            DelaunayIncrementalBuilderState::Owner(_)
        ));

        builder
            .try_set_validation_policy(ValidationPolicy::Always)
            .unwrap();
        let every_two = NonZeroUsize::new(2).unwrap();
        builder.set_delaunay_check_policy(DelaunayCheckPolicy::EveryN(every_two));

        assert_eq!(builder.number_of_vertices(), 3);
        assert_eq!(builder.number_of_simplices(), 1);
        assert_eq!(builder.dim(), 2);
        assert_eq!(builder.topology_guarantee(), TopologyGuarantee::PLManifold);
        assert_eq!(builder.global_topology(), GlobalTopology::Euclidean);
        assert_eq!(builder.topology_kind(), TopologyKind::Euclidean);
        assert_eq!(builder.validation_policy(), ValidationPolicy::Always);
        assert_eq!(
            builder.delaunay_check_policy(),
            DelaunayCheckPolicy::EveryN(every_two)
        );
        assert!(builder.vertex(first_key).is_some());
        assert!(builder.is_coherently_oriented());
        assert!(builder.validate_structure().is_ok());

        let error = builder
            .insert_vertex(vertex![0.0, 0.0].unwrap())
            .unwrap_err();
        assert!(matches!(
            error,
            DelaunayIncrementalBuilderError::Insertion { source }
                if matches!(source.as_ref(), InsertionError::DuplicateCoordinates { .. })
        ));
        assert_eq!(builder.number_of_vertices(), 3);
        assert!(builder.vertex(first_key).is_some());

        let triangulation = builder.finish().unwrap();
        assert_eq!(triangulation.validation_policy(), ValidationPolicy::Always);
        assert_eq!(
            triangulation.delaunay_check_policy(),
            DelaunayCheckPolicy::EveryN(every_two)
        );
        assert!(triangulation.validate().is_ok());
    }

    #[test]
    fn partial_bootstrap_builder_cannot_publish() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::new();
        builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();

        assert_eq!(
            builder.finish().unwrap_err(),
            DelaunayIncrementalBuilderError::IncompleteBootstrap {
                dimension: 2,
                vertex_count: 1,
            }
        );
    }

    #[test]
    fn failed_first_publication_restores_the_exact_bootstrap_owner() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::with_kernel_and_topology_context(
                AdaptiveKernel::new(),
                TopologyGuarantee::PLManifold,
                GlobalTopology::Euclidean,
            );
        let first_key = builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        let second_key = builder.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();

        let DelaunayIncrementalBuilderState::Bootstrap(state) = &mut builder.state else {
            panic!("two 2D vertices cannot publish");
        };
        state.validation_policy = ValidationPolicy::Never;

        let (owner_before, generation_before, insertion_count_before) = match &builder.state {
            DelaunayIncrementalBuilderState::Bootstrap(state) => (
                state.tds.topology_owner_id(),
                state.tds.topology_generation(),
                state.insertion_state.delaunay_repair_insertion_count,
            ),
            DelaunayIncrementalBuilderState::Owner(_) => {
                panic!("two 2D vertices cannot publish")
            }
        };

        let failure = builder
            .insert_vertex(vertex![0.0, 1.0].unwrap())
            .unwrap_err();
        assert!(
            matches!(
                &failure,
                DelaunayIncrementalBuilderError::TriangulationPublication { .. }
            ),
            "unexpected publication failure: {failure:?}"
        );

        let DelaunayIncrementalBuilderState::Bootstrap(state) = &builder.state else {
            panic!("failed publication must restore bootstrap state");
        };
        assert_eq!(state.tds.topology_owner_id(), owner_before);
        assert_eq!(state.tds.topology_generation(), generation_before);
        assert_eq!(
            state.insertion_state.delaunay_repair_insertion_count,
            insertion_count_before
        );
        assert_eq!(state.tds.number_of_vertices(), 2);
        assert_eq!(state.tds.number_of_simplices(), 0);
        assert!(state.tds.vertex(first_key).is_some());
        assert!(state.tds.vertex(second_key).is_some());

        let retry = builder
            .insert_vertex(vertex![0.0, 1.0].unwrap())
            .unwrap_err();
        assert!(
            matches!(
                &retry,
                DelaunayIncrementalBuilderError::TriangulationPublication { .. }
            ),
            "unexpected retry failure: {retry:?}"
        );
    }

    #[test]
    fn failed_bootstrap_assembly_restores_the_exact_preinsertion_workspace() {
        let mut builder: DelaunayIncrementalBuilder<_, (), (), 2> =
            DelaunayIncrementalBuilder::new();
        let first_key = builder.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
        let second_key = builder.insert_vertex(vertex![1.0, 0.0].unwrap()).unwrap();

        let DelaunayIncrementalBuilderState::Bootstrap(state) = &mut builder.state else {
            panic!("two 2D vertices cannot publish");
        };
        state.vertex_keys[1] = first_key;
        let owner_before = state.tds.topology_owner_id();
        let generation_before = state.tds.topology_generation();
        let vertex_keys_before = state.vertex_keys.clone();

        let failure = builder
            .insert_vertex(vertex![0.0, 1.0].unwrap())
            .unwrap_err();
        assert!(matches!(
            failure,
            DelaunayIncrementalBuilderError::BootstrapAssembly {
                source: TdsDraftInsertionError::SimplexCreation {
                    source: SimplexValidationError::DuplicateVertices,
                },
            }
        ));

        let DelaunayIncrementalBuilderState::Bootstrap(state) = &builder.state else {
            panic!("failed assembly must restore bootstrap state");
        };
        assert_eq!(state.tds.topology_owner_id(), owner_before);
        assert_eq!(state.tds.topology_generation(), generation_before);
        assert_eq!(state.vertex_keys, vertex_keys_before);
        assert_eq!(state.tds.number_of_vertices(), 2);
        assert_eq!(state.tds.number_of_simplices(), 0);
        assert!(state.tds.vertex(first_key).is_some());
        assert!(state.tds.vertex(second_key).is_some());
    }
}
