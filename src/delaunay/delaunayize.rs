//! End-to-end Delaunay refinement workflow.
//!
//! [`DelaunayRefinementBuilder`] is the single consuming transition from a
//! Levels 1–4 [`Triangulation`] to a Levels 1–5
//! [`DelaunayTriangulation`]. The default strict terminal checks only Level 5;
//! [`repair_by_flips`](DelaunayRefinementBuilder::repair_by_flips) selects
//! bounded transactional repair before certification.
//!
//! # Workflow
//!
//! 1. **Levels 1–4 proof consumption** — accepts the proof-bearing
//!    `Triangulation` without revalidating invariants encoded by its type.
//! 2. **Optional Delaunay flip repair** — when selected, runs transactional
//!    k=2/k=3 bistellar flips that preserve Levels 1–4 while restoring the
//!    empty-circumsphere property.
//! 3. **Optional fallback rebuild** — if configured, rebuilds the triangulation
//!    from its vertex set after Delaunay repair fails.
//! 4. **Level 5 certification** — publishes `DelaunayTriangulation` only after
//!    the refinement predicate succeeds.
//!
//! The flip stage follows the regular-triangulation framework of Edelsbrunner
//! and Shah \[4]. This API deliberately exposes a bounded, fallible conversion:
//! the cited result motivates incremental topological flipping, while rollback,
//! typed non-convergence, and final certification define this implementation's
//! supported contract. See `REFERENCES.md` for the numbered bibliography.
//!
//! # Example
//!
//! ```rust
//! use delaunay::prelude::delaunayize::*;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Delaunayize(#[from] delaunay::prelude::delaunayize::DelaunayizeError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! let vertices = vec![
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//! let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
//!
//! let result = DelaunayRefinementBuilder::new(tri)
//!     .repair_by_flips()
//!     .build()
//!     .map_err(RefinementError::into_reason)?;
//! assert!(result.triangulation.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
#![forbid(unsafe_code)]

// Re-export outcome/error field types so users can name the public contract
// without reaching into lower-level modules.
pub use crate::construction::DelaunayTriangulationConstructionError;
pub use crate::flips::{
    DelaunayRepairError, DelaunayRepairHeuristicRebuildFailure,
    DelaunayRepairHeuristicRebuildFailureKind, DelaunayRepairHeuristicVertexContext,
    DelaunayRepairOrientationCanonicalizationFailure,
    DelaunayRepairOrientationCanonicalizationFailureKind, DelaunayRepairPostconditionFailure,
    DelaunayRepairStats,
};
pub use crate::tds::SimplexValidationError;

use crate::builder::DelaunayTriangulationBuilder;
use crate::core::algorithms::flips::repair_delaunay_with_flips_k2_k3_run_in_transaction;
use crate::core::collections::{Entry, FastHashMap, SimplexVertexUuidBuffer};
use crate::core::operations::TopologicalOperation;
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, TdsMutationError};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::delaunay_model::DelaunayTriangulation;
use crate::draft::DelaunayTriangulationDraft;
use crate::geometry::kernel::ExactPredicates;
use crate::refinement::RefinementError;
use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
use crate::triangulation::Triangulation;
use crate::triangulation::rollback::TriangulationRollbackTransaction;
use crate::triangulation::validation::TopologyGuarantee;
use crate::validation::{
    DelaunayLevelFiveCertificate, DelaunayTriangulationValidationError,
    certify_level_five_for_refinement,
};
use thiserror::Error;

/// Type state for strict, non-mutating Level 5 certification.
#[derive(Clone, Copy, Debug, Default)]
#[doc(hidden)]
pub struct StrictDelaunayRefinement;

/// Type state for bounded flip repair before Level 5 certification.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct FlipRepairDelaunayRefinement {
    config: DelaunayizeConfig,
}

/// Fluent promotion from a Levels 1–4 [`Triangulation`] to a Levels 1–5
/// [`DelaunayTriangulation`].
///
/// [`new`](Self::new) defaults to strict publication, which checks only Level
/// 5, performs no mutation, and allocates no rollback snapshot.
/// [`repair_by_flips`](Self::repair_by_flips) explicitly selects bounded
/// transactional repair and enables its repair-only options.
///
/// Deserialized Levels 1–2 storage can be restored incrementally through the
/// same public proof boundaries:
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayResult, DelaunayTriangulationBuilder, GlobalTopology,
///     TopologyGuarantee,
/// };
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::prelude::triangulation::TriangulationBuilder;
/// use delaunay::DelaunayRefinementBuilder;
///
/// fn main() -> DelaunayResult<()> {
///     let vertices = [
///         delaunay::vertex![0.0, 0.0]?,
///         delaunay::vertex![1.0, 0.0]?,
///         delaunay::vertex![0.0, 1.0]?,
///     ];
///     let tds = DelaunayTriangulationBuilder::new(&vertices)
///         .build_triangulation()?
///         .into_tds();
///     let triangulation = TriangulationBuilder::new(tds, FastKernel::new())
///         .topology_guarantee(TopologyGuarantee::PLManifold)
///         .global_topology(GlobalTopology::Euclidean)
///         .build()
///         .expect("Levels 1-2 storage should satisfy Levels 3-4");
///     let reconstructed = DelaunayRefinementBuilder::new(triangulation)
///         .build()
///         .expect("fixture should satisfy Level 5");
///
///     assert_eq!(
///         reconstructed.topology_guarantee(),
///         TopologyGuarantee::PLManifold
///     );
///     assert_eq!(reconstructed.global_topology(), GlobalTopology::Euclidean);
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug)]
pub struct DelaunayRefinementBuilder<K, U, V, const D: usize, M = StrictDelaunayRefinement> {
    triangulation: Triangulation<K, U, V, D>,
    mode: M,
}

impl<K, U, V, const D: usize> DelaunayRefinementBuilder<K, U, V, D, StrictDelaunayRefinement> {
    /// Creates a strict, non-mutating Level 5 refinement request.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::delaunayize::DelaunayRefinementBuilder;
    /// use delaunay::prelude::geometry::AdaptiveKernel;
    /// use delaunay::prelude::triangulation::Triangulation;
    /// use delaunay::{DelaunayTriangulation, DelaunayTriangulationRefinementError};
    ///
    /// fn certify(
    ///     triangulation: Triangulation<AdaptiveKernel<f64>, (), (), 2>,
    /// ) -> Result<
    ///     DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>,
    ///     DelaunayTriangulationRefinementError<AdaptiveKernel<f64>, (), (), 2>,
    /// > {
    ///     DelaunayRefinementBuilder::new(triangulation).build()
    /// }
    /// ```
    #[must_use]
    pub const fn new(triangulation: Triangulation<K, U, V, D>) -> Self {
        Self {
            triangulation,
            mode: StrictDelaunayRefinement,
        }
    }

    /// Selects bounded bistellar-flip repair before Level 5 certification.
    #[must_use]
    pub fn repair_by_flips(
        self,
    ) -> DelaunayRefinementBuilder<K, U, V, D, FlipRepairDelaunayRefinement> {
        DelaunayRefinementBuilder {
            triangulation: self.triangulation,
            mode: FlipRepairDelaunayRefinement {
                config: DelaunayizeConfig::default(),
            },
        }
    }
}

impl<K, U, V, const D: usize> DelaunayRefinementBuilder<K, U, V, D, StrictDelaunayRefinement>
where
    K: crate::geometry::kernel::Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Certifies Level 5 without changing the triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationRefinementError`](crate::DelaunayTriangulationRefinementError)
    /// when the Level 5
    /// predicate rejects the candidate. The failure retains the unchanged
    /// Levels 1–4 triangulation.
    pub fn build(
        self,
    ) -> Result<
        DelaunayTriangulation<K, U, V, D>,
        crate::validation::DelaunayTriangulationRefinementError<K, U, V, D>,
    > {
        let Self {
            triangulation,
            mode: _,
        } = self;
        DelaunayTriangulationDraft::from_triangulation(triangulation).try_into_delaunay()
    }
}

impl<K, U, V, const D: usize> DelaunayRefinementBuilder<K, U, V, D, FlipRepairDelaunayRefinement> {
    /// Enables or disables fallback reconstruction after failed flip repair.
    #[must_use]
    pub const fn fallback_rebuild(mut self, fallback_rebuild: bool) -> Self {
        self.mode.config.fallback_rebuild = fallback_rebuild;
        self
    }

    /// Sets the per-attempt bistellar-flip budget cap.
    #[must_use]
    pub const fn max_flips(mut self, max_flips: usize) -> Self {
        self.mode.config.delaunay_max_flips = Some(max_flips);
        self
    }

    /// Clears the explicit flip cap and restores the dimension-dependent bound.
    #[must_use]
    pub const fn default_flip_budget(mut self) -> Self {
        self.mode.config.delaunay_max_flips = None;
        self
    }
}

impl<K, U, V, const D: usize> DelaunayRefinementBuilder<K, U, V, D, FlipRepairDelaunayRefinement>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Repairs and certifies the configured triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayizeRefinementError`] when repair, optional fallback,
    /// or final Level 5 certification fails. Transactional failure returns the
    /// original Levels 1–4 triangulation.
    #[expect(
        clippy::result_large_err,
        reason = "recoverable refinement failures retain the original proof-bearing owner"
    )]
    pub fn build(
        self,
    ) -> Result<DelaunayizeResult<K, U, V, D>, DelaunayizeRefinementError<K, U, V, D>> {
        let Self {
            triangulation,
            mode,
        } = self;
        delaunayize(triangulation, mode.config)
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DelaunayizeConfig {
    /// If `true`, rebuild the triangulation from the vertex set when flip-based
    /// Delaunay repair fails.
    ///
    /// Simplex-level user data (`V`) is restored for rebuilt simplices whose sorted
    /// vertex UUID set matches exactly one original simplex. Simplices that change
    /// during rebuild, have no original payload, or have ambiguous duplicate
    /// original signatures receive `None`.
    fallback_rebuild: bool,
    /// Optional per-attempt flip budget cap for Delaunay repair.
    ///
    /// `None` (default) uses the internal dimension-dependent budget.
    /// Set to `Some(n)` to limit each repair attempt to at most `n` flips.
    delaunay_max_flips: Option<usize>,
}

// =============================================================================
// OUTCOME
// =============================================================================

/// Outcome of successful flip-repair refinement.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::delaunayize::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Delaunayize(#[from] delaunay::prelude::delaunayize::DelaunayizeError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
///
/// let result = DelaunayRefinementBuilder::new(tri)
///     .repair_by_flips()
///     .build()
///     .map_err(RefinementError::into_reason)?;
/// assert!(!result.outcome.used_fallback_rebuild);
/// assert!(result.triangulation.validate().is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelaunayizeOutcome {
    /// Statistics from the flip-based Delaunay repair pass.
    ///
    /// If Delaunay repair fails but fallback rebuild succeeds, these preserve
    /// the counters available from the failed repair attempt. Use
    /// [`used_fallback_rebuild`](Self::used_fallback_rebuild) to distinguish
    /// successful rebuild recovery from direct Delaunay repair success.
    pub delaunay_repair: DelaunayRepairStats,
    /// Whether the fallback vertex-set rebuild was used.
    pub used_fallback_rebuild: bool,
}

/// A Delaunay-certified triangulation and the repair diagnostics that produced it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelaunayizeResult<K, U, V, const D: usize> {
    /// The Levels 1–5 domain value published after successful conversion.
    pub triangulation: DelaunayTriangulation<K, U, V, D>,
    /// Diagnostics from the bounded Delaunay repair workflow.
    pub outcome: DelaunayizeOutcome,
}

// =============================================================================
// ERRORS
// =============================================================================

/// Errors that can occur during flip-repair refinement.
///
/// There are five orthogonal failure modes:
/// - **Topology precondition** rejected an input without the PL-manifold proof
///   required by flip repair.
/// - **Delaunay repair** failed, with optional context about a
///   fallback rebuild attempt.
/// - **Fallback snapshot** failed before a fallback-eligible repair phase.
/// - **Fallback simplex-data recovery** failed while restoring simplex payloads
///   after a repair failure.
/// - **Final validation** rejected the repaired Levels 1–5 candidate.
///
/// # Orthogonality
///
/// The variants are mutually exclusive by failure mode:
/// - Delaunay repair, fallback not attempted -> [`DelaunayRepairFailed`](Self::DelaunayRepairFailed).
/// - Delaunay repair, fallback also failed   -> [`DelaunayRepairFailedWithRebuild`](Self::DelaunayRepairFailedWithRebuild).
/// - Delaunay repair, fallback rebuild succeeded but payload restore failed -> [`DelaunayRepairFailedWithRebuildRestore`](Self::DelaunayRepairFailedWithRebuildRestore).
/// - Fallback was enabled, but a fallback payload snapshot failed before a
///   repair phase could safely continue -> [`FallbackSimplexDataSnapshotFailed`](Self::FallbackSimplexDataSnapshotFailed).
///
/// Fallback payload snapshots are validated before the Delaunay repair result
/// can be accepted. Snapshot failures therefore report
/// [`FallbackSimplexDataSnapshotFailed`](Self::FallbackSimplexDataSnapshotFailed)
/// rather than being deferred into a Delaunay-repair failure variant.
///
/// Variants with secondary fallback failures preserve **both** the primary
/// repair error and the secondary construction, snapshot, or restore error as
/// typed values (no stringification), so consumers can inspect both errors via
/// pattern matching; the primary repair error is exposed via
/// [`Error::source`](std::error::Error::source).
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::delaunayize::*;
///
/// let err = DelaunayizeError::DelaunayRepairFailed {
///     source: DelaunayRepairError::PostconditionFailed {
///         reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
///     },
/// };
/// assert!(err.to_string().contains("Delaunay repair failed"));
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayizeError {
    /// The input is valid under its declared topology guarantee, but that
    /// guarantee is too weak for bistellar flip repair.
    #[error("Delaunay flip repair requires {required:?} topology, found {found:?}")]
    FlipTopologyNotAdmissible {
        /// Minimum topology proof required by the selected operation.
        required: TopologyGuarantee,
        /// Topology proof carried by the input triangulation.
        found: TopologyGuarantee,
    },

    /// Delaunay flip repair failed; no fallback rebuild was attempted
    /// (fallback disabled, or the caller's config did not request one).
    #[error("Delaunay repair failed: {source}")]
    DelaunayRepairFailed {
        /// The underlying flip-repair error.
        #[from]
        #[source]
        source: DelaunayRepairError,
    },

    /// Delaunay flip repair failed **and** the fallback vertex-set rebuild
    /// also failed.  Both errors are preserved as typed values.
    #[error("Delaunay repair failed ({source}); fallback rebuild also failed: {rebuild_error}")]
    DelaunayRepairFailedWithRebuild {
        /// The underlying flip-repair error that triggered the fallback.
        #[source]
        source: DelaunayRepairError,
        /// The construction error from the subsequent vertex-set rebuild attempt.
        rebuild_error: DelaunayTriangulationConstructionError,
    },

    /// Delaunay flip repair failed, the fallback vertex-set rebuild succeeded,
    /// but simplex-payload restoration from the rebuilt topology failed.
    #[error(
        "Delaunay repair failed ({source}); fallback rebuild succeeded but simplex-data restore failed: {restore_error}"
    )]
    DelaunayRepairFailedWithRebuildRestore {
        /// The underlying flip-repair error that triggered the fallback.
        #[source]
        source: DelaunayRepairError,
        /// The simplex-data restoration error from the rebuilt triangulation.
        restore_error: SimplexDataRestoreError,
    },

    /// Fallback rebuild was enabled, but a simplex-payload snapshot could not
    /// be collected before a fallback-eligible repair phase.
    #[error(
        "Fallback simplex-data snapshot failed; fallback rebuild cannot be attempted: {source}"
    )]
    FallbackSimplexDataSnapshotFailed {
        /// The simplex-data snapshot error from the current triangulation.
        #[from]
        #[source]
        source: SimplexValidationError,
    },

    /// Repair completed, but Level 5 certification rejected the result.
    #[error("Delaunayized triangulation failed Level 5 certification: {source}")]
    FinalValidationFailed {
        /// Underlying Delaunay-validation failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
}

/// Recoverable failure to repair and refine a Levels 1–4 triangulation.
///
/// Every variant retains the original triangulation after rollback, including
/// failures after successful flips but before final Level 5 publication.
pub type DelaunayizeRefinementError<K, U, V, const D: usize> =
    RefinementError<Triangulation<K, U, V, D>, DelaunayizeError>;

// =============================================================================
// HELPERS
// =============================================================================

/// Errors that can occur while restoring simplex payloads after a fallback rebuild.
///
/// Restoration first identifies rebuilt simplices by their sorted vertex UUID
/// set, then commits payloads through checked TDS mutation APIs. These are
/// separate failure modes so callers can distinguish corrupted simplex
/// identity data from stale mutation handles.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::delaunayize::SimplexDataRestoreError;
/// use delaunay::prelude::tds::{
///     SimplexKey, TdsError, TdsMutationError, VertexKey, SimplexValidationError,
/// };
/// use slotmap::KeyData;
///
/// let identity_error = SimplexDataRestoreError::SimplexIdentity {
///     source: SimplexValidationError::VertexKeyNotFound {
///         key: VertexKey::from(KeyData::from_ffi(0xBAD)),
///     },
/// };
/// std::assert_matches!(
///     identity_error,
///     SimplexDataRestoreError::SimplexIdentity { .. }
/// );
///
/// let mutation_error = TdsMutationError::from(TdsError::SimplexNotFound {
///     simplex_key: SimplexKey::from(KeyData::from_ffi(0xCAFE)),
///     context: "restore simplex payload".to_string(),
/// });
/// let assignment_error = SimplexDataRestoreError::PayloadAssignment {
///     source: mutation_error,
/// };
/// std::assert_matches!(
///     assignment_error,
///     SimplexDataRestoreError::PayloadAssignment { .. }
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimplexDataRestoreError {
    /// A rebuilt simplex could not resolve its vertex UUID identity.
    #[error("rebuilt simplex identity lookup failed: {source}")]
    SimplexIdentity {
        /// The simplex validation failure encountered while reading vertex UUIDs.
        #[from]
        #[source]
        source: SimplexValidationError,
    },

    /// A rebuilt simplex payload could not be assigned through the checked setter.
    #[error("rebuilt simplex payload assignment failed: {source}")]
    PayloadAssignment {
        /// The TDS mutation failure encountered while assigning simplex data.
        #[from]
        #[source]
        source: TdsMutationError,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SimplexDataMatch<V> {
    Unique(Option<V>),
    Ambiguous,
}

type SimplexDataByVertexUuids<V> = FastHashMap<SimplexVertexUuidBuffer, SimplexDataMatch<V>>;

/// Snapshot of the inputs needed for a fallback rebuild.
///
/// The snapshot stores only the preserved vertices and the simplex payload
/// signatures needed for rebuild input.
struct FallbackRebuildSnapshot<U, V, const D: usize> {
    vertices: Vec<Vertex<U, D>>,
    simplex_data: SimplexDataByVertexUuids<V>,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
}

impl<U, V, const D: usize> FallbackRebuildSnapshot<U, V, D> {
    /// Returns the preserved vertices used to seed fallback reconstruction.
    fn vertices(&self) -> &[Vertex<U, D>] {
        &self.vertices
    }

    /// Returns simplex payload signatures keyed by sorted vertex UUIDs.
    const fn simplex_data(&self) -> &SimplexDataByVertexUuids<V> {
        &self.simplex_data
    }
}

/// Captures the fallback rebuild inputs from the current TDS, including typed
/// failure if any simplex cannot resolve its vertex UUID identity.
///
/// # Errors
///
/// Returns [`SimplexValidationError`] if any simplex cannot resolve all vertex
/// UUIDs needed to build its order-independent payload signature.
fn snapshot_rebuild_state<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
) -> Result<FallbackRebuildSnapshot<U, V, D>, SimplexValidationError>
where
    U: Copy,
    V: Copy,
{
    let vertices = triangulation
        .vertices()
        .map(|(_, v)| Vertex::from_validated_point_with_uuid(*v.point(), v.uuid(), v.data))
        .collect::<Vec<_>>();
    let simplex_data = collect_simplex_data(triangulation)?;
    Ok(FallbackRebuildSnapshot {
        vertices,
        simplex_data,
        topology_guarantee: triangulation.topology_guarantee(),
        global_topology: triangulation.global_topology(),
    })
}

/// Hashes simplex payloads by sorted vertex UUIDs so fallback rebuilds can
/// recover payloads for simplices whose vertex set survives unchanged.
///
/// # Errors
///
/// Returns [`SimplexValidationError`] if a simplex references a vertex whose
/// UUID cannot be resolved.
fn collect_simplex_data<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
) -> Result<SimplexDataByVertexUuids<V>, SimplexValidationError>
where
    V: Copy,
{
    let mut simplex_data = FastHashMap::default();
    for (_, simplex) in triangulation.simplices() {
        let vertex_uuids = simplex_vertex_uuids(triangulation, simplex)?;
        match simplex_data.entry(vertex_uuids) {
            Entry::Vacant(entry) => {
                entry.insert(SimplexDataMatch::Unique(simplex.data().copied()));
            }
            Entry::Occupied(mut entry) => {
                entry.insert(SimplexDataMatch::Ambiguous);
            }
        }
    }
    Ok(simplex_data)
}

/// Builds the order-independent simplex identity used to match original and
/// rebuilt simplices across fallback reconstruction.
///
/// # Errors
///
/// Returns [`SimplexValidationError`] if any simplex vertex key cannot be
/// resolved to its stable vertex UUID.
fn simplex_vertex_uuids<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
    simplex: &Simplex<V, D>,
) -> Result<SimplexVertexUuidBuffer, SimplexValidationError> {
    let mut vertex_uuids = SimplexVertexUuidBuffer::new();
    for &vertex_key in simplex.vertices() {
        let vertex_uuid = triangulation
            .vertex_uuid_from_key(vertex_key)
            .ok_or(SimplexValidationError::VertexKeyNotFound { key: vertex_key })?;
        vertex_uuids.push(vertex_uuid);
    }
    vertex_uuids.sort_unstable();
    Ok(vertex_uuids)
}

/// Reattaches original simplex payloads to rebuilt simplices that retain the same
/// vertex UUID set after fallback reconstruction.
///
/// # Errors
///
/// Returns [`SimplexDataRestoreError`] if rebuilt simplex identity lookup fails
/// or if checked payload assignment rejects a rebuilt simplex key.
fn restore_simplex_data<K, U, V, const D: usize>(
    rebuilt: &mut DelaunayTriangulation<K, U, V, D>,
    original_simplex_data: &SimplexDataByVertexUuids<V>,
) -> Result<(), SimplexDataRestoreError>
where
    V: Copy,
{
    let mut assignments: Vec<(SimplexKey, V)> = Vec::new();
    for (simplex_key, simplex) in rebuilt.simplices() {
        let vertex_uuids = simplex_vertex_uuids(rebuilt.as_triangulation(), simplex)?;
        let Some(SimplexDataMatch::Unique(Some(data))) = original_simplex_data.get(&vertex_uuids)
        else {
            continue;
        };
        assignments.push((simplex_key, *data));
    }

    for (simplex_key, data) in assignments {
        rebuilt.set_simplex_data(simplex_key, Some(data))?;
    }

    Ok(())
}

/// Internal fallback rebuild failure before mapping into public conversion errors.
///
/// This private error keeps fallback construction and payload restoration
/// orthogonal while [`DelaunayizeError`] preserves both the original Delaunay
/// repair failure and the fallback failure.
#[derive(Clone, Debug, Error, PartialEq)]
enum FallbackRebuildError {
    /// Rebuilding from preserved vertices failed during triangulation construction.
    #[error("fallback rebuild failed: {source}")]
    Construction {
        #[from]
        #[source]
        source: DelaunayTriangulationConstructionError,
    },
    /// Rebuild succeeded but restoring simplex payloads failed.
    #[error("fallback simplex-data restore failed: {source}")]
    Restore {
        #[from]
        #[source]
        source: SimplexDataRestoreError,
    },
}

/// Maps a fallback rebuild failure while handling a Delaunay-repair failure
/// without erasing either typed source.
fn delaunay_rebuild_error(
    source: DelaunayRepairError,
    fallback_error: FallbackRebuildError,
) -> DelaunayizeError {
    match fallback_error {
        FallbackRebuildError::Construction {
            source: rebuild_error,
        } => DelaunayizeError::DelaunayRepairFailedWithRebuild {
            source,
            rebuild_error,
        },
        FallbackRebuildError::Restore {
            source: restore_error,
        } => DelaunayizeError::DelaunayRepairFailedWithRebuildRestore {
            source,
            restore_error,
        },
    }
}

/// Rebuilds a triangulation from preserved vertices while restoring any
/// simplex payloads whose vertex UUID signatures survive the rebuild unchanged.
///
/// # Errors
///
/// Returns [`FallbackRebuildError::Construction`] if triangulation construction
/// from preserved vertices fails, or [`FallbackRebuildError::Restore`] if
/// simplex payload restoration fails after a successful rebuild.
fn rebuild_preserving_data<K, U, V, const D: usize>(
    kernel: &K,
    snapshot: &FallbackRebuildSnapshot<U, V, D>,
) -> Result<DelaunayTriangulation<K, U, V, D>, FallbackRebuildError>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let builder = DelaunayTriangulationBuilder::new(snapshot.vertices())
        .simplex_data_type::<V>()
        .topology_guarantee(snapshot.topology_guarantee);
    let builder = match snapshot.global_topology {
        GlobalTopology::Toroidal {
            domain,
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        } => builder.toroidal(domain),
        global_topology => builder.global_topology(global_topology),
    };
    let mut rebuilt = builder.build_with_kernel(kernel)?;
    restore_simplex_data(&mut rebuilt, snapshot.simplex_data())?;
    Ok(rebuilt)
}

/// Runs the configured Delaunay repair strategy for the delaunayize workflow.
///
/// # Errors
///
/// Returns [`DelaunayRepairError`] from the selected flip-repair strategy when
/// repair does not converge or validation rejects the repaired triangulation.
fn run_configured_delaunay_repair<K, U, V, const D: usize>(
    transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
    kernel: &K,
    topology: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let run = repair_delaunay_with_flips_k2_k3_run_in_transaction(
        transaction,
        kernel,
        None,
        topology,
        global_topology,
        config.delaunay_max_flips,
    )?;
    transaction
        .triangulation_mut()
        .normalize_and_promote_positive_orientation()
        .map_err(
            |source| DelaunayRepairError::OrientationCanonicalizationFailed {
                reason: Box::new(
                    DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                        source: Box::new(source),
                    },
                ),
            },
        )?;
    Ok(run.stats)
}

trait DelaunayizeOperations<K, U, V, const D: usize>
where
    U: Clone,
    V: Clone,
{
    fn repair(
        &mut self,
        transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
        kernel: &K,
        topology: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
        config: DelaunayizeConfig,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError>;

    fn certify(
        &mut self,
        triangulation: &Triangulation<K, U, V, D>,
    ) -> Result<DelaunayLevelFiveCertificate<D>, DelaunayTriangulationValidationError>;
}

struct CanonicalDelaunayizeOperations;

impl<K, U, V, const D: usize> DelaunayizeOperations<K, U, V, D> for CanonicalDelaunayizeOperations
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    fn repair(
        &mut self,
        transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
        kernel: &K,
        topology: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
        config: DelaunayizeConfig,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError> {
        run_configured_delaunay_repair(transaction, kernel, topology, global_topology, config)
    }

    fn certify(
        &mut self,
        triangulation: &Triangulation<K, U, V, D>,
    ) -> Result<DelaunayLevelFiveCertificate<D>, DelaunayTriangulationValidationError> {
        certify_level_five_for_refinement(triangulation)
    }
}

/// Extracts Delaunay repair counters so fallback-recovered outcomes preserve
/// observable flip-repair diagnostics from the failed repair attempt.
fn failed_delaunay_repair_stats(source: &DelaunayRepairError) -> DelaunayRepairStats {
    match source {
        DelaunayRepairError::NonConvergent { diagnostics, .. } => DelaunayRepairStats {
            facets_checked: diagnostics.facets_checked,
            flips_performed: diagnostics.flips_performed,
            max_queue_len: diagnostics.max_queue_len,
        },
        DelaunayRepairError::PostconditionFailed { .. }
        | DelaunayRepairError::VerificationFailed { .. }
        | DelaunayRepairError::OrientationCanonicalizationFailed { .. }
        | DelaunayRepairError::InvalidTopology { .. }
        | DelaunayRepairError::HeuristicRebuildFailed { .. }
        | DelaunayRepairError::Flip { .. } => DelaunayRepairStats::default(),
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Converts a Levels 1–4 triangulation into a Levels 1–5 Delaunay triangulation.
///
/// The input is consumed so an intermediate repair state cannot escape under
/// the [`DelaunayTriangulation`] type. Conversion consumes the generic
/// triangulation's cumulative Levels 1–4 proof, performs bounded
/// invariant-preserving repair, then publishes the Delaunay owner only after
/// Level 5 certification succeeds.
///
/// The public [`DelaunayRefinementBuilder`] selects this implementation through
/// [`repair_by_flips`](DelaunayRefinementBuilder::repair_by_flips). Use its
/// strict mode when no repair should be attempted.
///
/// # Errors
///
/// Returns [`DelaunayizeRefinementError`] when conversion cannot converge or
/// final certification rejects the candidate. Every failure rolls back and
/// retains the original Levels 1–4 owner together with its typed
/// [`DelaunayizeError`] reason.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationBuilder,
///     DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::delaunayize::DelaunayRefinementBuilder;
/// use delaunay::prelude::geometry::CoordinateConversionError;
/// use delaunay::prelude::validation::DelaunayValidationError;
/// use delaunay::RefinementError;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     DelaunayProperty(#[from] DelaunayValidationError),
/// #     #[error(transparent)]
/// #     Delaunayize(#[from] delaunay::prelude::delaunayize::DelaunayizeError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![4.0, 0.0]?,
///     delaunay::vertex![4.0, 2.0]?,
///     delaunay::vertex![1.0, 2.0]?,
/// ];
/// let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
/// let triangulation =
///     DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
///         .map_err(DelaunayTriangulationConstructionError::from)?
///         .build_triangulation()?;
/// assert!(!triangulation.delaunay_violation_report(None)?.is_valid());
///
/// let result = DelaunayRefinementBuilder::new(triangulation)
///     .repair_by_flips()
///     .build()
///     .map_err(RefinementError::into_reason)?;
/// assert!(result.triangulation.validate().is_ok());
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::result_large_err,
    reason = "recoverable Delaunayize failures preserve typed sources on a cold conversion error path"
)]
fn delaunayize<K, U, V, const D: usize>(
    triangulation: Triangulation<K, U, V, D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayizeResult<K, U, V, D>, DelaunayizeRefinementError<K, U, V, D>>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    delaunayize_with_operations(triangulation, config, &mut CanonicalDelaunayizeOperations)
}

#[expect(
    clippy::result_large_err,
    reason = "recoverable Delaunayize failures preserve typed sources on a cold conversion error path"
)]
fn delaunayize_with_operations<K, U, V, O, const D: usize>(
    mut triangulation: Triangulation<K, U, V, D>,
    config: DelaunayizeConfig,
    operations: &mut O,
) -> Result<DelaunayizeResult<K, U, V, D>, DelaunayizeRefinementError<K, U, V, D>>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
    O: DelaunayizeOperations<K, U, V, D>,
{
    let operation = TopologicalOperation::FacetFlip;
    let topology = triangulation.topology_guarantee();
    if !operation.is_admissible_under(topology) {
        let reason = DelaunayizeError::FlipTopologyNotAdmissible {
            required: operation.required_topology(),
            found: topology,
        };
        return Err(RefinementError::new(triangulation, reason));
    }

    let fallback_snapshot = match config
        .fallback_rebuild
        .then(|| snapshot_rebuild_state(&triangulation))
        .transpose()
    {
        Ok(snapshot) => snapshot,
        Err(source) => {
            let reason = DelaunayizeError::FallbackSimplexDataSnapshotFailed { source };
            return Err(RefinementError::new(triangulation, reason));
        }
    };

    let kernel = triangulation.kernel.clone();
    let global_topology = triangulation.global_topology();
    let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);

    match operations.repair(&mut transaction, &kernel, topology, global_topology, config) {
        Ok(delaunay_repair) => {
            let certificate = match operations.certify(transaction.triangulation_mut()) {
                Ok(certificate) => certificate,
                Err(source) => {
                    transaction.rollback();
                    let reason = DelaunayizeError::FinalValidationFailed { source };
                    return Err(RefinementError::new(triangulation, reason));
                }
            };
            transaction.commit();
            let triangulation = DelaunayTriangulationDraft::from_triangulation(triangulation)
                .try_into_delaunay_with_certificate(Some(&certificate))
                .map_err(|failure| {
                    let (triangulation, source) = failure.into_parts();
                    RefinementError::new(
                        triangulation,
                        DelaunayizeError::FinalValidationFailed { source },
                    )
                })?;
            Ok(DelaunayizeResult {
                triangulation,
                outcome: DelaunayizeOutcome {
                    delaunay_repair,
                    used_fallback_rebuild: false,
                },
            })
        }
        Err(repair_error) => {
            let Some(fallback_snapshot) = fallback_snapshot else {
                transaction.rollback();
                return Err(RefinementError::new(
                    triangulation,
                    DelaunayizeError::from(repair_error),
                ));
            };
            let delaunay_repair = failed_delaunay_repair_stats(&repair_error);
            let triangulation = match rebuild_preserving_data(&kernel, &fallback_snapshot) {
                Ok(triangulation) => triangulation,
                Err(fallback_error) => {
                    let reason = delaunay_rebuild_error(repair_error, fallback_error);
                    transaction.rollback();
                    return Err(RefinementError::new(triangulation, reason));
                }
            };
            transaction.commit();
            Ok(DelaunayizeResult {
                triangulation,
                outcome: DelaunayizeOutcome {
                    delaunay_repair,
                    used_fallback_rebuild: true,
                },
            })
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{DelaunayRepairDiagnostics, RepairQueueOrder};
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::tds::{Tds, TdsError, VertexKey};
    use crate::validation::DelaunayVerificationError;
    use crate::vertex;
    use crate::{DelaunayTriangulationBuilder, TriangulationConstructionError};
    use slotmap::KeyData;
    use std::error::Error as StdError;

    #[must_use]
    fn synthetic_repair_error() -> DelaunayRepairError {
        DelaunayRepairError::NonConvergent {
            max_flips: 0,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 1,
                flips_performed: 0,
                max_queue_len: 1,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: RepairQueueOrder::Fifo,
            }),
        }
    }

    #[must_use]
    fn synthetic_final_validation_error() -> DelaunayTriangulationValidationError {
        DelaunayTriangulationValidationError::VerificationFailed {
            source: Box::new(DelaunayVerificationError::from(synthetic_repair_error())),
        }
    }

    struct FailRepairOperations;

    impl<K, U, V, const D: usize> DelaunayizeOperations<K, U, V, D> for FailRepairOperations
    where
        K: ExactPredicates<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        fn repair(
            &mut self,
            _transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
            _kernel: &K,
            _topology: TopologyGuarantee,
            _global_topology: GlobalTopology<D>,
            _config: DelaunayizeConfig,
        ) -> Result<DelaunayRepairStats, DelaunayRepairError> {
            Err(synthetic_repair_error())
        }

        fn certify(
            &mut self,
            triangulation: &Triangulation<K, U, V, D>,
        ) -> Result<DelaunayLevelFiveCertificate<D>, DelaunayTriangulationValidationError> {
            certify_level_five_for_refinement(triangulation)
        }
    }

    struct FailCertificationOperations;

    impl<K, U, V, const D: usize> DelaunayizeOperations<K, U, V, D> for FailCertificationOperations
    where
        K: ExactPredicates<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        fn repair(
            &mut self,
            transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
            kernel: &K,
            topology: TopologyGuarantee,
            global_topology: GlobalTopology<D>,
            config: DelaunayizeConfig,
        ) -> Result<DelaunayRepairStats, DelaunayRepairError> {
            run_configured_delaunay_repair(transaction, kernel, topology, global_topology, config)
        }

        fn certify(
            &mut self,
            _triangulation: &Triangulation<K, U, V, D>,
        ) -> Result<DelaunayLevelFiveCertificate<D>, DelaunayTriangulationValidationError> {
            Err(synthetic_final_validation_error())
        }
    }

    /// Snapshots deliberately malformed raw TDS fixtures for payload-disambiguation tests.
    fn snapshot_rebuild_state_from_tds<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
    ) -> Result<FallbackRebuildSnapshot<U, V, D>, SimplexValidationError>
    where
        U: Copy,
        V: Copy,
    {
        let vertices = tds
            .vertices()
            .map(|(_, v)| Vertex::from_validated_point_with_uuid(*v.point(), v.uuid(), v.data))
            .collect::<Vec<_>>();
        let simplex_data = collect_simplex_data_from_tds(tds)?;
        Ok(FallbackRebuildSnapshot {
            vertices,
            simplex_data,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            global_topology: GlobalTopology::DEFAULT,
        })
    }

    /// Hashes simplex payloads from invalid raw TDS fixtures by sorted vertex UUIDs.
    fn collect_simplex_data_from_tds<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
    ) -> Result<SimplexDataByVertexUuids<V>, SimplexValidationError>
    where
        V: Copy,
    {
        let mut simplex_data = FastHashMap::default();
        for (_, simplex) in tds.simplices() {
            let vertex_uuids = simplex_vertex_uuids_from_tds(tds, simplex)?;
            match simplex_data.entry(vertex_uuids) {
                Entry::Vacant(entry) => {
                    entry.insert(SimplexDataMatch::Unique(simplex.data().copied()));
                }
                Entry::Occupied(mut entry) => {
                    entry.insert(SimplexDataMatch::Ambiguous);
                }
            }
        }
        Ok(simplex_data)
    }

    /// Builds the sorted vertex-UUID simplex identity for invalid raw TDS fixtures.
    fn simplex_vertex_uuids_from_tds<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        simplex: &Simplex<V, D>,
    ) -> Result<SimplexVertexUuidBuffer, SimplexValidationError> {
        let mut vertex_uuids = simplex
            .vertex_uuid_iter(tds)
            .collect::<Result<SimplexVertexUuidBuffer, SimplexValidationError>>()?;
        vertex_uuids.sort_unstable();
        Ok(vertex_uuids)
    }

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    fn init_tracing() {
        let _ = tracing_subscriber::fmt::try_init();
    }

    fn construction_error() -> DelaunayTriangulationConstructionError {
        DelaunayTriangulationConstructionError::from(
            TriangulationConstructionError::FailedToCreateSimplex {
                message: "synthetic simplex creation failure".to_string(),
            },
        )
    }

    // =============================================================================
    // CONFIG DEFAULT TESTS
    // =============================================================================

    #[test]
    fn final_validation_error_preserves_typed_source_chain() {
        let validation_error = DelaunayTriangulationValidationError::VerificationFailed {
            source: Box::new(DelaunayVerificationError::from(synthetic_repair_error())),
        };
        let error = DelaunayizeError::FinalValidationFailed {
            source: validation_error.clone(),
        };

        assert_eq!(
            error,
            DelaunayizeError::FinalValidationFailed {
                source: validation_error,
            }
        );
        assert!(StdError::source(&error).is_some());
        assert!(error.to_string().contains("Level 5 certification"));
    }

    #[test]
    fn test_config_defaults() {
        init_tracing();
        let config = DelaunayizeConfig::default();
        assert!(!config.fallback_rebuild);
        assert!(config.delaunay_max_flips.is_none());
    }

    // =============================================================================
    // OUTCOME POPULATION TESTS
    // =============================================================================

    #[test]
    fn test_outcome_populated_on_success() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        let result = delaunayize(dt.into_triangulation(), DelaunayizeConfig::default()).unwrap();

        assert!(result.outcome.delaunay_repair.facets_checked > 0);
        assert_eq!(result.outcome.delaunay_repair.flips_performed, 0);
        assert!(result.outcome.delaunay_repair.max_queue_len > 0);
        assert!(!result.outcome.used_fallback_rebuild);
        assert!(result.triangulation.validate().is_ok());
    }

    // =============================================================================
    // ERROR PATH TESTS
    // =============================================================================

    #[test]
    fn test_simplex_vertex_uuids_missing_vertex() {
        let mut tds: Tds<(), i32, 2> = Tds::empty();
        let vertex_keys: Vec<_> = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ]
        .iter()
        .map(|vertex| tds.insert_vertex_with_mapping(*vertex).unwrap())
        .collect();
        let missing = vertex_keys[0];
        let simplex = Simplex::try_new_with_data(vertex_keys, Some(7)).unwrap();
        tds.remove_isolated_vertex(missing).unwrap();

        let err = simplex_vertex_uuids_from_tds(&tds, &simplex).unwrap_err();

        assert_eq!(
            err,
            SimplexValidationError::VertexKeyNotFound { key: missing }
        );
    }

    #[test]
    fn test_snapshot_error_source() {
        let source = SimplexValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
        };
        let err = DelaunayizeError::FallbackSimplexDataSnapshotFailed {
            source: source.clone(),
        };

        assert_eq!(
            err,
            DelaunayizeError::FallbackSimplexDataSnapshotFailed {
                source: source.clone()
            }
        );
        assert!(
            err.to_string()
                .contains("Fallback simplex-data snapshot failed")
        );
        assert!(
            err.to_string()
                .contains("fallback rebuild cannot be attempted")
        );
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn test_restore_error_sources() {
        let delaunay_source = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let restore_error = SimplexDataRestoreError::SimplexIdentity {
            source: SimplexValidationError::VertexKeyNotFound {
                key: VertexKey::from(KeyData::from_ffi(0xBAD)),
            },
        };

        let delaunay_err = DelaunayizeError::DelaunayRepairFailedWithRebuildRestore {
            source: delaunay_source.clone(),
            restore_error,
        };

        assert!(
            delaunay_err
                .to_string()
                .contains("simplex-data restore failed")
        );
        assert_eq!(
            StdError::source(&delaunay_err).unwrap().to_string(),
            delaunay_source.to_string()
        );
    }

    #[test]
    fn test_payload_assignment_restore_error_source() {
        let source = TdsMutationError::from(TdsError::SimplexNotFound {
            simplex_key: SimplexKey::from(KeyData::from_ffi(0xCAFE)),
            context: "restore simplex payload".to_string(),
        });
        let err = SimplexDataRestoreError::PayloadAssignment {
            source: source.clone(),
        };

        assert_eq!(
            err,
            SimplexDataRestoreError::PayloadAssignment {
                source: source.clone()
            }
        );
        assert!(err.to_string().contains("payload assignment failed"));
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn test_delaunay_rebuild_error_mapping() {
        let source = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let rebuild_error = construction_error();
        let restore_error = SimplexDataRestoreError::SimplexIdentity {
            source: SimplexValidationError::VertexKeyNotFound {
                key: VertexKey::from(KeyData::from_ffi(0xBAD)),
            },
        };

        let rebuild_err = delaunay_rebuild_error(
            source.clone(),
            FallbackRebuildError::Construction {
                source: rebuild_error.clone(),
            },
        );
        assert_eq!(
            rebuild_err,
            DelaunayizeError::DelaunayRepairFailedWithRebuild {
                source: source.clone(),
                rebuild_error,
            }
        );
        assert!(
            rebuild_err
                .to_string()
                .contains("fallback rebuild also failed")
        );

        let restore_err = delaunay_rebuild_error(
            source.clone(),
            FallbackRebuildError::Restore {
                source: restore_error.clone(),
            },
        );
        assert_eq!(
            restore_err,
            DelaunayizeError::DelaunayRepairFailedWithRebuildRestore {
                source,
                restore_error,
            }
        );
        assert!(
            restore_err
                .to_string()
                .contains("fallback rebuild succeeded but simplex-data restore failed")
        );
    }

    // =============================================================================
    // FALLBACK BEHAVIOR TESTS
    // =============================================================================

    #[test]
    fn final_certification_failure_rolls_back_mutating_repair_and_returns_triangulation() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([4.0, 0.0]).unwrap(),
            vertex!([4.0, 2.0]).unwrap(),
            vertex!([1.0, 2.0]).unwrap(),
        ];
        let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
        let triangulation =
            DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
                .unwrap()
                .build_triangulation()
                .unwrap();
        assert!(
            !triangulation
                .delaunay_violation_report(None)
                .unwrap()
                .is_valid()
        );

        let probe = delaunayize(triangulation.clone(), DelaunayizeConfig::default()).unwrap();
        assert!(probe.outcome.delaunay_repair.flips_performed > 0);

        let before = triangulation.tds.clone_for_rollback();
        let owner_before = triangulation.tds.topology_owner_id();
        let generation_before = triangulation.tds.generation();
        let failure = delaunayize_with_operations(
            triangulation,
            DelaunayizeConfig::default(),
            &mut FailCertificationOperations,
        )
        .expect_err("synthetic final certification must reject the repaired candidate");
        let (triangulation, reason) = failure.into_parts();

        assert!(matches!(
            reason,
            DelaunayizeError::FinalValidationFailed { .. }
        ));
        assert_eq!(triangulation.tds, before);
        assert_eq!(triangulation.tds.topology_owner_id(), owner_before);
        assert_eq!(triangulation.tds.generation(), generation_before);
        triangulation
            .validate_realization()
            .expect("failed Delaunay refinement must return the original Levels 1-4 owner");
    }

    #[test]
    fn delaunay_repair_fallback_rebuilds_after_unsupported_dimension() {
        init_tracing();
        let vertices = [vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt: DelaunayTriangulation<_, (), (), 1> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        let result = delaunayize(
            dt.into_triangulation(),
            DelaunayizeConfig {
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap();

        assert_eq!(result.outcome.delaunay_repair.facets_checked, 0);
        assert_eq!(result.outcome.delaunay_repair.flips_performed, 0);
        assert_eq!(result.outcome.delaunay_repair.max_queue_len, 0);
        assert!(result.outcome.used_fallback_rebuild);
        assert_eq!(result.triangulation.number_of_vertices(), vertices.len());
        assert!(result.triangulation.is_valid_structure().is_ok());
    }

    #[test]
    fn delaunay_repair_fallback_rebuilds_supported_2d_after_repair_failure() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        let result = delaunayize_with_operations(
            dt.into_triangulation(),
            DelaunayizeConfig {
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
            &mut FailRepairOperations,
        )
        .unwrap();

        assert!(result.outcome.used_fallback_rebuild);
        assert_eq!(result.outcome.delaunay_repair.facets_checked, 1);
        assert_eq!(result.outcome.delaunay_repair.flips_performed, 0);
        assert_eq!(result.outcome.delaunay_repair.max_queue_len, 1);
        assert_eq!(result.triangulation.number_of_vertices(), vertices.len());
        assert!(result.triangulation.validate().is_ok());
    }

    #[test]
    fn test_rebuild_restores_simplex_data() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let original_simplex_key = dt.simplices().next().unwrap().0;
        dt.set_simplex_data(original_simplex_key, Some(42)).unwrap();

        let snapshot = snapshot_rebuild_state(dt.as_triangulation()).unwrap();

        let rebuilt = rebuild_preserving_data(dt.kernel(), &snapshot).unwrap();

        let (_, rebuilt_simplex) = rebuilt.simplices().next().unwrap();
        assert_eq!(rebuilt_simplex.data(), Some(&42));
        assert!(rebuilt.validate().is_ok());
    }

    #[test]
    fn test_rebuild_drops_ambiguous_data() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut tds: Tds<(), i32, 2> = Tds::empty();
        let vertex_keys: Vec<_> = vertices
            .iter()
            .map(|vertex| tds.insert_vertex_with_mapping(*vertex).unwrap())
            .collect();

        let duplicate_a = Simplex::try_new_with_data(vertex_keys.clone(), Some(42)).unwrap();
        let duplicate_b = Simplex::try_new_with_data(vertex_keys, Some(42)).unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(duplicate_a)
            .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(duplicate_b)
            .unwrap();

        let snapshot = snapshot_rebuild_state_from_tds(&tds).unwrap();
        let kernel = AdaptiveKernel::new();
        let mut rebuilt: DelaunayTriangulation<_, (), i32, 2> =
            DelaunayTriangulationBuilder::new(snapshot.vertices())
                .simplex_data_type::<i32>()
                .build_with_kernel(&kernel)
                .unwrap();

        restore_simplex_data(&mut rebuilt, snapshot.simplex_data()).unwrap();

        let (_, rebuilt_simplex) = rebuilt.simplices().next().unwrap();
        assert_eq!(rebuilt_simplex.data(), None);
        assert!(rebuilt.validate().is_ok());
    }
}
