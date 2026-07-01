//! End-to-end "repair then delaunayize" workflow.
//!
//! This module provides [`delaunayize_by_flips`](crate::delaunayize::delaunayize_by_flips),
//! a single public entrypoint that takes an existing [`DelaunayTriangulation`],
//! performs bounded deterministic topology repair toward
//! [`TopologyGuarantee::PLManifold`](crate::TopologyGuarantee::PLManifold),
//! and then applies
//! standard flip-based Delaunay repair.
//!
//! # Workflow
//!
//! 1. **PL-manifold topology repair** — removes simplices that cause facet
//!    over-sharing, boundary-ridge multiplicity, ridge-link, or vertex-link
//!    violations using bounded deterministic [`PlManifoldRepairStage`] pruning
//!    stages.
//! 2. **Delaunay flip repair** — runs k=2/k=3 bistellar flips to restore the
//!    empty-circumsphere property.
//! 3. **Optional fallback rebuild** — if configured, rebuilds the triangulation
//!    from its vertex set after topology repair fails, or after Delaunay repair
//!    fails following successful topology repair.
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
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
//! assert!(outcome.topology_repair.succeeded);
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
pub use crate::{PlManifoldRepairError, PlManifoldRepairStage, PlManifoldRepairStats};

use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairConfig, repair_pl_manifold_topology,
};
use crate::core::collections::{Entry, FastHashMap, SimplexVertexUuidBuffer};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, Tds, TdsMutationError};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::delaunay_rollback::{DelaunayRollbackTransaction, DelaunaySpatialIndexRollback};
use crate::geometry::kernel::ExactPredicates;
use crate::repair::DelaunayRepairHeuristicConfig;
use crate::triangulation::DelaunayTriangulation;
use thiserror::Error;

#[cfg(test)]
mod test_hooks {
    use super::DelaunayRepairError;
    use crate::core::algorithms::flips::{DelaunayRepairDiagnostics, RepairQueueOrder};
    use std::cell::Cell as ThreadCell;

    thread_local! {
        static FORCE_DELAUNAY_REPAIR_FAILURE: ThreadCell<bool> = const { ThreadCell::new(false) };
    }

    /// Enables or disables a synthetic Delaunay repair failure for branch tests.
    pub(super) fn set_force_delaunay_repair_failure(enabled: bool) -> bool {
        FORCE_DELAUNAY_REPAIR_FAILURE.with(|flag| {
            let prior = flag.get();
            flag.set(enabled);
            prior
        })
    }

    /// Restores the previous synthetic Delaunay repair failure state.
    pub(super) fn restore_force_delaunay_repair_failure(prior: bool) {
        FORCE_DELAUNAY_REPAIR_FAILURE.with(|flag| flag.set(prior));
    }

    /// Reports whether synthetic Delaunay repair failure is enabled.
    pub(super) fn force_delaunay_repair_failure_enabled() -> bool {
        FORCE_DELAUNAY_REPAIR_FAILURE.with(ThreadCell::get)
    }

    /// Builds the synthetic non-convergence error used by fallback branch tests.
    pub(super) fn synthetic_repair_error() -> DelaunayRepairError {
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
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the [`delaunayize_by_flips`] workflow.
///
/// # Defaults
///
/// - `topology_max_iterations`: 64
/// - `topology_max_simplices_removed`: 10,000
/// - `fallback_rebuild`: false
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::delaunayize::DelaunayizeConfig;
///
/// let config = DelaunayizeConfig::default();
/// assert_eq!(config.topology_max_iterations, 64);
/// assert_eq!(config.topology_max_simplices_removed, 10_000);
/// assert!(!config.fallback_rebuild);
/// assert!(config.delaunay_max_flips.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelaunayizeConfig {
    /// Maximum number of topology-repair iterations.
    pub topology_max_iterations: usize,
    /// Maximum number of simplices that may be removed during topology repair.
    pub topology_max_simplices_removed: usize,
    /// If `true`, rebuild the triangulation from the vertex set when topology
    /// repair fails, or when flip-based Delaunay repair fails after topology
    /// repair succeeds.
    ///
    /// Simplex-level user data (`V`) is restored for rebuilt simplices whose sorted
    /// vertex UUID set matches exactly one original simplex. Simplices that change
    /// during rebuild, have no original payload, or have ambiguous duplicate
    /// original signatures receive `None`.
    pub fallback_rebuild: bool,
    /// Optional per-attempt flip budget cap for Delaunay repair.
    ///
    /// `None` (default) uses the internal dimension-dependent budget.
    /// Set to `Some(n)` to limit each repair attempt to at most `n` flips.
    pub delaunay_max_flips: Option<usize>,
}

impl Default for DelaunayizeConfig {
    fn default() -> Self {
        Self {
            topology_max_iterations: 64,
            topology_max_simplices_removed: 10_000,
            fallback_rebuild: false,
            delaunay_max_flips: None,
        }
    }
}

// =============================================================================
// OUTCOME
// =============================================================================

/// Outcome of a successful [`delaunayize_by_flips`] call.
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
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
/// assert!(outcome.topology_repair.succeeded);
/// assert!(!outcome.used_fallback_rebuild);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelaunayizeOutcome<U, V, const D: usize> {
    /// Statistics from the PL-manifold topology repair pass.
    ///
    /// If topology repair fails but fallback rebuild succeeds, these remain the
    /// failed/default repair stats for the repair attempt. Use
    /// [`used_fallback_rebuild`](Self::used_fallback_rebuild) to distinguish
    /// successful rebuild recovery from direct topology repair success.
    pub topology_repair: PlManifoldRepairStats<U, V, D>,
    /// Statistics from the flip-based Delaunay repair pass.
    pub delaunay_repair: DelaunayRepairStats,
    /// Whether the fallback vertex-set rebuild was used.
    pub used_fallback_rebuild: bool,
}

// =============================================================================
// ERRORS
// =============================================================================

/// Errors that can occur during the [`delaunayize_by_flips`] workflow.
///
/// There are four orthogonal failure modes:
/// - **Topology repair** failed (step 1).
/// - **Delaunay repair** failed (step 2), with optional context about a
///   fallback rebuild attempt.
/// - **Fallback snapshot** failed before any repair could run.
/// - **Fallback simplex-data recovery** failed while snapshotting or restoring
///   simplex payloads after a repair failure.
///
/// # Orthogonality
///
/// The variants are mutually exclusive by failure mode:
/// - Topology repair, fallback not attempted -> [`TopologyRepairFailed`](Self::TopologyRepairFailed).
/// - Topology repair, fallback also failed   -> [`TopologyRepairFailedWithRebuild`](Self::TopologyRepairFailedWithRebuild).
/// - Topology repair, fallback rebuild succeeded but payload restore failed -> [`TopologyRepairFailedWithRebuildRestore`](Self::TopologyRepairFailedWithRebuildRestore).
/// - Delaunay repair, fallback not attempted -> [`DelaunayRepairFailed`](Self::DelaunayRepairFailed).
/// - Delaunay repair, fallback payload snapshot failed -> [`DelaunayRepairFailedWithSimplexDataSnapshot`](Self::DelaunayRepairFailedWithSimplexDataSnapshot).
/// - Delaunay repair, fallback also failed   -> [`DelaunayRepairFailedWithRebuild`](Self::DelaunayRepairFailedWithRebuild).
/// - Delaunay repair, fallback rebuild succeeded but payload restore failed -> [`DelaunayRepairFailedWithRebuildRestore`](Self::DelaunayRepairFailedWithRebuildRestore).
/// - Fallback was enabled, but the pre-repair payload snapshot failed -> [`FallbackSimplexDataSnapshotFailed`](Self::FallbackSimplexDataSnapshotFailed).
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
    /// PL-manifold topology repair failed; no fallback rebuild was attempted
    /// (fallback disabled, or the caller's config did not request one).
    #[error("Topology repair failed: {source}")]
    TopologyRepairFailed {
        /// The underlying topology-repair error.
        #[from]
        #[source]
        source: PlManifoldRepairError,
    },

    /// PL-manifold topology repair failed **and** the fallback vertex-set
    /// rebuild also failed.  Both errors are preserved as typed values.
    #[error("Topology repair failed ({source}); fallback rebuild also failed: {rebuild_error}")]
    TopologyRepairFailedWithRebuild {
        /// The underlying topology-repair error that triggered the fallback.
        #[source]
        source: PlManifoldRepairError,
        /// The construction error from the subsequent vertex-set rebuild attempt.
        rebuild_error: DelaunayTriangulationConstructionError,
    },

    /// PL-manifold topology repair failed, the fallback vertex-set rebuild
    /// succeeded, but simplex-payload restoration from the rebuilt topology failed.
    #[error(
        "Topology repair failed ({source}); fallback rebuild succeeded but simplex-data restore failed: {restore_error}"
    )]
    TopologyRepairFailedWithRebuildRestore {
        /// The underlying topology-repair error that triggered the fallback.
        #[source]
        source: PlManifoldRepairError,
        /// The simplex-data restoration error from the rebuilt triangulation.
        restore_error: SimplexDataRestoreError,
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

    /// Delaunay flip repair failed and the fallback payload snapshot could not
    /// be collected from the current triangulation.
    #[error(
        "Delaunay repair failed ({source}); fallback simplex-data snapshot failed: {snapshot_error}"
    )]
    DelaunayRepairFailedWithSimplexDataSnapshot {
        /// The underlying flip-repair error that triggered the fallback.
        #[source]
        source: DelaunayRepairError,
        /// The simplex-data snapshot error from the current triangulation.
        snapshot_error: SimplexValidationError,
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

    /// Fallback rebuild was enabled, but the pre-repair simplex-payload snapshot
    /// could not be collected from the input triangulation. No topology or
    /// Delaunay repair was attempted.
    #[error("Fallback simplex-data snapshot failed before repair; no repair attempted: {source}")]
    FallbackSimplexDataSnapshotFailed {
        /// The simplex-data snapshot error from the input triangulation.
        #[from]
        #[source]
        source: SimplexValidationError,
    },
}

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
/// signatures needed for rebuild input. Transactional rollback remains owned by
/// [`DelaunayRollbackTransaction`], so this helper cannot become a second
/// rollback mechanism.
struct FallbackRebuildSnapshot<U, V, const D: usize> {
    vertices: Vec<Vertex<U, D>>,
    simplex_data: SimplexDataByVertexUuids<V>,
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
fn snapshot_rebuild_state<U, V, const D: usize>(
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
    let simplex_data = collect_simplex_data(tds)?;
    Ok(FallbackRebuildSnapshot {
        vertices,
        simplex_data,
    })
}

/// Hashes simplex payloads by sorted vertex UUIDs so fallback rebuilds can
/// recover payloads for simplices whose vertex set survives unchanged.
///
/// # Errors
///
/// Returns [`SimplexValidationError`] if a simplex references a vertex whose
/// UUID cannot be resolved.
fn collect_simplex_data<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<SimplexDataByVertexUuids<V>, SimplexValidationError>
where
    V: Copy,
{
    let mut simplex_data = FastHashMap::default();
    for (_, simplex) in tds.simplices() {
        let vertex_uuids = simplex_vertex_uuids(tds, simplex)?;
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
fn simplex_vertex_uuids<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex: &Simplex<V, D>,
) -> Result<SimplexVertexUuidBuffer, SimplexValidationError> {
    let mut vertex_uuids = simplex
        .vertex_uuid_iter(tds)
        .collect::<Result<SimplexVertexUuidBuffer, SimplexValidationError>>()?;
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
        let vertex_uuids = simplex_vertex_uuids(rebuilt.tds(), simplex)?;
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

/// Internal fallback rebuild failure before mapping into phase-specific public errors.
///
/// The public [`DelaunayizeError`] variants distinguish whether fallback was
/// triggered by topology repair or Delaunay repair. This private error keeps the
/// rebuild phase orthogonal so both call sites can preserve the typed source.
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

/// Maps a fallback rebuild failure while handling a topology-repair failure
/// without erasing either typed source.
fn topology_rebuild_error(
    source: PlManifoldRepairError,
    fallback_error: FallbackRebuildError,
) -> DelaunayizeError {
    match fallback_error {
        FallbackRebuildError::Construction {
            source: rebuild_error,
        } => DelaunayizeError::TopologyRepairFailedWithRebuild {
            source,
            rebuild_error,
        },
        FallbackRebuildError::Restore {
            source: restore_error,
        } => DelaunayizeError::TopologyRepairFailedWithRebuildRestore {
            source,
            restore_error,
        },
    }
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
    let mut rebuilt = DelaunayTriangulation::try_with_kernel(kernel, snapshot.vertices())?;
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
    dt: &mut DelaunayTriangulation<K, U, V, D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayRepairStats, DelaunayRepairError>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if let Some(max_flips) = config.delaunay_max_flips {
        dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig {
            max_flips: Some(max_flips),
            ..DelaunayRepairHeuristicConfig::default()
        })
        .map(|outcome| outcome.stats)
    } else {
        dt.repair_delaunay_with_flips()
    }
}

/// Finalizes Delaunay repair while preserving atomic rollback semantics.
///
/// The transaction commits on direct Delaunay repair success or successful
/// fallback rebuild. Every Delaunay repair, fallback snapshot, rebuild, or
/// restore failure rolls the transaction back before returning the typed public
/// [`DelaunayizeError`].
///
/// # Errors
///
/// Returns [`DelaunayizeError`] if Delaunay repair fails and fallback is
/// disabled, if fallback snapshotting fails after Delaunay repair failure, or
/// if fallback rebuild or payload restoration fails.
#[expect(
    clippy::result_large_err,
    reason = "DelaunayizeError preserves typed repair, rebuild, and restore errors so callers can pattern-match both primary and fallback failures; this is a cold transactional error path."
)]
fn finish_delaunayize_after_delaunay_repair<K, U, V, const D: usize>(
    mut transaction: DelaunayRollbackTransaction<'_, K, U, V, D>,
    topology_stats: PlManifoldRepairStats<U, V, D>,
    pre_delaunay_fallback_snapshot: Option<
        Result<FallbackRebuildSnapshot<U, V, D>, SimplexValidationError>,
    >,
    delaunay_result: Result<DelaunayRepairStats, DelaunayRepairError>,
) -> Result<DelaunayizeOutcome<U, V, D>, DelaunayizeError>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    match delaunay_result {
        Ok(delaunay_stats) => {
            let outcome = DelaunayizeOutcome {
                topology_repair: topology_stats,
                delaunay_repair: delaunay_stats,
                used_fallback_rebuild: false,
            };
            transaction.commit();
            Ok(outcome)
        }
        Err(repair_err) => {
            let Some(fallback_snapshot) = pre_delaunay_fallback_snapshot else {
                transaction.rollback();
                return Err(DelaunayizeError::from(repair_err));
            };
            let fallback_snapshot = match fallback_snapshot {
                Ok(state) => state,
                Err(snapshot_error) => {
                    transaction.rollback();
                    return Err(
                        DelaunayizeError::DelaunayRepairFailedWithSimplexDataSnapshot {
                            source: repair_err,
                            snapshot_error,
                        },
                    );
                }
            };

            let fallback_result = {
                let kernel = &transaction.delaunay_mut().as_triangulation().kernel;
                rebuild_preserving_data(kernel, &fallback_snapshot)
            };
            match fallback_result {
                Ok(rebuilt) => {
                    *transaction.delaunay_mut() = rebuilt;
                    transaction.commit();
                    Ok(DelaunayizeOutcome {
                        topology_repair: topology_stats,
                        delaunay_repair: DelaunayRepairStats::default(),
                        used_fallback_rebuild: true,
                    })
                }
                Err(fallback_error) => {
                    transaction.rollback();
                    Err(delaunay_rebuild_error(repair_err, fallback_error))
                }
            }
        }
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Performs bounded topology repair followed by flip-based Delaunay repair.
///
/// This is the primary public entrypoint for the "repair then delaunayize"
/// workflow described in the [module documentation](self).
///
/// # Type Constraints
///
/// The kernel must implement [`ExactPredicates`] (required by the underlying
/// Delaunay flip-repair engine). The default [`AdaptiveKernel`](crate::geometry::kernel::AdaptiveKernel)
/// satisfies this requirement.
///
/// # Errors
///
/// Returns [`DelaunayizeError`] if:
/// - Topology repair fails and no fallback rebuild was attempted
///   ([`TopologyRepairFailed`](DelaunayizeError::TopologyRepairFailed)).
/// - Topology repair fails **and** the fallback vertex-set rebuild also
///   fails
///   ([`TopologyRepairFailedWithRebuild`](DelaunayizeError::TopologyRepairFailedWithRebuild)).
/// - Topology repair fails, fallback rebuild succeeds, and simplex-payload
///   restoration fails
///   ([`TopologyRepairFailedWithRebuildRestore`](DelaunayizeError::TopologyRepairFailedWithRebuildRestore)).
/// - Delaunay flip repair fails and no fallback rebuild was attempted
///   ([`DelaunayRepairFailed`](DelaunayizeError::DelaunayRepairFailed)).
/// - Delaunay flip repair fails and the fallback simplex-payload snapshot fails
///   ([`DelaunayRepairFailedWithSimplexDataSnapshot`](DelaunayizeError::DelaunayRepairFailedWithSimplexDataSnapshot)).
/// - Delaunay flip repair fails **and** the fallback vertex-set rebuild also
///   fails
///   ([`DelaunayRepairFailedWithRebuild`](DelaunayizeError::DelaunayRepairFailedWithRebuild)).
/// - Delaunay flip repair fails, fallback rebuild succeeds, and simplex-payload
///   restoration fails
///   ([`DelaunayRepairFailedWithRebuildRestore`](DelaunayizeError::DelaunayRepairFailedWithRebuildRestore)).
/// - Fallback rebuild was enabled but the pre-repair simplex-payload snapshot
///   fails before repair starts
///   ([`FallbackSimplexDataSnapshotFailed`](DelaunayizeError::FallbackSimplexDataSnapshotFailed)).
///
/// When topology repair fails and fallback rebuild succeeds, this function
/// returns `Ok` with `used_fallback_rebuild = true` and
/// `topology_repair.succeeded = false`; the topology pass is not reported as
/// successful merely because rebuild recovered the workflow.
///
/// If any repair stage fails and no fallback rebuild succeeds, the
/// triangulation is restored to its pre-call state before the error is
/// returned. A successful direct repair or successful fallback rebuild commits
/// the resulting triangulation.
///
/// The `*WithRebuild` variants preserve both errors as typed fields so
/// consumers can inspect both typed errors;
/// [`Error::source`](std::error::Error::source) exposes the primary repair error.
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
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
/// assert!(outcome.topology_repair.succeeded);
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::result_large_err,
    reason = "DelaunayizeError preserves typed source and rebuild_error values on the *WithRebuild variants (no boxing) so callers can pattern-match both errors while Error::source exposes the primary repair error; this is a cold error path."
)]
pub fn delaunayize_by_flips<K, U, V, const D: usize>(
    dt: &mut DelaunayTriangulation<K, U, V, D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayizeOutcome<U, V, D>, DelaunayizeError>
where
    K: ExactPredicates<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let pl_config = PlManifoldRepairConfig {
        max_iterations: config.topology_max_iterations,
        max_simplices_removed: config.topology_max_simplices_removed,
    };
    let mut transaction =
        DelaunayRollbackTransaction::begin(dt, DelaunaySpatialIndexRollback::Restore);

    // Step 1: PL-manifold topology repair.
    let fallback_snapshot = if config.fallback_rebuild {
        let tds = &transaction.delaunay_mut().as_triangulation().tds;
        Some(
            snapshot_rebuild_state(tds)
                .map_err(|source| DelaunayizeError::FallbackSimplexDataSnapshotFailed { source })?,
        )
    } else {
        None
    };

    let global_topology = transaction.delaunay_mut().global_topology();
    let topology_result = {
        let delaunay = transaction.delaunay_mut();
        repair_pl_manifold_topology(delaunay.tds_mut_for_repair(), global_topology, &pl_config)
    };
    let topology_stats = match topology_result {
        Ok(stats) => stats,
        // Topology repair failed but fallback is enabled — try rebuilding.
        Err(topo_err) if config.fallback_rebuild => {
            let Some(fallback_snapshot) = fallback_snapshot else {
                transaction.rollback();
                return Err(topo_err.into());
            };
            let fallback_result = {
                let kernel = &transaction.delaunay_mut().as_triangulation().kernel;
                rebuild_preserving_data(kernel, &fallback_snapshot)
            };
            match fallback_result {
                Ok(rebuilt) => {
                    *transaction.delaunay_mut() = rebuilt;
                    transaction.commit();
                    return Ok(DelaunayizeOutcome {
                        topology_repair: PlManifoldRepairStats::default(),
                        delaunay_repair: DelaunayRepairStats::default(),
                        used_fallback_rebuild: true,
                    });
                }
                Err(fallback_error) => {
                    transaction.rollback();
                    return Err(topology_rebuild_error(topo_err, fallback_error));
                }
            }
        }
        Err(topo_err) => {
            transaction.rollback();
            return Err(topo_err.into());
        }
    };

    // Step 2: Flip-based Delaunay repair.
    // This is rebuild input only; rollback remains owned by `transaction`.
    let pre_delaunay_fallback_snapshot = if config.fallback_rebuild {
        let tds = &transaction.delaunay_mut().as_triangulation().tds;
        Some(snapshot_rebuild_state(tds))
    } else {
        None
    };
    #[cfg(test)]
    let delaunay_result = if test_hooks::force_delaunay_repair_failure_enabled() {
        Err(test_hooks::synthetic_repair_error())
    } else {
        run_configured_delaunay_repair(transaction.delaunay_mut(), config)
    };
    #[cfg(not(test))]
    let delaunay_result = run_configured_delaunay_repair(transaction.delaunay_mut(), config);

    finish_delaunayize_after_delaunay_repair(
        transaction,
        topology_stats,
        pre_delaunay_fallback_snapshot,
        delaunay_result,
    )
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::geometry::point::Point;
    use crate::tds::{TdsError, VertexKey};
    use crate::try_vertices_from_points;
    use crate::vertex;
    use crate::{DelaunayTriangulationBuilder, TriangulationConstructionError};
    use slotmap::KeyData;
    use std::assert_matches;
    use std::error::Error as StdError;
    use uuid::Uuid;

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

    /// Builds the canonical D-simplex vertex set used by fallback rebuild tests.
    fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut points = Vec::with_capacity(D + 1);
        points.push(Point::try_new([0.0; D]).expect("finite point coordinates"));
        for axis in 0..D {
            let mut coords = [0.0; D];
            coords[axis] = 1.0;
            points.push(Point::try_new(coords).expect("finite point coordinates"));
        }
        try_vertices_from_points(&points).expect("finite point coordinates")
    }

    fn insert_duplicate_simplex_copies<const D: usize>(
        dt: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
        copies: usize,
    ) {
        let duplicate_vertices = {
            let (_, existing_simplex) = dt.simplices().next().unwrap();
            existing_simplex.vertices().to_vec()
        };

        for _ in 0..copies {
            dt.tri
                .tds
                .insert_simplex_bypassing_topology_checks_for_test(
                    Simplex::try_new_with_data(duplicate_vertices.clone(), None).unwrap(),
                )
                .unwrap();
        }
    }

    /// Builds two tetrahedra sharing an edge but no facet.
    fn make_boundary_ridge_multiplicity_tds() -> Tds<(), (), 3> {
        let mut tds = Tds::empty();
        let shared_v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let shared_v1 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let tet1_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.1, 1.0, 0.2]).unwrap())
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.3, 1.3]).unwrap())
            .unwrap();
        let tet2_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.4, -1.1, 0.7]).unwrap())
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.6, 0.2, -1.4]).unwrap())
            .unwrap();

        for tet in [
            [shared_v0, shared_v1, tet1_v2, tet1_v3],
            [shared_v0, shared_v1, tet2_v2, tet2_v3],
        ] {
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![tet[0], tet[1], tet[2], tet[3]], None).unwrap(),
            )
            .unwrap();
        }

        tds
    }

    /// Builds a cone over a triangulated torus so targeted repair must mutate first.
    fn make_cone_on_torus_tds() -> Tds<(), (), 3> {
        const N: usize = 3;
        const M: usize = 3;

        let mut tds = Tds::empty();
        let mut grid: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in grid.iter_mut().enumerate() {
            for (j, vertex_key) in row.iter_mut().enumerate() {
                let i_f: f64 = u32::try_from(i).unwrap().into();
                let j_f: f64 = u32::try_from(j).unwrap().into();
                *vertex_key = tds
                    .insert_vertex_with_mapping(vertex!([i_f, j_f, 0.0]).unwrap())
                    .unwrap();
            }
        }
        let apex = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 1.0]).unwrap())
            .unwrap();

        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;
                let v00 = grid[i][j];
                let v10 = grid[i1][j];
                let v01 = grid[i][j1];
                let v11 = grid[i1][j1];
                for tri in [[v00, v10, v01], [v10, v11, v01]] {
                    tds.insert_simplex_with_mapping(
                        Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2], apex], None)
                            .unwrap(),
                    )
                    .unwrap();
                }
            }
        }

        tds
    }

    /// Creates a Delaunay wrapper whose TDS has a boundary-ridge violation.
    fn boundary_ridge_multiplicity_dt() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> {
        let vertices = unit_simplex_vertices::<3>();
        let mut dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        *dt.tds_mut() = make_boundary_ridge_multiplicity_tds();
        dt
    }

    /// Creates a Delaunay wrapper whose TDS has a vertex-link violation.
    fn cone_on_torus_dt() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> {
        let vertices = unit_simplex_vertices::<3>();
        let mut dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        *dt.tds_mut() = make_cone_on_torus_tds();
        dt
    }

    /// Returns simplex UUIDs in stable order for rollback identity checks.
    fn sorted_simplex_uuids<K, U, V, const D: usize>(
        dt: &DelaunayTriangulation<K, U, V, D>,
    ) -> Vec<Uuid> {
        let mut uuids = dt
            .simplices()
            .map(|(_, simplex)| simplex.uuid())
            .collect::<Vec<_>>();
        uuids.sort_unstable();
        uuids
    }

    /// Forces topology repair to fail on duplicate simplices, then checks fallback rebuild.
    fn assert_topology_repair_fallback_rebuilds_duplicate_simplex<const D: usize>()
    where
        AdaptiveKernel<f64>: ExactPredicates<D, Scalar = f64>,
    {
        init_tracing();
        let vertices = unit_simplex_vertices::<D>();
        let mut dt: DelaunayTriangulation<_, (), (), D> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        insert_duplicate_simplex_copies(&mut dt, 2);

        let outcome = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                topology_max_simplices_removed: 0,
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap();

        assert!(outcome.used_fallback_rebuild);
        assert!(
            !outcome.topology_repair.succeeded,
            "fallback rebuild should not mark the failed topology repair as succeeded"
        );
        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn topology_repair_failure_rolls_back_partial_mutation() {
        init_tracing();
        let vertices = unit_simplex_vertices::<2>();
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        insert_duplicate_simplex_copies(&mut dt, 3);

        let before_simplex_count = dt.number_of_simplices();
        let before_simplex_uuids = sorted_simplex_uuids(&dt);

        let err = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                topology_max_iterations: 1,
                topology_max_simplices_removed: usize::MAX,
                fallback_rebuild: false,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap_err();

        assert_matches!(
            err,
            DelaunayizeError::TopologyRepairFailed {
                source: PlManifoldRepairError::BudgetExhausted { .. }
            }
        );
        assert_eq!(dt.number_of_simplices(), before_simplex_count);

        assert_eq!(sorted_simplex_uuids(&dt), before_simplex_uuids);
    }

    #[test]
    fn targeted_topology_repair_failure_rolls_back_partial_mutation() {
        init_tracing();
        let mut dt = cone_on_torus_dt();
        let before_simplex_count = dt.number_of_simplices();
        let before_simplex_uuids = sorted_simplex_uuids(&dt);

        let err = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                topology_max_iterations: 1,
                topology_max_simplices_removed: usize::MAX,
                fallback_rebuild: false,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap_err();

        assert_matches!(
            err,
            DelaunayizeError::TopologyRepairFailed {
                source: PlManifoldRepairError::TargetedBudgetExhausted { .. }
            }
        );
        assert_eq!(dt.number_of_simplices(), before_simplex_count);

        assert_eq!(sorted_simplex_uuids(&dt), before_simplex_uuids);
    }

    #[test]
    fn delaunay_repair_failure_rolls_back_successful_topology_repair() {
        init_tracing();
        let mut dt = cone_on_torus_dt();
        let before_simplex_count = dt.number_of_simplices();
        let before_simplex_uuids = sorted_simplex_uuids(&dt);
        let _guard = ForceDelaunayRepairFailureGuard::enable();

        let err = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap_err();

        assert_matches!(
            err,
            DelaunayizeError::DelaunayRepairFailed {
                source: DelaunayRepairError::NonConvergent { .. }
            }
        );
        assert_eq!(dt.number_of_simplices(), before_simplex_count);
        assert_eq!(sorted_simplex_uuids(&dt), before_simplex_uuids);
    }

    struct ForceDelaunayRepairFailureGuard {
        prior: bool,
    }

    impl ForceDelaunayRepairFailureGuard {
        /// Enables synthetic Delaunay repair failure until the guard is dropped.
        fn enable() -> Self {
            Self {
                prior: test_hooks::set_force_delaunay_repair_failure(true),
            }
        }
    }

    impl Drop for ForceDelaunayRepairFailureGuard {
        fn drop(&mut self) {
            test_hooks::restore_force_delaunay_repair_failure(self.prior);
        }
    }

    // =============================================================================
    // CONFIG DEFAULT TESTS
    // =============================================================================

    #[test]
    fn test_config_defaults() {
        init_tracing();
        let config = DelaunayizeConfig::default();
        assert_eq!(config.topology_max_iterations, 64);
        assert_eq!(config.topology_max_simplices_removed, 10_000);
        assert!(!config.fallback_rebuild);
        assert!(config.delaunay_max_flips.is_none());
    }

    // =============================================================================
    // SUCCESS PATH TESTS
    // =============================================================================

    #[test]
    fn test_already_delaunay_3d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
        assert!(outcome.topology_repair.succeeded);
        assert!(!outcome.used_fallback_rebuild);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_already_delaunay_2d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
        assert!(outcome.topology_repair.succeeded);
        assert!(dt.validate().is_ok());
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
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
        assert!(outcome.topology_repair.succeeded);
        assert_eq!(outcome.topology_repair.simplices_removed, 0);
        assert!(!outcome.used_fallback_rebuild);
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

        let err = simplex_vertex_uuids(&tds, &simplex).unwrap_err();

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
        assert!(err.to_string().contains("no repair attempted"));
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn test_repair_snapshot_error_source() {
        let source = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let snapshot_error = SimplexValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
        };
        let err = DelaunayizeError::DelaunayRepairFailedWithSimplexDataSnapshot {
            source: source.clone(),
            snapshot_error: snapshot_error.clone(),
        };

        assert_eq!(
            err,
            DelaunayizeError::DelaunayRepairFailedWithSimplexDataSnapshot {
                source: source.clone(),
                snapshot_error,
            }
        );
        assert!(
            err.to_string()
                .contains("fallback simplex-data snapshot failed")
        );
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn delaunay_repair_snapshot_failure_rolls_back_transaction() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let before_simplex_count = dt.number_of_simplices();
        let before_simplex_uuids = sorted_simplex_uuids(&dt);
        let snapshot_error = SimplexValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
        };
        let transaction =
            DelaunayRollbackTransaction::begin(&mut dt, DelaunaySpatialIndexRollback::Restore);

        let err = finish_delaunayize_after_delaunay_repair(
            transaction,
            PlManifoldRepairStats {
                succeeded: true,
                ..PlManifoldRepairStats::default()
            },
            Some(Err(snapshot_error.clone())),
            Err(test_hooks::synthetic_repair_error()),
        )
        .unwrap_err();

        assert_matches!(
            err,
            DelaunayizeError::DelaunayRepairFailedWithSimplexDataSnapshot {
                source: DelaunayRepairError::NonConvergent { .. },
                snapshot_error: observed_snapshot_error,
            } if observed_snapshot_error == snapshot_error
        );
        assert_eq!(dt.number_of_simplices(), before_simplex_count);
        assert_eq!(sorted_simplex_uuids(&dt), before_simplex_uuids);
    }

    #[test]
    fn test_restore_error_sources() {
        let topology_source = PlManifoldRepairError::NoProgress {
            over_shared_facets: 2,
            iterations: 3,
            simplices_removed: 4,
        };
        let delaunay_source = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let restore_error = SimplexDataRestoreError::SimplexIdentity {
            source: SimplexValidationError::VertexKeyNotFound {
                key: VertexKey::from(KeyData::from_ffi(0xBAD)),
            },
        };

        let topology_err = DelaunayizeError::TopologyRepairFailedWithRebuildRestore {
            source: topology_source.clone(),
            restore_error: restore_error.clone(),
        };
        let delaunay_err = DelaunayizeError::DelaunayRepairFailedWithRebuildRestore {
            source: delaunay_source.clone(),
            restore_error,
        };

        assert!(
            topology_err
                .to_string()
                .contains("simplex-data restore failed")
        );
        assert_eq!(
            StdError::source(&topology_err).unwrap().to_string(),
            topology_source.to_string()
        );
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
    fn test_topology_rebuild_error_mapping() {
        let source = PlManifoldRepairError::NoProgress {
            over_shared_facets: 2,
            iterations: 3,
            simplices_removed: 4,
        };
        let rebuild_error = construction_error();
        let restore_error = SimplexDataRestoreError::SimplexIdentity {
            source: SimplexValidationError::VertexKeyNotFound {
                key: VertexKey::from(KeyData::from_ffi(0xBAD)),
            },
        };

        let rebuild_err = topology_rebuild_error(
            source.clone(),
            FallbackRebuildError::Construction {
                source: rebuild_error.clone(),
            },
        );
        assert_eq!(
            rebuild_err,
            DelaunayizeError::TopologyRepairFailedWithRebuild {
                source: source.clone(),
                rebuild_error,
            }
        );
        assert!(
            rebuild_err
                .to_string()
                .contains("fallback rebuild also failed")
        );

        let restore_err = topology_rebuild_error(
            source.clone(),
            FallbackRebuildError::Restore {
                source: restore_error.clone(),
            },
        );
        assert_eq!(
            restore_err,
            DelaunayizeError::TopologyRepairFailedWithRebuildRestore {
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
    fn test_fallback_disabled_by_default() {
        init_tracing();
        let config = DelaunayizeConfig::default();
        assert!(!config.fallback_rebuild);
    }

    #[test]
    fn test_fallback_enabled_on_valid_triangulation() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Fallback should not be triggered on a valid triangulation.
        let config = DelaunayizeConfig {
            fallback_rebuild: true,
            ..DelaunayizeConfig::default()
        };
        let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
        assert!(!outcome.used_fallback_rebuild);
    }

    #[test]
    fn targeted_topology_repair_fallback_rebuilds_after_budget_failure() {
        init_tracing();
        let mut dt = boundary_ridge_multiplicity_dt();
        let vertex_count = dt.number_of_vertices();

        let outcome = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                topology_max_simplices_removed: 0,
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap();

        assert!(outcome.used_fallback_rebuild);
        assert!(!outcome.topology_repair.succeeded);
        assert_eq!(outcome.topology_repair.simplices_removed, 0);
        assert_eq!(dt.number_of_vertices(), vertex_count);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn delaunay_repair_fallback_rebuilds_after_unsupported_dimension() {
        init_tracing();
        let vertices = [vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let mut dt: DelaunayTriangulation<_, (), (), 1> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap();

        assert!(outcome.topology_repair.succeeded);
        assert_eq!(outcome.topology_repair.simplices_removed, 0);
        assert_eq!(outcome.delaunay_repair.facets_checked, 0);
        assert_eq!(outcome.delaunay_repair.flips_performed, 0);
        assert_eq!(outcome.delaunay_repair.max_queue_len, 0);
        assert!(outcome.used_fallback_rebuild);
        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert!(dt.tds().is_valid().is_ok());
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
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let _guard = ForceDelaunayRepairFailureGuard::enable();
        let outcome = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                fallback_rebuild: true,
                ..DelaunayizeConfig::default()
            },
        )
        .unwrap();

        assert!(outcome.topology_repair.succeeded);
        assert!(outcome.used_fallback_rebuild);
        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn topology_repair_fallback_stats_failed() {
        assert_topology_repair_fallback_rebuilds_duplicate_simplex::<2>();
    }

    macro_rules! gen_topology_repair_fallback_tests {
        ($($dim:literal),* $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<topology_repair_fallback_rebuilds_duplicate_simplex_ $dim d>]() {
                        assert_topology_repair_fallback_rebuilds_duplicate_simplex::<$dim>();
                    }
                )*
            }
        };
    }

    gen_topology_repair_fallback_tests!(3, 4, 5);

    #[test]
    fn test_rebuild_restores_simplex_data() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        let original_simplex_key = dt.simplices().next().unwrap().0;
        dt.set_simplex_data(original_simplex_key, Some(42)).unwrap();

        let tds = &dt.as_triangulation().tds;
        let snapshot = snapshot_rebuild_state(tds).unwrap();

        let rebuilt = rebuild_preserving_data(&dt.as_triangulation().kernel, &snapshot).unwrap();

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

        let snapshot = snapshot_rebuild_state(&tds).unwrap();
        let kernel = AdaptiveKernel::new();
        let mut rebuilt: DelaunayTriangulation<_, (), i32, 2> =
            DelaunayTriangulation::try_with_kernel(&kernel, snapshot.vertices()).unwrap();

        restore_simplex_data(&mut rebuilt, snapshot.simplex_data()).unwrap();

        let (_, rebuilt_simplex) = rebuilt.simplices().next().unwrap();
        assert_eq!(rebuilt_simplex.data(), None);
        assert!(rebuilt.validate().is_ok());
    }

    // =============================================================================
    // DETERMINISM TESTS
    // =============================================================================

    #[test]
    fn test_deterministic_repeated_runs() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];

        let config = DelaunayizeConfig::default();

        let mut dt1: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let outcome1 = delaunayize_by_flips(&mut dt1, config).unwrap();

        let mut dt2: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let outcome2 = delaunayize_by_flips(&mut dt2, config).unwrap();

        assert_eq!(
            outcome1.topology_repair.simplices_removed,
            outcome2.topology_repair.simplices_removed
        );
        assert_eq!(
            outcome1.used_fallback_rebuild,
            outcome2.used_fallback_rebuild
        );
    }
}
