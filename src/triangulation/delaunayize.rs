//! End-to-end "repair then delaunayize" workflow.
//!
//! This module provides `delaunayize_by_flips`, a single public entrypoint that
//! takes an existing [`DelaunayTriangulation`], performs bounded deterministic
//! topology repair toward
//! [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold),
//! and then applies
//! standard flip-based Delaunay repair.
//!
//! # Workflow
//!
//! 1. **PL-manifold topology repair** — removes cells that cause facet
//!    over-sharing (codimension-1 facet degree > 2) using a bounded,
//!    deterministic pruning algorithm.
//! 2. **Delaunay flip repair** — runs k=2/k=3 bistellar flips to restore the
//!    empty-circumsphere property.
//! 3. **Optional fallback rebuild** — if configured and both repair passes
//!    fail, rebuilds the triangulation from its vertex set.
//!
//! # Example
//!
//! ```rust
//! use delaunay::prelude::triangulation::delaunayize::*;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let mut dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
//! assert!(outcome.topology_repair.succeeded);
//! ```
//!
//! # Explicitly Deferred
//!
//! - Dedicated targeted repair stages for boundary-ridge multiplicity,
//!   ridge-link manifoldness, and vertex-link manifoldness (#304).

#![forbid(unsafe_code)]

// Re-export outcome/error field types so users can name the public contract
// without reaching into lower-level modules.
pub use crate::core::algorithms::flips::{DelaunayRepairError, DelaunayRepairStats};
pub use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairError, PlManifoldRepairStats,
};
pub use crate::core::cell::CellValidationError;
pub use crate::triangulation::delaunay::DelaunayTriangulationConstructionError;

use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairConfig, repair_facet_oversharing,
};
use crate::core::cell::Cell;
use crate::core::collections::{Entry, FastHashMap, Uuid};
use crate::core::tds::{CellKey, Tds};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{ExactPredicates, Kernel};
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::triangulation::delaunay::{DelaunayRepairHeuristicConfig, DelaunayTriangulation};
use thiserror::Error;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the [`delaunayize_by_flips`] workflow.
///
/// # Defaults
///
/// - `topology_max_iterations`: 64
/// - `topology_max_cells_removed`: 10,000
/// - `fallback_rebuild`: false
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::delaunayize::DelaunayizeConfig;
///
/// let config = DelaunayizeConfig::default();
/// assert_eq!(config.topology_max_iterations, 64);
/// assert_eq!(config.topology_max_cells_removed, 10_000);
/// assert!(!config.fallback_rebuild);
/// assert!(config.delaunay_max_flips.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelaunayizeConfig {
    /// Maximum number of topology-repair iterations.
    pub topology_max_iterations: usize,
    /// Maximum number of cells that may be removed during topology repair.
    pub topology_max_cells_removed: usize,
    /// If `true`, rebuild the triangulation from the vertex set when both
    /// topology repair and flip-based Delaunay repair fail.
    ///
    /// Cell-level user data (`V`) is restored for rebuilt cells whose sorted
    /// vertex UUID set matches exactly one original cell. Cells that change
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
            topology_max_cells_removed: 10_000,
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
/// use delaunay::prelude::triangulation::delaunayize::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let mut dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulation::new(&vertices).unwrap();
///
/// let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
/// assert!(outcome.topology_repair.succeeded);
/// assert!(!outcome.used_fallback_rebuild);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DelaunayizeOutcome<T, U, V, const D: usize> {
    /// Statistics from the PL-manifold topology repair pass.
    ///
    /// If topology repair fails but fallback rebuild succeeds, these remain the
    /// failed/default repair stats for the repair attempt. Use
    /// [`used_fallback_rebuild`](Self::used_fallback_rebuild) to distinguish
    /// successful rebuild recovery from direct topology repair success.
    pub topology_repair: PlManifoldRepairStats<T, U, V, D>,
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
/// - **Fallback cell-data recovery** failed while snapshotting or restoring
///   cell payloads after a repair failure.
///
/// # Orthogonality
///
/// The variants are mutually exclusive by failure mode:
/// - Topology repair, fallback not attempted -> [`TopologyRepairFailed`](Self::TopologyRepairFailed).
/// - Topology repair, fallback also failed   -> [`TopologyRepairFailedWithRebuild`](Self::TopologyRepairFailedWithRebuild).
/// - Topology repair, fallback rebuild succeeded but payload restore failed -> [`TopologyRepairFailedWithRebuildRestore`](Self::TopologyRepairFailedWithRebuildRestore).
/// - Delaunay repair, fallback not attempted -> [`DelaunayRepairFailed`](Self::DelaunayRepairFailed).
/// - Delaunay repair, fallback payload snapshot failed -> [`DelaunayRepairFailedWithCellDataSnapshot`](Self::DelaunayRepairFailedWithCellDataSnapshot).
/// - Delaunay repair, fallback also failed   -> [`DelaunayRepairFailedWithRebuild`](Self::DelaunayRepairFailedWithRebuild).
/// - Delaunay repair, fallback rebuild succeeded but payload restore failed -> [`DelaunayRepairFailedWithRebuildRestore`](Self::DelaunayRepairFailedWithRebuildRestore).
/// - Fallback was enabled, but the pre-repair payload snapshot failed -> [`FallbackCellDataSnapshotFailed`](Self::FallbackCellDataSnapshotFailed).
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
/// use delaunay::prelude::triangulation::delaunayize::*;
///
/// let err = DelaunayizeError::DelaunayRepairFailed {
///     source: DelaunayRepairError::PostconditionFailed {
///         message: "still non-Delaunay after repair".to_string(),
///     },
/// };
/// assert!(err.to_string().contains("Delaunay repair failed"));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
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
    /// succeeded, but cell-payload restoration from the rebuilt topology failed.
    #[error(
        "Topology repair failed ({source}); fallback rebuild succeeded but cell-data restore failed: {restore_error}"
    )]
    TopologyRepairFailedWithRebuildRestore {
        /// The underlying topology-repair error that triggered the fallback.
        #[source]
        source: PlManifoldRepairError,
        /// The cell-data restoration error from the rebuilt triangulation.
        restore_error: CellValidationError,
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
        "Delaunay repair failed ({source}); fallback cell-data snapshot failed: {snapshot_error}"
    )]
    DelaunayRepairFailedWithCellDataSnapshot {
        /// The underlying flip-repair error that triggered the fallback.
        #[source]
        source: DelaunayRepairError,
        /// The cell-data snapshot error from the current triangulation.
        snapshot_error: CellValidationError,
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
    /// but cell-payload restoration from the rebuilt topology failed.
    #[error(
        "Delaunay repair failed ({source}); fallback rebuild succeeded but cell-data restore failed: {restore_error}"
    )]
    DelaunayRepairFailedWithRebuildRestore {
        /// The underlying flip-repair error that triggered the fallback.
        #[source]
        source: DelaunayRepairError,
        /// The cell-data restoration error from the rebuilt triangulation.
        restore_error: CellValidationError,
    },

    /// Fallback rebuild was enabled, but the pre-repair cell-payload snapshot
    /// could not be collected from the input triangulation. No topology or
    /// Delaunay repair was attempted.
    #[error("Fallback cell-data snapshot failed before repair; no repair attempted: {source}")]
    FallbackCellDataSnapshotFailed {
        /// The cell-data snapshot error from the input triangulation.
        #[from]
        #[source]
        source: CellValidationError,
    },
}

// =============================================================================
// HELPERS
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
enum CellDataMatch<V> {
    Unique(Option<V>),
    Ambiguous,
}

type CellDataByVertexUuids<V> = FastHashMap<Vec<Uuid>, CellDataMatch<V>>;
type FallbackRebuildState<T, U, V, const D: usize> =
    (Vec<Vertex<T, U, D>>, CellDataByVertexUuids<V>);

/// Captures the fallback rebuild inputs from the current TDS, including typed
/// failure if any cell cannot resolve its vertex UUID identity.
fn snapshot_rebuild_state<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<FallbackRebuildState<T, U, V, D>, CellValidationError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let vertices = tds
        .vertices()
        .map(|(_, v)| Vertex::new_with_uuid(*v.point(), v.uuid(), v.data))
        .collect::<Vec<_>>();
    let cell_data = collect_cell_data(tds)?;
    Ok((vertices, cell_data))
}

/// Hashes cell payloads by sorted vertex UUIDs so fallback rebuilds can
/// recover payloads for cells whose vertex set survives unchanged.
fn collect_cell_data<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<CellDataByVertexUuids<V>, CellValidationError>
where
    U: DataType,
    V: DataType,
{
    let mut cell_data = FastHashMap::default();
    for (_, cell) in tds.cells() {
        let vertex_uuids = cell_vertex_uuids(tds, cell)?;
        match cell_data.entry(vertex_uuids) {
            Entry::Vacant(entry) => {
                entry.insert(CellDataMatch::Unique(cell.data().copied()));
            }
            Entry::Occupied(mut entry) => {
                entry.insert(CellDataMatch::Ambiguous);
            }
        }
    }
    Ok(cell_data)
}

/// Builds the order-independent cell identity used to match original and
/// rebuilt cells across fallback reconstruction.
fn cell_vertex_uuids<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell: &Cell<T, U, V, D>,
) -> Result<Vec<Uuid>, CellValidationError>
where
    U: DataType,
    V: DataType,
{
    let mut vertex_uuids = cell
        .vertex_uuid_iter(tds)
        .collect::<Result<Vec<_>, CellValidationError>>()?;
    vertex_uuids.sort_unstable();
    Ok(vertex_uuids)
}

/// Reattaches original cell payloads to rebuilt cells that retain the same
/// vertex UUID set after fallback reconstruction.
fn restore_cell_data<K, U, V, const D: usize>(
    rebuilt: &mut DelaunayTriangulation<K, U, V, D>,
    original_cell_data: &CellDataByVertexUuids<V>,
) -> Result<(), CellValidationError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    let mut assignments: Vec<(CellKey, V)> = Vec::new();
    for (cell_key, cell) in rebuilt.cells() {
        let vertex_uuids = cell_vertex_uuids(rebuilt.tds(), cell)?;
        let Some(CellDataMatch::Unique(Some(data))) = original_cell_data.get(&vertex_uuids) else {
            continue;
        };
        assignments.push((cell_key, *data));
    }

    for (cell_key, data) in assignments {
        rebuilt.set_cell_data(cell_key, Some(data));
    }

    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
enum FallbackRebuildError {
    #[error("fallback rebuild failed: {source}")]
    Construction {
        #[from]
        #[source]
        source: DelaunayTriangulationConstructionError,
    },
    #[error("fallback cell-data restore failed: {source}")]
    Restore {
        #[from]
        #[source]
        source: CellValidationError,
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
/// cell payloads whose vertex UUID signatures survive the rebuild unchanged.
fn rebuild_preserving_data<K, U, V, const D: usize>(
    kernel: &K,
    vertices: &[Vertex<K::Scalar, U, D>],
    original_cell_data: &CellDataByVertexUuids<V>,
) -> Result<DelaunayTriangulation<K, U, V, D>, FallbackRebuildError>
where
    K: Kernel<D> + ExactPredicates,
    U: DataType,
    V: DataType,
{
    let mut rebuilt = DelaunayTriangulation::with_kernel(kernel, vertices)?;
    restore_cell_data(&mut rebuilt, original_cell_data)?;
    Ok(rebuilt)
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
/// - Delaunay flip repair fails and no fallback rebuild was attempted
///   ([`DelaunayRepairFailed`](DelaunayizeError::DelaunayRepairFailed)).
/// - Delaunay flip repair fails **and** the fallback vertex-set rebuild also
///   fails
///   ([`DelaunayRepairFailedWithRebuild`](DelaunayizeError::DelaunayRepairFailedWithRebuild)).
///
/// When topology repair fails and fallback rebuild succeeds, this function
/// returns `Ok` with `used_fallback_rebuild = true` and
/// `topology_repair.succeeded = false`; the topology pass is not reported as
/// successful merely because rebuild recovered the workflow.
///
/// The `*WithRebuild` variants preserve both errors as typed fields so
/// consumers can inspect both typed errors;
/// [`Error::source`](std::error::Error::source) exposes the primary repair error.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::delaunayize::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let mut dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulation::new(&vertices).unwrap();
///
/// let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
/// assert!(outcome.topology_repair.succeeded);
/// ```
#[expect(
    clippy::result_large_err,
    reason = "DelaunayizeError preserves typed source and rebuild_error values on the *WithRebuild variants (no boxing) so callers can pattern-match both errors while Error::source exposes the primary repair error; this is a cold error path."
)]
pub fn delaunayize_by_flips<K, U, V, const D: usize>(
    dt: &mut DelaunayTriangulation<K, U, V, D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayizeOutcome<K::Scalar, U, V, D>, DelaunayizeError>
where
    K: Kernel<D> + ExactPredicates,
    U: DataType,
    V: DataType,
{
    // Snapshot vertex and cell data before repair so fallback rebuild can use it.
    let fallback_state: Option<FallbackRebuildState<K::Scalar, U, V, D>> =
        if config.fallback_rebuild {
            let tds = &dt.as_triangulation().tds;
            Some(snapshot_rebuild_state(tds)?)
        } else {
            None
        };

    // Step 1: PL-manifold topology repair (facet over-sharing).
    let pl_config = PlManifoldRepairConfig {
        max_iterations: config.topology_max_iterations,
        max_cells_removed: config.topology_max_cells_removed,
    };
    let topology_stats = match repair_facet_oversharing(dt.tds_mut_for_repair(), &pl_config) {
        Ok(stats) => stats,
        // Topology repair failed but fallback is enabled — try rebuilding.
        Err(topo_err) if let Some((ref verts, ref cell_data)) = fallback_state => {
            match rebuild_preserving_data(&dt.as_triangulation().kernel, verts, cell_data) {
                Ok(rebuilt) => {
                    *dt = rebuilt;
                    return Ok(DelaunayizeOutcome {
                        topology_repair: PlManifoldRepairStats::default(),
                        delaunay_repair: DelaunayRepairStats::default(),
                        used_fallback_rebuild: true,
                    });
                }
                Err(fallback_error) => {
                    return Err(topology_rebuild_error(topo_err, fallback_error));
                }
            }
        }
        Err(topo_err) => return Err(topo_err.into()),
    };

    // Step 2: Flip-based Delaunay repair.
    let delaunay_result = if let Some(max_flips) = config.delaunay_max_flips {
        dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig {
            max_flips: Some(max_flips),
            ..DelaunayRepairHeuristicConfig::default()
        })
        .map(|outcome| outcome.stats)
    } else {
        dt.repair_delaunay_with_flips()
    };

    match delaunay_result {
        Ok(delaunay_stats) => Ok(DelaunayizeOutcome {
            topology_repair: topology_stats,
            delaunay_repair: delaunay_stats,
            used_fallback_rebuild: false,
        }),
        Err(repair_err) => {
            if config.fallback_rebuild {
                // Step 3 (optional): rebuild from vertex set.
                let tds = &dt.as_triangulation().tds;
                let (vertices, cell_data) = match snapshot_rebuild_state(tds) {
                    Ok(state) => state,
                    Err(snapshot_error) => {
                        return Err(DelaunayizeError::DelaunayRepairFailedWithCellDataSnapshot {
                            source: repair_err,
                            snapshot_error,
                        });
                    }
                };

                match rebuild_preserving_data(&dt.as_triangulation().kernel, &vertices, &cell_data)
                {
                    Ok(rebuilt) => {
                        *dt = rebuilt;
                        // The rebuild succeeded — return stats reflecting the fallback.
                        Ok(DelaunayizeOutcome {
                            topology_repair: topology_stats,
                            delaunay_repair: DelaunayRepairStats::default(),
                            used_fallback_rebuild: true,
                        })
                    }
                    Err(fallback_error) => Err(delaunay_rebuild_error(repair_err, fallback_error)),
                }
            } else {
                Err(DelaunayizeError::from(repair_err))
            }
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{tds::VertexKey, triangulation::TriangulationConstructionError};
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::triangulation::builder::DelaunayTriangulationBuilder;
    use crate::vertex;
    use slotmap::KeyData;
    use std::error::Error as StdError;

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    fn init_tracing() {
        let _ = tracing_subscriber::fmt::try_init();
    }

    fn construction_error() -> DelaunayTriangulationConstructionError {
        DelaunayTriangulationConstructionError::Triangulation(
            TriangulationConstructionError::FailedToCreateCell {
                message: "synthetic rebuild degeneracy".to_string(),
            },
        )
    }

    // =============================================================================
    // CONFIG DEFAULT TESTS
    // =============================================================================

    #[test]
    fn test_config_defaults() {
        init_tracing();
        let config = DelaunayizeConfig::default();
        assert_eq!(config.topology_max_iterations, 64);
        assert_eq!(config.topology_max_cells_removed, 10_000);
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
        assert!(outcome.topology_repair.succeeded);
        assert!(!outcome.used_fallback_rebuild);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_already_delaunay_2d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
        assert!(outcome.topology_repair.succeeded);
        assert_eq!(outcome.topology_repair.cells_removed, 0);
        assert!(!outcome.used_fallback_rebuild);
    }

    // =============================================================================
    // ERROR PATH TESTS
    // =============================================================================

    #[test]
    fn test_collect_cell_data_missing_vertex() {
        let mut tds: Tds<f64, (), i32, 2> = Tds::empty();
        let vertex_keys: Vec<_> = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ]
        .iter()
        .map(|vertex| tds.insert_vertex_with_mapping(*vertex).unwrap())
        .collect();
        let missing = vertex_keys[0];
        let cell = Cell::new(vertex_keys, Some(7)).unwrap();
        let dangling_cell = cell.clone();
        let cell_key = tds.insert_cell_with_mapping(cell).unwrap();
        assert_eq!(tds.remove_cells_by_keys(&[cell_key]), 1);
        tds.remove_isolated_vertex(missing);
        tds.cells_mut().insert(dangling_cell);

        let err = collect_cell_data(&tds).unwrap_err();

        assert_eq!(err, CellValidationError::VertexKeyNotFound { key: missing });
    }

    #[test]
    fn test_snapshot_error_source() {
        let source = CellValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
        };
        let err = DelaunayizeError::FallbackCellDataSnapshotFailed {
            source: source.clone(),
        };

        assert_eq!(
            err,
            DelaunayizeError::FallbackCellDataSnapshotFailed {
                source: source.clone()
            }
        );
        assert!(
            err.to_string()
                .contains("Fallback cell-data snapshot failed")
        );
        assert!(err.to_string().contains("no repair attempted"));
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn test_repair_snapshot_error_source() {
        let source = DelaunayRepairError::PostconditionFailed {
            message: "synthetic postcondition".to_string(),
        };
        let snapshot_error = CellValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
        };
        let err = DelaunayizeError::DelaunayRepairFailedWithCellDataSnapshot {
            source: source.clone(),
            snapshot_error: snapshot_error.clone(),
        };

        assert_eq!(
            err,
            DelaunayizeError::DelaunayRepairFailedWithCellDataSnapshot {
                source: source.clone(),
                snapshot_error,
            }
        );
        assert!(
            err.to_string()
                .contains("fallback cell-data snapshot failed")
        );
        let error_source = StdError::source(&err).unwrap();
        assert_eq!(error_source.to_string(), source.to_string());
    }

    #[test]
    fn test_restore_error_sources() {
        let topology_source = PlManifoldRepairError::NoProgress {
            over_shared_facets: 2,
            iterations: 3,
            cells_removed: 4,
        };
        let delaunay_source = DelaunayRepairError::PostconditionFailed {
            message: "synthetic postcondition".to_string(),
        };
        let restore_error = CellValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
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
                .contains("cell-data restore failed")
        );
        assert_eq!(
            StdError::source(&topology_err).unwrap().to_string(),
            topology_source.to_string()
        );
        assert!(
            delaunay_err
                .to_string()
                .contains("cell-data restore failed")
        );
        assert_eq!(
            StdError::source(&delaunay_err).unwrap().to_string(),
            delaunay_source.to_string()
        );
    }

    #[test]
    fn test_topology_rebuild_error_mapping() {
        let source = PlManifoldRepairError::NoProgress {
            over_shared_facets: 2,
            iterations: 3,
            cells_removed: 4,
        };
        let rebuild_error = construction_error();
        let restore_error = CellValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
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
                .contains("fallback rebuild succeeded but cell-data restore failed")
        );
    }

    #[test]
    fn test_delaunay_rebuild_error_mapping() {
        let source = DelaunayRepairError::PostconditionFailed {
            message: "synthetic postcondition".to_string(),
        };
        let rebuild_error = construction_error();
        let restore_error = CellValidationError::VertexKeyNotFound {
            key: VertexKey::from(KeyData::from_ffi(0xBAD)),
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
                .contains("fallback rebuild succeeded but cell-data restore failed")
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Fallback should not be triggered on a valid triangulation.
        let config = DelaunayizeConfig {
            fallback_rebuild: true,
            ..DelaunayizeConfig::default()
        };
        let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
        assert!(!outcome.used_fallback_rebuild);
    }

    #[test]
    fn topology_repair_fallback_stats_failed() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let (_, existing_cell) = dt.cells().next().unwrap();
        let duplicate_vertices = existing_cell.vertices().to_vec();
        dt.tri
            .tds
            .insert_cell_with_mapping(Cell::new(duplicate_vertices.clone(), None).unwrap())
            .unwrap();
        dt.tri
            .tds
            .insert_cell_with_mapping(Cell::new(duplicate_vertices, None).unwrap())
            .unwrap();

        let outcome = delaunayize_by_flips(
            &mut dt,
            DelaunayizeConfig {
                topology_max_cells_removed: 0,
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
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_rebuild_restores_cell_data() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        let original_cell_key = dt.cells().next().unwrap().0;
        dt.set_cell_data(original_cell_key, Some(42));

        let tds = &dt.as_triangulation().tds;
        let vertices: Vec<_> = tds
            .vertices()
            .map(|(_, v)| Vertex::new_with_uuid(*v.point(), v.uuid(), v.data))
            .collect();
        let cell_data = collect_cell_data(tds).unwrap();

        let rebuilt =
            rebuild_preserving_data(&dt.as_triangulation().kernel, &vertices, &cell_data).unwrap();

        let (_, rebuilt_cell) = rebuilt.cells().next().unwrap();
        assert_eq!(rebuilt_cell.data(), Some(&42));
        assert!(rebuilt.validate().is_ok());
    }

    #[test]
    fn test_rebuild_drops_ambiguous_data() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), i32, 2> = Tds::empty();
        let vertex_keys: Vec<_> = vertices
            .iter()
            .map(|vertex| tds.insert_vertex_with_mapping(*vertex).unwrap())
            .collect();

        let duplicate_a = Cell::new(vertex_keys.clone(), Some(42)).unwrap();
        let duplicate_b = Cell::new(vertex_keys, Some(42)).unwrap();
        tds.insert_cell_with_mapping(duplicate_a).unwrap();
        tds.insert_cell_with_mapping(duplicate_b).unwrap();

        let rebuild_vertices: Vec<_> = tds
            .vertices()
            .map(|(_, v)| Vertex::new_with_uuid(*v.point(), v.uuid(), v.data))
            .collect();
        let cell_data = collect_cell_data(&tds).unwrap();
        let kernel = AdaptiveKernel::new();
        let mut rebuilt: DelaunayTriangulation<_, (), i32, 2> =
            DelaunayTriangulation::with_kernel(&kernel, &rebuild_vertices).unwrap();

        restore_cell_data(&mut rebuilt, &cell_data).unwrap();

        let (_, rebuilt_cell) = rebuilt.cells().next().unwrap();
        assert_eq!(rebuilt_cell.data(), None);
        assert!(rebuilt.validate().is_ok());
    }

    // =============================================================================
    // DETERMINISM TESTS
    // =============================================================================

    #[test]
    fn test_deterministic_repeated_runs() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];

        let config = DelaunayizeConfig::default();

        let mut dt1: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let outcome1 = delaunayize_by_flips(&mut dt1, config).unwrap();

        let mut dt2: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let outcome2 = delaunayize_by_flips(&mut dt2, config).unwrap();

        assert_eq!(
            outcome1.topology_repair.cells_removed,
            outcome2.topology_repair.cells_removed
        );
        assert_eq!(
            outcome1.used_fallback_rebuild,
            outcome2.used_fallback_rebuild
        );
    }
}
