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
//! - Stronger cell-payload preservation in the fallback rebuild path (#305).

#![forbid(unsafe_code)]

// Re-export outcome field types so users can name them without reaching into
// internal modules.
pub use crate::core::algorithms::flips::DelaunayRepairStats;
pub use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairError, PlManifoldRepairStats,
};

use crate::core::algorithms::flips::DelaunayRepairError;
use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairConfig, repair_facet_oversharing,
};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{ExactPredicates, Kernel};
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::triangulation::delaunay::{
    DelaunayRepairHeuristicConfig, DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
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
/// use delaunay::triangulation::delaunayize::DelaunayizeConfig;
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
    /// **Warning:** the fallback rebuild discards cell-level user data (`V`).
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
/// There are two orthogonal failure modes:
/// - **Topology repair** failed (step 1).
/// - **Delaunay repair** failed (step 2), with optional context about a
///   fallback rebuild attempt.
///
/// # Orthogonality
///
/// The four variants are mutually exclusive by failure mode:
/// - Topology repair, fallback not attempted -> [`TopologyRepairFailed`](Self::TopologyRepairFailed).
/// - Topology repair, fallback also failed   -> [`TopologyRepairFailedWithRebuild`](Self::TopologyRepairFailedWithRebuild).
/// - Delaunay repair, fallback not attempted -> [`DelaunayRepairFailed`](Self::DelaunayRepairFailed).
/// - Delaunay repair, fallback also failed   -> [`DelaunayRepairFailedWithRebuild`](Self::DelaunayRepairFailedWithRebuild).
///
/// The `*WithRebuild` variants preserve **both** the primary repair error and
/// the secondary construction error as typed values (no stringification),
/// so consumers can traverse the full diagnostic chain via pattern
/// matching or [`Error::source`](std::error::Error::source).
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunayize::DelaunayizeError;
/// use delaunay::core::algorithms::flips::DelaunayRepairError;
/// use delaunay::core::triangulation::TopologyGuarantee;
///
/// let err = DelaunayizeError::DelaunayRepairFailed {
///     source: DelaunayRepairError::InvalidTopology {
///         required: TopologyGuarantee::PLManifold,
///         found: TopologyGuarantee::Pseudomanifold,
///         message: "requires manifold",
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
/// The `*WithRebuild` variants preserve both errors as typed fields so
/// consumers can walk the full diagnostic chain.
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
    reason = "DelaunayizeError deliberately preserves typed source and rebuild_error values on the *WithRebuild variants (no boxing) so consumers get the full diagnostic chain via pattern matching or Error::source; this is a cold error path."
)]
pub fn delaunayize_by_flips<K, U, V, const D: usize>(
    dt: &mut DelaunayTriangulation<K, U, V, D>,
    config: DelaunayizeConfig,
) -> Result<DelaunayizeOutcome<K::Scalar, U, V, D>, DelaunayizeError>
where
    K: Kernel<D> + ExactPredicates,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Snapshot vertex set before repair so fallback rebuild can use it.
    let fallback_vertices: Option<Vec<Vertex<K::Scalar, U, D>>> = if config.fallback_rebuild {
        Some(
            dt.as_triangulation()
                .tds
                .vertices()
                .map(|(_, v)| Vertex::new_with_uuid(*v.point(), v.uuid(), v.data))
                .collect(),
        )
    } else {
        None
    };

    // Step 1: PL-manifold topology repair (facet over-sharing).
    let pl_config = PlManifoldRepairConfig {
        max_iterations: config.topology_max_iterations,
        max_cells_removed: config.topology_max_cells_removed,
    };
    let topology_stats =
        match repair_facet_oversharing(&mut dt.as_triangulation_mut().tds, &pl_config) {
            Ok(stats) => stats,
            // Topology repair failed but fallback is enabled — try rebuilding.
            Err(topo_err) if let Some(ref verts) = fallback_vertices => {
                match DelaunayTriangulation::with_kernel(&dt.as_triangulation().kernel, verts) {
                    Ok(rebuilt) => {
                        *dt = rebuilt;
                        return Ok(DelaunayizeOutcome {
                            topology_repair: PlManifoldRepairStats::default(),
                            delaunay_repair: DelaunayRepairStats::default(),
                            used_fallback_rebuild: true,
                        });
                    }
                    Err(rebuild_err) => {
                        return Err(DelaunayizeError::TopologyRepairFailedWithRebuild {
                            source: topo_err,
                            rebuild_error: rebuild_err,
                        });
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
                let vertices: Vec<Vertex<K::Scalar, U, D>> = dt
                    .as_triangulation()
                    .tds
                    .vertices()
                    .map(|(_, v)| Vertex::new_with_uuid(*v.point(), v.uuid(), v.data))
                    .collect();

                match DelaunayTriangulation::with_kernel(&dt.as_triangulation().kernel, &vertices) {
                    Ok(rebuilt) => {
                        *dt = rebuilt;
                        // The rebuild succeeded — return stats reflecting the fallback.
                        Ok(DelaunayizeOutcome {
                            topology_repair: topology_stats,
                            delaunay_repair: DelaunayRepairStats::default(),
                            used_fallback_rebuild: true,
                        })
                    }
                    Err(rebuild_err) => Err(DelaunayizeError::DelaunayRepairFailedWithRebuild {
                        source: repair_err,
                        rebuild_error: rebuild_err,
                    }),
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
    use crate::vertex;

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    fn init_tracing() {
        let _ = tracing_subscriber::fmt::try_init();
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
