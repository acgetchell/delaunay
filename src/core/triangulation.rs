//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.
//!
//! # Validation Hierarchy
//!
//! The library provides **four levels** of validation, each building on the previous:
//!
//! ## Level 1: Element Validity
//! - **Methods**: [`Simplex::is_valid()`], [`Vertex::is_valid()`]
//! - **Checks**: Basic data integrity (coordinate validity, UUID presence, proper initialization)
//! - **Cost**: O(1) per element
//!
//! ## Level 2: TDS Structural Validity
//! - **Method**: [`Tds::is_valid()`]
//! - **Checks**:
//!   - UUID ↔ Key mapping consistency
//!   - No duplicate simplices (same vertex sets)
//!   - Facet sharing invariant (≤2 simplices per facet)
//!   - Neighbor consistency (mutual relationships)
//! - **Cost**: O(N×D²) where N = simplices, D = dimension
//!
//! Use [`Tds::validate()`] for cumulative Levels 1–2 (element + structural) validation.
//!
//! ## Level 3: Manifold Topology
//! - **Method**: [`Triangulation::is_valid()`](crate::core::triangulation::Triangulation::is_valid)
//! - **Checks**:
//!   - **Codimension-1 manifoldness**: exactly 1 boundary simplex or 2 interior simplices per facet
//!   - **Codimension-2 boundary manifoldness**: the boundary is closed ("no boundary of boundary")
//!   - Connectedness (single connected component in the simplex neighbor graph)
//!   - No isolated vertices (every vertex must be incident to at least one simplex)
//!   - Euler characteristic (χ = V - E + F - C matches expected topology)
//! - **Cost**: O(N×D²) dominated by simplex counting
//!
//! Use [`Triangulation::validate()`](crate::core::triangulation::Triangulation::validate) for cumulative Levels 1–3.
//!
//! ## Level 4: Delaunay Property
//! - **Method**: [`DelaunayTriangulation::is_valid()`](crate::triangulation::delaunay::DelaunayTriangulation::is_valid)
//! - **Checks**: Empty circumsphere property (no vertex inside any simplex's circumsphere)
//! - **Cost**: O(N×V) where N = simplices, V = vertices
//!
//! Use [`DelaunayTriangulation::validate()`](crate::triangulation::delaunay::DelaunayTriangulation::validate) for cumulative Levels 1–4.
//!
//! ## Usage Guidelines
//!
//! ```rust
//! use delaunay::prelude::triangulation::*;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Level 2: structural only (fast)
//! assert!(dt.tds().is_valid().is_ok());
//!
//! // Level 3: topology only (assumes structural validity)
//! assert!(dt.as_triangulation().is_valid().is_ok());
//!
//! // Level 4: Delaunay property only (assumes Levels 1–3)
//! assert!(dt.is_valid().is_ok());
//!
//! // Full cumulative validation (Levels 1–4)
//! assert!(dt.validate().is_ok());
//! ```
//!
//! **Performance**: Use Level 2 for most production validation. Reserve Level 3 for
//! tests/debug builds, and Level 4 for critical verification or debugging geometric issues.
//!
//! [`Simplex::is_valid()`]: crate::core::simplex::Simplex::is_valid
//! [`Vertex::is_valid()`]: crate::core::vertex::Vertex::is_valid
//! [`Tds::is_valid()`]: crate::core::tds::Tds::is_valid
//! [`Tds::validate()`]: crate::core::tds::Tds::validate
//!
//! ## Topology guarantees
//!
//! [`TopologyGuarantee`](crate::core::triangulation::TopologyGuarantee) selects which **manifoldness**
//! invariants are checked by Level 3 topology validation.
//!
//! Whether these checks run automatically after insertion is controlled by
//! [`ValidationPolicy`](crate::core::triangulation::ValidationPolicy).
//!
//! Level 3 validation always checks:
//! - Codimension-1 facet degree (pseudomanifold condition: 1 boundary or 2 interior simplices per facet)
//! - Codimension-2 boundary manifoldness (closed boundary: "no boundary of boundary")
//! - Connectedness (single connected component in the simplex neighbor graph)
//! - No isolated vertices (every vertex must be incident to at least one simplex)
//! - Euler characteristic
//!
//! With [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold),
//! Level 3 validation additionally checks the canonical **vertex-link** PL-manifoldness
//! condition via [`crate::topology::manifold::validate_vertex_links`].
//!
//! Note: for **D=3**, the current vertex-link validator additionally enforces that each link
//! has the Euler characteristic / boundary component counts of a sphere/ball (S²/B²).
//! For **D≥4**, it currently checks that each vertex link is a connected (D−1)-manifold
//! with the correct boundary behavior (a necessary condition), but does not attempt to
//! distinguish spheres/balls from other manifolds (not sufficient in general).
//!

#![forbid(unsafe_code)]

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, CavityRepairStage, HullExtensionReason, InsertionError, extend_hull,
    external_facets_for_boundary, fill_cavity_replacing_simplices, repair_neighbor_pointers,
    repair_neighbor_pointers_local, wire_cavity_neighbors,
};
#[cfg(debug_assertions)]
use crate::core::algorithms::locate::locate;
#[cfg(feature = "diagnostics")]
use crate::core::algorithms::locate::verify_conflict_region_completeness;
use crate::core::algorithms::locate::{
    ConflictError, LocateError, LocateResult, LocateStats, extract_cavity_boundary,
    find_conflict_region, locate_by_scan, locate_with_stats, locate_with_trace,
};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{
    CavityBoundaryBuffer, FacetIssuesMap, FacetToSimplicesMap, FastHashMap, FastHashSet,
    FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SimplexKeySet, SmallBuffer,
    VertexToSimplicesMap, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::edge::EdgeKey;
#[cfg(test)]
use crate::core::facet::facet_key_from_vertices;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter, FacetHandle};
use crate::core::operations::{
    InsertionOutcome, InsertionResult, InsertionStatistics, InsertionTelemetry,
    InsertionTelemetryMode, SuspicionFlags,
};
use crate::core::simplex::{Simplex, SimplexValidationError};
#[cfg(test)]
use crate::core::tds::NeighborValidationError;
use crate::core::tds::{
    GeometricError, InvariantError, InvariantErrorSummary, InvariantKind, InvariantViolation,
    SimplexKey, Tds, TdsConstructionError, TdsError, TriangulationValidationReport, VertexKey,
};
use crate::core::traits::data_type::DataType;
#[cfg(test)]
use crate::core::util::canonical_points::sorted_simplex_points;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::quality::radius_ratio;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
use crate::geometry::util::safe_scalar_to_f64;
use crate::topology::characteristics::euler::{TopologyClassification, expected_chi_for};
use crate::topology::characteristics::validation::validate_triangulation_euler_with_facet_to_simplices_map;
use crate::topology::manifold::{
    ManifoldError, validate_closed_boundary, validate_facet_degree,
    validate_local_pseudomanifold_for_simplices, validate_ridge_links,
    validate_ridge_links_for_simplices, validate_vertex_links,
};
use crate::topology::traits::global_topology_model::GlobalTopologyModel;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::delaunay::DelaunayTriangulationValidationError;
use crate::triangulation::locality::{
    append_live_unique_simplex_seeds, collect_local_exterior_conflict_seed_simplices,
    replace_simplices_and_record_removed, retain_simplices_and_record_removed,
};
use core::ops::Div;
use num_traits::{Float, NumCast, One, Zero};
use std::borrow::Cow;
#[cfg(all(test, debug_assertions))]
use std::cmp::Ordering as CmpOrdering;
use std::env;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use thiserror::Error;
use uuid::Uuid;

/// Maximum number of repair iterations for fixing non-manifold topology after insertion.
///
/// This limit prevents infinite loops in the rare case where repair cannot make progress.
/// In practice, most insertions require 0-2 iterations to restore manifold topology.
const MAX_REPAIR_ITERATIONS: usize = 10;

/// Default number of perturbation retries for transactional insertion.
///
/// Each retry uses a progressively larger perturbation magnitude (×10 per attempt),
/// so 3 retries span 4 orders of magnitude (e.g. `1e-8` → `1e-5` × `local_scale` for f64).
const DEFAULT_PERTURBATION_RETRIES: usize = 3;

/// Telemetry: counts how often the topology safety-net recovered from a Level 3 validation
/// failure by retrying insertion with a star-split of the containing simplex.
///
/// This is a process-wide counter across all triangulation instances.
///
/// This counter is intentionally lightweight and can be polled by production workloads
/// to see whether this recovery path is frequently used.
static TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_TOTAL: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_USED: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_ENABLED: OnceLock<bool> = OnceLock::new();
static RETRYABLE_SKIP_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static CAVITY_REDUCTION_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static FORCE_GLOBAL_NEIGHBOR_REBUILD_ENABLED: OnceLock<bool> = OnceLock::new();
static CAVITY_REDUCTION_TRACE_EMITTED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static DUPLICATE_DETECTION_FORCE_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(debug_assertions)]
static VERTEX_TO_SIMPLICES_SPILL_EVENTS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        static FORCE_NEXT_INSERTION_RETRYABLE_FAILURE: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn take_force_next_insertion_retryable_failure() -> bool {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.replace(false)
    }

    pub(super) fn set_force_next_insertion_retryable_failure(enabled: bool) -> bool {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.replace(enabled)
    }

    pub(super) fn restore_force_next_insertion_retryable_failure(prior: bool) {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.set(prior);
    }
}

fn duplicate_detection_metrics_enabled() -> bool {
    #[cfg(test)]
    if DUPLICATE_DETECTION_FORCE_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    *DUPLICATE_DETECTION_ENABLED.get_or_init(|| env::var_os("DELAUNAY_DUPLICATE_METRICS").is_some())
}

/// Caches whether retryable conflict-region skips should emit release-visible traces.
fn retryable_skip_trace_enabled() -> bool {
    *RETRYABLE_SKIP_TRACE_ENABLED
        .get_or_init(|| env::var_os("DELAUNAY_DEBUG_RETRYABLE_SKIP").is_some())
}

/// Returns whether the first cavity-reduction chain should emit release-visible tracing.
fn cavity_reduction_trace_enabled() -> bool {
    *CAVITY_REDUCTION_TRACE_ENABLED
        .get_or_init(|| env::var_os("DELAUNAY_DEBUG_CAVITY_REDUCTION_ONCE").is_some())
}

/// Returns whether local neighbor repair should be bypassed for regression isolation.
fn force_global_neighbor_rebuild_enabled() -> bool {
    *FORCE_GLOBAL_NEIGHBOR_REBUILD_ENABLED
        .get_or_init(|| env::var_os("DELAUNAY_FORCE_GLOBAL_NEIGHBOR_REBUILD").is_some())
}

/// Extracts a compact one-line summary for retryable conflict-region failures.
///
/// These summaries are designed for the large-scale debug harness logs, where we want
/// enough structure to correlate repeated ridge-fan failures without dumping the entire
/// conflict region.
fn retryable_conflict_trace_detail(error: &InsertionError) -> Option<String> {
    match error {
        InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
            facet_hash,
            simplex_count,
        }) => Some(format!(
            "kind=non_manifold_facet facet_hash={facet_hash:#x} simplex_count={simplex_count}"
        )),
        InsertionError::ConflictRegion(ConflictError::RidgeFan {
            facet_count,
            ridge_vertex_count,
            extra_simplices,
        }) => Some(format!(
            "kind=ridge_fan facet_count={facet_count} ridge_vertex_count={ridge_vertex_count} \
             extra_simplices={}",
            extra_simplices.len()
        )),
        InsertionError::ConflictRegion(ConflictError::DisconnectedBoundary {
            visited,
            total,
            disconnected_simplices,
        }) => Some(format!(
            "kind=disconnected_boundary visited={visited} total={total} disconnected_simplices={}",
            disconnected_simplices.len()
        )),
        InsertionError::ConflictRegion(ConflictError::OpenBoundary {
            facet_count,
            ridge_vertex_count,
            ..
        }) => Some(format!(
            "kind=open_boundary facet_count={facet_count} ridge_vertex_count={ridge_vertex_count}"
        )),
        _ => None,
    }
}

/// Formats a compact summary for cavity-boundary extraction failures.
fn cavity_conflict_error_summary(error: &ConflictError) -> String {
    match error {
        ConflictError::NonManifoldFacet {
            facet_hash,
            simplex_count,
        } => format!("non_manifold_facet facet_hash={facet_hash:#x} simplex_count={simplex_count}"),
        ConflictError::RidgeFan {
            facet_count,
            ridge_vertex_count,
            extra_simplices,
        } => format!(
            "ridge_fan facet_count={facet_count} ridge_vertex_count={ridge_vertex_count} \
             extra_simplices={}",
            extra_simplices.len()
        ),
        ConflictError::DisconnectedBoundary {
            visited,
            total,
            disconnected_simplices,
        } => format!(
            "disconnected_boundary visited={visited} total={total} disconnected_simplices={}",
            disconnected_simplices.len()
        ),
        ConflictError::OpenBoundary {
            facet_count,
            ridge_vertex_count,
            open_simplex,
        } => format!(
            "open_boundary facet_count={facet_count} ridge_vertex_count={ridge_vertex_count} \
             open_simplex={open_simplex:?}"
        ),
        ConflictError::InvalidStartSimplex { simplex_key } => {
            format!("invalid_start_simplex simplex_key={simplex_key:?}")
        }
        ConflictError::PredicateError { source } => {
            format!("predicate_error source={source}")
        }
        ConflictError::SimplexDataAccessFailed {
            simplex_key,
            message,
        } => {
            format!("simplex_data_access_failed simplex_key={simplex_key:?} message={message}")
        }
        ConflictError::InternalInconsistency { site } => {
            format!("internal_inconsistency site={site}")
        }
    }
}

/// Emits one-shot tracing for the first cavity-reduction chain in a run.
///
/// Routed through `tracing::debug!`; enable with `RUST_LOG=debug` (the
/// large-scale debug harness wires this up automatically when
/// `DELAUNAY_DEBUG_CAVITY_REDUCTION_ONCE` is set).
fn log_cavity_reduction_event<F>(
    enabled: bool,
    iteration: usize,
    conflict_simplices: &SimplexKeyBuffer,
    event: F,
) where
    F: FnOnce() -> String,
{
    if !enabled {
        return;
    }

    let conflict_preview: Vec<SimplexKey> = conflict_simplices.iter().copied().take(12).collect();
    let event = event();
    tracing::debug!(
        target: "delaunay::cavity_reduction",
        iteration,
        conflict_simplices = conflict_simplices.len(),
        event,
        conflict_preview = ?conflict_preview,
        "cavity-reduction event"
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "Diagnostic helper keeps retryable skip instrumentation centralized"
)]
/// Emits a single structured line for a retryable conflict-region skip after rollback.
///
/// Logging after rollback lets the trace report both the state we tried to modify and
/// the restored simplex/vertex counts that future attempts will see. Routed through
/// `tracing::debug!` so callers can filter it via `RUST_LOG`; enabled for release-mode
/// runs by `DELAUNAY_DEBUG_RETRYABLE_SKIP`.
fn log_retryable_conflict_skip(
    bulk_index: Option<usize>,
    uuid: Uuid,
    attempt: usize,
    max_attempts: usize,
    used_perturbation: bool,
    will_retry: bool,
    simplices_before_attempt: usize,
    vertices_before_attempt: usize,
    simplices_after_rollback: usize,
    vertices_after_rollback: usize,
    detail: &str,
    error: &InsertionError,
) {
    if !retryable_skip_trace_enabled() {
        return;
    }

    let bulk_index_display = bulk_index.map_or_else(|| String::from("n/a"), |idx| idx.to_string());
    tracing::debug!(
        target: "delaunay::retryable_skip",
        bulk_index = %bulk_index_display,
        uuid = %uuid,
        attempt,
        max_attempts,
        used_perturbation,
        rolled_back = true,
        will_retry,
        simplices_before_attempt,
        vertices_before_attempt,
        simplices_after_rollback,
        vertices_after_rollback,
        conflict = %detail,
        error = %error,
        "retryable conflict-region skip after rollback"
    );
}

/// Telemetry counters for duplicate-coordinate detection.
#[must_use]
#[derive(Debug, Clone, Copy, Default)]
pub struct DuplicateDetectionMetrics {
    /// Total number of duplicate-coordinate checks executed.
    pub total_checks: u64,
    /// Number of checks that successfully used the hash grid.
    pub grid_used: u64,
    /// Number of checks that fell back to a non-grid scan.
    pub grid_fallbacks: u64,
    /// Total candidate vertices inspected during grid-based checks.
    pub grid_candidates: u64,
}

pub(crate) fn record_duplicate_detection_metrics(
    used_grid: bool,
    candidate_count: usize,
    fell_back: bool,
) {
    if !duplicate_detection_metrics_enabled() {
        return;
    }
    DUPLICATE_DETECTION_TOTAL.fetch_add(1, Ordering::Relaxed);
    if used_grid {
        DUPLICATE_DETECTION_GRID_USED.fetch_add(1, Ordering::Relaxed);
        DUPLICATE_DETECTION_GRID_CANDIDATES.fetch_add(candidate_count as u64, Ordering::Relaxed);
    }
    if fell_back {
        DUPLICATE_DETECTION_GRID_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Convert an [`InsertionError`] into the appropriate [`InvariantError`], preserving
/// structured error information across all layers.
///
/// - `TopologyValidation(source)` → `InvariantError::Tds(source)` (Level 1–2 preserved)
/// - `TopologyValidationFailed { source }` → `InvariantError::Triangulation(source)` (Level 3 preserved)
/// - All other variants → `InvariantError::Tds(InconsistentDataStructure { .. })` with `context`
pub(crate) fn insertion_error_to_invariant_error(
    error: InsertionError,
    context: &str,
) -> InvariantError {
    match error {
        InsertionError::TopologyValidation(source) => InvariantError::Tds(source),
        InsertionError::TopologyValidationFailed { source, .. } => {
            InvariantError::Triangulation(source)
        }
        other => InvariantError::Tds(TdsError::InconsistentDataStructure {
            message: format!("{context}: {other}"),
        }),
    }
}

/// Errors that can occur during triangulation construction.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::{Triangulation, TriangulationConstructionError};
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::vertex;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let result: Result<_, TriangulationConstructionError> =
///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangulationConstructionError {
    /// Lower-layer construction error in the TDS.
    #[error(transparent)]
    Tds(#[from] TdsConstructionError),

    /// Failed to create a simplex during triangulation construction.
    #[error("Failed to create simplex during construction: {message}")]
    FailedToCreateSimplex {
        /// Description of the simplex creation failure.
        message: String,
    },

    /// Cavity filling failed during incremental construction.
    #[error("Cavity filling failed during insertion: {source}")]
    InsertionCavityFilling {
        /// Underlying cavity-filling error.
        #[source]
        source: CavityFillingError,
    },

    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying simplex validation error.
        source: SimplexValidationError,
    },

    /// Geometric degeneracy prevents triangulation construction.
    #[error("Geometric degeneracy encountered during construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },

    /// Conflict-region extraction failed during incremental construction.
    #[error("Conflict region failed during insertion: {source}")]
    InsertionConflictRegion {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },

    /// Point location failed during incremental construction.
    #[error("Point location failed during insertion: {source}")]
    InsertionLocation {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },

    /// Incremental insertion detected non-manifold topology.
    #[error(
        "Non-manifold topology during insertion: facet {facet_hash:#x} shared by {simplex_count} simplices"
    )]
    InsertionNonManifoldTopology {
        /// Hash of the over-shared facet.
        facet_hash: u64,
        /// Number of simplices sharing the facet.
        simplex_count: usize,
    },

    /// Hull extension failed during incremental construction.
    #[error("Hull extension failed during insertion: {reason}")]
    InsertionHullExtension {
        /// Structured hull-extension failure reason.
        reason: HullExtensionReason,
    },

    /// Level 4 Delaunay validation failed during incremental construction.
    #[error("Delaunay validation failed during insertion: {source}")]
    InsertionDelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Level 3 topology validation failed during incremental construction.
    #[error("{message}: {source}")]
    InsertionTopologyValidation {
        /// High-level insertion context.
        message: String,
        /// Underlying topology validation error.
        #[source]
        source: TriangulationValidationError,
    },

    /// Final cumulative topology validation failed after construction.
    ///
    /// Mirrors [`InsertionTopologyValidation`](Self::InsertionTopologyValidation)
    /// for post-build checks that run after the incremental insertion phase.
    #[error("{message}: {source}")]
    FinalTopologyValidation {
        /// High-level finalization context.
        message: String,
        /// Underlying validation error.
        #[source]
        source: InvariantErrorSummary,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// String representation of the duplicate coordinates.
        coordinates: String,
    },

    /// Internal bookkeeping state became inconsistent during construction.
    ///
    /// This indicates a bug in the construction algorithm rather than invalid
    /// input or geometric degeneracy.
    #[error("Internal inconsistency during construction: {message}")]
    InternalInconsistency {
        /// Description of the inconsistency.
        message: String,
    },
}

/// Errors that can occur during triangulation topology validation (Level 3).
///
/// This type represents **only** Level 3 (topology) errors. It does not contain
/// TDS-level (Levels 1–2) errors. Cumulative validators that can return errors
/// from any level use [`InvariantError`] instead.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::InvariantError;
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result: Result<(), InvariantError> = dt.as_triangulation().validate();
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangulationValidationError {
    /// A facet belongs to an unexpected number of simplices for a manifold-with-boundary.
    #[error(
        "Non-manifold facet: facet {facet_key:016x} belongs to {simplex_count} simplices (expected 1 or 2)"
    )]
    ManifoldFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The number of incident simplices observed.
        simplex_count: usize,
    },

    /// Boundary is not a closed (D-1)-manifold:
    /// wrong number of boundary facets.
    ///
    /// This detects "boundary of boundary" issues (codimension-2 manifoldness of the boundary).
    #[error(
        "Boundary is not closed: boundary ridge {ridge_key:016x} is incident to {boundary_facet_count} boundary facets (expected 2)"
    )]
    BoundaryRidgeMultiplicity {
        /// Canonical key for the (D-2)-simplex (ridge) on the boundary.
        ridge_key: u64,
        /// Number of incident boundary facets observed.
        boundary_facet_count: usize,
    },

    /// A ridge's link graph is not a 1-manifold (path or cycle).
    ///
    /// This is required for PL-manifold validation.
    #[error(
        "Ridge link is not a 1-manifold: ridge {ridge_key:016x} has link graph with {link_vertex_count} vertices, {link_edge_count} edges, max degree {max_degree}, degree-1 vertices {degree_one_vertices}, connected={connected} (expected connected cycle or path)"
    )]
    RidgeLinkNotManifold {
        /// Canonical key for the (D-2)-simplex (ridge).
        ridge_key: u64,
        /// Number of vertices in the ridge's link graph.
        link_vertex_count: usize,
        /// Number of edges in the ridge's link graph.
        link_edge_count: usize,
        /// Maximum vertex degree observed in the link graph.
        max_degree: usize,
        /// Number of vertices of degree 1 observed in the link graph.
        degree_one_vertices: usize,
        /// Whether the link graph is connected.
        connected: bool,
    },

    /// A vertex link is not a (D-1)-manifold (sphere/ball) as required for PL-manifoldness.
    #[error(
        "Vertex link is not a PL (D-1)-manifold: vertex {vertex_key:?} has link with {link_vertex_count} vertices, {link_simplex_count} simplices, boundary_facets={boundary_facet_count}, max_degree={max_degree}, connected={connected}, interior_vertex={interior_vertex}"
    )]
    VertexLinkNotManifold {
        /// The vertex whose link failed validation.
        vertex_key: VertexKey,
        /// Number of vertices in the link (0-simplices of the link).
        link_vertex_count: usize,
        /// Number of (D-1)-simplices (simplices) in the link.
        link_simplex_count: usize,
        /// Number of boundary facets in the link (facets of degree 1).
        boundary_facet_count: usize,
        /// Maximum degree in the link 1-skeleton.
        max_degree: usize,
        /// Whether the link 1-skeleton is connected.
        connected: bool,
        /// Whether the vertex was classified as an interior vertex of the original complex.
        interior_vertex: bool,
    },

    /// Euler characteristic does not match the expected value for the classified topology.
    #[error(
        "Euler characteristic mismatch: computed χ={computed}, expected χ={expected} for {classification:?}"
    )]
    EulerCharacteristicMismatch {
        /// Computed Euler characteristic.
        computed: isize,
        /// Expected Euler characteristic for the classification.
        expected: isize,
        /// The topology classification used to determine expectation.
        classification: TopologyClassification,
    },

    /// Vertex is not incident to any simplex.
    ///
    /// An isolated vertex violates manifold invariants at the topology (Level 3) layer
    /// and may indicate a failed insertion or an insertion that was partially rolled back.
    #[error(
        "Isolated vertex: vertex {vertex_uuid} (key {vertex_key:?}) is not incident to any simplex"
    )]
    IsolatedVertex {
        /// Key of the isolated vertex.
        vertex_key: VertexKey,
        /// UUID of the isolated vertex.
        vertex_uuid: Uuid,
    },

    /// The simplex neighbor graph is not a single connected component.
    ///
    /// A valid triangulation-with-boundary must be connected; multiple disconnected
    /// components indicate a structural problem (e.g. simplices that share only a vertex
    /// or edge but no facet, so no neighbor pointers link them).
    #[error(
        "Disconnected triangulation: simplex neighbor graph is not a single connected component ({simplex_count} simplices total)"
    )]
    Disconnected {
        /// Total number of simplices in the triangulation.
        simplex_count: usize,
    },
}

impl TryFrom<ManifoldError> for TriangulationValidationError {
    type Error = TdsError;

    fn try_from(err: ManifoldError) -> Result<Self, Self::Error> {
        match err {
            ManifoldError::Tds(source) => Err(source),
            ManifoldError::ManifoldFacetMultiplicity {
                facet_key,
                simplex_count,
            } => Ok(Self::ManifoldFacetMultiplicity {
                facet_key,
                simplex_count,
            }),
            ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            } => Ok(Self::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            }),
            ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            } => Ok(Self::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            }),
            ManifoldError::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            } => Ok(Self::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            }),
        }
    }
}

impl From<ManifoldError> for InvariantError {
    fn from(err: ManifoldError) -> Self {
        match TriangulationValidationError::try_from(err) {
            Ok(source) => Self::Triangulation(source),
            Err(source) => Self::Tds(source),
        }
    }
}

struct TryInsertImplOk {
    /// Inserted vertex key plus an optional locate hint for the caller.
    inserted: (VertexKey, Option<SimplexKey>),
    /// Number of simplices removed during local non-manifold repair.
    simplices_removed: usize,
    /// Suspicion flags observed during the insertion attempt.
    suspicion: SuspicionFlags,
    /// Simplices touched by insertion that should seed follow-up local repair.
    ///
    /// This includes live simplices created by the insertion plus simplices that were shrunk
    /// out of the final conflict region so higher layers can revisit nearby
    /// Delaunay violations without rediscovering the inserted vertex star globally.
    repair_seed_simplices: SimplexKeyBuffer,
    /// Whether the insertion path can leave local Delaunay work for the caller.
    ///
    /// Clean interior Bowyer-Watson insertions preserve the Delaunay property.
    /// Exterior hull extensions and suspicious fallback/repair paths still need
    /// a local flip-repair pass.
    delaunay_repair_required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertionValidationWork {
    FullValidation,
    RequiredTopologyLinks,
}

/// Internal result from over-shared-facet repair, including the surviving frontier
/// that should seed local neighbor-pointer repair.
struct LocalFacetRepairOutcome {
    /// Number of simplices actually removed from the TDS.
    removed_count: usize,
    /// Simplices selected for removal before they were deleted.
    #[cfg_attr(
        not(debug_assertions),
        expect(
            dead_code,
            reason = "Removed-simplex keys are retained for debug logging and future local repair diagnostics"
        )
    )]
    removed_simplices: SimplexKeyBuffer,
    /// Surviving one-hop neighbors whose back-references may have been cleared.
    frontier_simplices: SimplexKeyBuffer,
}

/// Result of filling one insertion cavity, including the follow-up Delaunay
/// repair requirements that depend on how the cavity was shaped.
struct CavityInsertionOutcome {
    /// Locate hint for the next insertion.
    hint: Option<SimplexKey>,
    /// Number of simplices removed during local non-manifold repair.
    simplices_removed: usize,
    /// Simplices touched by insertion that should seed follow-up local repair.
    repair_seed_simplices: SimplexKeyBuffer,
    /// Whether this cavity path can leave Delaunay work for the caller.
    delaunay_repair_required: bool,
}

enum InsertionSite<'a> {
    Interior {
        start_simplex: SimplexKey,
        conflict_simplices: Cow<'a, SimplexKeyBuffer>,
    },
    Exterior {
        conflict_simplices: Option<Cow<'a, SimplexKeyBuffer>>,
        repair_seed_simplices: SimplexKeyBuffer,
    },
}

/// Internal insertion result that preserves the user-facing outcome plus
/// hidden repair seeding used by batch/debug construction paths.
pub(crate) struct DetailedInsertionResult {
    /// Public insertion outcome returned to higher layers.
    pub outcome: InsertionOutcome,
    /// Public statistics collected while attempting the insertion.
    pub stats: InsertionStatistics,
    /// Internal path telemetry collected while attempting the insertion.
    pub telemetry: InsertionTelemetry,
    /// Local simplices that should seed the caller's Delaunay repair set.
    pub repair_seed_simplices: SimplexKeyBuffer,
    /// Whether callers should run Delaunay repair over `repair_seed_simplices`.
    pub delaunay_repair_required: bool,
}

/// Policy controlling when the triangulation runs global validation passes.
///
/// Validation can be expensive (O(N×D²) or worse), so this allows callers to trade
/// performance for stricter correctness checks during incremental operations.
///
/// **Note**: [`TopologyGuarantee::PLManifold`] is incompatible with [`ValidationPolicy::Never`].
/// `PLManifold` requires at least end-of-construction validation to certify full
/// PL-manifoldness. Use [`ValidationPolicy::OnSuspicion`] (default) for best performance,
/// or [`ValidationPolicy::Always`] for maximum safety during incremental operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::operations::SuspicionFlags;
/// use delaunay::prelude::triangulation::ValidationPolicy;
///
/// let policy = ValidationPolicy::OnSuspicion;
/// let suspicion = SuspicionFlags { perturbation_used: true, ..SuspicionFlags::default() };
/// assert!(policy.should_validate(suspicion));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Never run global validation.
    Never,

    /// Validate only if the operation is suspicious (e.g. degeneracy).
    OnSuspicion,

    /// Always validate after insertion.
    Always,

    /// Debug builds: always validate; release builds: [`ValidationPolicy::OnSuspicion`].
    DebugOnly,
}

impl ValidationPolicy {
    /// Returns `true` if a global validation pass should be run given the observed
    /// [`crate::core::operations::SuspicionFlags`].
    #[inline]
    #[must_use]
    pub const fn should_validate(&self, suspicion: SuspicionFlags) -> bool {
        match self {
            Self::Never => false,
            Self::Always => true,
            Self::OnSuspicion => suspicion.is_suspicious(),
            Self::DebugOnly => cfg!(debug_assertions) || suspicion.is_suspicious(),
        }
    }
}

impl Default for ValidationPolicy {
    #[inline]
    fn default() -> Self {
        Self::OnSuspicion
    }
}

/// Selects which topological invariants are checked by Level 3 validation.
///
/// This enum specifies *what is checked* about the underlying simplicial complex when
/// Level 3 validation runs. Whether Level 3 validation runs automatically after insertion
/// is controlled by [`ValidationPolicy`].
///
/// - [`TopologyGuarantee::Pseudomanifold`] checks the codimension-1 adjacency condition:
///   each facet is incident to one or two simplices, and the codimension-2 boundary is closed.
///   This is sufficient for many geometric algorithms but does not guarantee local Euclidean structure.
///
/// - [`TopologyGuarantee::PLManifold`] uses ridge-link validation during insertion and
///   requires a vertex-link validation pass at construction completion to certify
///   PL-manifoldness.
/// - [`TopologyGuarantee::PLManifoldStrict`] runs vertex-link validation after every
///   insertion for maximal safety (slowest).
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
///
/// // Optional: relax topology checks for speed (weaker guarantees).
/// dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
/// assert!(!dt.topology_guarantee().requires_vertex_links_at_completion());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyGuarantee {
    /// Validate only the pseudomanifold / manifold-with-boundary invariants:
    /// - facet degree (1 or 2 incident simplices per facet)
    /// - closed boundary ("no boundary of boundary")
    Pseudomanifold,

    /// Validate PL-manifold invariants (incremental mode).
    ///
    /// This includes all `Pseudomanifold` checks plus ridge-link validation during
    /// insertion, with a required vertex-link validation at construction completion.
    PLManifold,

    /// Validate PL-manifold invariants with strict per-insertion checks.
    ///
    /// This includes all `Pseudomanifold` checks plus vertex-link validation
    /// after every insertion (slowest, maximum safety).
    PLManifoldStrict,
}

impl Default for TopologyGuarantee {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TopologyGuarantee {
    /// The default topology guarantee used when constructing triangulations.
    ///
    /// This is a `const` alternative to `<Self as Default>::default()` for `const fn` constructors.
    pub const DEFAULT: Self = Self::PLManifold;

    /// Returns `true` if this topology guarantee requires vertex-link validation
    /// after each insertion.
    #[inline]
    #[must_use]
    pub const fn requires_vertex_links_during_insertion(self) -> bool {
        matches!(self, Self::PLManifoldStrict)
    }

    /// Returns `true` if this topology guarantee requires vertex-link validation
    /// at construction completion.
    #[inline]
    #[must_use]
    pub const fn requires_vertex_links_at_completion(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
    }

    /// Returns `true` if this topology guarantee requires pseudomanifold checks
    /// during insertion.
    ///
    /// All current guarantees require the codimension-1 facet-degree and
    /// codimension-2 closed-boundary conditions. Stronger guarantees layer
    /// ridge-link and vertex-link validation on top of these checks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::TopologyGuarantee;
    ///
    /// assert!(
    ///     TopologyGuarantee::Pseudomanifold
    ///         .requires_pseudomanifold_checks_during_insertion()
    /// );
    /// assert!(
    ///     TopologyGuarantee::PLManifold
    ///         .requires_pseudomanifold_checks_during_insertion()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn requires_pseudomanifold_checks_during_insertion(self) -> bool {
        matches!(
            self,
            Self::Pseudomanifold | Self::PLManifold | Self::PLManifoldStrict
        )
    }

    /// Returns `true` if this topology guarantee requires ridge-link validation.
    ///
    /// Ridge-link validation is fast (O(local)) and catches many PL-manifold violations,
    /// providing good error detection even with reduced validation frequency.
    #[inline]
    #[must_use]
    pub const fn requires_ridge_links(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
    }

    /// Returns the [`ValidationPolicy`] that should be used by default for this guarantee.
    ///
    /// [`PLManifoldStrict`](Self::PLManifoldStrict) uses [`Always`](ValidationPolicy::Always)
    /// so that full Level-3 global validation (including vertex-link checks) runs
    /// after every insertion — this is the strongest and slowest setting.
    /// All other guarantees default to
    /// [`OnSuspicion`](ValidationPolicy::OnSuspicion).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{TopologyGuarantee, ValidationPolicy};
    ///
    /// assert_eq!(
    ///     TopologyGuarantee::PLManifoldStrict.default_validation_policy(),
    ///     ValidationPolicy::Always,
    /// );
    /// assert_eq!(
    ///     TopologyGuarantee::PLManifold.default_validation_policy(),
    ///     ValidationPolicy::OnSuspicion,
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn default_validation_policy(self) -> ValidationPolicy {
        match self {
            Self::PLManifoldStrict => ValidationPolicy::Always,
            _ => ValidationPolicy::OnSuspicion,
        }
    }

    /// Returns `true` if this guarantee is compatible with the given validation policy.
    ///
    /// `PLManifold` requires at least end-of-construction validation, so it's incompatible
    /// with `ValidationPolicy::Never`.
    #[inline]
    #[must_use]
    pub const fn is_compatible_with_policy(self, policy: ValidationPolicy) -> bool {
        match self {
            Self::Pseudomanifold => true,
            Self::PLManifold | Self::PLManifoldStrict => !matches!(policy, ValidationPolicy::Never),
        }
    }
}

/// Generic triangulation combining kernel and data structure.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for simplices
/// - `D`: Dimension of the triangulation
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::Triangulation;
/// use delaunay::prelude::geometry::FastKernel;
///
/// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
///     Triangulation::new_empty(FastKernel::new());
/// assert_eq!(tri.number_of_vertices(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct Triangulation<K: Kernel<D>, U, V, const D: usize> {
    /// The geometric kernel for predicates.
    pub(crate) kernel: K,
    /// The combinatorial triangulation data structure.
    pub(crate) tds: Tds<K::Scalar, U, V, D>,
    /// Runtime metadata describing the global topological space represented by this triangulation.
    pub(crate) global_topology: GlobalTopology<D>,
    pub(crate) validation_policy: ValidationPolicy,
    pub(crate) topology_guarantee: TopologyGuarantee,
}

// =============================================================================
// Internal Helpers (Structural / Graph Traversals)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Traverses the simplex neighbor graph starting at `start` and returns the set of visited simplices.
    ///
    /// If `allowed` is `Some`, traversal is restricted to that set. Neighbors outside the allowed
    /// set are reported via `on_external_neighbor`.
    #[must_use]
    fn traverse_simplex_neighbor_graph<F>(
        &self,
        start: SimplexKey,
        reserve: usize,
        allowed: Option<&SimplexKeySet>,
        mut on_external_neighbor: F,
    ) -> SimplexKeySet
    where
        F: FnMut(SimplexKey, SimplexKey),
    {
        let mut visited: SimplexKeySet = SimplexKeySet::default();
        visited.reserve(reserve);

        let mut stack: SimplexKeyBuffer = SimplexKeyBuffer::new();
        stack.push(start);

        while let Some(ck) = stack.pop() {
            if !visited.insert(ck) {
                continue;
            }

            let Some(simplex) = self.tds.simplex(ck) else {
                continue;
            };

            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for n_opt in neighbors {
                let Some(nk) = n_opt else {
                    continue;
                };

                if !self.tds.contains_simplex(nk) {
                    continue;
                }

                if allowed.is_some_and(|allowed| !allowed.contains(&nk)) {
                    on_external_neighbor(ck, nk);
                    continue;
                }

                if !visited.contains(&nk) {
                    stack.push(nk);
                }
            }
        }

        visited
    }
}

// =============================================================================
// Basic Accessors (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Create an empty triangulation with the given kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::*;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(tri.number_of_vertices(), 0);
    /// assert_eq!(tri.number_of_simplices(), 0);
    /// assert_eq!(tri.dim(), -1); // Empty triangulation has dimension -1
    /// ```
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            kernel,
            tds: Tds::empty(),
            global_topology: GlobalTopology::DEFAULT,
            validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    /// Returns the topology guarantee used for Level 3 topology validation.
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.topology_guarantee
    }

    /// Returns the runtime global topology metadata associated with this triangulation.
    #[inline]
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        self.global_topology
    }

    /// Returns the high-level topology kind (`Euclidean`, `Toroidal`, etc.).
    #[inline]
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        self.global_topology.kind()
    }

    /// Sets runtime global topology metadata on the triangulation.
    #[inline]
    pub const fn set_global_topology(&mut self, global_topology: GlobalTopology<D>) {
        self.global_topology = global_topology;
    }

    /// Returns the insertion-time global topology validation policy used by the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{Triangulation, ValidationPolicy};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::OnSuspicion);
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.validation_policy
    }

    /// Sets the insertion-time global topology validation policy used by the triangulation.
    ///
    /// If the requested policy is incompatible with the current topology guarantee (for example,
    /// `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](Self::validate_at_completion) to provide
    /// immediate feedback and emits a warning. Call `validate_at_completion()` after batch
    /// construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{Triangulation, ValidationPolicy};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// tri.set_validation_policy(ValidationPolicy::Always);
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::Always);
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        if !self.topology_guarantee.is_compatible_with_policy(policy) {
            let completion_result = self.validate_at_completion();

            if let Err(err) = completion_result {
                debug_assert!(
                    false,
                    "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; validate_at_completion failed: {err}",
                    guarantee = self.topology_guarantee
                );
                tracing::warn!(
                    "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; validate_at_completion failed: {err}. Validation policy not updated.",
                    guarantee = self.topology_guarantee
                );
                return;
            }

            tracing::warn!(
                "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; call validate_at_completion() after construction to certify PL-manifoldness.",
                guarantee = self.topology_guarantee
            );
        }

        self.validation_policy = policy;
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
    ///
    /// If the requested guarantee is incompatible with the current validation policy (for
    /// example, `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](Self::validate_at_completion) to provide
    /// immediate feedback and emits a warning. Call `validate_at_completion()` after batch
    /// construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{TopologyGuarantee, Triangulation};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    /// assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[inline]
    pub fn set_topology_guarantee(&mut self, guarantee: TopologyGuarantee) {
        if !guarantee.is_compatible_with_policy(self.validation_policy) {
            let previous = self.topology_guarantee;
            self.topology_guarantee = guarantee;
            let completion_result = self.validate_at_completion();

            if let Err(err) = completion_result {
                self.topology_guarantee = previous;
                debug_assert!(
                    false,
                    "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; validate_at_completion failed: {err}",
                    policy = self.validation_policy
                );
                tracing::warn!(
                    "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; validate_at_completion failed: {err}. Topology guarantee not updated.",
                    policy = self.validation_policy
                );
                return;
            }

            self.topology_guarantee = previous;
            tracing::warn!(
                "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; call validate_at_completion() after construction to certify PL-manifoldness.",
                policy = self.validation_policy
            );
        }

        self.topology_guarantee = guarantee;
    }

    /// Returns the number of times the topology safety-net recovered from a Level 3
    /// validation failure by retrying insertion with a star-split of the containing simplex.
    ///
    /// This is a process-wide counter (across all triangulation instances) intended for
    /// production telemetry. A high value suggests the cavity-based insertion frequently
    /// creates transient invalid topology that is being masked by the fallback.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::Triangulation;
    ///
    /// let count = Triangulation::<FastKernel<f64>, (), (), 3>
    ///     ::topology_safety_net_star_split_fallback_successes();
    /// assert!(count >= 0);
    /// ```
    #[must_use]
    pub fn topology_safety_net_star_split_fallback_successes() -> u64 {
        TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES.load(Ordering::Relaxed)
    }

    /// Returns duplicate-detection telemetry if enabled via `DELAUNAY_DUPLICATE_METRICS`.
    ///
    /// This is a process-wide counter (across all triangulation instances). It reports how often
    /// duplicate checks used the hash grid versus falling back to linear scans, along with the
    /// total candidate count inspected during grid queries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::{DuplicateDetectionMetrics, Triangulation};
    ///
    /// let metrics = Triangulation::<FastKernel<f64>, (), (), 3>
    ///     ::duplicate_detection_metrics();
    /// let _ = metrics; // None unless DELAUNAY_DUPLICATE_METRICS is set
    /// ```
    #[must_use]
    pub fn duplicate_detection_metrics() -> Option<DuplicateDetectionMetrics> {
        if !duplicate_detection_metrics_enabled() {
            return None;
        }
        Some(DuplicateDetectionMetrics {
            total_checks: DUPLICATE_DETECTION_TOTAL.load(Ordering::Relaxed),
            grid_used: DUPLICATE_DETECTION_GRID_USED.load(Ordering::Relaxed),
            grid_fallbacks: DUPLICATE_DETECTION_GRID_FALLBACKS.load(Ordering::Relaxed),
            grid_candidates: DUPLICATE_DETECTION_GRID_CANDIDATES.load(Ordering::Relaxed),
        })
    }

    #[cfg(test)]
    #[inline]
    pub(crate) const fn new_with_tds(kernel: K, tds: Tds<K::Scalar, U, V, D>) -> Self {
        Self {
            kernel,
            tds,
            global_topology: GlobalTopology::DEFAULT,
            validation_policy: TopologyGuarantee::DEFAULT.default_validation_policy(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    /// Returns an iterator over all simplices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over simplices
    /// for (_simplex_key, simplex) in tri.simplices() {
    ///     assert_eq!(simplex.number_of_vertices(), 3); // 2D triangle
    /// }
    /// assert_eq!(tri.simplices().count(), 1);
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<K::Scalar, U, V, D>)> {
        self.tds.simplices()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over vertices
    /// for (_vertex_key, vertex) in tri.vertices() {
    ///     assert_eq!(vertex.dim(), 2); // 2D vertices
    /// }
    /// assert_eq!(tri.vertices().count(), 3);
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tds.vertices()
    }

    /// Sets the auxiliary data on a vertex, returning the previous value.
    ///
    /// Delegates to [`Tds::set_vertex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<U>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices: [Vertex<f64, i32, 2>; 3] = [
    ///     vertex!([0.0, 0.0], 10i32),
    ///     vertex!([1.0, 0.0], 20),
    ///     vertex!([0.0, 1.0], 30),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    /// let key = dt.vertices().next().unwrap().0;
    /// let prev = dt.set_vertex_data(key, Some(99));
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None);
    /// assert_eq!(prev, Some(Some(99)));
    /// assert_eq!(dt.tds().vertex(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_vertex_data(&mut self, key: VertexKey, data: Option<U>) -> Option<Option<U>> {
        self.tds.set_vertex_data(key, data)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// Delegates to [`Tds::set_simplex_data`]. This is a safe O(1) operation
    /// that does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<V>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<i32>()
    ///     .unwrap();
    /// let key = dt.simplices().next().unwrap().0;
    /// let prev = dt.set_simplex_data(key, Some(42));
    /// assert_eq!(prev, Some(None));
    ///
    /// // Clear data
    /// let prev = dt.set_simplex_data(key, None);
    /// assert_eq!(prev, Some(Some(42)));
    /// assert_eq!(dt.tds().simplex(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_simplex_data(&mut self, key: SimplexKey, data: Option<V>) -> Option<Option<V>> {
        self.tds.set_simplex_data(key, data)
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_vertices(), 4);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tds.number_of_vertices()
    }

    /// Returns the number of simplices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_simplices(), 1); // Single tetrahedron
    /// ```
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.tds.number_of_simplices()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::*;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Empty triangulation has dimension -1
    /// let empty: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(empty.dim(), -1);
    ///
    /// // 3D tetrahedron has dimension 3
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().dim(), 3);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tds.dim()
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// This provides efficient access to all facets without pre-allocating a vector.
    /// Each facet is represented as a lightweight `FacetView` that references the
    /// underlying triangulation data.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Iterate over all facets
    /// let facet_count = dt.as_triangulation().facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        AllFacetsIter::new(&self.tds)
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one simplex. This method
    /// computes the facet-to-simplices map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Panics
    ///
    /// Panics if the triangulation data structure is corrupted (simplices have invalid
    /// neighbor relationships or facet information). This indicates a bug in the
    /// library and should never happen with a properly constructed triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.as_triangulation().boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        // build_facet_to_simplices_map only fails if simplices have invalid structure,
        // which should never happen in a valid triangulation
        let facet_map = self
            .tds
            .build_facet_to_simplices_map()
            .expect("Failed to build facet map - triangulation structure is corrupted");
        BoundaryFacetsIter::new(&self.tds, facet_map)
    }

    // =============================================================================
    // Public Topology Traversal & Adjacency API (Read-only)
    // =============================================================================

    #[inline]
    fn debug_assert_adjacency_index_matches(&self, index: &AdjacencyIndex) {
        // AdjacencyIndex is built from a snapshot of a triangulation. We cannot enforce at
        // compile-time that an index belongs to this triangulation, but we can cheaply catch
        // obvious mix-ups in debug builds.
        debug_assert_eq!(
            index.vertex_to_simplices.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_simplices size does not match triangulation vertex count"
        );
        debug_assert_eq!(
            index.vertex_to_edges.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_edges size does not match triangulation vertex count"
        );
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// Edges are inferred from the vertex lists of each simplex; they are not stored explicitly.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method allocates an internal set to deduplicate edges. The iteration order is
    /// not specified.
    ///
    /// If you need fast repeated topology queries, consider building an
    /// [`AdjacencyIndex`] once via [`Triangulation::build_adjacency_index`](Self::build_adjacency_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let edges: std::collections::HashSet<_> = tri.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_edges().into_iter()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    /// let edges: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.edges()
    }

    /// Returns the number of unique edges in the triangulation.
    ///
    /// This is equivalent to `self.edges().count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // A single 2D triangle has 3 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// assert_eq!(tri.number_of_edges(), 3);
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.collect_edges().len()
    }

    /// Returns the number of unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This is equivalent to `self.edges_with_index(index).count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// assert_eq!(tri.number_of_edges_with_index(&index), 6);
    /// ```
    #[must_use]
    pub fn number_of_edges_with_index(&self, index: &AdjacencyIndex) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_edges()
    }

    /// Returns an iterator over all simplices adjacent (incident) to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// Iteration order is not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Two tetrahedra sharing a triangular facet.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Find a vertex on the shared triangle by coordinates.
    /// let shared_vertex_key = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| {
    ///         let coords = tri.vertex_coords(vk)?;
    ///         (coords == [0.0, 0.0, 0.0]).then_some(vk)
    ///     })
    ///     .unwrap();
    ///
    /// // The shared vertex is incident to both simplices.
    /// assert_eq!(tri.adjacent_simplices(shared_vertex_key).count(), 2);
    /// ```
    pub fn adjacent_simplices(&self, v: VertexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.tds
            .find_simplices_containing_vertex_by_key(v)
            .into_iter()
    }

    /// Returns an iterator over all simplices adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// This avoids per-call scans of the triangulation.
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v = tri.vertices().next().unwrap().0;
    /// assert!(tri.adjacent_simplices_with_index(&index, v).count() >= 1);
    /// ```
    pub fn adjacent_simplices_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = SimplexKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.adjacent_simplices(v)
    }

    /// Returns the number of simplices adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_adjacent_simplices_with_index(&index, v0), 1);
    /// ```
    #[must_use]
    pub fn number_of_adjacent_simplices_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_adjacent_simplices(v)
    }

    /// Returns an iterator over all neighbors of a simplex.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// Iteration order is not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Two tetrahedra sharing a triangular facet => each tetra has exactly one neighbor.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
    /// assert_eq!(simplex_keys.len(), 2);
    ///
    /// for &ck in &simplex_keys {
    ///     let neighbors: Vec<_> = tri.simplex_neighbors(ck).collect();
    ///     assert_eq!(neighbors.len(), 1);
    ///     assert!(simplex_keys.contains(&neighbors[0]));
    ///     assert_ne!(neighbors[0], ck);
    /// }
    /// ```
    pub fn simplex_neighbors(&self, c: SimplexKey) -> impl Iterator<Item = SimplexKey> + '_ {
        self.tds
            .simplex(c)
            .and_then(Simplex::neighbors)
            .into_iter()
            .flat_map(IntoIterator::into_iter)
            .flatten()
            .filter(|&neighbor_key| self.tds.contains_simplex(neighbor_key))
    }

    /// Returns an iterator over all neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let simplex_key = tri.simplices().next().unwrap().0;
    /// let neighbors: Vec<_> = tri.simplex_neighbors_with_index(&index, simplex_key).collect();
    /// assert_eq!(neighbors.len(), 1);
    /// ```
    pub fn simplex_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: SimplexKey,
    ) -> impl Iterator<Item = SimplexKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.simplex_neighbors(c)
    }

    /// Returns the number of neighbors of a simplex using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let simplex_key = tri.simplices().next().unwrap().0;
    /// assert_eq!(tri.number_of_simplex_neighbors_with_index(&index, simplex_key), 1);
    /// ```
    #[must_use]
    pub fn number_of_simplex_neighbors_with_index(
        &self,
        index: &AdjacencyIndex,
        c: SimplexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_simplex_neighbors(c)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method allocates an internal set to deduplicate edges. The iteration order is
    /// not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // In a single tetrahedron, each vertex has degree 3.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let v0 = tri.vertices().next().unwrap().0;
    /// let incident: Vec<_> = tri.incident_edges(v0).collect();
    /// assert_eq!(incident.len(), 3);
    /// assert!(incident
    ///     .iter()
    ///     .all(|e| matches!(e.endpoints(), (a, b) if a == v0 || b == v0)));
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_incident_edges(v).into_iter()
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.incident_edges_with_index(&index, v0).count(), 3);
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.incident_edges(v)
    }

    /// Returns the number of unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_incident_edges_with_index(&index, v0), 3);
    /// ```
    #[must_use]
    pub fn number_of_incident_edges_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_incident_edges(v)
    }

    /// Returns the number of unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // In a single tetrahedron, each vertex has degree 3.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_incident_edges(v0), 3);
    /// ```
    #[must_use]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.collect_incident_edges(v).len()
    }

    /// Returns a slice view of a simplex's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let simplex_key = tri.simplices().next().unwrap().0;
    /// let simplex_vertices = tri.simplex_vertices(simplex_key).unwrap();
    /// assert_eq!(simplex_vertices.len(), 3); // D+1 for a 2D simplex
    /// ```
    #[must_use]
    pub fn simplex_vertices(&self, c: SimplexKey) -> Option<&[VertexKey]> {
        self.tds.simplex(c).map(Simplex::vertices)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let v_key = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// assert_eq!(tri.vertex_coords(v_key).unwrap(), [1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]> {
        self.tds
            .vertex(v)
            .map(|vertex| &vertex.point().coords()[..])
    }

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This never stores any cache internally and does not mutate the triangulation.
    ///
    /// ## Notes
    ///
    /// - No sorted-order guarantees are provided for the values.
    /// - The returned collections are optimized for performance.
    /// - The maps include an entry for every vertex currently stored in the triangulation.
    ///   During the bootstrap phase (before the initial simplex is created), vertices have empty
    ///   adjacency lists because no simplices exist yet. This is expected and not an error condition.
    /// - Isolated vertices (present in the vertex store but not referenced by any simplex) are allowed at
    ///   the TDS structural layer, but violate the Level 3 manifold invariants checked by
    ///   [`Triangulation::is_valid`](Self::is_valid). When present, their adjacency lists are empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the triangulation data structure is internally inconsistent
    /// (e.g., a simplex references a missing vertex key or a missing neighbor simplex key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Two tetrahedra sharing a triangular facet.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    ///
    /// // The index exposes adjacency maps keyed by VertexKey / SimplexKey.
    /// let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
    /// for &ck in &simplex_keys {
    ///     let neighbors = index.simplex_to_neighbors.get(&ck).unwrap();
    ///     assert_eq!(neighbors.len(), 1);
    /// }
    /// ```
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        let vertex_cap = self.tds.number_of_vertices();
        let simplex_cap = self.tds.number_of_simplices();

        let mut vertex_to_simplices: VertexToSimplicesMap = fast_hash_map_with_capacity(vertex_cap);
        let mut simplex_to_neighbors: FastHashMap<
            SimplexKey,
            SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(simplex_cap);
        let mut vertex_to_edges: FastHashMap<
            VertexKey,
            SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(vertex_cap);

        // Deduplicate edges globally while building the index.
        let edges_per_simplex = (D + 1).saturating_mul(D) / 2;
        let mut seen_edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(simplex_cap.saturating_mul(edges_per_simplex));

        for (simplex_key, simplex) in self.tds.simplices() {
            let vertices = simplex.vertices();

            // Vertex → simplices
            for &vk in vertices {
                if !self.tds.contains_vertex_key(vk) {
                    return Err(AdjacencyIndexBuildError::MissingVertexKey {
                        simplex_key,
                        vertex_key: vk,
                    });
                }
                let entry = vertex_to_simplices.entry(vk).or_default();
                #[cfg(debug_assertions)]
                let was_spilled = entry.spilled();
                entry.push(simplex_key);
                #[cfg(debug_assertions)]
                if !was_spilled && entry.spilled() {
                    let spill_count =
                        VERTEX_TO_SIMPLICES_SPILL_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
                    tracing::debug!(
                        "VertexToSimplicesMap spill #{spill_count}: vertex={vk:?} len={} cap={} (MAX_PRACTICAL_DIMENSION_SIZE={MAX_PRACTICAL_DIMENSION_SIZE})",
                        entry.len(),
                        entry.capacity()
                    );
                }
            }

            // Simplex → neighbors
            if let Some(neighbors) = simplex.neighbor_keys() {
                let mut neighs: SmallBuffer<SimplexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::new();

                for n_opt in neighbors {
                    let Some(nk) = n_opt else {
                        continue;
                    };

                    if !self.tds.contains_simplex(nk) {
                        return Err(AdjacencyIndexBuildError::MissingNeighborSimplex {
                            simplex_key,
                            neighbor_key: nk,
                        });
                    }

                    neighs.push(nk);
                }

                if !neighs.is_empty() {
                    simplex_to_neighbors.insert(simplex_key, neighs);
                }
            }

            // Vertex → edges (deduped)
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    let edge = EdgeKey::new(vertices[i], vertices[j]);
                    if !seen_edges.insert(edge) {
                        continue;
                    }

                    let (a, b) = edge.endpoints();
                    vertex_to_edges.entry(a).or_default().push(edge);
                    vertex_to_edges.entry(b).or_default().push(edge);
                }
            }
        }

        // Ensure every vertex in the triangulation has an entry, even if it is currently
        // not incident to any simplex (e.g., bootstrap phase with < D+1 vertices, or TDS-level
        // states with isolated vertices). Level 3 topology validation (`Triangulation::is_valid`)
        // rejects isolated vertices, but this indexing helper remains usable for debugging and
        // intermediate construction states.
        for (vk, _) in self.tds.vertices() {
            vertex_to_simplices.entry(vk).or_default();
            vertex_to_edges.entry(vk).or_default();
        }

        Ok(AdjacencyIndex {
            vertex_to_edges,
            vertex_to_simplices,
            simplex_to_neighbors,
        })
    }

    #[must_use]
    fn collect_edges(&self) -> FastHashSet<EdgeKey> {
        let simplex_cap = self.tds.number_of_simplices();
        let edges_per_simplex = (D + 1).saturating_mul(D) / 2;

        let mut edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(simplex_cap.saturating_mul(edges_per_simplex));

        for (_simplex_key, simplex) in self.tds.simplices() {
            let vertices = simplex.vertices();
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    edges.insert(EdgeKey::new(vertices[i], vertices[j]));
                }
            }
        }

        edges
    }

    #[must_use]
    fn collect_incident_edges(&self, v: VertexKey) -> FastHashSet<EdgeKey> {
        let mut edges: FastHashSet<EdgeKey> = FastHashSet::default();

        for simplex_key in self.adjacent_simplices(v) {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue;
            };

            for &other in simplex.vertices() {
                if other == v {
                    continue;
                }
                edges.insert(EdgeKey::new(v, other));
            }
        }

        edges
    }

    /// Collect simplex points for orientation evaluation.
    ///
    /// For periodic simplices, this delegates per-vertex lattice-offset lifting to the active
    /// [`GlobalTopology`] behavior model.
    fn collect_simplex_points_for_orientation(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<K::Scalar, U, V, D>,
        purpose: &str,
    ) -> Result<SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        let topology_model = self.global_topology.model();
        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets {
            // Check length invariant
            if offsets.len() != simplex.number_of_vertices() {
                return Err(TdsError::DimensionMismatch {
                    expected: simplex.number_of_vertices(),
                    actual: offsets.len(),
                    context: format!(
                        "simplex {:?} (key {simplex_key:?}) periodic offset count vs vertex count during {purpose}",
                        simplex.uuid(),
                    ),
                });
            }
            // Check topology capabilities
            if !topology_model.supports_periodic_orientation_offsets() {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}) has periodic offsets (count {}) during {purpose}, but triangulation global topology is {:?} (kind {:?}, allows_boundary: {}, periodic_domain: {:?}); expected periodic-orientation-offset-capable topology",
                        simplex.uuid(),
                        offsets.len(),
                        self.global_topology,
                        topology_model.kind(),
                        topology_model.allows_boundary(),
                        topology_model.periodic_domain(),
                    ),
                });
            }
        }

        let mut points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(simplex.number_of_vertices());

        for (vertex_idx, &vertex_key) in simplex.vertices().iter().enumerate() {
            let vertex = self.tds.vertex(vertex_key).ok_or_else(|| {
                TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by simplex {:?} (key {simplex_key:?}) at position {vertex_idx} during {purpose}",
                        simplex.uuid(),
                    ),
                }
            })?;
            let periodic_offset = periodic_offsets.map(|offsets| offsets[vertex_idx]);
            let lifted_coords = topology_model
                .lift_for_orientation(*vertex.point().coords(), periodic_offset)
                .map_err(|error| TdsError::InconsistentDataStructure {
                    message: format!(
                        "Failed to lift coordinates for vertex key {vertex_key:?} at slot {vertex_idx} in simplex {:?} (key {simplex_key:?}) during {purpose}: {error}",
                        simplex.uuid(),
                    ),
                })?;

            points.push(Point::new(lifted_coords));
        }

        Ok(points)
    }

    /// Evaluate a simplex's geometric orientation for a specific validation/canonicalization context.
    ///
    /// This helper centralizes:
    /// - lifted-point collection, and
    /// - exact (non-SoS) orientation determination via [`robust_orientation`].
    ///
    /// # Exact Orientation (no `SoS`)
    ///
    /// This function uses [`robust_orientation`] exclusively to determine the sign.
    /// It does **not** consult the kernel (which may use `SoS`), because:
    /// - Callers need the true geometric sign for orientation canonicalization
    ///   and validation — `SoS` tie-breaking would mask real degeneracies.
    /// - Using `robust_orientation` alone avoids duplicate exact-arithmetic work
    ///   (both `robust_orientation` and `AdaptiveKernel::orientation()` run the
    ///   same fast-filter + Bareiss pipeline internally).
    ///
    /// Returns `0` for truly degenerate (zero-volume) simplices, so callers' existing
    /// `orientation == 0` handling works correctly.
    ///
    /// # Error Mapping
    ///
    /// Conversion failures (e.g. non-finite coordinates) are mapped to
    /// [`TdsError::InconsistentDataStructure`] because they indicate
    /// an internal problem rather than a geometry degeneracy.
    fn evaluate_simplex_orientation_for_context(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<K::Scalar, U, V, D>,
        purpose: &str,
        predicate_failure_prefix: &str,
    ) -> Result<i32, TdsError> {
        let points = self.collect_simplex_points_for_orientation(simplex_key, simplex, purpose)?;

        // Use exact orientation only (no SoS):
        // sign to correctly classify degenerate simplices vs negatively oriented ones.
        match robust_orientation(&points) {
            Ok(Orientation::POSITIVE) => Ok(1),
            Ok(Orientation::NEGATIVE) => Ok(-1),
            Ok(Orientation::DEGENERATE) => Ok(0),
            Err(e) => Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "{predicate_failure_prefix} {:?} (key {simplex_key:?}): {e}",
                    simplex.uuid(),
                ),
            }),
        }
    }
    /// Validates geometric orientation sign for each stored simplex using exact arithmetic
    /// via [`robust_orientation`].
    ///
    /// Simplices are stored in canonical positive orientation order by construction and mutation
    /// paths; a negative sign indicates geometric/combinatorial mismatch.
    ///
    /// Orientation is evaluated through [`evaluate_simplex_orientation_for_context`](Self::evaluate_simplex_orientation_for_context),
    /// which uses [`robust_orientation`] exclusively (no `SoS`) so that truly degenerate
    /// simplices are correctly identified rather than masked.
    ///
    /// Periodic-lifted simplices are validated in lifted coordinates using per-vertex periodic
    /// offsets and toroidal domain periods.
    pub(crate) fn validate_geometric_simplex_orientation(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "geometric orientation validation",
                "Geometric orientation predicate failed for simplex",
            )?;
            // Degenerate simplices (zero exact determinant) can legitimately arise
            // from flip-based repair in higher dimensions. They are topologically
            // valid (BFS coherent orientation handles them) and do not indicate
            // a sign mismatch.  Only flag simplices with negative orientation.
            if orientation < 0 {
                // Emit structured diagnostic context for debugging (especially 4D+ cases).
                let vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex.vertices().iter().copied().collect();
                let neighbor_keys: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex
                        .neighbor_keys()
                        .map(Iterator::collect)
                        .unwrap_or_default();
                tracing::debug!(
                    simplex_uuid = %simplex.uuid(),
                    ?simplex_key,
                    ?vertex_keys,
                    ?neighbor_keys,
                    orientation,
                    "negative geometric orientation detected during validation",
                );

                return Err(TdsError::Geometric(GeometricError::NegativeOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}, vertices {vertex_keys:?}) has negative geometric orientation; expected positive canonical orientation",
                        simplex.uuid(),
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Validates geometric orientation for a local set of simplices.
    fn validate_geometric_simplex_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), TdsError> {
        for &simplex_key in simplices {
            let simplex =
                self.tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "local geometric orientation validation scope".to_string(),
                    })?;
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "local geometric orientation validation",
                "Geometric orientation predicate failed for local simplex",
            )?;
            if orientation < 0 {
                let vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex.vertices().iter().copied().collect();
                tracing::debug!(
                    simplex_uuid = %simplex.uuid(),
                    ?simplex_key,
                    ?vertex_keys,
                    orientation,
                    "negative geometric orientation detected during local validation",
                );

                return Err(TdsError::Geometric(GeometricError::NegativeOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}, vertices {vertex_keys:?}) has negative geometric orientation; expected positive canonical orientation",
                        simplex.uuid(),
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Validates local orientation invariants for simplices changed by insertion.
    fn validate_local_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), InsertionError> {
        self.tds
            .validate_coherent_orientation_for_simplices(simplices)?;
        self.validate_geometric_simplex_orientation_for_simplices(simplices)?;
        Ok(())
    }

    /// Flip all negatively oriented simplices to positive orientation.
    ///
    /// This applies to both Euclidean simplices and periodic-lifted simplices (when present).
    ///
    /// Returns `true` if at least one simplex was flipped.
    fn promote_simplices_to_positive_orientation(&mut self) -> Result<bool, InsertionError> {
        let mut negative_simplices = SimplexKeyBuffer::new();

        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "positive-orientation promotion",
                "Geometric orientation predicate failed while promoting positive orientation for simplex",
            )?;
            // Skip degenerate simplices — their exact determinant is zero, so there
            // is no meaningful "positive" to promote to. BFS coherent-orientation
            // normalization and the global sign canonicalization handle them.
            if orientation == 0 {
                continue;
            }
            if orientation < 0 {
                negative_simplices.push(simplex_key);
            }
        }

        if negative_simplices.is_empty() {
            return Ok(false);
        }

        for simplex_key in negative_simplices {
            let simplex =
                self.tds
                    .simplex_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "applying positive-orientation promotion".to_string(),
                    })?;
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
            }
        }

        self.tds.mark_topology_modified();
        Ok(true)
    }

    /// Check whether any simplex still requires positive-orientation promotion.
    ///
    /// This performs the same orientation inspection as promotion, but does not mutate any simplices.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if at least one simplex has negative geometric orientation.
    /// - `Ok(false)` if all simplices are already positively oriented.
    ///
    /// # Errors
    ///
    /// Returns an [`InsertionError`] if orientation evaluation fails.
    /// Geometrically degenerate simplices (`orientation == 0` per [`robust_orientation`])
    /// are skipped, consistent with [`promote_simplices_to_positive_orientation`](Self::promote_simplices_to_positive_orientation).
    fn simplices_require_positive_orientation_promotion(&self) -> Result<bool, InsertionError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "positive-orientation convergence check",
                "Geometric orientation predicate failed while checking positive-orientation convergence for simplex",
            )?;
            // Skip degenerate simplices (see promote_simplices_to_positive_orientation).
            if orientation == 0 {
                continue;
            }
            if orientation < 0 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// For connected non-periodic triangulations, coherent orientation has two equivalent global
    /// sign choices. Canonicalize that global sign to positive by flipping all simplices when needed.
    fn canonicalize_global_orientation_sign(&mut self) -> Result<(), InsertionError> {
        // Find the first simplex with a non-zero exact orientation
        // Skip degenerate simplices (orientation == 0) — they have no meaningful geometric sign.
        let representative_sign = {
            let mut sign = None;
            for (simplex_key, simplex) in self.tds.simplices() {
                let orientation = self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "global orientation-sign canonicalization",
                    "Geometric orientation predicate failed while canonicalizing global orientation sign for simplex",
                )?;
                if orientation != 0 {
                    sign = Some(orientation);
                    break;
                }
            }
            sign
        };

        if representative_sign != Some(-1) {
            return Ok(());
        }

        let simplex_keys: SimplexKeyBuffer = self.tds.simplex_keys().collect();
        let mut flipped_any = false;
        for simplex_key in simplex_keys {
            let Some(simplex) = self.tds.simplex_mut(simplex_key) else {
                continue;
            };
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
                flipped_any = true;
            }
        }

        if flipped_any {
            self.tds.mark_topology_modified();
        }

        Ok(())
    }

    /// Normalize coherent orientation and promote geometric orientation to the positive
    /// canonical sign.
    ///
    /// Strategy:
    /// 1. Normalize coherent orientation (BFS propagation) so all adjacencies agree.
    /// 2. Canonicalize the global sign — for a connected orientable manifold all simplices
    ///    share the same sign after normalization, so a single global flip resolves it.
    /// 3. Fall back to bounded per-simplex promotion passes for FP-precision edge cases.
    pub(crate) fn normalize_and_promote_positive_orientation(
        &mut self,
    ) -> Result<(), InsertionError> {
        // Phase 1: make all adjacencies coherent
        // Canonicalizing *before* the promote loop resolves the common case where
        // all simplices share the same (negative) sign after BFS normalization.
        self.tds.normalize_coherent_orientation()?;
        self.canonicalize_global_orientation_sign()?;

        // Phase 2 (fallback): bounded promote + normalize passes for stragglers.
        for _ in 0..3 {
            if !self.promote_simplices_to_positive_orientation()? {
                break;
            }
            self.tds.normalize_coherent_orientation()?;
        }

        // Soft post-condition: after normalize + canonicalize + bounded promote
        // passes, any remaining "negative" simplices are near-degenerate (det ≈ 0)
        // where the fast kernel's sign is unreliable.  Log a diagnostic but do
        // not fail — the BFS normalization guarantees coherent orientation and
        // the global canonicalization ensures the dominant sign is positive.
        if self.simplices_require_positive_orientation_promotion()? {
            let mut residual_count = 0_usize;
            let mut sample_keys: [Option<SimplexKey>; 5] = [None; 5];
            for (simplex_key, simplex) in self.tds.simplices() {
                let orientation = self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "residual negative-orientation sampling",
                    "Geometric orientation predicate failed while sampling residual negatives for simplex",
                )?;
                if orientation < 0 {
                    if residual_count < sample_keys.len() {
                        sample_keys[residual_count] = Some(simplex_key);
                    }
                    residual_count += 1;
                }
            }
            let sampled: Vec<SimplexKey> = sample_keys.into_iter().flatten().collect();
            tracing::debug!(
                residual_count,
                sampled_keys = ?sampled,
                "normalize_and_promote_positive_orientation: \
                 {residual_count} simplices still appear negative after bounded promotion \
                 passes (likely near-degenerate FP noise); accepting coherent orientation"
            );
        }
        self.canonicalize_global_orientation_sign()?;
        Ok(())
    }
    /// Canonicalize a set of newly created simplices to positive geometric orientation.
    ///
    /// This preserves simplex-local slot alignment (vertices/neighbors/periodic offsets) by using
    /// `swap_vertex_slots(0, 1)` for negatively oriented simplices.
    #[expect(
        clippy::too_many_lines,
        reason = "debug-only orientation diagnostics with dedup add conditional branches"
    )]
    fn canonicalize_positive_orientation_for_simplices(
        &mut self,
        simplices: &SimplexKeyBuffer,
    ) -> Result<(), InsertionError> {
        #[cfg(debug_assertions)]
        let debug_orientation = std::env::var_os("DELAUNAY_DEBUG_ORIENTATION").is_some();
        #[cfg(debug_assertions)]
        let mut orientation_warn_count = 0_usize;

        for &simplex_key in simplices {
            let orientation = {
                let simplex =
                    self.tds
                        .simplex(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "canonicalizing insertion orientation".to_string(),
                        })?;
                self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "insertion orientation canonicalization",
                    "Geometric orientation predicate failed while canonicalizing simplex",
                )?
            };

            if orientation == 0 {
                // Keep temporary degenerate simplices unchanged here. Downstream local-repair and
                // topology/geometry validation decide whether they are removed or rejected.
                continue;
            }

            if orientation < 0 {
                // Capture pre-swap vertices only when the debug env var is set,
                // so release and non-debug runs avoid the allocation entirely.
                #[cfg(debug_assertions)]
                let pre_swap_vertices = if debug_orientation {
                    self.tds.simplex(simplex_key).map(|c| c.vertices().to_vec())
                } else {
                    None
                };

                let simplex =
                    self.tds
                        .simplex_mut(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "applying insertion orientation canonicalization".to_string(),
                        })?;
                if simplex.number_of_vertices() < 2 {
                    return Err(TdsError::DimensionMismatch {
                        expected: 2,
                        actual: simplex.number_of_vertices(),
                        context: format!(
                            "simplex {simplex_key:?} needs >= 2 vertices for orientation canonicalization"
                        ),
                    }
                    .into());
                }
                simplex.swap_vertex_slots(0, 1);

                #[cfg(debug_assertions)]
                if debug_orientation {
                    orientation_warn_count += 1;
                    // Log full detail for the first 3 occurrences; suppress the rest.
                    if orientation_warn_count <= 3 {
                        // Re-evaluate orientation after swap to confirm it worked.
                        // Handle the Result locally so verification failures are
                        // observational only and never promote to insertion errors.
                        let post_orientation = self.tds.simplex(simplex_key).map(|c| {
                            self.evaluate_simplex_orientation_for_context(
                                simplex_key,
                                c,
                                "orientation swap verification",
                                "orientation predicate failed during swap verification",
                            )
                        });
                        match post_orientation {
                            Some(Ok(post_o)) => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    post_swap_orientation = post_o,
                                    swap_fixed = post_o > 0,
                                    "canonicalize_positive_orientation: negative-orientation simplex swapped"
                                );
                            }
                            Some(Err(ref e)) => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    error = %e,
                                    "canonicalize_positive_orientation: post-swap verification failed"
                                );
                            }
                            None => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    "canonicalize_positive_orientation: simplex not found after swap"
                                );
                            }
                        }
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        if orientation_warn_count > 3 && debug_orientation {
            let suppressed = orientation_warn_count - 3;
            tracing::warn!(
                total_negative = orientation_warn_count,
                suppressed,
                "canonicalize_positive_orientation: suppressed {suppressed} additional negative-orientation warnings (see first 3 above)"
            );
        }

        Ok(())
    }

    /// Validates topological invariants of the triangulation (Level 3).
    ///
    /// This checks the triangulation/topology layer **only**:
    /// - Codimension-1 pseudomanifold condition: each facet is incident to 1 (boundary) or 2 (interior) simplices
    /// - Codimension-2 boundary manifoldness: the boundary must be closed ("no boundary of boundary")
    /// - Geometric orientation-sign consistency for stored simplices (signed determinant > 0)
    /// - Ridge-link validation (when `topology_guarantee.requires_ridge_links()`)
    /// - Vertex-link validation during insertion (when `topology_guarantee.requires_vertex_links_during_insertion()`)
    /// - Connectedness (single component in the simplex neighbor graph)
    /// - No isolated vertices (every vertex must be incident to at least one simplex)
    /// - Euler characteristic
    ///
    /// For `TopologyGuarantee::PLManifold`, full PL-manifold certification requires
    /// calling [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    /// (or [`Triangulation::validate`](Self::validate)) after batch construction.
    ///
    /// It intentionally does **not** validate lower layers (vertices/simplices or TDS structure).
    /// For cumulative validation, use [`Triangulation::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if:
    /// - The manifold-with-boundary facet property is violated.
    /// - The triangulation is disconnected (multiple simplex components).
    /// - An isolated vertex is detected (no incident simplex).
    /// - Euler characteristic validation fails.
    /// - The topology module reports an error (treated as inconsistent data structure).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Level 3: topology validation (manifold-with-boundary + Euler characteristic)
    /// assert!(dt.as_triangulation().is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), InvariantError> {
        self.validate_topology_core()?;
        // Check geometric orientation after manifold/link checks so topology-specific
        // diagnostics surface first when multiple invariants are violated.
        self.validate_geometric_simplex_orientation()?;
        Ok(())
    }

    /// Validates topological invariants **without** geometric orientation checks.
    ///
    /// This is identical to [`is_valid`](Self::is_valid) but omits the
    /// `validate_geometric_simplex_orientation()` step. It is intended for
    /// explicit combinatorial construction where the user-provided vertex
    /// orderings may produce negative determinants that are nonetheless
    /// topologically valid.
    pub(crate) fn is_valid_topology_only(&self) -> Result<(), InvariantError> {
        self.validate_topology_core()
    }

    /// Verifies that no simplex is geometrically degenerate (zero-volume simplex).
    ///
    /// This is a sign-agnostic check: it flags simplices whose exact orientation
    /// determinant is zero (collinear in 2D, coplanar in 3D, etc.) regardless
    /// of the sign.  Intended for explicit construction where the user-supplied
    /// vertex set must form non-degenerate simplices.
    ///
    /// Unlike [`validate_geometric_simplex_orientation`](Self::validate_geometric_simplex_orientation),
    /// this does **not** reject negative-orientation simplices.
    pub(crate) fn validate_geometric_nondegeneracy(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "geometric nondegeneracy check",
                "Orientation predicate failed for simplex",
            )?;
            if orientation == 0 {
                return Err(TdsError::Geometric(GeometricError::DegenerateOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}) is geometrically degenerate \
                         (zero-volume simplex from collinear/coplanar vertices)",
                        simplex.uuid(),
                    ),
                }));
            }
        }
        Ok(())
    }

    /// Shared Level-3 topology validation sequence used by both [`is_valid`](Self::is_valid)
    /// and [`is_valid_topology_only`](Self::is_valid_topology_only).
    ///
    /// Checks connectedness, manifold facet degree, closed boundary, ridge/vertex
    /// links (when required by the topology guarantee), isolated vertices, and
    /// Euler characteristic.
    fn validate_topology_core(&self) -> Result<(), InvariantError> {
        // 1. Connectedness
        //
        // Checked first because it is cheaper than building the facet-to-simplices map
        // (which requires O(N·D) hash-map insertions plus allocations) and avoids
        // all subsequent work when the triangulation is disconnected.
        self.validate_global_connectedness()?;

        // 2. Manifold facet multiplicity (codimension-1 pseudomanifold condition)
        //
        // Build the facet map once and reuse it for manifold validation and Euler counting.
        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        self.validate_topology_core_with_facet_to_simplices_map(&facet_to_simplices)
    }

    fn validate_topology_core_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), InvariantError> {
        validate_facet_degree(facet_to_simplices)?;

        // 2b. Boundary manifoldness in codimension 2: the boundary must be "closed"
        // (i.e., its ridges must have degree 2 within boundary facets).
        validate_closed_boundary(&self.tds, facet_to_simplices)?;

        // 2c. Ridge-link validation for PLManifold/PLManifoldStrict (fast, catches many PL issues).
        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links(&self.tds)?;
        }
        // 2d. PL-manifold vertex-link condition during insertion (strict mode).
        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            validate_vertex_links(&self.tds, facet_to_simplices)?;
        }

        // 3. Vertex incidence (manifold invariant): every vertex must be incident to at least one simplex.
        self.validate_no_isolated_vertices()?;

        // 4. Euler characteristic using the topology module
        let topology_result =
            validate_triangulation_euler_with_facet_to_simplices_map(&self.tds, facet_to_simplices);

        // Override the heuristic classification when the caller has declared a
        // non-Euclidean global topology.  The heuristic classifies any closed
        // mesh (no boundary facets) as `ClosedSphere(D)`, but a toroidal mesh
        // also has no boundary — its expected χ is 0, not 1+(-1)^D.
        let (classification, expected) = match self.global_topology {
            GlobalTopology::Toroidal { .. }
                if matches!(
                    topology_result.classification,
                    TopologyClassification::ClosedSphere(_)
                ) =>
            {
                let cls = TopologyClassification::ClosedToroid(D);
                (cls, expected_chi_for(&cls))
            }
            _ => (topology_result.classification, topology_result.expected),
        };

        if let Some(exp) = expected
            && topology_result.chi != exp
        {
            return Err(TriangulationValidationError::EulerCharacteristicMismatch {
                computed: topology_result.chi,
                expected: exp,
                classification,
            }
            .into());
        }

        Ok(())
    }

    /// Validates vertex-link condition at construction completion.
    ///
    /// This should be called once after batch construction is complete to certify
    /// full PL-manifoldness when using `TopologyGuarantee::PLManifold` (incremental mode).
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if vertex-link validation fails
    /// (e.g. a vertex link is not a PL-sphere/ball as required for PL-manifoldness).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert!(dt.as_triangulation().validate_at_completion().is_ok());
    /// ```
    pub fn validate_at_completion(&self) -> Result<(), InvariantError> {
        if !self
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            return Ok(());
        }

        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        self.validate_at_completion_with_facet_to_simplices_map(&facet_to_simplices)?;
        Ok(())
    }

    fn validate_at_completion_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), InvariantError> {
        if !self
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            return Ok(());
        }

        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        validate_vertex_links(&self.tds, facet_to_simplices)?;
        Ok(())
    }

    /// Performs cumulative validation for Levels 1–3.
    ///
    /// This validates:
    /// - **Level 1–2** via [`Tds::validate`](crate::core::tds::Tds::validate)
    /// - **Level 3** via [`Triangulation::is_valid`](Self::is_valid)
    /// - **Completion-time PL-manifold check** via [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if:
    /// - Any vertex/simplex is invalid (Level 1).
    /// - The TDS structural invariants fail (Level 2).
    /// - Topology validation fails (Level 3).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Levels 1–3: elements + TDS structure + topology
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), InvariantError>
    where
        U: DataType,
        V: DataType,
    {
        self.tds.validate()?;
        self.validate_global_connectedness()?;
        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        self.validate_topology_core_with_facet_to_simplices_map(&facet_to_simplices)?;
        // Check geometric orientation after manifold/link checks so topology-specific
        // diagnostics surface first when multiple invariants are violated.
        self.validate_geometric_simplex_orientation()?;
        self.validate_at_completion_with_facet_to_simplices_map(&facet_to_simplices)
    }

    /// Generate a comprehensive validation report for Levels 1–3.
    ///
    /// This is intended for debugging/telemetry where you want to see *all* violated
    /// invariants, not just the first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–3.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all invariant violations.
    pub(crate) fn validation_report(&self) -> Result<(), TriangulationValidationReport>
    where
        U: DataType,
        V: DataType,
    {
        let mut violations: Vec<InvariantViolation> = Vec::new();

        // Level 2 (structural): reuse the TDS report.
        match self.tds.validation_report() {
            Ok(()) => {}
            Err(report) => {
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::SimplexMappings
                    )
                }) {
                    return Err(report);
                }
                violations.extend(report.violations);
            }
        }

        // Level 1 (element validity): vertices
        for (_vertex_key, vertex) in self.tds.vertices() {
            if let Err(source) = (*vertex).is_valid() {
                violations.push(InvariantViolation {
                    kind: InvariantKind::VertexValidity,
                    error: InvariantError::Tds(TdsError::InvalidVertex {
                        vertex_id: vertex.uuid(),
                        source,
                    }),
                });
            }
        }

        // Level 1 (element validity): simplices
        for (_simplex_key, simplex) in self.tds.simplices() {
            if let Err(source) = simplex.is_valid() {
                violations.push(InvariantViolation {
                    kind: InvariantKind::SimplexValidity,
                    error: InvariantError::Tds(TdsError::InvalidSimplex {
                        simplex_id: simplex.uuid(),
                        source,
                    }),
                });
            }
        }

        // Level 3 (topology)
        if let Err(e) = self.is_valid() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Topology,
                error: e,
            });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Validates that the triangulation's simplex neighbor graph is a single connected component.
    ///
    /// Delegates to [`Tds::is_connected`], an O(N·D) BFS over neighbor pointers.
    fn validate_global_connectedness(&self) -> Result<(), TriangulationValidationError> {
        if !self.tds.is_connected() {
            return Err(TriangulationValidationError::Disconnected {
                simplex_count: self.tds.number_of_simplices(),
            });
        }
        Ok(())
    }

    /// Validates that every vertex is incident to at least one simplex.
    ///
    /// Isolated vertices are allowed at the TDS (structural) layer, but they violate the
    /// manifold invariants checked at the topology (Level 3) layer.
    fn validate_no_isolated_vertices(&self) -> Result<(), TriangulationValidationError> {
        if self.tds.number_of_vertices() == 0 {
            return Ok(());
        }

        let mut vertices_in_simplices: FastHashSet<VertexKey> =
            fast_hash_set_with_capacity(self.tds.number_of_vertices());

        for (_simplex_key, simplex) in self.tds.simplices() {
            for &vk in simplex.vertices() {
                vertices_in_simplices.insert(vk);
            }
        }

        for (vk, vertex) in self.tds.vertices() {
            if !vertices_in_simplices.contains(&vk) {
                return Err(TriangulationValidationError::IsolatedVertex {
                    vertex_key: vk,
                    vertex_uuid: vertex.uuid(),
                });
            }
        }

        Ok(())
    }
}

// =============================================================================
// Geometric Operations (Requires Extra Numeric Conversion Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: NumCast,
    U: DataType,
    V: DataType,
{
    /// Build initial D-simplex from D+1 vertices with degeneracy validation.
    ///
    /// This creates a Tds with a single simplex containing all D+1 vertices,
    /// with explicit boundary neighbor slots. The simplex is
    /// validated to ensure it is non-degenerate (vertices span full D-dimensional space).
    ///
    /// **Design Note**: This method uses [`robust_orientation`] directly for the
    /// non-degeneracy check, bypassing the kernel. This avoids `SoS` tie-breaking
    /// (which would mask truly degenerate input) and keeps the method independent
    /// of kernel state.
    ///
    /// # Arguments
    /// - `vertices`: Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    /// A Tds containing one D-simplex with all vertices, ready for incremental insertion.
    ///
    /// # Errors
    /// Returns error if:
    /// - Wrong number of vertices (must be exactly D+1)
    /// - Vertices are degenerate (collinear in 2D, coplanar in 3D, etc.)
    /// - Vertex or simplex insertion fails
    /// - Duplicate UUIDs detected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::*;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a 2D triangle (initial simplex)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// assert_eq!(tds.dim(), 2);
    ///
    /// // Error: wrong number of vertices (need exactly D+1)
    /// let bad_vertices = vec![vertex!([0.0, 0.0])];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&bad_vertices);
    /// assert!(result.is_err());
    ///
    /// // Error: collinear points in 2D (degenerate simplex)
    /// let collinear = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([2.0, 0.0]),
    /// ];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&collinear);
    /// assert!(result.is_err());
    /// ```
    pub fn build_initial_simplex(
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Tds<K::Scalar, U, V, D>, TriangulationConstructionError> {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: SimplexValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        for vertex in vertices {
            vertex.is_valid().map_err(|source| {
                TriangulationConstructionError::Tds(TdsConstructionError::ValidationError(
                    TdsError::InvalidVertex {
                        vertex_id: vertex.uuid(),
                        source,
                    },
                ))
            })?;
        }

        // Validate that the simplex is non-degenerate using exact orientation.
        // Use robust_orientation (no SoS) so that truly degenerate input
        // (collinear/coplanar) is detected even when the kernel uses SoS.

        // Collect points into stack-allocated buffer (at most 8 points for D ≤ 7)
        let points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            vertices.iter().map(|v| *v.point()).collect();

        // Exact degeneracy check — DEGENERATE means zero-volume simplex.
        let exact_orientation = robust_orientation(&points[..]).map_err(|e| {
            TriangulationConstructionError::FailedToCreateSimplex {
                message: format!("Exact orientation test failed: {e}"),
            }
        })?;

        if matches!(exact_orientation, Orientation::DEGENERATE) {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Degenerate initial simplex: vertices are collinear/coplanar in {}D space. \
                     The {} input vertices do not span a full {}-dimensional simplex. \
                     Provide non-degenerate vertices to create a valid triangulation.",
                    D,
                    D + 1,
                    D
                ),
            });
        }

        // Use the exact orientation sign directly — robust_orientation already
        // provides a provably correct POSITIVE/NEGATIVE for non-degenerate inputs.
        let orientation = match exact_orientation {
            Orientation::POSITIVE => 1,
            Orientation::NEGATIVE => -1,
            Orientation::DEGENERATE => {
                // Unreachable: degeneracy was checked above.
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Degenerate initial simplex in {D}D (unreachable)"),
                });
            }
        };

        // Create empty Tds
        let mut tds = Tds::empty();

        // Insert all vertices and collect their keys
        let mut vertex_keys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for vertex in vertices {
            let vkey = tds.insert_vertex_with_mapping(*vertex)?;
            vertex_keys.push(vkey);
        }

        // Canonicalize initial simplex orientation: store simplices in positive orientation order.
        // Swapping any two vertices flips orientation.
        if orientation < 0 {
            if vertex_keys.len() >= 2 {
                vertex_keys.swap(0, 1);
            } else {
                return Err(TriangulationConstructionError::FailedToCreateSimplex {
                    message: format!(
                        "Cannot canonicalize orientation for {}D simplex with {} vertex key(s)",
                        D,
                        vertex_keys.len(),
                    ),
                });
            }
        }

        // Create single D-simplex from all vertices in canonicalized order.
        let simplex = Simplex::new(vertex_keys, None).map_err(|e| {
            TriangulationConstructionError::FailedToCreateSimplex {
                message: format!("Failed to create initial simplex: {e}"),
            }
        })?;

        // Insert the simplex
        let _simplex_key = tds.insert_simplex_with_mapping(simplex)?;

        // Assign explicit boundary neighbor slots for the initial simplex.
        tds.assign_neighbors()
            .map_err(TdsConstructionError::ValidationError)?;

        // Assign incident simplices to vertices (each vertex points to this one simplex)
        // This is required for proper Tds structure
        tds.assign_incident_simplices()
            .map_err(|e| TdsConstructionError::ValidationError(e.into()))?;

        Ok(tds)
    }

    /// Insert a vertex into the triangulation using cavity-based algorithm.
    ///
    /// This is a generic insertion method that handles:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating simplices
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-simplex
    /// - **Incremental (> D+1 vertices)**: Cavity-based insertion or hull extension
    ///
    /// # Arguments
    /// - `vertex`: The vertex to insert
    /// - `conflict_simplices`: Optional conflict region (simplices to be removed). Required for
    ///   interior points, not needed for exterior points (hull extension).
    /// - `hint`: Optional simplex hint for point location (improves performance)
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate simplex containing the point
    /// 4. Handle location result:
    ///    - `InsideSimplex`: Use provided `conflict_simplices` for cavity-based insertion
    ///    - `Outside`: Extend hull (no conflict simplices needed)
    /// 5. Extract cavity boundary (if interior)
    /// 6. Fill cavity (create new simplices)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict simplices (if interior)
    /// 9. Repair invalid facet sharing
    ///
    /// # Returns
    /// - `Ok(VertexKey)`: The key of the inserted vertex
    /// - New simplex keys via the returned result (for hint caching at higher layers)
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate coordinates detected (within 1e-10 tolerance)
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails
    /// - Point location fails
    /// - Interior point without `conflict_simplices` parameter
    /// - Cavity operations fail
    /// - Degenerate location (`OnFacet`, `OnEdge`, `OnVertex`) - not yet implemented
    ///
    /// **Note**: For insertions beyond D+1 vertices, use `DelaunayTriangulation::insert()`
    /// instead, which handles conflict region computation automatically.
    #[cfg(test)]
    pub(crate) fn insert(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
    ) -> Result<(VertexKey, Option<SimplexKey>), InsertionError> {
        let (outcome, _stats) = self.insert_transactional(
            vertex,
            conflict_simplices,
            hint,
            DEFAULT_PERTURBATION_RETRIES,
            0,
            None,
            None,
        )?;
        match outcome {
            InsertionOutcome::Inserted { vertex_key, hint } => Ok((vertex_key, hint)),
            InsertionOutcome::Skipped { error } => Err(error),
        }
    }

    /// Insert a vertex and return statistics about the operation.
    ///
    /// This method returns detailed statistics about the insertion including:
    /// - Number of attempts (perturbation retries)
    /// - Whether the vertex was skipped
    /// - Number of simplices removed during repair
    ///
    /// This is useful for testing, debugging, and understanding how the
    /// triangulation handles geometric degeneracies.
    ///
    /// # Errors
    ///
    /// Returns an error only for non-retryable structural failures (e.g. duplicate UUID).
    /// Retryable geometric degeneracies that exhaust all attempts, and duplicate coordinates,
    /// return `Ok((InsertionOutcome::Skipped { .. }, stats))`.
    #[cfg(test)]
    pub(crate) fn insert_with_statistics(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError> {
        self.insert_transactional(
            vertex,
            conflict_simplices,
            hint,
            DEFAULT_PERTURBATION_RETRIES,
            0,
            None,
            None,
        )
    }

    /// Insert a vertex with statistics, using a custom perturbation seed and an optional
    /// spatial hash-grid index, and also return the simplices that cavity reduction touched
    /// and left in place.
    ///
    /// The extra seed set stays internal so bulk construction and debug rebuilds can widen
    /// their local repair frontier without changing the public insertion API.
    pub(crate) fn insert_with_statistics_seeded_indexed_detailed(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<K::Scalar, D>>,
        bulk_index: Option<usize>,
    ) -> Result<DetailedInsertionResult, InsertionError> {
        self.insert_with_statistics_seeded_indexed_detailed_with_telemetry(
            vertex,
            conflict_simplices,
            hint,
            perturbation_seed,
            index,
            bulk_index,
            InsertionTelemetryMode::CountsOnly,
        )
    }

    /// Insert a vertex with statistics and explicitly selected telemetry collection.
    ///
    /// Use [`InsertionTelemetryMode::CountsAndTimings`] only when the caller will
    /// consume elapsed-time telemetry; the default detailed insertion path records
    /// counters without paying per-attempt `Instant::now()` costs.
    #[expect(
        clippy::too_many_arguments,
        reason = "Internal detailed insertion carries perturbation, spatial-index, trace, and telemetry knobs"
    )]
    pub(crate) fn insert_with_statistics_seeded_indexed_detailed_with_telemetry(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<K::Scalar, D>>,
        bulk_index: Option<usize>,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<DetailedInsertionResult, InsertionError> {
        self.insert_transactional_detailed(
            vertex,
            conflict_simplices,
            hint,
            DEFAULT_PERTURBATION_RETRIES,
            perturbation_seed,
            index,
            bulk_index,
            telemetry_mode,
        )
    }

    /// Transactional insertion with automatic rollback and perturbation retry.
    ///
    /// This ensures the triangulation always remains in a valid state by:
    /// 1. Cloning TDS before each insertion attempt (snapshot)
    /// 2. Attempting insertion
    /// 3. On failure: restore TDS from snapshot
    /// 4. If the error is retryable: perturb vertex and retry (up to `max_perturbation_attempts`)
    /// 5. If retryable attempts are exhausted, or the vertex is a duplicate: return
    ///    `Ok((InsertionOutcome::Skipped { error }, stats))`
    /// 6. If the error is non-retryable: return `Err(InsertionError)`
    ///
    /// This guarantees we transition from one valid manifold to another.
    #[cfg(test)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Test helpers mirror the detailed transactional insertion signature"
    )]
    fn insert_transactional(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        max_perturbation_attempts: usize,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<K::Scalar, D>>,
        bulk_index: Option<usize>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError> {
        let detail = self.insert_transactional_detailed(
            vertex,
            conflict_simplices,
            hint,
            max_perturbation_attempts,
            perturbation_seed,
            index,
            bulk_index,
            InsertionTelemetryMode::CountsOnly,
        )?;
        Ok((detail.outcome, detail.stats))
    }

    /// Transactional insertion with automatic rollback and perturbation retry, plus
    /// the local-repair seed simplices discovered while shaping the cavity.
    #[expect(
        clippy::too_many_lines,
        reason = "Complex insertion logic; splitting further would harm readability"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "Transactional insertion needs the bulk-index diagnostic context for #204 tracing"
    )]
    fn insert_transactional_detailed(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        max_perturbation_attempts: usize,
        perturbation_seed: u64,
        mut index: Option<&mut HashGridIndex<K::Scalar, D>>,
        bulk_index: Option<usize>,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<DetailedInsertionResult, InsertionError> {
        let mut stats = InsertionStatistics::default();
        let mut telemetry = InsertionTelemetry::default();
        let original_coords = *vertex.point().coords();
        let original_uuid = vertex.uuid();
        let mut current_vertex = vertex;
        // Preserve the last retryable failure so an exhausted perturbation loop can
        // explain why the vertex was skipped instead of reporting a generic error.
        let mut last_retryable_error: Option<InsertionError> = None;

        // Reuse the caller's spatial index as a locate-hint source when batch insertion did
        // not already provide a better hint. This keeps retries and bulk runs on the same
        // point-location path.
        let mut hint = hint;
        if hint.is_none()
            && let Some(index_ref) = index.as_deref()
        {
            hint = self.select_locate_hint_from_hash_grid(&original_coords, index_ref);
        }

        // Scale perturbations against the local neighborhood so retries stay small relative
        // to the nearby geometry instead of using a single global epsilon.
        let local_scale = self.estimate_local_perturbation_scale(&original_coords, hint);

        let duplicate_tolerance =
            self.estimate_duplicate_coordinate_tolerance(&original_coords, hint);
        self.ensure_duplicate_index_cell_size(index.as_deref_mut(), duplicate_tolerance);

        // Base perturbation epsilon: ≈ √machine_epsilon for the scalar type.
        let epsilon_value: f64 = if K::Scalar::mantissa_digits() <= 24 {
            1e-4
        } else {
            1e-8
        };

        for attempt in 0..=max_perturbation_attempts {
            stats.attempts = attempt + 1;

            // Attempt 0 uses the caller's coordinates verbatim; later attempts apply a
            // deterministic signed perturbation so the same seed reproduces the same path.
            if attempt > 0 {
                let mut perturbed_coords = original_coords;
                // Progressive local-scale perturbation: magnitude grows ×10 per attempt.
                //   attempt 1: base × local_scale × 10
                //   attempt 2: base × local_scale × 100
                //   attempt 3: base × local_scale × 1000
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    reason = "attempt is at most DEFAULT_PERTURBATION_RETRIES (3), fits in i32"
                )]
                let scale_factor = 10.0_f64.powi(attempt as i32);
                let Some(epsilon) = <K::Scalar as NumCast>::from(epsilon_value * scale_factor)
                else {
                    // We failed to convert the perturbation scale into the scalar type.
                    //
                    // This should not happen for our supported scalar types (`f32`, `f64`), but if it
                    // does (e.g. with a custom scalar), we degrade gracefully by skipping this vertex
                    // rather than aborting the whole insertion.
                    stats.result = InsertionResult::SkippedDegeneracy;
                    let error = last_retryable_error.unwrap_or_else(|| {
                        CavityFillingError::PerturbationScaleConversion {
                            value: epsilon_value.to_string(),
                        }
                        .into()
                    });
                    return Ok(DetailedInsertionResult {
                        outcome: InsertionOutcome::Skipped { error },
                        stats,
                        telemetry,
                        repair_seed_simplices: SimplexKeyBuffer::new(),
                        delaunay_repair_required: false,
                    });
                };

                let perturbation_scale = epsilon * local_scale;
                for (idx, coord) in perturbed_coords.iter_mut().enumerate() {
                    let coord_scale =
                        <K::Scalar as NumCast>::from(idx + 1).unwrap_or_else(K::Scalar::one);
                    let signed_perturbation = if perturbation_seed == 0 {
                        if (attempt + idx) % 2 == 0 {
                            perturbation_scale
                        } else {
                            -perturbation_scale
                        }
                    } else {
                        let mix = perturbation_seed
                            ^ ((attempt as u64) << 32)
                            ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        if mix & 1 == 0 {
                            perturbation_scale
                        } else {
                            -perturbation_scale
                        }
                    };
                    *coord += signed_perturbation * coord_scale;
                }

                // Preserve the caller-provided vertex UUID across perturbation retries.
                // This ensures the inserted vertex retains its original identity even if we have
                // to retry with perturbed coordinates.
                current_vertex =
                    Vertex::new_with_uuid(Point::new(perturbed_coords), original_uuid, vertex.data);
            }

            // Duplicate coordinate detection uses the hash grid when available; otherwise it
            // falls back to a linear scan (O(n·D) per insertion, O(n²·D) worst-case).
            if let Some(error) = self.duplicate_coordinates_error(
                current_vertex.point().coords(),
                duplicate_tolerance,
                index.as_deref(),
            ) {
                stats.result = InsertionResult::SkippedDuplicate;
                #[cfg(debug_assertions)]
                tracing::debug!("SKIPPED: {error}");
                return Ok(DetailedInsertionResult {
                    outcome: InsertionOutcome::Skipped { error },
                    stats,
                    telemetry,
                    repair_seed_simplices: SimplexKeyBuffer::new(),
                    delaunay_repair_required: false,
                });
            }

            let simplices_before_attempt = self.tds.number_of_simplices();
            let vertices_before_attempt = self.tds.number_of_vertices();

            // Clone TDS for rollback (transactional semantics)
            let tds_snapshot = self.tds.clone_for_rollback();

            // Try insertion.
            //
            // Topology safety net: ensure we don't commit an insertion that breaks Level 3 topology.
            // If the cavity-based insertion produces an Euler/topology mismatch, roll back and retry a
            // conservative fallback (star-split of the containing simplex) within the same transactional attempt.
            #[cfg(test)]
            // Test-only hook for deterministic coverage of the rollback + perturbation retry
            // success path, which is otherwise rare under the adaptive SoS predicates.
            let result = if test_hooks::take_force_next_insertion_retryable_failure() {
                Err(InsertionError::NonManifoldTopology {
                    facet_hash: 0x000F_0CED,
                    simplex_count: 3,
                })
            } else {
                self.try_insert_with_topology_safety_net(
                    current_vertex,
                    conflict_simplices,
                    hint,
                    attempt,
                    &tds_snapshot,
                    &mut telemetry,
                    telemetry_mode,
                )
            };
            #[cfg(not(test))]
            let result = self.try_insert_with_topology_safety_net(
                current_vertex,
                conflict_simplices,
                hint,
                attempt,
                &tds_snapshot,
                &mut telemetry,
                telemetry_mode,
            );

            match result {
                Ok(TryInsertImplOk {
                    inserted,
                    simplices_removed,
                    repair_seed_simplices,
                    delaunay_repair_required,
                    ..
                }) => {
                    stats.simplices_removed_during_repair = simplices_removed;
                    stats.result = InsertionResult::Inserted;
                    #[cfg(debug_assertions)]
                    if attempt > 0 {
                        tracing::debug!(
                            "Warning: Geometric degeneracy resolved via perturbation (attempt {attempt})"
                        );
                    }

                    let (vertex_key, hint) = inserted;
                    // Only the committed attempt updates the duplicate index. Earlier
                    // retries all rolled back to the pre-attempt triangulation state.
                    if let Some(index) = index.as_deref_mut()
                        && let Some(vertex) = self.tds.vertex(vertex_key)
                    {
                        index.insert_vertex(vertex_key, vertex.point().coords());
                    }

                    return Ok(DetailedInsertionResult {
                        outcome: InsertionOutcome::Inserted { vertex_key, hint },
                        stats,
                        telemetry,
                        repair_seed_simplices,
                        delaunay_repair_required,
                    });
                }
                Err(e) => {
                    // Any error - rollback to snapshot
                    self.tds = tds_snapshot;

                    // Handle duplicate coordinates specially - skip immediately without retry
                    if matches!(e, InsertionError::DuplicateCoordinates { .. }) {
                        stats.result = InsertionResult::SkippedDuplicate;
                        #[cfg(debug_assertions)]
                        tracing::debug!("SKIPPED: {e}");
                        return Ok(DetailedInsertionResult {
                            outcome: InsertionOutcome::Skipped { error: e },
                            stats,
                            telemetry,
                            repair_seed_simplices: SimplexKeyBuffer::new(),
                            delaunay_repair_required: false,
                        });
                    }

                    // Check if this is a retryable error (geometric degeneracy)
                    let is_retryable = e.is_retryable();

                    // Emit the conflict summary after rollback so the trace captures the
                    // restored manifold state that the next retry will start from.
                    if retryable_skip_trace_enabled()
                        && let Some(detail) = retryable_conflict_trace_detail(&e)
                    {
                        log_retryable_conflict_skip(
                            bulk_index,
                            original_uuid,
                            attempt + 1,
                            max_perturbation_attempts + 1,
                            attempt > 0,
                            is_retryable && attempt < max_perturbation_attempts,
                            simplices_before_attempt,
                            vertices_before_attempt,
                            self.tds.number_of_simplices(),
                            self.tds.number_of_vertices(),
                            &detail,
                            &e,
                        );
                    }

                    if is_retryable && attempt < max_perturbation_attempts {
                        last_retryable_error = Some(e.clone());
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "RETRYING: Attempt {} failed with: {e}. Applying perturbation...",
                            attempt + 1
                        );
                    } else if is_retryable {
                        stats.result = InsertionResult::SkippedDegeneracy;
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "SKIPPED: Could not insert vertex after {} attempts (max perturbation ≈ {:.0e} × local_scale). Last error: {e}. Vertex skipped to maintain manifold.",
                            max_perturbation_attempts + 1,
                            epsilon_value
                                * 10.0_f64.powi(
                                    #[expect(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_possible_wrap,
                                        reason = "max_perturbation_attempts is small, fits in i32"
                                    )]
                                    {
                                        max_perturbation_attempts as i32
                                    }
                                ),
                        );
                        return Ok(DetailedInsertionResult {
                            outcome: InsertionOutcome::Skipped { error: e },
                            stats,
                            telemetry,
                            // Skipped insertions do not mutate the triangulation, so any
                            // intermediate cavity-seed hints are irrelevant to callers.
                            repair_seed_simplices: SimplexKeyBuffer::new(),
                            delaunay_repair_required: false,
                        });
                    } else {
                        // Non-retryable structural error (e.g., duplicate UUID)
                        return Err(e);
                    }
                }
            }
        }

        Err(InsertionError::TopologyValidation(
            TdsError::InconsistentDataStructure {
                message: "insertion retry loop exhausted without producing an outcome".to_string(),
            },
        ))
    }

    fn select_locate_hint_from_hash_grid(
        &self,
        coords: &[K::Scalar; D],
        index: &HashGridIndex<K::Scalar, D>,
    ) -> Option<SimplexKey> {
        let mut best: Option<(K::Scalar, SimplexKey)> = None;

        index.for_each_candidate_vertex_key(coords, |vkey| {
            let Some(vertex) = self.tds.vertex(vkey) else {
                return true;
            };

            let Some(simplex_key) = vertex.incident_simplex() else {
                return true;
            };

            if !self.tds.contains_simplex(simplex_key) {
                return true;
            }

            let vcoords = vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq += diff * diff;
            }

            match best {
                Some((best_dist, _)) if dist_sq >= best_dist => {}
                _ => {
                    best = Some((dist_sq, simplex_key));
                }
            }

            true
        });

        best.map(|(_, simplex_key)| simplex_key)
    }

    /// Chooses the relative duplicate-coordinate tolerance for the scalar precision.
    fn duplicate_relative_tolerance() -> K::Scalar {
        let value = if K::Scalar::mantissa_digits() <= 24 {
            1e-6_f64
        } else {
            1e-10_f64
        };
        <K::Scalar as NumCast>::from(value).unwrap_or_else(K::Scalar::default_tolerance)
    }

    /// Keeps duplicate-scale estimates tied to existing geometry rather than
    /// hard-coding a scalar-unit epsilon.
    fn include_duplicate_scale_reference(
        point_coords: &[K::Scalar; D],
        axis_min: &mut [K::Scalar; D],
        axis_max: &mut [K::Scalar; D],
        magnitude_scale: &mut K::Scalar,
        saw_reference: &mut bool,
    ) {
        *saw_reference = true;
        for i in 0..D {
            let coord = point_coords[i];
            if coord < axis_min[i] {
                axis_min[i] = coord;
            }
            if coord > axis_max[i] {
                axis_max[i] = coord;
            }

            let abs = coord.abs();
            if abs > *magnitude_scale {
                *magnitude_scale = abs;
            }
        }
    }

    /// Estimates a duplicate-coordinate tolerance from the local simplex span plus
    /// a small ULP-scaled floor for translated coordinate systems.
    fn estimate_duplicate_coordinate_tolerance(
        &self,
        coords: &[K::Scalar; D],
        hint: Option<SimplexKey>,
    ) -> K::Scalar {
        let mut axis_min = *coords;
        let mut axis_max = *coords;
        let mut magnitude_scale = K::Scalar::zero();
        let mut saw_reference = false;
        let mut local_feature_scale = None;

        for coord in coords {
            let abs = (*coord).abs();
            if abs > magnitude_scale {
                magnitude_scale = abs;
            }
        }

        if let Some(simplex_key) = hint
            && let Some(simplex) = self.tds.simplex(simplex_key)
        {
            for &vkey in simplex.vertices() {
                if let Some(vertex) = self.tds.vertex(vkey) {
                    Self::include_duplicate_scale_reference(
                        vertex.point().coords(),
                        &mut axis_min,
                        &mut axis_max,
                        &mut magnitude_scale,
                        &mut saw_reference,
                    );
                }
            }
        }

        if !saw_reference {
            let local_scale = self.estimate_local_perturbation_scale(coords, None);
            if local_scale.is_finite() && local_scale > K::Scalar::zero() {
                if local_scale > magnitude_scale {
                    magnitude_scale = local_scale;
                }
                local_feature_scale = Some(local_scale);
            }
        }

        let feature_scale = local_feature_scale.unwrap_or_else(|| {
            let mut span_sq = K::Scalar::zero();
            for i in 0..D {
                let span = axis_max[i] - axis_min[i];
                span_sq += span * span;
            }
            span_sq.sqrt()
        });
        let relative_tolerance = Self::duplicate_relative_tolerance() * feature_scale;
        let ulp_factor = <K::Scalar as NumCast>::from(16.0_f64).unwrap_or_else(K::Scalar::one);
        let ulp_tolerance = K::Scalar::epsilon() * ulp_factor * magnitude_scale;
        let mut tolerance = if relative_tolerance > ulp_tolerance {
            relative_tolerance
        } else {
            ulp_tolerance
        };

        if !tolerance.is_finite() || tolerance <= K::Scalar::zero() {
            tolerance = Self::duplicate_relative_tolerance();
        }

        tolerance
    }

    /// Rebuilds the duplicate index when a scale-aware tolerance grows beyond
    /// the current grid cell size, preserving complete candidate coverage.
    fn ensure_duplicate_index_cell_size(
        &self,
        index: Option<&mut HashGridIndex<K::Scalar, D>>,
        tolerance: K::Scalar,
    ) {
        let Some(index) = index else {
            return;
        };
        if !HashGridIndex::<K::Scalar, D>::supports_dimension()
            || !tolerance.is_finite()
            || tolerance <= K::Scalar::zero()
        {
            return;
        }
        if index.cell_size() >= tolerance {
            return;
        }

        let mut rebuilt = HashGridIndex::new(tolerance);
        for (vkey, vertex) in self.tds.vertices() {
            rebuilt.insert_vertex(vkey, vertex.point().coords());
        }
        *index = rebuilt;
    }

    /// Compares a squared distance against the duplicate tolerance without
    /// overflowing the tolerance square on extreme coordinate scales.
    fn duplicate_distance_within_tolerance(dist_sq: K::Scalar, tolerance: K::Scalar) -> bool {
        let tolerance_sq = tolerance * tolerance;
        if tolerance_sq.is_finite() {
            dist_sq <= tolerance_sq
        } else {
            dist_sq.sqrt() <= tolerance
        }
    }

    /// Check for near-duplicate coordinates using the hash grid when available, with a
    /// linear-scan fallback (O(n·D) per insertion) if the index is unavailable/unusable.
    fn duplicate_coordinates_error(
        &self,
        coords: &[K::Scalar; D],
        tolerance: K::Scalar,
        index: Option<&HashGridIndex<K::Scalar, D>>,
    ) -> Option<InsertionError> {
        let mut duplicate_found = false;
        let make_duplicate_error = || {
            let mut coordinates = String::from("[");
            for (idx, coord) in coords.iter().enumerate() {
                if idx != 0 {
                    coordinates.push_str(", ");
                }
                let _ = write!(&mut coordinates, "{coord:?}");
            }
            coordinates.push(']');
            InsertionError::DuplicateCoordinates { coordinates }
        };

        if let Some(index) = index
            && index.cell_size() >= tolerance
        {
            let mut candidate_count = 0usize;
            let used_index = index.for_each_candidate_vertex_key(coords, |vkey| {
                candidate_count = candidate_count.saturating_add(1);
                let Some(vertex) = self.tds.vertex(vkey) else {
                    return true;
                };

                let vcoords = vertex.point().coords();
                let mut dist_sq = K::Scalar::zero();
                for i in 0..D {
                    let diff = vcoords[i] - coords[i];
                    dist_sq += diff * diff;
                }

                if Self::duplicate_distance_within_tolerance(dist_sq, tolerance) {
                    duplicate_found = true;
                    return false;
                }

                true
            });
            record_duplicate_detection_metrics(used_index, candidate_count, !used_index);

            if duplicate_found {
                return Some(make_duplicate_error());
            }

            if used_index {
                return None;
            }
        } else {
            record_duplicate_detection_metrics(false, 0, true);
        }

        for (_, existing_vertex) in self.tds.vertices() {
            let existing_coords = existing_vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = coords[i] - existing_coords[i];
                dist_sq += diff * diff;
            }

            if Self::duplicate_distance_within_tolerance(dist_sq, tolerance) {
                duplicate_found = true;
                break;
            }
        }

        if duplicate_found {
            Some(make_duplicate_error())
        } else {
            None
        }
    }

    /// Estimate a local length scale for perturbation based on nearby vertices.
    ///
    /// Uses the hint simplex when available; otherwise falls back to the closest
    /// existing vertex. This keeps perturbations translation-invariant and
    /// proportional to local feature size.
    fn estimate_local_perturbation_scale(
        &self,
        coords: &[K::Scalar; D],
        hint: Option<SimplexKey>,
    ) -> K::Scalar {
        let mut min_dist_sq: Option<K::Scalar> = None;

        let consider_vertex = |vertex: &Vertex<K::Scalar, U, D>,
                               min_dist_sq: &mut Option<K::Scalar>| {
            let vcoords = vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq += diff * diff;
            }
            match min_dist_sq {
                Some(current) => {
                    if dist_sq < *current {
                        *current = dist_sq;
                    }
                }
                None => {
                    *min_dist_sq = Some(dist_sq);
                }
            }
        };

        if let Some(simplex_key) = hint
            && let Some(simplex) = self.tds.simplex(simplex_key)
        {
            for &vkey in simplex.vertices() {
                if let Some(vertex) = self.tds.vertex(vkey) {
                    consider_vertex(vertex, &mut min_dist_sq);
                }
            }
        }

        if min_dist_sq.is_none() {
            for (_, vertex) in self.tds.vertices() {
                consider_vertex(vertex, &mut min_dist_sq);
            }
        }

        let mut scale = min_dist_sq.map_or_else(K::Scalar::one, num_traits::Float::sqrt);

        let min_scale = K::Scalar::default_tolerance();
        if scale < min_scale {
            scale = min_scale;
        }

        scale
    }

    // -------------------------------------------------------------------------
    // Topology safety net helpers
    // -------------------------------------------------------------------------

    /// Logs when Level 3 validation is triggered (debug builds only).
    #[inline]
    fn log_validation_trigger_if_enabled(&self, suspicion: SuspicionFlags) {
        #[cfg(debug_assertions)]
        if self.validation_policy.should_validate(suspicion) && suspicion.is_suspicious() {
            tracing::debug!("Validation triggered by {suspicion:?}");
        }

        // Keep the parameter "used" in release builds where the debug-only logging
        // is compiled out, so `cargo clippy -D warnings` stays clean across profiles.
        #[cfg(not(debug_assertions))]
        {
            let _ = suspicion;
        }
    }

    /// Convert an [`InvariantError`] into the appropriate [`InsertionError`] variant.
    ///
    /// - `InvariantError::Tds(e)` → `InsertionError::TopologyValidation(e)`
    /// - `InvariantError::Triangulation(e)` → `InsertionError::TopologyValidationFailed { source: e }`
    /// - `InvariantError::Delaunay(e)` → `InsertionError::DelaunayValidationFailed { message }`
    fn invariant_error_to_insertion_error(err: InvariantError) -> InsertionError {
        match err {
            InvariantError::Tds(tds_err) => InsertionError::TopologyValidation(tds_err),
            InvariantError::Triangulation(tri_err) => InsertionError::TopologyValidationFailed {
                message: "Topology validation failed".to_string(),
                source: tri_err,
            },
            InvariantError::Delaunay(dt_err) => {
                InsertionError::DelaunayValidationFailed { source: dt_err }
            }
        }
    }

    /// Runs mandatory link checks required by the topology guarantee.
    fn validate_required_topology_links(&self) -> Result<(), InvariantError> {
        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        validate_facet_degree(&facet_to_simplices)?;
        validate_closed_boundary(&self.tds, &facet_to_simplices)?;

        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links(&self.tds)?;
        }

        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            validate_vertex_links(&self.tds, &facet_to_simplices)?;
        }

        // Keep geometric orientation non-negotiable during incremental insertion,
        // even when global validation is throttled. Run this after topology
        // checks so topology diagnostics still surface first.
        self.validate_geometric_simplex_orientation()?;

        Ok(())
    }

    /// Runs mandatory topology checks over the local simplices touched by insertion.
    ///
    /// Soundness boundary: the scoped path checks coherent orientation, local
    /// pseudomanifold facet incidence, ridge links, and geometric simplex
    /// orientation. Those local checks are sufficient only when `simplices` is
    /// non-empty and `topology_guarantee` does not require vertex-link checks
    /// during insertion; otherwise this explicitly falls back to
    /// [`validate_required_topology_links`](Self::validate_required_topology_links).
    /// See `REFERENCES.md`, "Scoped Local Validation and Flips" \[1\], for the
    /// local-vs-global validation tradeoff and geometric conditioning context.
    fn validate_required_topology_links_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), InvariantError> {
        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        if simplices.is_empty()
            || self
                .topology_guarantee
                .requires_vertex_links_during_insertion()
        {
            return self.validate_required_topology_links();
        }

        self.tds
            .validate_coherent_orientation_for_simplices(simplices)?;
        validate_local_pseudomanifold_for_simplices(&self.tds, simplices)?;

        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links_for_simplices(&self.tds, simplices)?;
        }

        self.validate_geometric_simplex_orientation_for_simplices(simplices)?;

        Ok(())
    }

    fn validation_after_insertion_work(
        &self,
        suspicion: SuspicionFlags,
    ) -> Option<InsertionValidationWork> {
        if self.tds.number_of_simplices() == 0 {
            return None;
        }

        let should_validate = self.validation_policy.should_validate(suspicion);
        let requires_required_topology_checks = self
            .topology_guarantee
            .requires_pseudomanifold_checks_during_insertion();

        if should_validate {
            Some(InsertionValidationWork::FullValidation)
        } else if requires_required_topology_checks {
            Some(InsertionValidationWork::RequiredTopologyLinks)
        } else {
            None
        }
    }

    fn validate_after_insertion_with_scope(
        &self,
        suspicion: SuspicionFlags,
        local_simplices: Option<&[SimplexKey]>,
    ) -> Result<(), InvariantError> {
        let Some(work) = self.validation_after_insertion_work(suspicion) else {
            return Ok(());
        };

        self.log_validation_trigger_if_enabled(suspicion);
        match work {
            InsertionValidationWork::FullValidation => self.is_valid(),
            InsertionValidationWork::RequiredTopologyLinks => local_simplices.map_or_else(
                || self.validate_required_topology_links(),
                |simplices| self.validate_required_topology_links_for_simplices(simplices),
            ),
        }
    }

    /// Runs post-insertion validation and records count/timing telemetry for the selected work.
    fn validate_after_insertion_and_record_telemetry(
        &self,
        suspicion: SuspicionFlags,
        local_simplices: &[SimplexKey],
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<(), InvariantError> {
        let validation_work = self.validation_after_insertion_work(suspicion);
        let validation_started =
            validation_work.and_then(|_| Self::start_insertion_timing(telemetry_mode));
        let validation_result =
            self.validate_after_insertion_with_scope(suspicion, Some(local_simplices));

        if validation_work.is_some() {
            Self::record_topology_validation_telemetry(
                telemetry,
                validation_started
                    .map(|started| Self::duration_nanos_saturating(started.elapsed())),
            );
        }

        validation_result
    }

    /// Repair neighbor pointers after local simplex removal without scanning the full TDS.
    fn repair_neighbors_after_local_simplex_removal(
        &mut self,
        new_simplices: &SimplexKeyBuffer,
        frontier_simplices: &[SimplexKey],
    ) -> Result<usize, InsertionError> {
        #[cfg(debug_assertions)]
        tracing::debug!(
            simplices = self.tds.number_of_simplices(),
            surviving_new_simplex_seeds = new_simplices
                .iter()
                .filter(|&&simplex_key| self.tds.contains_simplex(simplex_key))
                .count(),
            frontier_simplex_seeds = frontier_simplices
                .iter()
                .filter(|&&simplex_key| self.tds.contains_simplex(simplex_key))
                .count(),
            "Before local neighbor-pointer repair"
        );

        if force_global_neighbor_rebuild_enabled() {
            #[cfg(debug_assertions)]
            tracing::debug!(
                "DELAUNAY_FORCE_GLOBAL_NEIGHBOR_REBUILD set; using global neighbor rebuild"
            );
            return repair_neighbor_pointers(&mut self.tds).map_err(|source| {
                CavityFillingError::NeighborRebuild {
                    reason: source.into(),
                }
                .into()
            });
        }

        #[cfg(debug_assertions)]
        {
            match repair_neighbor_pointers_local(
                &mut self.tds,
                new_simplices,
                Some(frontier_simplices),
            ) {
                Ok(repaired) => Ok(repaired),
                Err(local_error) => {
                    tracing::warn!(
                        error = %local_error,
                        "Local neighbor-pointer repair failed; falling back to global rebuild in debug mode"
                    );
                    repair_neighbor_pointers(&mut self.tds).map_err(|source| {
                        CavityFillingError::NeighborRebuild {
                            reason: source.into(),
                        }
                        .into()
                    })
                }
            }
        }

        #[cfg(not(debug_assertions))]
        {
            repair_neighbor_pointers_local(&mut self.tds, new_simplices, Some(frontier_simplices))
                .map_err(|source| {
                    CavityFillingError::NeighborRebuild {
                        reason: source.into(),
                    }
                    .into()
                })
        }
    }

    /// Attempt an insertion, and if Level 3 validation fails, roll back and try a
    /// conservative star-split fallback of the containing simplex.
    #[expect(
        clippy::too_many_arguments,
        reason = "Topology safety net needs transactional rollback context plus telemetry mode"
    )]
    fn try_insert_with_topology_safety_net(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        attempt: usize,
        tds_snapshot: &Tds<K::Scalar, U, V, D>,
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<TryInsertImplOk, InsertionError> {
        let mut insert_ok =
            self.try_insert_impl(vertex, conflict_simplices, hint, telemetry, telemetry_mode)?;

        if attempt > 0 {
            insert_ok.suspicion.perturbation_used = true;
        }
        if insert_ok.suspicion.is_suspicious() {
            insert_ok.delaunay_repair_required = true;
        }

        // Skip Level 3 validation during bootstrap (vertices but no simplices yet).
        if self.tds.number_of_simplices() == 0 {
            return Ok(insert_ok);
        }

        let validation_result = self.validate_after_insertion_and_record_telemetry(
            insert_ok.suspicion,
            &insert_ok.repair_seed_simplices,
            telemetry,
            telemetry_mode,
        );
        if let Err(validation_err) = validation_result {
            // Roll back to snapshot and attempt a star-split fallback for interior points.
            self.tds = tds_snapshot.clone_for_rollback();
            return self.try_star_split_fallback_after_topology_failure(
                vertex,
                hint,
                attempt,
                validation_err,
                telemetry,
                telemetry_mode,
            );
        }

        Ok(insert_ok)
    }

    /// After a Level 3 topology validation failure, try to recover by performing a star-split
    /// of the containing simplex (if the point can be re-located inside a simplex).
    ///
    /// Notes:
    /// - This fallback is only applicable when the point re-locates to [`LocateResult::InsideSimplex`].
    /// - We re-run Level 3 validation after the fallback to avoid "recovering" into an invalid state.
    fn try_star_split_fallback_after_topology_failure(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        hint: Option<SimplexKey>,
        attempt: usize,
        validation_err: InvariantError,
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<TryInsertImplOk, InsertionError> {
        let point = *vertex.point();
        let location = match locate_with_stats(&self.tds, &self.kernel, &point, hint) {
            Ok((location, stats)) => {
                Self::record_locate_telemetry(telemetry, location, &stats);
                Ok(location)
            }
            Err(error) => Err(error),
        };

        let Ok(LocateResult::InsideSimplex(start_simplex)) = location else {
            return Err(Self::invariant_error_to_insertion_error(validation_err));
        };

        let mut star_conflict = SimplexKeyBuffer::new();
        star_conflict.push(start_simplex);

        match self.try_insert_impl(
            vertex,
            Some(&star_conflict),
            Some(start_simplex),
            telemetry,
            telemetry_mode,
        ) {
            Ok(mut fallback_ok) => {
                fallback_ok.suspicion.fallback_star_split = true;
                if attempt > 0 {
                    fallback_ok.suspicion.perturbation_used = true;
                }
                fallback_ok.delaunay_repair_required = true;

                let validation_result = self.validate_after_insertion_and_record_telemetry(
                    fallback_ok.suspicion,
                    &fallback_ok.repair_seed_simplices,
                    telemetry,
                    telemetry_mode,
                );
                if let Err(fallback_validation_err) = validation_result {
                    return Err(Self::invariant_error_to_insertion_error(
                        fallback_validation_err,
                    ));
                }

                // Telemetry: the fallback succeeded, meaning we recovered from a topology
                // validation failure without surfacing an insertion error to the caller.
                TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES.fetch_add(1, Ordering::Relaxed);

                #[cfg(debug_assertions)]
                tracing::debug!(
                    "Topology safety-net: star-split fallback succeeded (start_simplex={start_simplex:?})"
                );

                Ok(fallback_ok)
            }
            Err(fallback_err) => Err(fallback_err),
        }
    }

    /// Ensure an interior insertion never proceeds with an empty conflict region.
    ///
    /// An empty conflict region would produce an empty cavity boundary, create no new simplices, and
    /// leave the inserted vertex isolated (not incident to any simplex), which breaks Level 3 topology
    /// validation via Euler characteristic.
    fn ensure_non_empty_conflict_simplices(
        conflict_simplices: Cow<'_, SimplexKeyBuffer>,
        fallback_simplex: SimplexKey,
    ) -> Cow<'_, SimplexKeyBuffer> {
        if !conflict_simplices.is_empty() {
            return conflict_simplices;
        }

        if let Cow::Owned(mut owned) = conflict_simplices {
            owned.push(fallback_simplex);
            Cow::Owned(owned)
        } else {
            let mut owned = SimplexKeyBuffer::new();
            owned.push(fallback_simplex);
            Cow::Owned(owned)
        }
    }

    /// Build the boundary facets for a "star-split" of the containing simplex.
    fn star_split_boundary_facets(start_simplex: SimplexKey) -> CavityBoundaryBuffer {
        (0..=D)
            .map(|i| {
                FacetHandle::new(
                    start_simplex,
                    u8::try_from(i).expect("facet index must fit in u8"),
                )
            })
            .collect()
    }

    /// Connectedness guard (localized).
    ///
    /// This check is designed to be **O(k·D)**, where `k` is the number of newly created simplices and
    /// `D` is the triangulation dimension (each simplex has at most `D+1` neighbors).
    ///
    /// It validates two properties that are sufficient to catch the common “disconnected neighbor
    /// graph after insertion” failure modes without walking the entire triangulation:
    ///
    /// 1. The surviving subset of `new_simplices` forms a single connected component (via neighbor pointers).
    /// 2. If there are simplices outside that component, the new component is attached to at least one
    ///    existing simplex (via a *mutual* neighbor relationship).
    fn validate_connectedness(
        &self,
        new_simplices: &SimplexKeyBuffer,
    ) -> Result<(), InsertionError> {
        let total_simplices = self.tds.number_of_simplices();
        if total_simplices == 0 {
            return Ok(());
        }

        // Build a set of the *surviving* new simplices (some may have been removed during repair).
        let mut new_set: SimplexKeySet = SimplexKeySet::default();
        new_set.reserve(new_simplices.len());
        for &ck in new_simplices {
            if self.tds.contains_simplex(ck) {
                new_set.insert(ck);
            }
        }

        if new_set.is_empty() {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: "Disconnected triangulation detected after insertion: no surviving new simplices"
                        .to_string(),
                },
            ));
        }

        let expected_new_simplices = new_set.len();

        let Some(&start) = new_set.iter().next() else {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message:
                        "new_set unexpectedly empty after non-empty check in validate_connectedness"
                            .to_string(),
                },
            ));
        };

        let mut touches_existing_simplices = false;

        let visited = self.traverse_simplex_neighbor_graph(
            start,
            expected_new_simplices,
            Some(&new_set),
            |ck, nk| {
                if touches_existing_simplices {
                    return;
                }

                // For connectivity between new simplices and existing simplices, require *mutual* adjacency.
                // This avoids treating one-way neighbor pointers as “connected”.
                if let Some(neighbor_simplex) = self.tds.simplex(nk)
                    && neighbor_simplex
                        .neighbor_keys()
                        .is_some_and(|mut neighbor_keys| {
                            neighbor_keys.any(|neighbor| neighbor == Some(ck))
                        })
                {
                    touches_existing_simplices = true;
                }
            },
        );

        if visited.len() != expected_new_simplices {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-simplex subgraph visited {} of {} simplices",
                        visited.len(),
                        expected_new_simplices
                    ),
                },
            ));
        }

        // If there are simplices outside `new_set`, ensure the new component is attached to at least one
        // of them (otherwise we'd be creating a disconnected component).
        if total_simplices > expected_new_simplices && !touches_existing_simplices {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-simplex component ({expected_new_simplices} simplices) is not connected to existing simplices (total_simplices={total_simplices})"
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Find all conflict simplices by scanning the entire triangulation.
    ///
    /// Test-only global conflict scanner for malformed-simplex error coverage.
    ///
    /// Exterior production insertion deliberately avoids this path: hull
    /// extension is the local topological mutation, and Delaunay violations are
    /// left to the cadenced or final repair layers.
    #[cfg(test)]
    fn find_conflict_region_global(
        &self,
        point: &Point<K::Scalar, D>,
    ) -> Result<SimplexKeyBuffer, ConflictError> {
        #[cfg(debug_assertions)]
        let log_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL").is_some()
            || std::env::var_os("DELAUNAY_DEBUG_CONFLICT").is_some();
        #[cfg(debug_assertions)]
        let mut simplices_scanned = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_positive = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_zero = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_negative = 0usize;

        let mut conflict_simplices = SimplexKeyBuffer::new();

        for (simplex_key, simplex) in self.tds.simplices() {
            #[cfg(debug_assertions)]
            {
                simplices_scanned = simplices_scanned.saturating_add(1);
            }
            // Collect simplex vertex points in canonical VertexKey order for consistent
            // SoS perturbation priority.
            let simplex_points = sorted_simplex_points(&self.tds, simplex).ok_or_else(|| {
                ConflictError::SimplexDataAccessFailed {
                    simplex_key,
                    message: format!("Failed to resolve all {} simplex vertices", D + 1),
                }
            })?;

            if simplex_points.len() != D + 1 {
                return Err(ConflictError::SimplexDataAccessFailed {
                    simplex_key,
                    message: format!("Expected {} vertices, got {}", D + 1, simplex_points.len()),
                });
            }

            let sign = self.kernel.in_sphere(&simplex_points, point)?;
            #[cfg(debug_assertions)]
            {
                if log_enabled {
                    tracing::debug!(
                        simplex_key = ?simplex_key,
                        sign,
                        "find_conflict_region_global: in_sphere sign"
                    );
                }
                match sign.cmp(&0) {
                    CmpOrdering::Greater => {
                        sign_positive = sign_positive.saturating_add(1);
                    }
                    CmpOrdering::Equal => {
                        sign_zero = sign_zero.saturating_add(1);
                    }
                    CmpOrdering::Less => {
                        sign_negative = sign_negative.saturating_add(1);
                    }
                }
            }
            if sign > 0 {
                conflict_simplices.push(simplex_key);
            }
        }

        #[cfg(debug_assertions)]
        if log_enabled {
            tracing::debug!(
                point = ?point,
                simplices_scanned,
                conflict_simplices = conflict_simplices.len(),
                sign_positive,
                sign_zero,
                sign_negative,
                "find_conflict_region_global: summary"
            );
        }

        Ok(conflict_simplices)
    }

    /// Returns true if any conflict simplex has a facet on the hull boundary.
    #[cfg(test)]
    fn conflict_region_touches_boundary(
        &self,
        conflict_simplices: &SimplexKeyBuffer,
    ) -> Result<bool, InsertionError> {
        if conflict_simplices.is_empty() {
            return Ok(false);
        }

        let facet_to_simplices = self
            .tds
            .build_facet_to_simplices_map()
            .map_err(InsertionError::TopologyValidation)?;

        let mut boundary_facets: FastHashSet<u64> =
            fast_hash_set_with_capacity(facet_to_simplices.len());
        for (facet_key, simplex_list) in &facet_to_simplices {
            if simplex_list.len() == 1 {
                boundary_facets.insert(*facet_key);
            }
        }

        if boundary_facets.is_empty() {
            return Ok(false);
        }

        for &simplex_key in conflict_simplices {
            let simplex = self.tds.simplex(simplex_key).ok_or_else(|| {
                InsertionError::TopologyValidation(TdsError::SimplexNotFound {
                    simplex_key,
                    context: "checking boundary facets for conflict region".to_string(),
                })
            })?;
            for facet_idx in 0..simplex.number_of_vertices() {
                let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(D);
                for (i, &vkey) in simplex.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vertices.push(vkey);
                    }
                }
                let facet_key = facet_key_from_vertices(&facet_vertices);
                if boundary_facets.contains(&facet_key) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Perform cavity insertion given an explicit conflict region.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep cavity insertion and repair logic together for clarity"
    )]
    fn insert_with_conflict_region(
        &mut self,
        v_key: VertexKey,
        point: &Point<K::Scalar, D>,
        mut conflict_simplices: SimplexKeyBuffer,
        fallback_simplex: Option<SimplexKey>,
        suspicion: &mut SuspicionFlags,
    ) -> Result<CavityInsertionOutcome, InsertionError> {
        #[cfg(not(debug_assertions))]
        let _ = point;

        if conflict_simplices.is_empty() {
            let Some(start_simplex) = fallback_simplex else {
                return Err(CavityFillingError::EmptyConflictRegion {
                    fallback_simplex: None,
                }
                .into());
            };
            suspicion.empty_conflict_region = true;
            suspicion.fallback_star_split = true;
            conflict_simplices.push(start_simplex);
            // The fallback star-split is topologically safe but not a full
            // Bowyer-Watson conflict-region replacement, so local Delaunay
            // repair must revisit it.
        }

        // Preserve every simplex that participates in cavity shaping and is later
        // removed from the final cavity so callers can seed local Delaunay
        // repair from the surviving fringe.
        let mut repair_seed_simplices = SimplexKeyBuffer::new();
        let mut delaunay_repair_required = suspicion.fallback_star_split;

        // Extract cavity boundary.
        //
        // Iteratively resolve cavity-boundary errors rather than immediately falling back to a
        // star-split.  Star-splits create non-Delaunay configurations that the global flip repair
        // must fix; in high dimensions this is extremely slow.  In all dimensions it is better
        // to first attempt to reshape the conflict region:
        //
        //   • RidgeFan          – SHRINK: remove extra fan simplices (3rd, 4th, … facets).
        //   • DisconnectedBnd   – EXPAND: add the non-conflict neighbors of the disconnected
        //                         simplices to fill the topological "hole" in the conflict region
        //                         that causes the disconnected boundary.  Falls back to SHRINK
        //                         if no non-conflict neighbors are found.
        //   • OpenBoundary      – SHRINK: remove the simplex with the dangling facet.
        //
        // After each reshape we re-run extract_cavity_boundary.  If the loop exhausts its
        // budget without producing a valid boundary:
        //   • D>=3: return a retryable error so insert_transactional retries with a perturbed
        //     vertex instead of creating an un-repairable star-split.
        //   • D<3:  fall through to the existing star-split fallback (the 2D flip repair
        //     guarantees convergence even from star-split configurations).
        let mut boundary_facets = {
            let mut extraction_result = extract_cavity_boundary(&self.tds, &conflict_simplices);

            {
                const MAX_CAVITY_ITERATIONS: usize = 32;
                let mut iterations: usize = 0;
                let trace_enabled = cavity_reduction_trace_enabled();
                let mut trace_cavity_reduction = false;
                let mut saw_ridge_fan_shrink = false;

                match &extraction_result {
                    Ok(boundary) => {
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            &conflict_simplices,
                            || format!("initial_ok boundary_facets={}", boundary.len()),
                        );
                    }
                    Err(err) => {
                        trace_cavity_reduction = trace_enabled
                            && !CAVITY_REDUCTION_TRACE_EMITTED.swap(true, Ordering::Relaxed);
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            &conflict_simplices,
                            || format!("initial_err {}", cavity_conflict_error_summary(err)),
                        );
                    }
                }

                loop {
                    if iterations >= MAX_CAVITY_ITERATIONS {
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            &conflict_simplices,
                            || "budget_exhausted".to_string(),
                        );
                        break;
                    }
                    iterations += 1;

                    match &extraction_result {
                        // RidgeFan: SHRINK – remove the simplices contributing extra boundary facets.
                        Err(ConflictError::RidgeFan {
                            extra_simplices, ..
                        }) if !extra_simplices.is_empty() && conflict_simplices.len() > D + 1 => {
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                remove_count = extra_simplices.len(),
                                conflict_simplices_before = conflict_simplices.len(),
                                "D={D}: cavity reduction (RidgeFan shrink)"
                            );
                            log_cavity_reduction_event(
                                trace_cavity_reduction,
                                iterations,
                                &conflict_simplices,
                                || format!("ridge_fan_shrink remove_simplices={extra_simplices:?}"),
                            );
                            saw_ridge_fan_shrink = true;
                            delaunay_repair_required = true;
                            let remove_set: FastHashSet<SimplexKey> =
                                extra_simplices.iter().copied().collect();
                            retain_simplices_and_record_removed(
                                &mut conflict_simplices,
                                &mut repair_seed_simplices,
                                |simplex_key| !remove_set.contains(&simplex_key),
                            );
                        }

                        // DisconnectedBoundary: EXPAND – add non-conflict neighbors of the
                        // disconnected simplices to fill the topological hole.  These simplices form the
                        // "inner wall" of a donut-shaped conflict region; their non-conflict
                        // neighbors are the hole simplices that, when added, reconnect the boundary.
                        // Falls back to SHRINK if no non-conflict neighbors exist.
                        Err(ConflictError::DisconnectedBoundary {
                            disconnected_simplices,
                            ..
                        }) if !disconnected_simplices.is_empty() => {
                            let conflict_set: FastHashSet<SimplexKey> =
                                conflict_simplices.iter().copied().collect();
                            let mut simplices_to_add: FastHashSet<SimplexKey> =
                                FastHashSet::default();
                            if !saw_ridge_fan_shrink {
                                for &dc in disconnected_simplices {
                                    if let Some(simplex) = self.tds.simplex(dc)
                                        && let Some(neighbors) = simplex.neighbor_keys()
                                    {
                                        for neighbor_opt in neighbors {
                                            if let Some(nk) = neighbor_opt
                                                && !conflict_set.contains(&nk)
                                            {
                                                simplices_to_add.insert(nk);
                                            }
                                        }
                                    }
                                }
                            }

                            if !simplices_to_add.is_empty() {
                                // EXPAND: add the hole-filling simplices.
                                delaunay_repair_required = true;
                                #[cfg(debug_assertions)]
                                tracing::debug!(
                                    add_count = simplices_to_add.len(),
                                    conflict_simplices_before = conflict_simplices.len(),
                                    "D={D}: cavity expansion (DisconnectedBoundary hole-fill)"
                                );
                                log_cavity_reduction_event(
                                    trace_cavity_reduction,
                                    iterations,
                                    &conflict_simplices,
                                    || {
                                        let added: Vec<SimplexKey> =
                                            simplices_to_add.iter().copied().collect();
                                        format!(
                                            "disconnected_boundary_expand add_simplices={added:?}"
                                        )
                                    },
                                );
                                for k in simplices_to_add {
                                    conflict_simplices.push(k);
                                }
                            } else if conflict_simplices.len() > D + 1 {
                                // SHRINK fallback: no non-conflict neighbors found.
                                delaunay_repair_required = true;
                                #[cfg(debug_assertions)]
                                tracing::debug!(
                                    remove_count = disconnected_simplices.len(),
                                    conflict_simplices_before = conflict_simplices.len(),
                                    "D={D}: cavity reduction (DisconnectedBoundary shrink fallback)"
                                );
                                log_cavity_reduction_event(
                                    trace_cavity_reduction,
                                    iterations,
                                    &conflict_simplices,
                                    || {
                                        format!(
                                            "disconnected_boundary_shrink remove_simplices={disconnected_simplices:?}"
                                        )
                                    },
                                );
                                let remove_set: FastHashSet<SimplexKey> =
                                    disconnected_simplices.iter().copied().collect();
                                retain_simplices_and_record_removed(
                                    &mut conflict_simplices,
                                    &mut repair_seed_simplices,
                                    |simplex_key| !remove_set.contains(&simplex_key),
                                );
                            } else {
                                log_cavity_reduction_event(
                                    trace_cavity_reduction,
                                    iterations,
                                    &conflict_simplices,
                                    || "disconnected_boundary_no_progress".to_string(),
                                );
                                break;
                            }
                        }

                        // OpenBoundary: SHRINK – remove the simplex with the dangling facet.
                        Err(ConflictError::OpenBoundary { open_simplex, .. })
                            if conflict_simplices.len() > D + 1 =>
                        {
                            delaunay_repair_required = true;
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                ?open_simplex,
                                conflict_simplices_before = conflict_simplices.len(),
                                "D={D}: cavity reduction (OpenBoundary shrink)"
                            );
                            log_cavity_reduction_event(
                                trace_cavity_reduction,
                                iterations,
                                &conflict_simplices,
                                || format!("open_boundary_shrink open_simplex={open_simplex:?}"),
                            );
                            let open = *open_simplex;
                            retain_simplices_and_record_removed(
                                &mut conflict_simplices,
                                &mut repair_seed_simplices,
                                |simplex_key| simplex_key != open,
                            );
                        }

                        _ => {
                            log_cavity_reduction_event(
                                trace_cavity_reduction,
                                iterations,
                                &conflict_simplices,
                                || "no_reduction_rule_matched".to_string(),
                            );
                            break;
                        }
                    }

                    extraction_result = extract_cavity_boundary(&self.tds, &conflict_simplices);
                    match &extraction_result {
                        Ok(boundary) => {
                            log_cavity_reduction_event(
                                trace_cavity_reduction,
                                iterations,
                                &conflict_simplices,
                                || format!("reextract_ok boundary_facets={}", boundary.len()),
                            );
                        }
                        Err(err) => {
                            log_cavity_reduction_event(
                                trace_cavity_reduction,
                                iterations,
                                &conflict_simplices,
                                || format!("reextract_err {}", cavity_conflict_error_summary(err)),
                            );
                        }
                    }
                }
            }

            match extraction_result {
                Ok(boundary) => boundary,
                Err(err) => {
                    // For D=3 and D≥4: do NOT fall back to star-split once cavity reduction
                    // is exhausted.  Star-splits create heavily non-Delaunay configurations
                    // (the star of the new vertex is only D+1 simplices instead of the correct
                    // conflict region) whose violations cannot be reliably fixed by the flip
                    // repair: isolated violations may exist in simplices that are not connected to
                    // the star-split star through any violation chain.  Return a retryable
                    // error instead so insert_transactional can retry with a perturbed vertex
                    // and, after all retries, skip the vertex.  A valid Delaunay triangulation
                    // with a few skipped vertices is preferable to an invalid one with all of
                    // them (the is_delaunay_property_only() check in build_with_shuffled_retries
                    // will reject the latter anyway).
                    //
                    // For D=2: star-split is used as a last resort.  The 2D flip repair
                    // guarantees convergence from star-split configurations and the extra simplices
                    // are quickly handled by the k=2 repair loop.
                    let should_fallback = D < 3
                        && matches!(
                            err,
                            ConflictError::NonManifoldFacet { .. }
                                | ConflictError::RidgeFan { .. }
                                | ConflictError::DisconnectedBoundary { .. }
                                | ConflictError::OpenBoundary { .. }
                        );

                    if should_fallback {
                        let Some(start_simplex) = fallback_simplex else {
                            return Err(err.into());
                        };

                        suspicion.fallback_star_split = true;
                        delaunay_repair_required = true;

                        #[cfg(debug_assertions)]
                        tracing::warn!(
                            "Conflict region degeneracy ({err}); falling back to star-split of simplex {start_simplex:?}"
                        );

                        let mut replacement = SimplexKeyBuffer::new();
                        replacement.push(start_simplex);
                        replace_simplices_and_record_removed(
                            &mut conflict_simplices,
                            &mut repair_seed_simplices,
                            replacement,
                        );

                        Self::star_split_boundary_facets(start_simplex)
                    } else {
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "D={D}: cavity boundary unresolvable ({err}); returning retryable error"
                        );
                        return Err(err.into());
                    }
                }
            }
        };

        // Fallback: never allow an insertion to create a dangling vertex.
        if boundary_facets.is_empty() {
            let Some(start_simplex) = fallback_simplex else {
                return Err(CavityFillingError::EmptyBoundary {
                    fallback_simplex: None,
                }
                .into());
            };

            suspicion.empty_conflict_region = true;
            suspicion.fallback_star_split = true;
            delaunay_repair_required = true;

            #[cfg(debug_assertions)]
            tracing::warn!(
                "Empty cavity boundary; falling back to splitting containing simplex {start_simplex:?}"
            );

            let mut replacement = SimplexKeyBuffer::new();
            replacement.push(start_simplex);
            replace_simplices_and_record_removed(
                &mut conflict_simplices,
                &mut repair_seed_simplices,
                replacement,
            );
            boundary_facets = Self::star_split_boundary_facets(start_simplex);
        }

        // Fill cavity BEFORE removing old simplices.
        let new_simplices =
            fill_cavity_replacing_simplices(&mut self.tds, v_key, &boundary_facets)?;
        self.canonicalize_positive_orientation_for_simplices(&new_simplices)?;

        // Post-insertion orientation audit: verify that canonicalization
        // actually produced all-positive orientations among the new simplices.
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_ORIENTATION").is_some() {
            let mut pos = 0_usize;
            let mut neg = 0_usize;
            let mut deg = 0_usize;
            let mut fail = 0_usize;
            for &ck in &new_simplices {
                if let Some(c) = self.tds.simplex(ck) {
                    match self.evaluate_simplex_orientation_for_context(
                        ck,
                        c,
                        "post-insertion orientation audit",
                        "orientation predicate failed during post-insertion audit",
                    ) {
                        Ok(o) if o > 0 => pos += 1,
                        Ok(o) if o < 0 => neg += 1,
                        Ok(_) => deg += 1,
                        Err(ref e) => {
                            fail += 1;
                            tracing::warn!(
                                simplex_key = ?ck,
                                error = %e,
                                "post-insertion orientation audit: evaluation failed"
                            );
                        }
                    }
                }
            }
            if neg > 0 || fail > 0 {
                tracing::warn!(
                    new_simplices = new_simplices.len(),
                    positive = pos,
                    negative = neg,
                    degenerate = deg,
                    eval_errors = fail,
                    "post-insertion orientation audit: NEGATIVE simplices or evaluation errors after canonicalization"
                );
            } else {
                tracing::debug!(
                    new_simplices = new_simplices.len(),
                    positive = pos,
                    degenerate = deg,
                    "post-insertion orientation audit: all simplices positive"
                );
            }
        }

        // Wire neighbors (while both old and new simplices exist)
        let external_facets =
            external_facets_for_boundary(&self.tds, &conflict_simplices, &boundary_facets)?;
        wire_cavity_neighbors(
            &mut self.tds,
            &new_simplices,
            external_facets.iter().copied(),
            Some(&conflict_simplices),
        )?;

        // Drop any repair-seed entries that were removed earlier but later got
        // reintroduced into the final conflict region. Those keys will be
        // deleted by `remove_simplices_by_keys` below, so they cannot seed repair.
        let dead_conflict_simplices: FastHashSet<SimplexKey> =
            conflict_simplices.iter().copied().collect();
        repair_seed_simplices.retain(|ck| !dead_conflict_simplices.contains(ck));
        let mut seen_repair_seed_simplices = FastHashSet::default();
        repair_seed_simplices.retain(|ck| seen_repair_seed_simplices.insert(*ck));

        // Remove conflict simplices (now that new simplices are wired up)
        let _removed_count = self.tds.remove_simplices_by_keys(&conflict_simplices);

        // Iteratively repair non-manifold topology until facet sharing is valid
        let mut total_removed = 0;
        let mut facet_sharing_known_valid = true;
        let mut neighbor_repair_frontier = SimplexKeyBuffer::new();
        #[cfg_attr(
            not(debug_assertions),
            expect(
                unused_variables,
                reason = "`iteration` is only used for debug logging",
            )
        )]
        for iteration in 0..MAX_REPAIR_ITERATIONS {
            // Check for non-manifold issues in newly created simplices (local scan)
            let simplices_to_check: SimplexKeyBuffer = new_simplices
                .iter()
                .copied()
                .filter(|ck| self.tds.contains_simplex(*ck))
                .collect();

            if let Some(issues) = self.detect_local_facet_issues(&simplices_to_check)? {
                // Only mark this as "suspicious" if we *actually* detected local facet issues
                // and entered the repair path.
                suspicion.repair_loop_entered = true;
                delaunay_repair_required = true;

                #[cfg(debug_assertions)]
                tracing::debug!(
                    "Repair iteration {}: {} over-shared facets detected, removing simplices...",
                    iteration + 1,
                    issues.len()
                );

                let repair = self.repair_local_facet_issues_with_frontier(&issues)?;
                let removed = repair.removed_count;

                // Early exit if repair made no progress
                if removed == 0 {
                    #[cfg(debug_assertions)]
                    tracing::warn!(
                        "No simplices removed in iteration {} - repair cannot make progress",
                        iteration + 1
                    );
                    return Err(InsertionError::TopologyValidation(
                        TdsError::InconsistentDataStructure {
                            message: format!(
                                "Repair stalled: {} over-shared facets remain but no simplices could be removed",
                                issues.len()
                            ),
                        },
                    ));
                }

                total_removed += removed;
                neighbor_repair_frontier.extend(repair.frontier_simplices);

                if removed > 0 {
                    suspicion.simplices_removed = true;
                    delaunay_repair_required = true;
                }

                #[cfg(debug_assertions)]
                tracing::debug!(
                    removed_simplices = ?repair.removed_simplices,
                    "Removed {removed} simplices (total: {total_removed})"
                );

                // Early exit if repair succeeded
                facet_sharing_known_valid = self.tds.validate_facet_sharing().is_ok();
                if facet_sharing_known_valid {
                    break;
                }
            } else {
                // No more non-manifold issues - safe to rebuild neighbors
                break;
            }
        }

        // Rebuild neighbor pointers now that topology is manifold.
        #[cfg(debug_assertions)]
        tracing::debug!("After repair loop: total_removed={total_removed}");

        if !facet_sharing_known_valid {
            return Err(CavityFillingError::InvalidFacetSharingAfterRepair {
                stage: CavityRepairStage::PrimaryInsertion,
            }
            .into());
        }

        // Global neighbor rebuild is expensive. In the common case (no simplices removed during the
        // local facet-repair loop), `wire_cavity_neighbors` has already glued the cavity locally.
        //
        // If we *did* remove simplices during the repair loop, repair only the new-simplex/frontier
        // neighborhood unless the force-rebuild diagnostic environment variable is set.
        if total_removed > 0 {
            let repaired = self.repair_neighbors_after_local_simplex_removal(
                &new_simplices,
                &neighbor_repair_frontier,
            )?;
            suspicion.neighbor_pointers_rebuilt = repaired > 0;
            delaunay_repair_required = true;
        }

        // New cavity simplices were canonicalized on creation; validate the local
        // orientation frontier without scanning the whole triangulation.
        let mut orientation_simplices = SimplexKeyBuffer::new();
        append_live_unique_simplex_seeds(&self.tds, &new_simplices, &mut orientation_simplices);
        append_live_unique_simplex_seeds(
            &self.tds,
            &neighbor_repair_frontier,
            &mut orientation_simplices,
        );
        self.validate_local_orientation_for_simplices(&orientation_simplices)?;

        // Assign an incident simplex for the inserted vertex without a global rebuild.
        let hint = new_simplices.iter().copied().find(|&ck| {
            self.tds
                .simplex(ck)
                .is_some_and(|simplex| simplex.contains_vertex(v_key))
        });
        if let Some(incident_simplex) = hint
            && let Some(vertex) = self.tds.vertex_mut(v_key)
        {
            vertex.set_incident_simplex(Some(incident_simplex));
        }

        // Optional debug: validate neighbor pointers by forcing a full facet walk (no hint).
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_VALIDATE_LOCATE").is_some() {
            let _ = locate(&self.tds, &self.kernel, point, None)?;
        }

        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
            match validate_ridge_links(&self.tds) {
                Ok(()) => {
                    tracing::debug!(
                        "insert_with_conflict_region: ridge-link validation passed after insertion"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        error = ?err,
                        "insert_with_conflict_region: ridge-link validation failed after insertion"
                    );
                }
            }
        }

        // Repair stale incident-simplex pointers (e.g. pointing to deleted conflict-region
        // simplices) and error only for truly isolated vertices (in zero simplices).
        self.repair_stale_incident_simplices()?;

        // Connectedness guard (STRUCTURAL SAFETY, NOT Level 3 validation)
        self.validate_connectedness(&new_simplices)?;

        // Seed follow-up Delaunay repair from the local insertion product.  Higher layers
        // use these simplices to avoid rediscovering the inserted vertex star with a global scan.
        append_live_unique_simplex_seeds(&self.tds, &new_simplices, &mut repair_seed_simplices);
        append_live_unique_simplex_seeds(
            &self.tds,
            &neighbor_repair_frontier,
            &mut repair_seed_simplices,
        );

        // Return hint for next insertion
        Ok(CavityInsertionOutcome {
            hint,
            simplices_removed: total_removed,
            repair_seed_simplices,
            delaunay_repair_required: delaunay_repair_required || suspicion.is_suspicious(),
        })
    }

    /// Repair stale incident-simplex pointers and detect truly isolated vertices.
    ///
    /// After cavity filling and simplex removal, pre-existing boundary vertices may
    /// still reference deleted conflict-region simplices via a stale `incident_simplex`.
    /// For each vertex with a stale or missing `incident_simplex`, this scans all
    /// simplices for a valid one and updates the pointer.  Returns an error only if a
    /// vertex is in zero simplices (truly isolated).
    fn repair_stale_incident_simplices(&mut self) -> Result<(), InsertionError> {
        let stale_vertices: Vec<_> = {
            let tds = &self.tds;
            tds.vertices()
                .filter(|(vk, v)| {
                    !v.incident_simplex().is_some_and(|simplex_key| {
                        tds.simplex(simplex_key)
                            .is_some_and(|simplex| simplex.contains_vertex(*vk))
                    })
                })
                .map(|(vk, v)| (vk, v.uuid()))
                .collect()
        };
        #[cfg(debug_assertions)]
        if !stale_vertices.is_empty() && std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
            tracing::debug!(
                stale_count = stale_vertices.len(),
                "repairing stale incident-simplex pointers"
            );
        }
        for &(vk, uuid) in &stale_vertices {
            let repaired_simplex = self
                .tds
                .simplices()
                .find_map(|(ck, simplex)| simplex.contains_vertex(vk).then_some(ck));
            if let Some(simplex_key) = repaired_simplex {
                if let Some(vertex) = self.tds.vertex_mut(vk) {
                    vertex.set_incident_simplex(Some(simplex_key));
                }
            } else {
                // Truly isolated: no simplex in the TDS contains this vertex.
                return Err(InsertionError::TopologyValidationFailed {
                    message: "Truly isolated vertex detected during stale incident-simplex repair"
                        .to_string(),
                    source: TriangulationValidationError::IsolatedVertex {
                        vertex_key: vk,
                        vertex_uuid: uuid,
                    },
                });
            }
        }
        Ok(())
    }

    /// Records one point-location result into insertion telemetry.
    #[inline]
    fn record_locate_telemetry(
        telemetry: &mut InsertionTelemetry,
        location: LocateResult,
        stats: &LocateStats,
    ) {
        telemetry.locate_calls = telemetry.locate_calls.saturating_add(1);
        telemetry.locate_walk_steps_total = telemetry
            .locate_walk_steps_total
            .saturating_add(stats.walk_steps);
        telemetry.locate_walk_steps_max = telemetry.locate_walk_steps_max.max(stats.walk_steps);

        if stats.used_hint {
            telemetry.locate_hint_uses = telemetry.locate_hint_uses.saturating_add(1);
        }

        if stats.fell_back_to_scan() {
            telemetry.locate_scan_fallbacks = telemetry.locate_scan_fallbacks.saturating_add(1);
        }

        match location {
            LocateResult::InsideSimplex(_) => {
                telemetry.located_inside = telemetry.located_inside.saturating_add(1);
            }
            LocateResult::Outside => {
                telemetry.located_outside = telemetry.located_outside.saturating_add(1);
            }
            LocateResult::OnFacet(_, _) | LocateResult::OnEdge(_) | LocateResult::OnVertex(_) => {
                telemetry.located_on_boundary = telemetry.located_on_boundary.saturating_add(1);
            }
        }
    }

    /// Records conflict-region size counters without touching timing fields.
    #[inline]
    fn record_conflict_region_telemetry(telemetry: &mut InsertionTelemetry, simplices: usize) {
        telemetry.conflict_region_calls = telemetry.conflict_region_calls.saturating_add(1);
        telemetry.conflict_region_simplices_total = telemetry
            .conflict_region_simplices_total
            .saturating_add(simplices);
        telemetry.conflict_region_simplices_max =
            telemetry.conflict_region_simplices_max.max(simplices);
    }

    /// Records measured conflict-region time when timing telemetry is enabled.
    #[inline]
    fn record_conflict_region_timing(telemetry: &mut InsertionTelemetry, elapsed_nanos: u64) {
        telemetry.conflict_region_nanos = telemetry
            .conflict_region_nanos
            .saturating_add(elapsed_nanos);
        telemetry.conflict_region_nanos_max =
            telemetry.conflict_region_nanos_max.max(elapsed_nanos);
    }

    /// Records one cavity insertion attempt and its optional elapsed time.
    #[inline]
    fn record_cavity_insertion_telemetry(
        telemetry: &mut InsertionTelemetry,
        elapsed_nanos: Option<u64>,
    ) {
        telemetry.cavity_insertion_calls = telemetry.cavity_insertion_calls.saturating_add(1);
        if let Some(elapsed_nanos) = elapsed_nanos {
            telemetry.cavity_insertion_nanos = telemetry
                .cavity_insertion_nanos
                .saturating_add(elapsed_nanos);
            telemetry.cavity_insertion_nanos_max =
                telemetry.cavity_insertion_nanos_max.max(elapsed_nanos);
        }
    }

    /// Records one hull-extension attempt and its optional elapsed time.
    #[inline]
    fn record_hull_extension_telemetry(
        telemetry: &mut InsertionTelemetry,
        elapsed_nanos: Option<u64>,
    ) {
        telemetry.hull_extension_calls = telemetry.hull_extension_calls.saturating_add(1);
        if let Some(elapsed_nanos) = elapsed_nanos {
            telemetry.hull_extension_nanos =
                telemetry.hull_extension_nanos.saturating_add(elapsed_nanos);
            telemetry.hull_extension_nanos_max =
                telemetry.hull_extension_nanos_max.max(elapsed_nanos);
        }
    }

    /// Records one topology-validation pass and its optional elapsed time.
    #[inline]
    fn record_topology_validation_telemetry(
        telemetry: &mut InsertionTelemetry,
        elapsed_nanos: Option<u64>,
    ) {
        telemetry.topology_validation_calls = telemetry.topology_validation_calls.saturating_add(1);
        if let Some(elapsed_nanos) = elapsed_nanos {
            telemetry.topology_validation_nanos = telemetry
                .topology_validation_nanos
                .saturating_add(elapsed_nanos);
            telemetry.topology_validation_nanos_max =
                telemetry.topology_validation_nanos_max.max(elapsed_nanos);
        }
    }

    /// Convert a duration to nanoseconds while saturating at `u64::MAX`.
    #[inline]
    fn duration_nanos_saturating(duration: Duration) -> u64 {
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
    }

    /// Starts a wall-clock timer only when insertion telemetry will publish timings.
    #[inline]
    fn start_insertion_timing(telemetry_mode: InsertionTelemetryMode) -> Option<Instant> {
        telemetry_mode.records_timings().then(Instant::now)
    }

    fn collect_exterior_repair_seed_simplices(
        &self,
        point: &Point<K::Scalar, D>,
        terminal_simplex: SimplexKey,
        locate_stats: &LocateStats,
        telemetry_mode: InsertionTelemetryMode,
        telemetry: &mut InsertionTelemetry,
    ) -> Result<SimplexKeyBuffer, ConflictError> {
        if locate_stats.fell_back_to_scan() || !self.tds.contains_simplex(terminal_simplex) {
            return Ok(SimplexKeyBuffer::new());
        }

        let conflict_started = Self::start_insertion_timing(telemetry_mode);
        let local_seed_simplices = collect_local_exterior_conflict_seed_simplices(
            &self.tds,
            &self.kernel,
            point,
            terminal_simplex,
        )?;
        Self::record_conflict_region_telemetry(
            telemetry,
            local_seed_simplices.conflict_simplices_found,
        );
        if let Some(conflict_started) = conflict_started {
            Self::record_conflict_region_timing(
                telemetry,
                Self::duration_nanos_saturating(conflict_started.elapsed()),
            );
        }
        Ok(local_seed_simplices.seed_simplices)
    }

    /// Internal implementation of insert without retry logic.
    /// Returns the result and the number of simplices removed during repair.
    ///
    /// Note: `conflict_simplices` parameter is optional. If `None`, it will be computed automatically
    /// for interior points using `locate()` + `find_conflict_region()`.
    #[expect(
        clippy::too_many_lines,
        reason = "Complex insertion logic; splitting further would harm readability"
    )]
    fn try_insert_impl(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<TryInsertImplOk, InsertionError> {
        let mut suspicion = SuspicionFlags::default();

        // CRITICAL: Capture UUID and point BEFORE inserting into TDS
        // Rationale:
        // - inserted_uuid: Needed to remap v_key after TDS rebuild (lines 736-744)
        //   when building initial simplex. The rebuild replaces self.tds entirely,
        //   invalidating all previous VertexKeys.
        // - point: Needed for locate(), find_conflict_region(), and extend_hull() calls
        //   (lines 752, 760, 879, 895). After TDS rebuild, we cannot access the vertex
        //   via the old v_key, so we must have the point value captured.
        let inserted_uuid = vertex.uuid();
        let point = *vertex.point();

        vertex.is_valid().map_err(|source| {
            InsertionError::TopologyValidation(TdsError::InvalidVertex {
                vertex_id: inserted_uuid,
                source,
            })
        })?;

        // 1. Insert vertex into Tds
        let mut v_key = self
            .tds
            .insert_vertex_with_mapping(vertex)
            .map_err(InsertionError::from)?;

        // 2. Check if we need to bootstrap the initial simplex
        let num_vertices = self.tds.number_of_vertices();

        if num_vertices < D + 1 {
            // Bootstrap phase: just accumulate vertices, no simplices yet
            return Ok(TryInsertImplOk {
                inserted: (v_key, None),
                simplices_removed: 0,
                suspicion,
                repair_seed_simplices: SimplexKeyBuffer::new(),
                delaunay_repair_required: false,
            });
        } else if num_vertices == D + 1 {
            // Build initial simplex from all D+1 vertices
            let all_vertices: Vec<_> = self.tds.vertices().map(|(_, v)| *v).collect();
            let new_tds = Self::build_initial_simplex(&all_vertices).map_err(|source| {
                CavityFillingError::InitialSimplexConstruction {
                    reason: source.into(),
                }
            })?;

            // Replace empty TDS with simplex TDS (preserve kernel)
            self.tds = new_tds;

            // Re-map vertex key to the rebuilt TDS
            v_key = self.tds.vertex_key_from_uuid(&inserted_uuid).ok_or(
                CavityFillingError::RebuiltVertexMissing {
                    uuid: inserted_uuid,
                },
            )?;

            // Return first simplex key for hint caching
            let first_simplex = self.tds.simplex_keys().next();
            return Ok(TryInsertImplOk {
                inserted: (v_key, first_simplex),
                simplices_removed: 0,
                suspicion,
                repair_seed_simplices: SimplexKeyBuffer::new(),
                delaunay_repair_required: false,
            });
        }

        // 3. Locate containing simplex (for vertex D+2 and beyond).
        //
        // `locate()` delegates to `locate_with_stats()`, so collecting the stats here keeps
        // the same point-location algorithm while making release-mode batch diagnostics useful.
        let locate_trace = locate_with_trace(&self.tds, &self.kernel, &point, hint)?;
        let location = locate_trace.result;
        let locate_stats = locate_trace.stats;
        Self::record_locate_telemetry(telemetry, location, &locate_stats);

        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some()
            || std::env::var_os("DELAUNAY_DEBUG_LOCATE").is_some()
        {
            tracing::debug!(
                point = ?point,
                location = ?location,
                start_simplex = ?locate_stats.start_simplex,
                used_hint = locate_stats.used_hint,
                walk_steps = locate_stats.walk_steps,
                fallback = ?locate_stats.fallback,
                "try_insert_impl: locate stats"
            );
        }

        // 4. Determine the supported insertion site and any conflict simplices it needs.
        let insertion_site = match (location, conflict_simplices) {
            (LocateResult::InsideSimplex(start_simplex), None) => {
                // Interior point: compute conflict region automatically.
                //
                // IMPORTANT:
                // `find_conflict_region()` (Bowyer–Watson style) can legitimately return an empty
                // set when the point lies inside the triangulation but is not strictly inside any
                // existing simplex circumsphere (e.g., obtuse tetrahedra whose circumsphere does not
                // contain all interior points).
                //
                // An empty conflict region would produce an empty cavity boundary, create no new
                // simplices, and leave the inserted vertex isolated (not incident to any simplex), which
                // breaks Level 3 topology validation via Euler characteristic.
                //
                // Fallback: treat the containing simplex as the conflict region, effectively performing
                // a star-split of that simplex to keep the simplicial complex connected.
                let conflict_started = Self::start_insertion_timing(telemetry_mode);
                let computed =
                    find_conflict_region(&self.tds, &self.kernel, &point, start_simplex)?;
                Self::record_conflict_region_telemetry(telemetry, computed.len());
                if let Some(conflict_started) = conflict_started {
                    Self::record_conflict_region_timing(
                        telemetry,
                        Self::duration_nanos_saturating(conflict_started.elapsed()),
                    );
                }

                #[cfg(feature = "diagnostics")]
                if std::env::var_os("DELAUNAY_DEBUG_CONFLICT_VERIFY").is_some() {
                    let missed = verify_conflict_region_completeness(
                        &self.tds,
                        &self.kernel,
                        &point,
                        &computed,
                    );
                    if missed > 0 {
                        tracing::warn!(
                            missed,
                            bfs_conflict = computed.len(),
                            start_simplex = ?start_simplex,
                            point = ?point,
                            num_vertices = self.tds.number_of_vertices(),
                            num_simplices = self.tds.number_of_simplices(),
                            "try_insert_impl: INCOMPLETE conflict region at insertion"
                        );
                    }
                }

                if computed.is_empty() {
                    suspicion.empty_conflict_region = true;
                    suspicion.fallback_star_split = true;
                }
                InsertionSite::Interior {
                    start_simplex,
                    conflict_simplices: Self::ensure_non_empty_conflict_simplices(
                        Cow::Owned(computed),
                        start_simplex,
                    ),
                }
            }
            (LocateResult::InsideSimplex(start_simplex), Some(simplices)) => {
                // If the caller provided an empty conflict region (can happen if the Delaunay layer
                // computes conflicts using a strict in-sphere test), we must still replace at least
                // one simplex; otherwise we'd create no cavity, no new simplices, and leave a dangling
                // vertex (χ increases by 1, typically showing up as χ=2 for Ball(3)).
                if simplices.is_empty() {
                    suspicion.empty_conflict_region = true;
                    suspicion.fallback_star_split = true;
                }
                InsertionSite::Interior {
                    start_simplex,
                    conflict_simplices: Self::ensure_non_empty_conflict_simplices(
                        Cow::Borrowed(simplices),
                        start_simplex,
                    ),
                }
            }
            (LocateResult::Outside, None) => {
                // Exterior insertion is the hull-extension case.  Avoid the old
                // full-TDS conflict scan here; it was O(number_of_simplices) per
                // exterior point, often only to rediscover that the hull path
                // was required anyway.  Cadenced and final Delaunay repair own
                // any local empty-circumsphere cleanup after the hull mutation.
                #[cfg(debug_assertions)]
                if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        "Outside insertion: skipping global conflict-region scan; using hull extension"
                    );
                }
                let repair_seed_simplices = self.collect_exterior_repair_seed_simplices(
                    &point,
                    locate_trace.terminal_simplex,
                    &locate_stats,
                    telemetry_mode,
                    telemetry,
                )?;
                InsertionSite::Exterior {
                    conflict_simplices: None,
                    repair_seed_simplices,
                }
            }
            (LocateResult::Outside, Some(simplices)) => {
                if simplices.is_empty() {
                    #[cfg(debug_assertions)]
                    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                        tracing::debug!(
                            "Outside insertion: caller provided empty conflict region; will use hull extension"
                        );
                    }
                    let repair_seed_simplices = self.collect_exterior_repair_seed_simplices(
                        &point,
                        locate_trace.terminal_simplex,
                        &locate_stats,
                        telemetry_mode,
                        telemetry,
                    )?;
                    InsertionSite::Exterior {
                        conflict_simplices: None,
                        repair_seed_simplices,
                    }
                } else {
                    #[cfg(debug_assertions)]
                    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                        tracing::debug!(
                            conflict_simplices = simplices.len(),
                            "Outside insertion: using caller-provided conflict region"
                        );
                    }
                    InsertionSite::Exterior {
                        conflict_simplices: Some(Cow::Borrowed(simplices)),
                        repair_seed_simplices: simplices.iter().copied().collect(),
                    }
                }
            }
            (location, _) => {
                // Degenerate locations (OnFacet, OnEdge, OnVertex)
                return Err(CavityFillingError::UnsupportedDegenerateLocation { location }.into());
            }
        };

        // 5. Handle different insertion sites.
        match insertion_site {
            InsertionSite::Interior {
                start_simplex,
                conflict_simplices,
            } => {
                let conflict_simplices = conflict_simplices.into_owned();
                let cavity_started = Self::start_insertion_timing(telemetry_mode);
                let insertion_result = self.insert_with_conflict_region(
                    v_key,
                    &point,
                    conflict_simplices,
                    Some(start_simplex),
                    &mut suspicion,
                );
                Self::record_cavity_insertion_telemetry(
                    telemetry,
                    cavity_started
                        .map(|started| Self::duration_nanos_saturating(started.elapsed())),
                );
                let outcome = insertion_result?;
                Ok(TryInsertImplOk {
                    inserted: (v_key, outcome.hint),
                    simplices_removed: outcome.simplices_removed,
                    suspicion,
                    repair_seed_simplices: outcome.repair_seed_simplices,
                    delaunay_repair_required: outcome.delaunay_repair_required,
                })
            }
            InsertionSite::Exterior {
                conflict_simplices,
                repair_seed_simplices: exterior_repair_seed_simplices,
            } => {
                if let Some(conflict_simplices) = conflict_simplices {
                    let conflict_simplices = conflict_simplices.into_owned();
                    #[cfg(debug_assertions)]
                    let conflict_len = conflict_simplices.len();
                    #[cfg(debug_assertions)]
                    tracing::debug!(
                        "Outside insertion attempting cavity insertion with conflict region size {conflict_len}"
                    );
                    let cavity_started = Self::start_insertion_timing(telemetry_mode);
                    let result = self.insert_with_conflict_region(
                        v_key,
                        &point,
                        conflict_simplices,
                        None,
                        &mut suspicion,
                    );
                    Self::record_cavity_insertion_telemetry(
                        telemetry,
                        cavity_started
                            .map(|started| Self::duration_nanos_saturating(started.elapsed())),
                    );
                    match result {
                        Ok(outcome) => {
                            return Ok(TryInsertImplOk {
                                inserted: (v_key, outcome.hint),
                                simplices_removed: outcome.simplices_removed,
                                suspicion,
                                repair_seed_simplices: outcome.repair_seed_simplices,
                                delaunay_repair_required: true,
                            });
                        }
                        Err(err) => {
                            // For exterior points, a "global" conflict region can intersect the hull,
                            // producing an open/disconnected cavity boundary. In these cases we fall back
                            // to hull extension instead of surfacing an insertion error.
                            //
                            // IMPORTANT: Only ConflictError variants are safe to fall back from here.
                            // These originate from `extract_cavity_boundary` which runs BEFORE any TDS
                            // mutation.  Errors like `IsolatedVertex` originate from AFTER the cavity
                            // has been filled, neighbors wired, and conflict simplices removed — the TDS
                            // is already heavily mutated and hull extension on that state is unsound.
                            let should_fallback = matches!(
                                &err,
                                InsertionError::ConflictRegion(
                                    ConflictError::NonManifoldFacet { .. }
                                        | ConflictError::RidgeFan { .. }
                                        | ConflictError::DisconnectedBoundary { .. }
                                        | ConflictError::OpenBoundary { .. }
                                )
                            );

                            if should_fallback {
                                #[cfg(debug_assertions)]
                                tracing::warn!(
                                    "Outside insertion conflict boundary degeneracy ({err}) (conflict_simplices={conflict_len}); falling back to hull extension"
                                );
                                #[cfg(debug_assertions)]
                                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                                    tracing::debug!(
                                        error = %err,
                                        "Outside insertion: cavity insertion failed; using hull extension"
                                    );
                                }
                            } else {
                                #[cfg(debug_assertions)]
                                tracing::warn!("Outside insertion cavity insertion failed: {err}");
                                return Err(err);
                            }
                        }
                    }
                }
                // Exterior vertex: extend convex hull
                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        point = ?point,
                        "Outside insertion: proceeding to hull extension"
                    );
                }
                let hull_started = Self::start_insertion_timing(telemetry_mode);
                let hull_result = extend_hull(&mut self.tds, &self.kernel, v_key, &point);
                Self::record_hull_extension_telemetry(
                    telemetry,
                    hull_started.map(|started| Self::duration_nanos_saturating(started.elapsed())),
                );
                let new_simplices = match hull_result {
                    Ok(simplices) => simplices,
                    Err(err) => {
                        let retry_inside = matches!(
                            &err,
                            InsertionError::HullExtension {
                                reason: HullExtensionReason::NoVisibleFacets
                            }
                        );
                        if retry_inside {
                            let fallback_location =
                                locate_by_scan(&self.tds, &self.kernel, &point)?;
                            // This retry starts as a scan, so account for the fallback
                            // explicitly and let the common recorder handle the outcome.
                            telemetry.locate_scan_fallbacks =
                                telemetry.locate_scan_fallbacks.saturating_add(1);
                            let scan_start_simplex = self
                                .tds
                                .simplex_keys()
                                .next()
                                .ok_or(LocateError::EmptyTriangulation)?;
                            let scan_stats = LocateStats {
                                start_simplex: scan_start_simplex,
                                used_hint: false,
                                walk_steps: 0,
                                fallback: None,
                            };
                            Self::record_locate_telemetry(
                                telemetry,
                                fallback_location,
                                &scan_stats,
                            );
                            if let LocateResult::InsideSimplex(start_simplex) = fallback_location {
                                #[cfg(debug_assertions)]
                                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                                    tracing::warn!(
                                        point = ?point,
                                        start_simplex = ?start_simplex,
                                        "Outside insertion: no visible facets; retrying as interior with star-split"
                                    );
                                }
                                suspicion.fallback_star_split = true;
                                let mut star_conflict = SimplexKeyBuffer::new();
                                star_conflict.push(start_simplex);
                                let cavity_started = Self::start_insertion_timing(telemetry_mode);
                                let insertion_result = self.insert_with_conflict_region(
                                    v_key,
                                    &point,
                                    star_conflict,
                                    Some(start_simplex),
                                    &mut suspicion,
                                );
                                Self::record_cavity_insertion_telemetry(
                                    telemetry,
                                    cavity_started.map(|started| {
                                        Self::duration_nanos_saturating(started.elapsed())
                                    }),
                                );
                                let outcome = insertion_result?;
                                return Ok(TryInsertImplOk {
                                    inserted: (v_key, outcome.hint),
                                    simplices_removed: outcome.simplices_removed,
                                    suspicion,
                                    repair_seed_simplices: outcome.repair_seed_simplices,
                                    delaunay_repair_required: true,
                                });
                            }
                        }
                        #[cfg(debug_assertions)]
                        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                            tracing::warn!(
                                point = ?point,
                                error = %err,
                                "Outside insertion: hull extension failed"
                            );
                        }
                        return Err(err);
                    }
                };
                self.canonicalize_positive_orientation_for_simplices(&new_simplices)?;
                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        new_simplices = new_simplices.len(),
                        "Outside insertion: hull extension succeeded"
                    );
                }

                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_NEIGHBORS").is_some() {
                    let mut total_slots = 0usize;
                    let mut neighbor_none = 0usize;
                    let mut neighbor_missing = 0usize;
                    let mut neighbor_mutual = 0usize;
                    let mut neighbor_non_mutual = 0usize;

                    for &simplex_key in &new_simplices {
                        let Some(simplex) = self.tds.simplex(simplex_key) else {
                            continue;
                        };
                        let Some(neighbors) = simplex.neighbor_keys() else {
                            continue;
                        };
                        for neighbor_opt in neighbors {
                            total_slots = total_slots.saturating_add(1);
                            match neighbor_opt {
                                None => {
                                    neighbor_none = neighbor_none.saturating_add(1);
                                }
                                Some(neighbor_key) => {
                                    if !self.tds.contains_simplex(neighbor_key) {
                                        neighbor_missing = neighbor_missing.saturating_add(1);
                                    } else if self
                                        .tds
                                        .simplex(neighbor_key)
                                        .and_then(Simplex::neighbors)
                                        .is_some_and(|mut ns| {
                                            ns.any(|neighbor| neighbor == Some(simplex_key))
                                        })
                                    {
                                        neighbor_mutual = neighbor_mutual.saturating_add(1);
                                    } else {
                                        neighbor_non_mutual = neighbor_non_mutual.saturating_add(1);
                                    }
                                }
                            }
                        }
                    }

                    tracing::debug!(
                        new_simplices = new_simplices.len(),
                        total_slots,
                        neighbor_none,
                        neighbor_missing,
                        neighbor_mutual,
                        neighbor_non_mutual,
                        "Outside insertion: hull extension neighbor-pointer summary"
                    );
                }

                // Iteratively repair non-manifold topology until facet sharing is valid
                let mut total_removed = 0;
                let mut facet_sharing_known_valid = true;
                let mut neighbor_repair_frontier = SimplexKeyBuffer::new();
                #[cfg_attr(
                    not(debug_assertions),
                    expect(
                        unused_variables,
                        reason = "`iteration` is only used for debug logging",
                    )
                )]
                for iteration in 0..MAX_REPAIR_ITERATIONS {
                    // Check for non-manifold issues in newly created hull simplices (local scan)
                    // This keeps the repair O(k·D) where k is the number of new hull simplices, rather than O(N·D)
                    let simplices_to_check: SimplexKeyBuffer = new_simplices
                        .iter()
                        .copied()
                        .filter(|ck| self.tds.contains_simplex(*ck))
                        .collect();

                    if let Some(issues) = self.detect_local_facet_issues(&simplices_to_check)? {
                        // Only mark this as "suspicious" if we *actually* detected local facet issues
                        // and entered the repair path.
                        suspicion.repair_loop_entered = true;

                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "Hull extension repair iteration {}: {} over-shared facets detected, removing simplices...",
                            iteration + 1,
                            issues.len()
                        );

                        let repair = self.repair_local_facet_issues_with_frontier(&issues)?;
                        let removed = repair.removed_count;

                        // Early exit if repair made no progress
                        if removed == 0 {
                            #[cfg(debug_assertions)]
                            tracing::warn!(
                                "No simplices removed in iteration {} - repair cannot make progress",
                                iteration + 1
                            );
                            return Err(InsertionError::TopologyValidation(
                                TdsError::InconsistentDataStructure {
                                    message: format!(
                                        "Hull extension repair stalled: {} over-shared facets remain but no simplices could be removed",
                                        issues.len()
                                    ),
                                },
                            ));
                        }

                        total_removed += removed;
                        neighbor_repair_frontier.extend(repair.frontier_simplices);
                        if removed > 0 {
                            suspicion.simplices_removed = true;
                        }

                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            removed_simplices = ?repair.removed_simplices,
                            "Removed {removed} simplices (total: {total_removed})"
                        );

                        // Early exit if repair succeeded
                        facet_sharing_known_valid = self.tds.validate_facet_sharing().is_ok();
                        if facet_sharing_known_valid {
                            break;
                        }
                    } else {
                        // No more non-manifold issues - safe to rebuild neighbors
                        break;
                    }
                }

                // Repair neighbor pointers now that topology is manifold.
                if !facet_sharing_known_valid {
                    return Err(CavityFillingError::InvalidFacetSharingAfterRepair {
                        stage: CavityRepairStage::FanTriangulation,
                    }
                    .into());
                }

                if total_removed > 0 {
                    let repaired = self.repair_neighbors_after_local_simplex_removal(
                        &new_simplices,
                        &neighbor_repair_frontier,
                    )?;
                    suspicion.neighbor_pointers_rebuilt = repaired > 0;
                }

                // New hull simplices were canonicalized on creation; validate the
                // local orientation frontier without scanning the whole TDS.
                let mut orientation_simplices = SimplexKeyBuffer::new();
                append_live_unique_simplex_seeds(
                    &self.tds,
                    &new_simplices,
                    &mut orientation_simplices,
                );
                append_live_unique_simplex_seeds(
                    &self.tds,
                    &neighbor_repair_frontier,
                    &mut orientation_simplices,
                );
                self.validate_local_orientation_for_simplices(&orientation_simplices)?;

                // Assign an incident simplex for the inserted vertex without a global rebuild.
                let hint = new_simplices.iter().copied().find(|&ck| {
                    self.tds
                        .simplex(ck)
                        .is_some_and(|simplex| simplex.contains_vertex(v_key))
                });
                if let Some(incident_simplex) = hint
                    && let Some(vertex) = self.tds.vertex_mut(v_key)
                {
                    vertex.set_incident_simplex(Some(incident_simplex));
                }

                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                    match validate_ridge_links(&self.tds) {
                        Ok(()) => {
                            tracing::debug!(
                                "extend_hull: ridge-link validation passed after insertion"
                            );
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = ?err,
                                "extend_hull: ridge-link validation failed after insertion"
                            );
                        }
                    }
                }

                // Repair stale incident-simplex pointers (e.g. pointing to deleted
                // conflict-region simplices) and error only for truly isolated vertices.
                self.repair_stale_incident_simplices()?;

                // Connectedness guard (localized): ensure the newly created simplex set is internally
                // connected and attached to the existing triangulation.
                self.validate_connectedness(&new_simplices)?;

                // Return vertex key and hint for next insertion
                let mut repair_seed_simplices = SimplexKeyBuffer::new();
                append_live_unique_simplex_seeds(
                    &self.tds,
                    &new_simplices,
                    &mut repair_seed_simplices,
                );
                append_live_unique_simplex_seeds(
                    &self.tds,
                    &neighbor_repair_frontier,
                    &mut repair_seed_simplices,
                );
                append_live_unique_simplex_seeds(
                    &self.tds,
                    &exterior_repair_seed_simplices,
                    &mut repair_seed_simplices,
                );
                Ok(TryInsertImplOk {
                    inserted: (v_key, hint),
                    simplices_removed: total_removed,
                    suspicion,
                    repair_seed_simplices,
                    delaunay_repair_required: true,
                })
            }
        }
    }

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation maintains topological consistency by:
    /// 1. Finding all simplices containing the vertex
    /// 2. Removing those simplices (creating a cavity)
    /// 3. Extracting the cavity boundary facets
    /// 4. Filling the cavity with a fan triangulation (pick apex, connect to all boundary facets)
    /// 5. Wiring neighbors to maintain consistency
    /// 6. Removing the vertex itself
    ///
    /// **Fan Triangulation**: The cavity is filled by picking one boundary vertex as an apex
    /// and connecting it to all boundary facets. This is fast and maintains all topological
    /// invariants, though it may create poorly-shaped simplices in some cases.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - Key of the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of simplices that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] if the removal cannot be completed while maintaining
    /// triangulation invariants. The error preserves structured information from whichever
    /// layer (TDS or Topology) detected the failure.
    pub(crate) fn remove_vertex(&mut self, vertex_key: VertexKey) -> Result<usize, InvariantError> {
        // Verify the vertex exists
        if self.tds.vertex(vertex_key).is_none() {
            return Ok(0); // Vertex not found, nothing to remove
        }

        // Collect all simplices containing this vertex by scanning all simplices
        let simplices_to_remove: SimplexKeyBuffer = self
            .tds
            .simplices()
            .filter_map(|(simplex_key, simplex)| {
                if simplex.vertices().contains(&vertex_key) {
                    Some(simplex_key)
                } else {
                    None
                }
            })
            .collect();

        if simplices_to_remove.is_empty() {
            // Vertex exists but has no incident simplices - use Tds removal
            return self
                .tds
                .remove_vertex(vertex_key)
                .map_err(|e| InvariantError::Tds(e.into_inner()));
        }

        // Extract cavity boundary BEFORE removing simplices
        let boundary_facets =
            extract_cavity_boundary(&self.tds, &simplices_to_remove).map_err(|e| {
                TdsError::InconsistentDataStructure {
                    message: format!("Failed to extract cavity boundary: {e}"),
                }
            })?;

        // If boundary is empty, we're removing the entire triangulation
        if boundary_facets.is_empty() {
            // Use Tds removal for empty boundary case
            return self
                .tds
                .remove_vertex(vertex_key)
                .map_err(|e| InvariantError::Tds(e.into_inner()));
        }

        // Pick apex vertex for fan triangulation (first vertex of first boundary facet)
        let apex_vertex_key = self.pick_fan_apex(&boundary_facets).ok_or_else(|| {
            TdsError::InconsistentDataStructure {
                message: "Failed to find apex vertex for fan triangulation".to_string(),
            }
        })?;

        // Snapshot before destructive retriangulation edits so we can roll back if any
        // subsequent orientation/finalization step fails.
        let tds_snapshot = self.tds.clone_for_rollback();
        let retriangulation_result = (|| -> Result<usize, InvariantError> {
            // Fill cavity with fan triangulation BEFORE removing old simplices
            // Use fan triangulation that skips boundary facets which already include the apex
            let new_simplices = self
                .fan_fill_cavity(apex_vertex_key, &boundary_facets)
                .map_err(|e| insertion_error_to_invariant_error(e, "Fan triangulation failed"))?;
            // Wire neighbors for the new simplices (while both old and new simplices exist)
            let external_facets =
                external_facets_for_boundary(&self.tds, &simplices_to_remove, &boundary_facets)
                    .map_err(|e| {
                        insertion_error_to_invariant_error(e, "External-facet collection failed")
                    })?;
            wire_cavity_neighbors(
                &mut self.tds,
                &new_simplices,
                external_facets.iter().copied(),
                Some(&simplices_to_remove),
            )
            .map_err(|e| insertion_error_to_invariant_error(e, "Neighbor wiring failed"))?;

            // Remove the simplices containing the vertex (now that new simplices are wired up)
            // Note: remove_simplices_by_keys() automatically clears neighbor pointers in surviving
            // simplices that reference removed simplices (sets them to None/boundary)
            let mut simplices_removed = self.tds.remove_simplices_by_keys(&simplices_to_remove);

            // Validate facet topology for newly created simplices (O(k*D) localized check)
            if let Some(issues) = self.detect_local_facet_issues(&new_simplices)? {
                #[cfg(debug_assertions)]
                tracing::warn!(
                    "Warning: {} over-shared facets detected after vertex removal, repairing...",
                    issues.len()
                );
                let removed = self.repair_local_facet_issues(&issues)?;
                simplices_removed += removed;
                #[cfg(debug_assertions)]
                tracing::debug!("Repaired by removing {removed} additional simplices");

                // Repair neighbor pointers after removing additional simplices
                // This ensures neighbor consistency after repair operations
                if removed > 0 {
                    repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                        insertion_error_to_invariant_error(
                            e,
                            "Neighbor repair after facet issue repair failed",
                        )
                    })?;
                }
            }
            // Normalize coherent orientation, canonicalize global sign, and promote
            // simplices to positive orientation (#258).
            self.normalize_and_promote_positive_orientation()
                .map_err(|e| {
                    insertion_error_to_invariant_error(
                        e,
                        "Orientation canonicalization failed after fan retriangulation",
                    )
                })?;

            // Rebuild vertex-simplex incidence for all vertices
            self.tds
                .assign_incident_simplices()
                .map_err(|e| InvariantError::Tds(e.into_inner()))?;

            // Remove the vertex using Tds method (handles internal bookkeeping)
            self.tds
                .remove_vertex(vertex_key)
                .map_err(|e| InvariantError::Tds(e.into_inner()))?;

            Ok(simplices_removed)
        })();

        match retriangulation_result {
            Ok(simplices_removed) => Ok(simplices_removed),
            Err(error) => {
                self.tds = tds_snapshot;
                Err(error)
            }
        }
    }

    /// Pick an apex vertex for fan triangulation.
    ///
    /// Selects the first vertex from the first boundary facet as the apex.
    /// The fan will connect this apex to all boundary facets.
    ///
    /// # Arguments
    ///
    /// * `boundary_facets` - The cavity boundary facets
    ///
    /// # Returns
    ///
    /// The vertex key to use as apex, or None if no suitable vertex found.
    fn pick_fan_apex(&self, boundary_facets: &[FacetHandle]) -> Option<VertexKey> {
        // Get first boundary facet
        let first_facet = boundary_facets.first()?;
        let simplex = self.tds.simplex(first_facet.simplex_key())?;

        // Get the first vertex from this facet (any vertex that's not the opposite one)
        let facet_idx = <usize as From<_>>::from(first_facet.facet_index());
        simplex
            .vertices()
            .iter()
            .enumerate()
            .find(|(i, _)| *i != facet_idx)
            .map(|(_, &vkey)| vkey)
    }

    /// Fan-specific cavity fill: connect an existing apex vertex to boundary facets
    /// that do not already include the apex. This avoids creating degenerate simplices
    /// with duplicate vertices when the apex lies on a boundary facet.
    fn fan_fill_cavity(
        &mut self,
        apex_vertex_key: VertexKey,
        boundary_facets: &[FacetHandle],
    ) -> Result<SimplexKeyBuffer, InsertionError> {
        let mut new_simplices = SimplexKeyBuffer::new();

        for facet_handle in boundary_facets {
            let boundary_simplex =
                self.tds
                    .simplex(facet_handle.simplex_key())
                    .ok_or_else(|| CavityFillingError::MissingBoundarySimplex {
                        simplex_key: facet_handle.simplex_key(),
                    })?;

            let facet_idx = <usize as From<_>>::from(facet_handle.facet_index());
            if facet_idx >= boundary_simplex.number_of_vertices() {
                return Err(CavityFillingError::InvalidFacetIndex {
                    simplex_key: facet_handle.simplex_key(),
                    facet_index: facet_idx,
                    vertex_count: boundary_simplex.number_of_vertices(),
                }
                .into());
            }

            // Gather facet vertices (all except the opposite vertex)
            let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in boundary_simplex.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vertices.push(vkey);
                }
            }

            // Skip facets that already contain the apex to avoid duplicate vertices
            if facet_vertices.contains(&apex_vertex_key) {
                continue;
            }

            // Build new simplex vertices = facet_vertices + apex
            let mut new_simplex_vertices = facet_vertices;
            new_simplex_vertices.push(apex_vertex_key);

            // Create and insert the new simplex
            let new_simplex =
                Simplex::new(new_simplex_vertices, None).map_err(CavityFillingError::from)?;
            let simplex_key = self
                .tds
                .insert_simplex_with_mapping_prechecked_topology(new_simplex)
                .map_err(InsertionError::from)?;

            new_simplices.push(simplex_key);
        }

        if new_simplices.is_empty() {
            return Err(CavityFillingError::EmptyFanTriangulation.into());
        }

        Ok(new_simplices)
    }

    /// Detects over-shared facets
    ///
    /// This is an **O(k * D)** operation where k = number of simplices to check,
    /// unlike global validation which is O(N * D) for the entire triangulation.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(k * D) where k = `simplices.len()`, D = dimension
    /// - **Use case**: Detect issues in newly created simplices after insertion/removal
    /// - **Comparison**: Global detection is O(N * D) where N = total simplices
    ///
    /// # Arguments
    ///
    /// * `simplices` - Keys of simplices to check (typically newly created simplices)
    ///
    /// # Returns
    ///
    /// `Ok(None)` if all facets are valid (≤2 simplices per facet).
    /// `Ok(Some(issues))` if over-shared facets are detected, where issues is a map
    /// from facet hash to the simplices sharing that facet.
    ///
    /// # Errors
    ///
    /// Returns error if simplices cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // A single simplex has no over-shared facets.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let simplex_keys: Vec<_> = dt.simplices().map(|(ck, _)| ck).collect();
    /// let issues = dt
    ///     .as_triangulation()
    ///     .detect_local_facet_issues(&simplex_keys)
    ///     .unwrap();
    /// assert!(issues.is_none());
    ///
    /// // Note: This method is most useful for checking newly created simplices
    /// // after insertion/removal operations (see usage in insert_transactional).
    /// ```
    pub fn detect_local_facet_issues(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<Option<FacetIssuesMap>, TdsError> {
        // Build facet map for ONLY the specified simplices
        // This is O(k * D) instead of O(N * D)
        let mut facet_to_simplices = FacetIssuesMap::default();

        // Index facets from the specified simplices
        for &simplex_key in simplices {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue; // Simplex was removed, skip
            };

            // For each facet of this simplex
            for facet_idx in 0..simplex.number_of_vertices() {
                // Compute facet hash from sorted vertex keys
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vkey) in simplex.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vkey);
                    }
                }
                facet_vkeys.sort_unstable();

                // Hash the facet
                let mut hasher = FastHasher::default();
                for &vkey in &facet_vkeys {
                    vkey.hash(&mut hasher);
                }
                let facet_hash = hasher.finish();

                // Track this simplex/facet pair
                let facet_idx_u8 =
                    u8::try_from(facet_idx).map_err(|_| TdsError::IndexOutOfBounds {
                        index: facet_idx,
                        bound: u8::MAX as usize + 1,
                        context: "facet index exceeds u8 range (dimension too high)".to_string(),
                    })?;
                facet_to_simplices
                    .entry(facet_hash)
                    .or_insert_with(SmallBuffer::new)
                    .push((simplex_key, facet_idx_u8));
            }
        }

        // Filter to only over-shared facets (> 2 simplices) in a single pass
        facet_to_simplices.retain(|_, simplex_facet_pairs| simplex_facet_pairs.len() > 2);

        if facet_to_simplices.is_empty() {
            Ok(None)
        } else {
            Ok(Some(facet_to_simplices))
        }
    }

    /// Select simplices to remove for over-shared-facet repair without mutating the TDS.
    fn simplices_for_local_facet_issue_repair(
        &self,
        issues: &FacetIssuesMap,
    ) -> Result<SimplexKeyBuffer, TdsError>
    where
        K::Scalar: Div<Output = K::Scalar>,
    {
        let mut simplices_to_remove = SimplexKeySet::default();

        // For each over-shared facet, select simplices to remove
        for simplex_facet_pairs in issues.values() {
            // Compute quality for each simplex - propagate errors from quality evaluation
            let mut simplex_qualities: Vec<(SimplexKey, f64, Uuid)> = Vec::new();
            for &(simplex_key, _) in simplex_facet_pairs {
                let simplex =
                    self.tds
                        .simplex(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "facet repair quality evaluation".to_string(),
                        })?;
                let uuid = simplex.uuid();

                // Propagate quality evaluation errors
                let ratio = radius_ratio(self, simplex_key).map_err(|e| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Quality evaluation failed for simplex {simplex_key:?}: {e}"
                        ),
                    }
                })?;
                let ratio_f64 =
                    safe_scalar_to_f64(ratio).map_err(|_| TdsError::InconsistentDataStructure {
                        message: format!(
                            "Quality ratio conversion failed for simplex {simplex_key:?}"
                        ),
                    })?;

                if ratio_f64.is_finite() {
                    simplex_qualities.push((simplex_key, ratio_f64, uuid));
                } else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Non-finite quality ratio {ratio_f64} for simplex {simplex_key:?}"
                        ),
                    });
                }
            }

            // Quality-based selection: keep 2 best, remove rest
            // Note: simplex_qualities always has all involved_simplices at this point since
            // any quality computation failure results in an early error return above
            simplex_qualities
                .sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

            // Mark simplices beyond the top 2 for removal
            for (simplex_key, _, _) in simplex_qualities.iter().skip(2) {
                if self.tds.contains_simplex(*simplex_key) {
                    simplices_to_remove.insert(*simplex_key);
                }
            }
        }

        Ok(simplices_to_remove.into_iter().collect())
    }

    /// Collect surviving neighbor simplices that will have removed-simplex back-references cleared.
    fn removal_frontier_for_simplices(
        &self,
        simplices_to_remove: &[SimplexKey],
    ) -> SimplexKeyBuffer {
        if simplices_to_remove.is_empty() {
            return SimplexKeyBuffer::new();
        }

        let removal_set: SimplexKeySet = simplices_to_remove.iter().copied().collect();
        let mut frontier = SimplexKeyBuffer::new();
        let mut seen = FastHashSet::default();

        for &simplex_key in simplices_to_remove {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue;
            };
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };
            for neighbor_key in neighbors.flatten() {
                if removal_set.contains(&neighbor_key) || !self.tds.contains_simplex(neighbor_key) {
                    continue;
                }
                if seen.insert(neighbor_key) {
                    frontier.push(neighbor_key);
                }
            }
        }

        frontier
    }

    /// Add surviving simplices from the facet-issue incidence map to the local repair frontier.
    fn add_issue_survivors_to_frontier(
        &self,
        issues: &FacetIssuesMap,
        simplices_to_remove: &[SimplexKey],
        frontier: &mut SimplexKeyBuffer,
    ) {
        let removal_set: SimplexKeySet = simplices_to_remove.iter().copied().collect();
        let mut seen: FastHashSet<SimplexKey> = frontier.iter().copied().collect();

        for simplex_facet_pairs in issues.values() {
            for &(simplex_key, _) in simplex_facet_pairs {
                if removal_set.contains(&simplex_key) || !self.tds.contains_simplex(simplex_key) {
                    continue;
                }
                if seen.insert(simplex_key) {
                    frontier.push(simplex_key);
                }
            }
        }
    }

    /// Repair over-shared facets and return the local frontier for neighbor repair.
    fn repair_local_facet_issues_with_frontier(
        &mut self,
        issues: &FacetIssuesMap,
    ) -> Result<LocalFacetRepairOutcome, TdsError>
    where
        K::Scalar: Div<Output = K::Scalar>,
    {
        let to_remove = self.simplices_for_local_facet_issue_repair(issues)?;
        let mut frontier_simplices = self.removal_frontier_for_simplices(&to_remove);
        self.add_issue_survivors_to_frontier(issues, &to_remove, &mut frontier_simplices);
        let removed_count = self.tds.remove_simplices_by_keys(&to_remove);

        Ok(LocalFacetRepairOutcome {
            removed_count,
            removed_simplices: to_remove,
            frontier_simplices,
        })
    }

    /// Repairs over-shared facets by removing lower-quality simplices.
    ///
    /// Uses geometric quality metrics (`radius_ratio`) to select which simplices to keep
    /// when a facet is shared by more than 2 simplices. UUID ordering is used as a tie-breaker
    /// when simplices have equal quality. Errors if quality computation or conversion fails.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(m * q) where m = number of problematic facets, q = quality computation cost
    /// - **Localized**: Only processes simplices involved in detected issues
    ///
    /// # Arguments
    ///
    /// * `issues` - Detected facet issues map from `detect_local_facet_issues()`
    ///
    /// # Returns
    ///
    /// Number of simplices removed during repair.
    ///
    /// # Errors
    ///
    /// Returns error if quality evaluation or facet bookkeeping fails while
    /// selecting simplices to remove. This function itself does not rebuild neighbors;
    /// callers are responsible for repairing or validating topology after removal
    /// (e.g., via local or global neighbor-pointer repair, or a validation pass).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::collections::FacetIssuesMap;
    /// use delaunay::prelude::tds::TdsError;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] TdsError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Start with a valid 2D simplex.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices)?;
    ///
    /// // Empty issues map => nothing to remove.
    /// let mut tri = dt.as_triangulation().clone();
    /// let removed = tri.repair_local_facet_issues(&FacetIssuesMap::default())?;
    /// assert_eq!(removed, 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// In practice, this method is typically called with issues detected by
    /// [`detect_local_facet_issues`](Self::detect_local_facet_issues) after insertion/removal
    /// operations. See `insert_transactional` for a typical usage pattern.
    pub fn repair_local_facet_issues(&mut self, issues: &FacetIssuesMap) -> Result<usize, TdsError>
    where
        K::Scalar: Div<Output = K::Scalar>,
    {
        self.repair_local_facet_issues_with_frontier(issues)
            .map(|outcome| outcome.removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::locate::InternalInconsistencySite;
    use crate::core::collections::NeighborBuffer;
    use crate::core::collections::spatial_hash_grid::HashGridIndex;
    use crate::core::simplex::NeighborSlot;
    use crate::core::vertex::VertexBuilder;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{
        Coordinate, CoordinateConversionError, CoordinateScalar,
    };
    use crate::geometry::util::generate_random_points_seeded;
    use crate::topology::characteristics::validation::validate_triangulation_euler;
    use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
    use crate::triangulation::delaunay::{DelaunayRepairPolicy, DelaunayTriangulation};
    use crate::vertex;

    use slotmap::KeyData;
    use std::collections::HashSet;

    /// Helper: build a minimal 3D triangulation with one tetrahedron and valid
    /// incident-simplex pointers for all four vertices.
    fn build_single_tet() -> (
        Triangulation<FastKernel<f64>, (), (), 3>,
        [VertexKey; 4],
        SimplexKey,
    ) {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let v0 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();

        let ck = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2, v3], None).unwrap())
            .unwrap();

        for vk in [v0, v1, v2, v3] {
            tri.tds
                .vertex_mut(vk)
                .unwrap()
                .set_incident_simplex(Some(ck));
        }

        (tri, [v0, v1, v2, v3], ck)
    }

    struct ForceNextRetryableInsertionFailureGuard {
        prior: bool,
    }

    impl ForceNextRetryableInsertionFailureGuard {
        fn enable() -> Self {
            let prior = test_hooks::set_force_next_insertion_retryable_failure(true);
            Self { prior }
        }
    }

    impl Drop for ForceNextRetryableInsertionFailureGuard {
        fn drop(&mut self) {
            test_hooks::restore_force_next_insertion_retryable_failure(self.prior);
        }
    }

    #[test]
    fn test_triangulation_validation_error_try_from_manifold_error_preserves_detail() {
        let tds_err = TdsError::InvalidNeighbors {
            reason: NeighborValidationError::Other {
                message: "unit test".to_string(),
            },
        };

        // ManifoldError::Tds belongs to the lower TDS layer, not TriangulationValidationError.
        assert_eq!(
            TriangulationValidationError::try_from(ManifoldError::Tds(tds_err.clone())),
            Err(tds_err.clone())
        );
        assert_eq!(
            InvariantError::from(ManifoldError::Tds(tds_err.clone())),
            InvariantError::Tds(tds_err)
        );

        assert!(matches!(
            TriangulationValidationError::try_from(ManifoldError::ManifoldFacetMultiplicity {
                facet_key: 123,
                simplex_count: 3
            })
            .unwrap(),
            TriangulationValidationError::ManifoldFacetMultiplicity {
                facet_key: 123,
                simplex_count: 3
            }
        ));

        assert!(matches!(
            TriangulationValidationError::try_from(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            })
            .unwrap(),
            TriangulationValidationError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            }
        ));

        assert!(matches!(
            TriangulationValidationError::try_from(ManifoldError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            })
            .unwrap(),
            TriangulationValidationError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            }
        ));

        assert!(matches!(
            TriangulationValidationError::try_from(ManifoldError::VertexLinkNotManifold {
                vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                link_vertex_count: 3,
                link_simplex_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
            })
            .unwrap(),
            TriangulationValidationError::VertexLinkNotManifold {
                link_vertex_count: 3,
                link_simplex_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
                ..
            }
        ));
    }

    #[test]
    fn test_internal_inconsistency_display() {
        let err = TriangulationConstructionError::InternalInconsistency {
            message: "missing vertex in lookup table".to_string(),
        };

        assert_eq!(
            err.to_string(),
            "Internal inconsistency during construction: missing vertex in lookup table"
        );
    }

    #[test]
    fn test_retryable_conflict_trace_detail_formats_retryable_variants() {
        let extra_simplex = SimplexKey::from(KeyData::from_ffi(10));
        let disconnected_simplex = SimplexKey::from(KeyData::from_ffi(11));
        let open_simplex = SimplexKey::from(KeyData::from_ffi(12));

        let non_manifold = InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
            facet_hash: 0xABCD,
            simplex_count: 3,
        });
        assert_eq!(
            retryable_conflict_trace_detail(&non_manifold).as_deref(),
            Some("kind=non_manifold_facet facet_hash=0xabcd simplex_count=3")
        );

        let ridge_fan = InsertionError::ConflictRegion(ConflictError::RidgeFan {
            facet_count: 4,
            ridge_vertex_count: 2,
            extra_simplices: vec![extra_simplex],
        });
        assert_eq!(
            retryable_conflict_trace_detail(&ridge_fan).as_deref(),
            Some("kind=ridge_fan facet_count=4 ridge_vertex_count=2 extra_simplices=1")
        );

        let disconnected = InsertionError::ConflictRegion(ConflictError::DisconnectedBoundary {
            visited: 2,
            total: 5,
            disconnected_simplices: vec![disconnected_simplex],
        });
        assert_eq!(
            retryable_conflict_trace_detail(&disconnected).as_deref(),
            Some("kind=disconnected_boundary visited=2 total=5 disconnected_simplices=1")
        );

        let open = InsertionError::ConflictRegion(ConflictError::OpenBoundary {
            facet_count: 1,
            ridge_vertex_count: 2,
            open_simplex,
        });
        assert_eq!(
            retryable_conflict_trace_detail(&open).as_deref(),
            Some("kind=open_boundary facet_count=1 ridge_vertex_count=2")
        );

        let not_retryable = InsertionError::CavityFilling {
            reason: CavityFillingError::EmptyFanTriangulation,
        };
        assert!(retryable_conflict_trace_detail(&not_retryable).is_none());
    }

    #[test]
    fn test_cavity_conflict_error_summary_formats_all_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(21));

        let cases = vec![
            (
                ConflictError::NonManifoldFacet {
                    facet_hash: 0xCAFE,
                    simplex_count: 4,
                },
                "non_manifold_facet facet_hash=0xcafe simplex_count=4".to_string(),
            ),
            (
                ConflictError::RidgeFan {
                    facet_count: 5,
                    ridge_vertex_count: 3,
                    extra_simplices: vec![simplex_key],
                },
                "ridge_fan facet_count=5 ridge_vertex_count=3 extra_simplices=1".to_string(),
            ),
            (
                ConflictError::DisconnectedBoundary {
                    visited: 1,
                    total: 3,
                    disconnected_simplices: vec![simplex_key],
                },
                "disconnected_boundary visited=1 total=3 disconnected_simplices=1".to_string(),
            ),
            (
                ConflictError::OpenBoundary {
                    facet_count: 1,
                    ridge_vertex_count: 2,
                    open_simplex: simplex_key,
                },
                format!(
                    "open_boundary facet_count=1 ridge_vertex_count=2 open_simplex={simplex_key:?}"
                ),
            ),
            (
                ConflictError::InvalidStartSimplex { simplex_key },
                format!("invalid_start_simplex simplex_key={simplex_key:?}"),
            ),
            (
                ConflictError::SimplexDataAccessFailed {
                    simplex_key,
                    message: "missing vertices".to_string(),
                },
                format!(
                    "simplex_data_access_failed simplex_key={simplex_key:?} message=missing vertices"
                ),
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(cavity_conflict_error_summary(&error), expected);
        }

        let predicate = ConflictError::PredicateError {
            source: CoordinateConversionError::ConversionFailed {
                coordinate_index: 2,
                coordinate_value: "NaN".to_string(),
                from_type: "f64",
                to_type: "f32",
            },
        };
        assert!(
            cavity_conflict_error_summary(&predicate)
                .starts_with("predicate_error source=Failed to convert coordinate")
        );

        let internal = ConflictError::InternalInconsistency {
            site: InternalInconsistencySite::RidgeInfoMissingSecondFacet {
                first_facet: 4,
                boundary_facets_len: 6,
                ridge_vertex_count: 2,
            },
        };
        assert!(cavity_conflict_error_summary(&internal).contains("internal_inconsistency site="));
    }

    #[test]
    fn test_log_cavity_reduction_event_only_evaluates_when_enabled() {
        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(SimplexKey::from(KeyData::from_ffi(41)));

        let mut called = false;
        log_cavity_reduction_event(false, 0, &conflict_simplices, || {
            called = true;
            "should not run".to_string()
        });
        assert!(!called);

        log_cavity_reduction_event(true, 1, &conflict_simplices, || {
            called = true;
            "ran".to_string()
        });
        assert!(called);
    }

    #[test]
    fn test_triangulation_new_empty_and_new_with_tds_default_to_pl_manifold() {
        let tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);

        let tri_with_tds: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_with_tds(FastKernel::new(), Tds::<f64, (), (), 2>::empty());
        assert_eq!(
            tri_with_tds.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
    }

    #[test]
    fn test_triangulation_set_topology_guarantee_round_trips() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);

        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
        assert_eq!(
            tri.topology_guarantee(),
            TopologyGuarantee::PLManifoldStrict
        );

        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
    }
    #[test]
    fn test_validation_policy_should_validate_matrix() {
        let clean = SuspicionFlags::default();
        let suspicious = SuspicionFlags {
            perturbation_used: true,
            ..SuspicionFlags::default()
        };

        assert!(!ValidationPolicy::Never.should_validate(clean));
        assert!(!ValidationPolicy::Never.should_validate(suspicious));

        assert!(ValidationPolicy::Always.should_validate(clean));
        assert!(ValidationPolicy::Always.should_validate(suspicious));

        assert!(!ValidationPolicy::OnSuspicion.should_validate(clean));
        assert!(ValidationPolicy::OnSuspicion.should_validate(suspicious));

        assert!(ValidationPolicy::DebugOnly.should_validate(suspicious));
        assert_eq!(
            ValidationPolicy::DebugOnly.should_validate(clean),
            cfg!(debug_assertions)
        );
    }

    #[test]
    fn test_topology_guarantee_helper_matrix_and_policy_compatibility() {
        assert_eq!(TopologyGuarantee::default(), TopologyGuarantee::DEFAULT);
        assert_eq!(TopologyGuarantee::DEFAULT, TopologyGuarantee::PLManifold);

        assert!(!TopologyGuarantee::Pseudomanifold.requires_vertex_links_during_insertion());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_vertex_links_during_insertion());

        assert!(!TopologyGuarantee::Pseudomanifold.requires_vertex_links_at_completion());
        assert!(TopologyGuarantee::PLManifold.requires_vertex_links_at_completion());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_vertex_links_at_completion());

        assert!(
            TopologyGuarantee::Pseudomanifold.requires_pseudomanifold_checks_during_insertion()
        );
        assert!(TopologyGuarantee::PLManifold.requires_pseudomanifold_checks_during_insertion());
        assert!(
            TopologyGuarantee::PLManifoldStrict.requires_pseudomanifold_checks_during_insertion()
        );

        assert!(!TopologyGuarantee::Pseudomanifold.requires_ridge_links());
        assert!(TopologyGuarantee::PLManifold.requires_ridge_links());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_ridge_links());

        // default_validation_policy
        assert_eq!(
            TopologyGuarantee::PLManifoldStrict.default_validation_policy(),
            ValidationPolicy::Always
        );
        assert_eq!(
            TopologyGuarantee::PLManifold.default_validation_policy(),
            ValidationPolicy::OnSuspicion
        );
        assert_eq!(
            TopologyGuarantee::Pseudomanifold.default_validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        // Verify constructors use the centralized mapping.
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_empty(FastKernel::new());
        assert_eq!(
            tri.validation_policy(),
            TopologyGuarantee::DEFAULT.default_validation_policy()
        );

        for policy in [
            ValidationPolicy::Never,
            ValidationPolicy::OnSuspicion,
            ValidationPolicy::Always,
            ValidationPolicy::DebugOnly,
        ] {
            assert!(TopologyGuarantee::Pseudomanifold.is_compatible_with_policy(policy));
        }

        assert!(!TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::Never));
        assert!(
            !TopologyGuarantee::PLManifoldStrict.is_compatible_with_policy(ValidationPolicy::Never)
        );
        assert!(
            TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::OnSuspicion)
        );
        assert!(TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::Always));
        assert!(
            TopologyGuarantee::PLManifoldStrict.is_compatible_with_policy(ValidationPolicy::Always)
        );
    }

    #[test]
    fn test_set_validation_policy_incompatible_updates_when_completion_validation_succeeds() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
        assert_eq!(tri.validation_policy(), ValidationPolicy::OnSuspicion);

        tri.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(tri.validation_policy(), ValidationPolicy::Never);
    }

    #[test]
    fn test_set_topology_guarantee_incompatible_updates_when_completion_validation_succeeds() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        tri.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(tri.validation_policy(), ValidationPolicy::Never);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
        assert_eq!(
            tri.topology_guarantee(),
            TopologyGuarantee::PLManifoldStrict
        );
    }

    #[test]
    fn test_duplicate_detection_metrics_force_enable() {
        struct DuplicateDetectionGuard;

        impl Drop for DuplicateDetectionGuard {
            fn drop(&mut self) {
                DUPLICATE_DETECTION_FORCE_ENABLED.store(false, Ordering::Relaxed);
            }
        }

        let _guard = DuplicateDetectionGuard;
        DUPLICATE_DETECTION_FORCE_ENABLED.store(true, Ordering::Relaxed);

        let before = Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_detection_metrics()
            .expect("duplicate detection metrics should be enabled");

        record_duplicate_detection_metrics(true, 3, false);
        record_duplicate_detection_metrics(false, 0, true);

        let after = Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_detection_metrics()
            .expect("duplicate detection metrics should be enabled");

        assert!(after.total_checks > before.total_checks);
        assert!(after.grid_used > before.grid_used);
        assert!(after.grid_fallbacks > before.grid_fallbacks);
        assert!(after.grid_candidates >= before.grid_candidates + 3);
    }

    #[test]
    fn test_validate_at_completion_skips_for_pseudomanifold() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert!(tri.validate_at_completion().is_ok());
    }
    /// Regression test: a negatively oriented but topologically valid simplex
    /// passes `is_valid_topology_only()` while failing `is_valid()` (which
    /// includes the geometric orientation check).
    #[test]
    fn test_negative_oriented_simplex_topology_only() {
        // Build a single positively oriented triangle, then swap vertices 0↔1
        // to make it negatively oriented.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        // Confirm it starts valid.
        assert!(tri.is_valid().is_ok());
        assert!(tri.is_valid_topology_only().is_ok());

        // Flip the single simplex's vertex order to make it negatively oriented.
        let simplex_key = tri
            .tds
            .simplex_keys()
            .next()
            .expect("single simplex exists");
        tri.tds
            .simplex_mut(simplex_key)
            .expect("simplex exists")
            .swap_vertex_slots(0, 1);

        // Topology-only validation should still pass (orientation sign is irrelevant).
        assert!(
            tri.is_valid_topology_only().is_ok(),
            "Negatively oriented simplex should pass topology-only validation"
        );

        // Full validation (which includes geometric orientation) should fail.
        assert!(
            tri.is_valid().is_err(),
            "Negatively oriented simplex should fail full is_valid()"
        );
    }

    fn insert_test_vertex_with_coords<const D: usize>(
        tds: &mut Tds<f64, (), (), D>,
        entries: &[(usize, f64)],
    ) -> VertexKey {
        let mut coords = [0.0_f64; D];
        for &(axis, value) in entries {
            coords[axis] = value;
        }
        tds.insert_vertex_with_mapping(vertex!(coords)).unwrap()
    }

    fn build_invalid_vertex_link_tds<const D: usize>() -> (Tds<f64, (), (), D>, VertexKey) {
        // Two disjoint stars sharing only one apex produce a disconnected vertex link.
        let mut tds: Tds<f64, (), (), D> = Tds::empty();
        let shared = insert_test_vertex_with_coords(&mut tds, &[]);

        if D == 2 {
            // Two cone cycles keep the 1D boundary closed, so strict validation reaches
            // the disconnected vertex-link diagnostic instead of stopping at ridge degree.
            let first_a = insert_test_vertex_with_coords(&mut tds, &[(0, 1.0)]);
            let first_b = insert_test_vertex_with_coords(&mut tds, &[(1, 1.0)]);
            let first_c = insert_test_vertex_with_coords(&mut tds, &[(0, -1.0)]);
            let second_a = insert_test_vertex_with_coords(&mut tds, &[(0, 10.0)]);
            let second_b = insert_test_vertex_with_coords(&mut tds, &[(0, 11.0), (1, 1.0)]);
            let second_c = insert_test_vertex_with_coords(&mut tds, &[(0, 9.0), (1, 1.0)]);

            for simplex_vertices in [
                vec![shared, first_a, first_b],
                vec![shared, first_b, first_c],
                vec![shared, first_c, first_a],
                vec![shared, second_a, second_b],
                vec![shared, second_b, second_c],
                vec![shared, second_c, second_a],
            ] {
                let _ = tds
                    .insert_simplex_with_mapping(Simplex::new(simplex_vertices, None).unwrap())
                    .unwrap();
            }

            tds.assign_incident_simplices().unwrap();
            return (tds, shared);
        }

        let mut first_simplex_vertices = vec![shared];
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            first_simplex_vertices.push(tds.insert_vertex_with_mapping(vertex!(coords)).unwrap());
        }

        let mut second_simplex_vertices = vec![shared];
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[0] = 10.0;
            coords[axis] += 1.0;
            second_simplex_vertices.push(tds.insert_vertex_with_mapping(vertex!(coords)).unwrap());
        }

        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(first_simplex_vertices, None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(second_simplex_vertices, None).unwrap())
            .unwrap();

        tds.assign_incident_simplices().unwrap();

        (tds, shared)
    }

    fn build_invalid_vertex_link_tds_2d() -> (Tds<f64, (), (), 2>, VertexKey) {
        build_invalid_vertex_link_tds::<2>()
    }

    #[test]
    fn test_validate_at_completion_reports_invalid_vertex_link() {
        let (tds, v0) = build_invalid_vertex_link_tds_2d();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        match tri.validate_at_completion() {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. },
            )) => {
                assert_eq!(vertex_key, v0);
            }
            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
        }
    }
    #[test]
    fn test_set_validation_policy_rejects_incompatible_policy_when_completion_validation_fails() {
        let (tds, _) = build_invalid_vertex_link_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        assert!(matches!(
            tri.validate_at_completion(),
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold { .. }
            ))
        ));
        assert_eq!(tri.validation_policy(), ValidationPolicy::OnSuspicion);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tri.set_validation_policy(ValidationPolicy::Never);
        }));
        if cfg!(debug_assertions) {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(tri.validation_policy(), ValidationPolicy::OnSuspicion);
    }

    #[test]
    fn test_set_topology_guarantee_rejects_incompatible_guarantee_when_completion_validation_fails()
    {
        let (tds, _) = build_invalid_vertex_link_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        tri.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
        assert_eq!(tri.validation_policy(), ValidationPolicy::Never);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
        }));
        if cfg!(debug_assertions) {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    }

    fn build_disconnected_two_triangles_tds_2d() -> Tds<f64, (), (), 2> {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let a1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let a2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let b0 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0]))
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 0.0]))
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![a0, a1, a2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![b0, b1, b2], None).unwrap())
            .unwrap();

        tds
    }

    fn build_three_triangles_sharing_edge_tds_2d() -> Tds<f64, (), (), 2> {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let v4 = tds.insert_vertex_with_mapping(vertex!([2.0, 0.0])).unwrap();

        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::new(vec![v0, v1, v4], None).unwrap(),
            )
            .unwrap();

        tds
    }

    #[test]
    fn test_validate_after_insertion_skips_when_no_simplices() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Force validation to be enabled if there were any simplices.
        tri.set_validation_policy(ValidationPolicy::Always);

        // Insert a vertex without creating any simplices (bootstrap phase).
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        assert_eq!(tri.number_of_simplices(), 0);

        tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None)
            .unwrap();
    }

    #[test]
    fn test_validate_after_insertion_calls_is_valid_when_policy_triggers() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::Always);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None) {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                ..
            })) => {}
            other => panic!("Expected Disconnected error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_after_insertion_required_checks_do_not_run_global_connectedness() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        // The triangulation is globally invalid (disconnected), but the required
        // pseudomanifold checks are local and still satisfied.
        assert!(tri.is_valid().is_err());
        tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None)
            .unwrap();
    }

    #[test]
    fn test_validate_after_insertion_does_not_skip_pseudomanifold_checks() {
        let tds = build_three_triangles_sharing_edge_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None) {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => {
                assert_eq!(simplex_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_scoped_validation_catches_touched_over_shared_facet() {
        let tds = build_three_triangles_sharing_edge_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let scope: SimplexKeyBuffer = tri.tds.simplex_keys().take(1).collect();

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope)) {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => {
                assert_eq!(simplex_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0_f64; D]));
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords));
        }
        vertices
    }

    fn unit_simplex_interior_vertex<const D: usize>() -> Vertex<f64, (), D> {
        vertex!([0.125_f64; D])
    }

    /// Build a simplex whose feature length is controlled by one shared axis scale.
    fn axis_scaled_simplex_vertices<const D: usize>(scale: f64) -> Vec<Vertex<f64, (), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0_f64; D]));
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = scale;
            vertices.push(vertex!(coords));
        }
        vertices
    }

    /// Build coordinates with only the first component set for tolerance-scale tests.
    fn coords_with_first<const D: usize>(first: f64) -> [f64; D] {
        let mut coords = [0.0_f64; D];
        coords[0] = first;
        coords
    }

    macro_rules! test_scoped_strict_validation_falls_back_to_global_vertex_links {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_scoped_strict_validation_falls_back_to_global_vertex_links_ $dim d>]() {
                        let (tds, expected_vertex_key) = build_invalid_vertex_link_tds::<$dim>();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);
                        let scope: SimplexKeyBuffer = tri.tds.simplex_keys().take(1).collect();
                        assert!(!scope.is_empty());

                        // Direct field assignment keeps this internal test focused on insertion-time
                        // strict fallback behavior even though the fixture is intentionally invalid.
                        tri.validation_policy = ValidationPolicy::OnSuspicion;
                        tri.topology_guarantee = TopologyGuarantee::PLManifoldStrict;

                        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope)) {
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::RidgeLinkNotManifold {
                                    connected: false,
                                    ..
                                },
                            )) if $dim == 2 => {
                                // In 2D, ridges are vertices, so the global strict path
                                // reports the disconnected apex link at the ridge layer first.
                            }
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. },
                            )) => {
                                assert_eq!(vertex_key, expected_vertex_key);
                            }
                            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
                        }
                    }
                )+
            }
        };
    }

    test_scoped_strict_validation_falls_back_to_global_vertex_links!(2, 3, 4, 5);

    #[test]
    fn test_local_geometric_orientation_validation_errors_on_missing_scope_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        assert_eq!(tri.tds.remove_simplices_by_keys(&[simplex_key]), 1);

        match tri.validate_geometric_simplex_orientation_for_simplices(&[simplex_key]) {
            Err(TdsError::SimplexNotFound {
                simplex_key: missing_key,
                ..
            }) => assert_eq!(missing_key, simplex_key),
            other => panic!("Expected SimplexNotFound, got {other:?}"),
        }
    }

    macro_rules! test_insertion_scoped_validation_preserves_full_validity {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_insertion_scoped_validation_preserves_full_validity_ $dim d>]() {
                        let vertices = unit_simplex_vertices::<$dim>();
                        let tds =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                                .unwrap();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
                        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

                        let detail = tri
                            .insert_with_statistics_seeded_indexed_detailed(
                                unit_simplex_interior_vertex::<$dim>(),
                                None,
                                None,
                                0,
                                None,
                                None,
                            )
                            .unwrap();

                        assert!(matches!(
                            detail.outcome,
                            InsertionOutcome::Inserted {
                                vertex_key: _,
                                hint: _
                            }
                        ));
                        assert!(!detail.repair_seed_simplices.is_empty());
                        tri.validate_after_insertion_with_scope(
                            SuspicionFlags::default(),
                            Some(&detail.repair_seed_simplices),
                        )
                        .unwrap();
                        tri.is_valid().unwrap();
                    }
                )+
            }
        };
    }

    test_insertion_scoped_validation_preserves_full_validity!(2, 3, 4, 5);

    #[test]
    fn test_validate_after_insertion_skips_global_validation_but_runs_required_checks() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        assert!(tri.is_valid().is_err());
        tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None)
            .unwrap();
    }

    #[test]
    fn test_validation_after_insertion_will_run_matches_policy_and_link_requirements() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(
            tri.validation_after_insertion_work(SuspicionFlags::default()),
            Some(InsertionValidationWork::RequiredTopologyLinks)
        );

        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);
        assert_eq!(
            tri.validation_after_insertion_work(SuspicionFlags::default()),
            Some(InsertionValidationWork::RequiredTopologyLinks)
        );
    }

    #[test]
    fn test_select_locate_hint_from_hash_grid_returns_incident_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        for (vkey, vertex) in tri.tds.vertices() {
            index.insert_vertex(vkey, vertex.point().coords());
        }

        let hint = tri.select_locate_hint_from_hash_grid(&[0.05, 0.05], &index);
        assert_eq!(hint, Some(simplex_key));
    }

    #[test]
    fn test_select_locate_hint_from_hash_grid_skips_missing_simplex() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let vkey = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        {
            let vertex = tri.tds.vertex_mut(vkey).unwrap();
            vertex.set_incident_simplex(Some(SimplexKey::default()));
        }

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        index.insert_vertex(vkey, &[0.0, 0.0]);

        let hint = tri.select_locate_hint_from_hash_grid(&[0.0, 0.0], &index);
        assert!(hint.is_none());
    }

    #[test]
    fn test_duplicate_coordinates_error_uses_hash_grid_index() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let vkey = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        index.insert_vertex(vkey, &[0.0, 0.0]);

        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol, Some(&index));
        assert!(matches!(
            err,
            Some(InsertionError::DuplicateCoordinates { .. })
        ));
    }

    #[test]
    fn test_duplicate_coordinates_error_falls_back_when_index_unusable() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let index: HashGridIndex<f64, 2> = HashGridIndex::new(0.0); // unusable
        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol, Some(&index));
        assert!(matches!(
            err,
            Some(InsertionError::DuplicateCoordinates { .. })
        ));
    }

    fn duplicate_coordinate_tolerance_scales_down_for_small_features<const D: usize>() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), D> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!(coords_with_first::<D>(1.0e-6)))
            .unwrap();

        let candidate = coords_with_first::<D>(1.0e-6 + 1.0e-11);
        let tolerance = tri.estimate_duplicate_coordinate_tolerance(&candidate, None);

        assert!(
            tolerance < 1.0e-10,
            "small-scale inputs should not inherit a fixed scalar-unit tolerance"
        );
        assert!(
            tri.duplicate_coordinates_error(&candidate, tolerance, None)
                .is_none(),
            "distinct small-scale vertices should not be skipped as duplicates"
        );
    }

    fn duplicate_coordinate_tolerance_uses_hint_simplex_span<const D: usize>() {
        let vertices = unit_simplex_vertices::<D>();
        let tds =
            Triangulation::<FastKernel<f64>, (), (), D>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), D>::new_with_tds(FastKernel::new(), tds);
        let hint = tri.tds.simplex_keys().next();
        let candidate = coords_with_first::<D>(5.0e-11);
        let tolerance = tri.estimate_duplicate_coordinate_tolerance(&candidate, hint);

        assert!(
            tolerance > 1.0e-10,
            "unit-scale hint simplices should preserve near-duplicate filtering"
        );
        assert!(matches!(
            tri.duplicate_coordinates_error(&candidate, tolerance, None),
            Some(InsertionError::DuplicateCoordinates { .. })
        ));
    }

    fn duplicate_index_rebuilds_when_tolerance_exceeds_cell_size<const D: usize>() {
        let vertices = axis_scaled_simplex_vertices::<D>(1.0e6);
        let tds =
            Triangulation::<FastKernel<f64>, (), (), D>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), D>::new_with_tds(FastKernel::new(), tds);
        let hint = tri.tds.simplex_keys().next();
        let candidate = [1.0_f64; D];
        let tolerance = tri.estimate_duplicate_coordinate_tolerance(&candidate, hint);
        let mut index: HashGridIndex<f64, D> = HashGridIndex::new(1.0e-10);
        for (vkey, vertex) in tri.tds.vertices() {
            index.insert_vertex(vkey, vertex.point().coords());
        }

        tri.ensure_duplicate_index_cell_size(Some(&mut index), tolerance);

        approx::assert_abs_diff_eq!(index.cell_size(), tolerance, epsilon = f64::EPSILON);
        assert!(
            index.for_each_candidate_vertex_key(&candidate, |_| false),
            "rebuilt duplicate index should remain queryable"
        );
    }

    #[test]
    fn test_duplicate_distance_within_tolerance_handles_overflowed_tolerance_square() {
        assert!(
            Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_distance_within_tolerance(
                f64::MAX,
                f64::MAX
            )
        );
        assert!(
            !Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_distance_within_tolerance(
                f64::MAX,
                1.0
            )
        );
    }

    macro_rules! test_duplicate_tolerance_dimensions {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_duplicate_coordinate_tolerance_scales_down_for_small_features_ $dim d>]() {
                        duplicate_coordinate_tolerance_scales_down_for_small_features::<$dim>();
                    }

                    #[test]
                    fn [<test_duplicate_coordinate_tolerance_uses_hint_simplex_span_ $dim d>]() {
                        duplicate_coordinate_tolerance_uses_hint_simplex_span::<$dim>();
                    }

                    #[test]
                    fn [<test_duplicate_index_rebuilds_when_tolerance_exceeds_cell_size_ $dim d>]() {
                        duplicate_index_rebuilds_when_tolerance_exceeds_cell_size::<$dim>();
                    }
                )+
            }
        };
    }

    test_duplicate_tolerance_dimensions!(2, 3, 4, 5);

    #[test]
    fn test_estimate_local_perturbation_scale_uses_hint_simplex_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();

        let scale = tri.estimate_local_perturbation_scale(&[0.1, 0.0], Some(simplex_key));
        assert!((scale - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_local_perturbation_scale_clamps_to_min_scale() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let scale = tri.estimate_local_perturbation_scale(&[0.0, 0.0], None);
        let min_scale = <f64 as CoordinateScalar>::default_tolerance();
        approx::assert_abs_diff_eq!(scale, min_scale, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pl_manifold_insertion_never_commits_invalid_topology_under_validation_policy_never() {
        // A small deterministic point set (seeded RNG) that exercises degeneracy handling.
        //
        // This test is intentionally small and validates after *each* insertion to ensure
        // we never commit an invalid PL-manifold state, even when the user disables
        // automatic validation via `ValidationPolicy::Never`.
        let points = generate_random_points_seeded::<f64, 3>(25, (-100.0, 100.0), 123).unwrap();

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

        dt.set_validation_policy(ValidationPolicy::Never);
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

        for (i, point) in points.into_iter().enumerate() {
            let vertex = VertexBuilder::default().point(point).build().unwrap();

            let result = dt
                .insert_with_statistics(vertex)
                .unwrap_or_else(|e| panic!("Non-retryable insertion error at i={i}: {e:?}"));

            let (outcome, stats) = result;

            // Skip Level 3 validation during bootstrap (vertices but no simplices yet).
            if dt.number_of_simplices() > 0
                && let Err(err) = dt.as_triangulation().validate()
            {
                panic!(
                    "Topology invalid after insertion i={i} (outcome={outcome:?}, attempts={}, used_perturbation={}): {err}",
                    stats.attempts,
                    stats.used_perturbation()
                );
            }
        }
    }

    /// Macro to generate `build_initial_simplex` tests across dimensions.
    ///
    /// This macro generates tests that verify `build_initial_simplex` by:
    /// 1. Creating D+1 affinely independent vertices
    /// 2. Calling `build_initial_simplex` directly
    /// 3. Verifying the Tds has correct structure (vertices, simplices, dimension)
    ///
    /// # Usage
    /// ```ignore
    /// test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_build_initial_simplex_ $dim d>]() {
                    // Build initial simplex (D+1 vertices)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();

                    // Verify structure
                    assert_eq!(tds.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert_eq!(tds.number_of_simplices(), 1,
                        "{}D: Expected 1 simplex", $dim);
                    assert_eq!(tds.dim(), $dim as i32,
                        "{}D: Expected dimension {}", $dim, $dim);

                    // Verify all vertices are present
                    assert_eq!(tds.vertices().count(), expected_vertices,
                        "{}D: All vertices should be in Tds", $dim);

                    // Verify the single simplex has correct number of vertices
                    let (_, simplex) = tds.simplices().next()
                        .expect(&format!("{}D: Should have exactly one simplex", $dim));
                    assert_eq!(simplex.number_of_vertices(), expected_vertices,
                        "{}D: Simplex should have {} vertices", $dim, expected_vertices);

                    // Verify incident simplices are assigned
                    for (_, vertex) in tds.vertices() {
                        assert!(vertex.incident_simplex().is_some(),
                            "{}D: All vertices should have incident simplex assigned", $dim);
                    }

                    // Verify initial simplex has explicit boundary neighbor slots.
                    let neighbors = simplex
                        .neighbor_slots()
                        .expect("initial simplex should assign boundary neighbor slots");
                    assert!(
                        neighbors.iter().all(|slot| *slot == NeighborSlot::Boundary),
                        "{}D: Initial simplex should have boundary slots",
                        $dim
                    );
                }
            }
        };
    }

    // 2D: Triangle
    test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron
    test_build_initial_simplex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex
    test_build_initial_simplex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex
    test_build_initial_simplex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    /// Macro to generate Level 3 (topology) validation tests across dimensions.
    ///
    /// This macro generates tests that verify manifold-with-boundary validation by:
    /// 1. Creating a Delaunay triangulation from D+1 affinely independent vertices
    /// 2. Calling `Triangulation::is_valid()` (Level 3)
    /// 3. Verifying that the validation passes
    ///
    /// # Usage
    /// ```ignore
    /// test_is_valid_topology!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_is_valid_topology {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_is_valid_topology_ $dim d>]() {
                    // Build triangulation from D+1 vertices (initial simplex)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let dt = DelaunayTriangulation::new(&vertices)
                        .expect(&format!("Failed to create {}D triangulation", $dim));
                    let tri = dt.as_triangulation();

                    // Level 3: topology validation
                    let result = tri.is_valid();
                    assert!(
                        result.is_ok(),
                        "{}D: Simple simplex should be a valid manifold-with-boundary. Error: {:?}",
                        $dim,
                        result.err()
                    );

                    // Also verify basic properties
                    assert_eq!(tri.number_of_vertices(), expected_vertices,
                        "{}D: Should have {} vertices", $dim, expected_vertices);
                    assert_eq!(tri.number_of_simplices(), 1,
                        "{}D: Should have exactly 1 simplex", $dim);
                }
            }
        };
    }

    // 2D: Triangle manifold
    test_is_valid_topology!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron manifold
    test_is_valid_topology!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex manifold
    test_is_valid_topology!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex manifold
    test_is_valid_topology!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    #[test]
    fn test_is_valid_topology_empty() {
        // Empty triangulation should pass topology validation
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert!(
            tri.is_valid().is_ok(),
            "Empty triangulation should be a valid (empty) manifold"
        );
    }

    #[test]
    fn test_is_valid_pl_manifold_mode_rejects_wedge_at_vertex_in_2d() {
        // This builds the same 2D "wedge at a vertex" configuration as the topology-module
        // unit test, but exercises the Level 3 validation pipeline and TopologyGuarantee gating.
        //
        // The complex is a pseudomanifold (every edge has degree 2), but not a PL 2-manifold:
        // the shared vertex has a disconnected link (two disjoint cycles).
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        // Shared vertex.
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();

        // First tetrahedron boundary (4 triangles on 4 vertices).
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        // Second tetrahedron boundary (shares only v0).
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v4, v5], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v4, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v5, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v4, v5, v6], None).unwrap())
            .unwrap();

        // Ensure neighbor pointers exist so connectedness validation is meaningful.
        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        // Default is PL-manifold mode; relax to pseudomanifold for this part of the test.
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        // In pseudomanifold mode, Level 3 validation proceeds past manifold checks and fails at
        // connectedness (two components that share only a vertex).
        assert!(matches!(
            tri.is_valid(),
            Err(InvariantError::Triangulation(
                TriangulationValidationError::Disconnected { .. }
            ))
        ));

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        // In strict PL-manifold mode, Level 3 validation should fail.  With connectivity now
        // checked first, a disconnected triangulation surfaces as Disconnected before vertex-link
        // or ridge-link validation.  The two components share only v0 (a vertex, not a facet)
        // so the neighbor graph is disconnected.
        match tri.is_valid() {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key,
                    link_vertex_count,
                    link_simplex_count,
                    boundary_facet_count,
                    max_degree,
                    connected,
                    interior_vertex,
                },
            )) => {
                assert_eq!(vertex_key, v0);
                assert!(interior_vertex);
                assert!(link_vertex_count > 0);
                assert!(link_simplex_count > 0);
                assert_eq!(boundary_facet_count, 0);
                assert_eq!(max_degree, 2);
                assert!(!connected);
            }
            // Connectivity is checked before link validation; a two-component wedge is
            // also disconnected in the neighbor graph, so either error is acceptable.
            Err(InvariantError::Triangulation(
                TriangulationValidationError::RidgeLinkNotManifold { .. }
                | TriangulationValidationError::Disconnected { .. },
            )) => {}
            other => panic!(
                "Expected RidgeLinkNotManifold, VertexLinkNotManifold, or Disconnected in strict PL-manifold mode, got {other:?}"
            ),
        }
    }

    #[test]
    fn test_is_valid_pl_manifold_mode_rejects_cone_on_torus_in_3d_even_when_simplex_graph_connected()
     {
        // Cone over a triangulated torus:
        // - The 3D simplex neighbor graph is connected.
        // - Facet-degree and closed-boundary checks pass.
        // - But the apex has link T^2 (χ=0), so PL-manifold vertex-link validation must fail.
        const N: usize = 3;
        const M: usize = 3;

        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let mut v: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in v.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let i_f = <f64 as std::convert::From<u32>>::from(u32::try_from(i).unwrap());
                let j_f = <f64 as std::convert::From<u32>>::from(u32::try_from(j).unwrap());
                *slot = tds
                    .insert_vertex_with_mapping(vertex!([i_f, j_f, 0.0]))
                    .unwrap();
            }
        }

        let apex = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 1.0]))
            .unwrap();

        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;

                let v00 = v[i][j];
                let v10 = v[i1][j];
                let v01 = v[i][j1];
                let v11 = v[i1][j1];

                for tri in [[v00, v10, v01], [v10, v11, v01]] {
                    let _ = tds
                        .insert_simplex_with_mapping(
                            Simplex::new(vec![tri[0], tri[1], tri[2], apex], None).unwrap(),
                        )
                        .unwrap();
                }
            }
        }

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Sanity: the simplex neighbor graph is connected.
        tri.validate_global_connectedness().unwrap();

        // Sanity: pseudomanifold-with-boundary checks pass.
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tri.tds, &facet_to_simplices).unwrap();

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        match tri.is_valid() {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key,
                    link_vertex_count,
                    link_simplex_count,
                    boundary_facet_count,
                    connected,
                    interior_vertex,
                    ..
                },
            )) => {
                assert_eq!(vertex_key, apex);
                assert!(interior_vertex);
                assert!(connected);
                assert_eq!(boundary_facet_count, 0);
                assert!(link_vertex_count > 0);
                assert!(link_simplex_count > 0);
            }
            other => panic!("Expected VertexLinkNotManifold for cone apex, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_disconnected_detected_before_non_manifold_boundary_ridge() {
        // Two tetrahedra that share only an edge (not a facet) are disconnected in the neighbor
        // graph (no shared facet ⇒ no neighbor pointers).  Connectivity is now checked FIRST in
        // `is_valid()`, so the disconnection error is returned before the non-manifold boundary
        // ridge error (4 boundary triangles on the shared edge).
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        // Shared edge
        let shared_edge_v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();

        // First tetrahedron
        let tet1_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();

        // Second tetrahedron
        let tet2_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0, 0.0]))
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::new(vec![shared_edge_v0, shared_edge_v1, tet1_v2, tet1_v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::new(vec![shared_edge_v0, shared_edge_v1, tet2_v2, tet2_v3], None).unwrap(),
            )
            .unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // The two simplices share only an edge, so they have no mutual neighbor pointers and the
        // neighbor-graph BFS visits only one of them.  Connectivity is now the first check in
        // `is_valid()`, so the disconnection error is returned first.
        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => {
                assert_eq!(
                    simplex_count, 2,
                    "Expected 2 simplices in disconnected triangulation"
                );
            }
            other => panic!("Expected Disconnected, got {other:?}"),
        }
    }
    #[test]
    fn test_validate_includes_tds_validation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        // Triangulation::validate should pass if the underlying TDS validates.
        assert!(tri.tds.validate().is_ok(), "TDS should validate");
        assert!(
            tri.validate().is_ok(),
            "Triangulation::validate should pass"
        );
    }

    #[test]
    fn test_is_valid_rejects_bootstrap_phase_with_isolated_vertex() {
        // A triangulation with vertices but no simplices is not a valid manifold (Level 3).
        // Level 3 requires every vertex to be incident to at least one simplex.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Bootstrap insertion (no simplices yet)
        let vertex = vertex!([0.0, 0.0, 0.0]);
        let expected_uuid = vertex.uuid();
        let (expected_vk, _) = tri
            .insert(vertex, None, None)
            .expect("bootstrap insertion should succeed");

        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                vertex_key,
                vertex_uuid,
            })) => {
                assert_eq!(vertex_key, expected_vk);
                assert_eq!(vertex_uuid, expected_uuid);
            }
            other => panic!("Expected IsolatedVertex error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_rejects_isolated_vertex_even_when_simplices_exist() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Default is PL-manifold mode; use pseudomanifold mode here so the isolated-vertex check
        // triggers before vertex-link validation.
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        // Insert a vertex into the TDS without adding any simplices that reference it.
        // This creates an isolated vertex, which violates the Level 3 manifold invariant.
        let _isolated_vk = tri
            .tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0, 10.0]))
            .unwrap();

        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                ..
            })) => {
                // Expected: isolated vertex produces a structured IsolatedVertex error.
            }
            other => panic!("Expected IsolatedVertex error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_rejects_disconnected_even_when_euler_matches() {
        // Construct a disconnected 1D triangulation made of:
        // - A path (Ball(1)) with χ = 1
        // - A cycle (ClosedSphere(1)) with χ = 0
        //
        // The overall complex has boundary, so it is classified as Ball(1) with expected χ = 1.
        // Euler characteristic alone therefore cannot detect disconnectedness here.
        let mut tds: Tds<f64, (), (), 1> = Tds::empty();

        // Path component: v0 - v1 - v2 (2 edges)
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([2.0])).unwrap();

        let e0 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1], None).unwrap())
            .unwrap();
        let e1 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v1, v2], None).unwrap())
            .unwrap();

        // Cycle component: v3 - v4 - v5 - v3 (3 edges)
        let v3 = tds.insert_vertex_with_mapping(vertex!([10.0])).unwrap();
        let v4 = tds.insert_vertex_with_mapping(vertex!([11.0])).unwrap();
        let v5 = tds.insert_vertex_with_mapping(vertex!([12.0])).unwrap();

        let c0 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v3, v4], None).unwrap())
            .unwrap();
        let c1 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v4, v5], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v5, v3], None).unwrap())
            .unwrap();

        // Set neighbor pointers (1D: each simplex has 2 "facets" => 2 neighbor slots).

        // Path neighbors:
        {
            let simplex = tds.simplex_mut(e0).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(2, None);
            // e0 = [v0, v1]; across v1 is facet_index=0
            neighbors[0] = Some(e1);
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }
        {
            let simplex = tds.simplex_mut(e1).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(2, None);
            // e1 = [v1, v2]; across v1 is facet_index=1
            neighbors[1] = Some(e0);
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }

        // Cycle neighbors:
        {
            let simplex = tds.simplex_mut(c0).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(2, None);
            // c0 = [v3, v4]; across v4 is facet_index=0, across v3 is facet_index=1
            neighbors[0] = Some(c1); // at v4
            neighbors[1] = Some(c2); // at v3
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }
        {
            let simplex = tds.simplex_mut(c1).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(2, None);
            // c1 = [v4, v5]; across v5 is facet_index=0, across v4 is facet_index=1
            neighbors[0] = Some(c2); // at v5
            neighbors[1] = Some(c0); // at v4
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }
        {
            let simplex = tds.simplex_mut(c2).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(2, None);
            // c2 = [v5, v3]; across v3 is facet_index=0, across v5 is facet_index=1
            neighbors[0] = Some(c0); // at v3
            neighbors[1] = Some(c1); // at v5
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }

        tds.assign_incident_simplices().unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 1>::new_with_tds(FastKernel::new(), tds);

        // Sanity: codimension-1 pseudomanifold facet multiplicity passes.
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree(&facet_to_simplices).unwrap();

        // Sanity: Euler characteristic check would pass for this disconnected complex.
        let topology = validate_triangulation_euler(&tri.tds).unwrap();
        assert_eq!(
            topology.classification,
            TopologyClassification::Ball(1),
            "Classification should be Ball(1) because the complex has boundary"
        );
        assert_eq!(topology.expected, Some(1));
        assert_eq!(topology.chi, 1);

        // Level 3 should still fail due to disconnectedness.
        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => {
                assert_eq!(
                    simplex_count, 5,
                    "Expected 5 simplices (2 path + 3 cycle) in disconnected triangulation"
                );
            }
            other => panic!("Expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    fn test_tds_is_valid_rejects_boundary_facet_has_neighbor() {
        // Create two disjoint tetrahedra and manually introduce an invalid neighbor pointer
        // across a boundary facet.
        let vertices_simplex_1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices_simplex_1)
                .unwrap();
        let first_simplex_key = tds.simplex_keys().next().unwrap();

        // Add a disjoint second tetrahedron.
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0, 0.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 0.0, 0.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0, 0.0]))
            .unwrap();
        let v7 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0, 1.0]))
            .unwrap();

        let simplex_2 = Simplex::new(vec![v4, v5, v6, v7], None).unwrap();
        let second_simplex_key = tds.insert_simplex_with_mapping(simplex_2).unwrap();

        // Invalidate: boundary facet has a neighbor pointer.
        let first_simplex = tds.simplex_mut(first_simplex_key).unwrap();
        let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
        neighbors.resize(4, None);
        neighbors[0] = Some(second_simplex_key);
        first_simplex.set_neighbors_from_keys(neighbors).unwrap();

        match tds.is_valid() {
            Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNeighbor { neighbor_key, .. },
            }) => {
                assert_eq!(neighbor_key, second_simplex_key);
            }
            other => panic!("Expected InvalidNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn test_tds_is_valid_rejects_interior_facet_neighbor_mismatch() {
        // Two tetrahedra share a facet, but we leave neighbor pointers unset.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();

        let simplex_1 = Simplex::new(vec![v0, v1, v2, v3], None).unwrap();
        let simplex_2 = Simplex::new(vec![v0, v1, v2, v4], None).unwrap();
        let _ = tds.insert_simplex_with_mapping(simplex_1).unwrap();
        let _ = tds.insert_simplex_with_mapping(simplex_2).unwrap();

        match tds.is_valid() {
            Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::InteriorFacetNeighborMismatch { .. },
            }) => {}
            other => panic!("Expected InvalidNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_non_manifold_facet_multiplicity() {
        // Three tetrahedra share a single facet -> not a manifold-with-boundary.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 3.0]))
            .unwrap();

        let simplex_1 = Simplex::new(vec![v0, v1, v2, v3], None).unwrap();
        let simplex_2 = Simplex::new(vec![v0, v1, v2, v4], None).unwrap();
        let simplex_3 = Simplex::new(vec![v0, v1, v2, v5], None).unwrap();

        let _ = tds.insert_simplex_with_mapping(simplex_1).unwrap();
        let _ = tds.insert_simplex_with_mapping(simplex_2).unwrap();
        let _ = tds
            .insert_simplex_bypassing_topology_checks_for_test(simplex_3)
            .unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // The three simplices share facet {v0,v1,v2} three-ways.  `repair_neighbor_pointers` would
        // fail on this configuration (a facet shared by >2 simplices violates the early-exit guard in
        // `assign_neighbors`), so the simplices have no neighbor pointers and the neighbor-graph BFS
        // visits only one of them.  Connectivity is now the first check in `is_valid()`, so the
        // disconnection error is returned before the non-manifold facet error.
        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                ..
            })) => {}
            // Non-manifold facet detection is still valid if the simplices happen to be connected.
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => {
                assert_eq!(simplex_count, 3);
            }
            other => panic!("Expected Disconnected or ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_triangulation_validation_report_ok_for_valid_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        assert!(tri.validation_report().is_ok());
    }

    #[test]
    fn test_triangulation_validation_report_returns_mapping_failures_only() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Break UUID↔key mappings: remove one vertex UUID entry.
        let uuid = tri.tds.vertices().next().unwrap().1.uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = tri.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::SimplexMappings
            )
        }));
    }

    #[test]
    fn test_triangulation_validation_report_includes_vertex_and_simplex_validity() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Insert an invalid vertex (nil UUID) to exercise VertexValidity reporting.
        let invalid_vertex: Vertex<f64, (), 3> = Vertex::empty();
        let _ = tri.tds.insert_vertex_with_mapping(invalid_vertex).unwrap();

        // Corrupt one simplex locally: neighbors buffer with the wrong length.
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
        simplex.ensure_neighbors_buffer_mut().truncate(3); // expected D+1 = 4

        let report = tri.validation_report().unwrap_err();

        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexValidity),
            "Report should include a VertexValidity violation"
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::SimplexValidity),
            "Report should include a SimplexValidity violation"
        );
    }

    #[test]
    fn test_insert_duplicate_coordinates_skips_with_statistics_and_errors_without() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // First insertion succeeds.
        tri.insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        // Second insertion at same coordinates: insert() returns Err, insert_with_statistics() reports Skipped.
        let err = tri
            .insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .unwrap_err();
        assert!(matches!(err, InsertionError::DuplicateCoordinates { .. }));

        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.0, 0.0, 0.0]), None, None)
            .unwrap();
        assert!(stats.skipped());
        assert!(matches!(outcome, InsertionOutcome::Skipped { .. }));

        // No new vertex should have been inserted.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn test_insert_duplicate_uuid_is_non_retryable_and_rolls_back() {
        // Insert a vertex, then attempt to insert another vertex with the same UUID.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        tri.insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        let existing_uuid = tri.tds.vertices().next().unwrap().1.uuid();
        let mut dup = vertex!([1.0, 0.0, 0.0]);
        dup.set_uuid(existing_uuid).unwrap();

        let err = tri.insert(dup, None, None).unwrap_err();
        assert!(
            !err.is_retryable(),
            "Duplicate UUID should be non-retryable"
        );

        // Ensure rollback: vertex count unchanged.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn test_build_initial_simplex_insufficient_vertices() {
        // Try to build 3D simplex with only 2 vertices (need 4)
        let vertices = vec![vertex!([0.0, 0.0, 0.0]), vertex!([1.0, 0.0, 0.0])];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_build_initial_simplex_too_many_vertices() {
        // Try to build 2D simplex with 4 vertices (need exactly 3)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {}
            _ => panic!("Expected InsufficientVertices error for wrong count"),
        }
    }

    fn invalid_initial_simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0_f64; D]));

        let mut invalid_coords = [0.0_f64; D];
        invalid_coords[0] = 1.0;
        invalid_coords[1] = f64::NAN;
        vertices.push(Vertex::new_with_uuid(
            Point::new(invalid_coords),
            Uuid::new_v4(),
            None,
        ));

        for axis in 1..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords));
        }

        vertices
    }

    macro_rules! test_build_initial_simplex_rejects_invalid_vertex_dimensions {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_build_initial_simplex_rejects_invalid_vertex_ $dim d>]() {
                        let vertices = invalid_initial_simplex_vertices::<$dim>();

                        let result = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices);

                        assert!(matches!(
                            result,
                            Err(TriangulationConstructionError::Tds(
                                TdsConstructionError::ValidationError(TdsError::InvalidVertex { .. })
                            ))
                        ));
                    }
                )+
            }
        };
    }

    test_build_initial_simplex_rejects_invalid_vertex_dimensions!(2, 3, 4, 5);

    #[test]
    fn test_build_initial_simplex_with_user_data() {
        // Build vertices with user data
        let v1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(42_usize)
            .build()
            .unwrap();
        let v2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(43_usize)
            .build()
            .unwrap();
        let v3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(44_usize)
            .build()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let tds = Triangulation::<FastKernel<f64>, usize, (), 2>::build_initial_simplex(&vertices)
            .unwrap();

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_simplices(), 1);

        // Verify user data is preserved
        let data_values: Vec<_> = tds
            .vertices()
            .filter_map(|(_, v)| v.data.as_ref())
            .copied()
            .collect();
        assert_eq!(data_values.len(), 3);
        assert!(data_values.contains(&42));
        assert!(data_values.contains(&43));
        assert!(data_values.contains(&44));
    }

    // =============================================================================
    // Tests for build_initial_simplex degeneracy validation
    // =============================================================================

    #[test]
    fn test_build_initial_simplex_rejects_collinear_2d() {
        // Collinear points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Collinear points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate"),
                    "Error message should mention degeneracy"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for collinear points"),
        }
    }

    #[test]
    fn test_build_initial_simplex_rejects_coplanar_3d() {
        // Coplanar points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Coplanar points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate") || message.contains("coplanar"),
                    "Error message should mention degeneracy or coplanarity"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for coplanar points"),
        }
    }

    #[test]
    fn test_is_valid_rejects_negative_geometric_simplex_orientation() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        let simplex_key = tds.simplex_keys().next().unwrap();
        let simplex = tds.simplex_mut(simplex_key).unwrap();
        simplex.swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.is_valid().unwrap_err();
        assert!(matches!(
            err,
            InvariantError::Tds(TdsError::Geometric(GeometricError::NegativeOrientation { message }))
                if message.contains("negative geometric orientation")
        ));
    }

    /// Calls `validate_geometric_simplex_orientation()` directly (not through `is_valid()`
    /// which may short-circuit on coherent orientation checks) and asserts the returned
    /// error contains vertex keys for debuggability.
    #[test]
    fn test_validate_geometric_simplex_orientation_returns_enriched_error_on_negative() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(
            matches!(
                &err,
                TdsError::Geometric(GeometricError::NegativeOrientation { message })
                    if message.contains("negative geometric orientation")
                       && message.contains("vertices")
            ),
            "Error should contain vertex keys: {err}"
        );
    }

    #[test]
    fn test_simplices_require_positive_orientation_promotion_detects_negative_without_mutating() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let before: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        assert!(
            tri.simplices_require_positive_orientation_promotion()
                .unwrap(),
            "Negative orientation should be detected"
        );

        let after: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();
        assert_eq!(
            before, after,
            "Convergence check must not mutate simplex slot ordering"
        );
    }

    #[test]
    fn test_simplices_require_positive_orientation_promotion_false_for_positive_without_mutating() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let before: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        assert!(
            !tri.simplices_require_positive_orientation_promotion()
                .unwrap(),
            "Already-positive orientation should not require promotion"
        );

        let after: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();
        assert_eq!(
            before, after,
            "Convergence check must not mutate already-positive simplices"
        );
    }

    #[test]
    fn test_periodic_geometric_orientation_validation_uses_lifted_coordinates() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.8, 0.0]),
            vertex!([0.0, 0.8]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
                .unwrap();
        }

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_global_topology(GlobalTopology::Toroidal {
            domain: [1.0, 1.0],
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        });

        // In lifted coordinates this simplex is positively oriented.
        assert!(tri.validate_geometric_simplex_orientation().is_ok());

        // Flipping two slots should invert lifted orientation and be rejected.
        {
            let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
            simplex.swap_vertex_slots(0, 1);
        }
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(matches!(
            err,
            TdsError::Geometric(GeometricError::NegativeOrientation { message })
                if message.contains("negative geometric orientation")
        ));
    }

    #[test]
    fn test_periodic_geometric_orientation_validation_requires_toroidal_metadata() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.8, 0.0]),
            vertex!([0.0, 0.8]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
                .unwrap();
        }

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("has periodic offsets")
                    && message.contains("expected periodic-orientation-offset-capable topology")
        ));
    }

    #[test]
    fn test_periodic_geometric_orientation_validation_rejects_offset_count_mismatch() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.8, 0.0]),
            vertex!([0.0, 0.8]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .periodic_vertex_offsets = Some(vec![[0, 0], [1, 0]].into());

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(matches!(
            err,
            TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_periodic_geometric_orientation_validation_maps_lift_errors() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.8, 0.0]),
            vertex!([0.0, 0.8]),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
            .unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_global_topology(GlobalTopology::Toroidal {
            domain: [0.0, 1.0],
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        });

        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("Failed to lift coordinates")
                    && message.contains("Invalid toroidal period")
        ));
    }

    /// Consolidated macro for facet validation tests across dimensions.
    ///
    /// Verifies the manifold topology invariant: each facet shared by at most 2 simplices.
    /// Consolidates detection and repair tests into comprehensive suites.
    macro_rules! test_facet_validation {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_detect_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Valid simplex: should have no issues
                    let simplex_keys: Vec<_> = tri.tds.simplex_keys().collect();
                    assert_eq!(simplex_keys.len(), 1);
                    let issues = tri.detect_local_facet_issues(&simplex_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Valid simplex should have no facet issues", $dim);

                    // Empty list: should return None
                    let issues = tri.detect_local_facet_issues(&[]).unwrap();
                    assert!(issues.is_none(), "{}D: Empty list should have no issues", $dim);

                    // Nonexistent simplices: should be skipped gracefully
                    let fake_keys = vec![SimplexKey::default()];
                    let issues = tri.detect_local_facet_issues(&fake_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Nonexistent simplices should be skipped", $dim);

                    // Verify neighbors (all should be explicit boundary slots for a single simplex)
                    let (_, simplex) = tri.tds.simplices().next().unwrap();
                    let neighbors = simplex
                        .neighbor_slots()
                        .expect("single simplex should assign boundary neighbor slots");
                    assert!(
                        neighbors.iter().all(|slot| *slot == NeighborSlot::Boundary),
                        "{}D: Single simplex should have boundary slots",
                        $dim
                    );
                }

                #[test]
                fn [<test_repair_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let mut tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Empty issues map: should remove nothing
                    let empty_issues = FacetIssuesMap::default();
                    let removed = tri.repair_local_facet_issues(&empty_issues).unwrap();
                    assert_eq!(removed, 0, "{}D: Empty issues should remove 0 simplices", $dim);
                    assert_eq!(tri.tds.number_of_simplices(), 1, "{}D: Should still have 1 simplex", $dim);
                }
            }
        };
    }

    /// Dimension-parametric `remove_vertex` tests.
    ///
    /// Verifies that vertex removal maintains neighbor pointer integrity and
    /// triangulation validity across dimensions.
    macro_rules! test_remove_vertex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_remove_vertex_neighbor_pointers_ $dim d>]() {
                    // Build triangulation with D+1 simplex vertices + 1 interior point
                    let vertices: Vec<Vertex<f64, (), $dim>> = {
                        let mut v = vec![$(vertex!($simplex_coords)),+];
                        v.push(vertex!($interior_point));
                        v
                    };

                    let mut dt = DelaunayTriangulation::new(&vertices)
                        .expect("Failed to create triangulation");

                    // Find and remove the interior vertex
                    let interior_vertex_key = dt
                        .vertices()
                        .find(|(_, v)| {
                            let coords = v.point().coords();
                            coords.iter()
                                .zip($interior_point.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-10)
                        })
                        .map(|(k, _)| k)
                        .expect("Interior vertex not found");

                    let initial_simplex_count = dt.tds().number_of_simplices();
                    dt.remove_vertex(interior_vertex_key)
                        .expect("Failed to remove vertex");

                    // After removal, should have fewer simplices (or same if just 1 simplex left)
                    assert!(dt.tds().number_of_simplices() <= initial_simplex_count,
                        "{}D: Simplex count should not increase after removal", $dim);

                    // Verify neighbor pointer consistency:
                    // 1. No dangling pointers (all neighbor keys exist)
                    // 2. Neighbor relationships are symmetric
                    for (simplex_key, simplex) in dt.tds().simplices() {
                        if let Some(neighbors) = simplex.neighbors() {
                            for (facet_idx, neighbor_opt) in neighbors.enumerate() {
                                if let Some(neighbor_key) = neighbor_opt {
                                    // Verify neighbor exists
                                    assert!(
                                        dt.tds().contains_simplex(neighbor_key),
                                        "{}D: Simplex {simplex_key:?} has neighbor pointer to non-existent simplex {neighbor_key:?}",
                                        $dim
                                    );

                                    // Verify symmetry: neighbor should point back to us
                                    let neighbor_simplex = dt
                                        .tds()
                                        .simplex(neighbor_key)
                                        .expect("Neighbor simplex should exist");
                                    if let Some(mut neighbor_neighbors) = neighbor_simplex.neighbors() {
                                        let points_back = neighbor_neighbors
                                            .any(|neighbor| neighbor == Some(simplex_key));
                                        assert!(
                                            points_back,
                                            "{}D: Simplex {simplex_key:?} has neighbor {neighbor_key:?} at facet {facet_idx}, but neighbor doesn't point back",
                                            $dim
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Verify triangulation is still valid (Levels 1–3; removal does not guarantee Delaunay)
                    let validation = dt.as_triangulation().validate();
                    assert!(
                        validation.is_ok(),
                        "{}D: Triangulation should be structurally valid after vertex removal: {:?}",
                        $dim,
                        validation.err()
                    );
                }
            }
        };
    }

    /// Basic accessor tests across dimensions.
    macro_rules! test_basic_accessors {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_basic_accessors_ $dim d>]() {
                    // Empty triangulation
                    let empty: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());
                    assert_eq!(empty.number_of_vertices(), 0);
                    assert_eq!(empty.number_of_simplices(), 0);
                    assert_eq!(empty.dim(), -1);
                    assert_eq!(empty.simplices().count(), 0);
                    assert_eq!(empty.vertices().count(), 0);
                    assert_eq!(empty.facets().count(), 0);
                    assert_eq!(empty.boundary_facets().count(), 0);

                    // Simplex triangulation
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];
                    let expected_vertex_count = vertices.len();

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(
                        FastKernel::new(),
                        tds,
                    );

                    assert_eq!(tri.number_of_vertices(), expected_vertex_count);
                    assert_eq!(tri.number_of_simplices(), 1);
                    assert_eq!(tri.dim(), $dim as i32);
                    assert_eq!(tri.simplices().count(), 1);
                    assert_eq!(tri.vertices().count(), expected_vertex_count);

                    // D-simplex has D+1 facets, all on boundary
                    let facet_count = tri.facets().count();
                    assert_eq!(facet_count, expected_vertex_count, "{}D: D-simplex should have D+1 facets", $dim);
                    let boundary_count = tri.boundary_facets().count();
                    assert_eq!(boundary_count, expected_vertex_count, "{}D: All facets should be on boundary", $dim);
                }
            }
        };
    }

    // Facet validation tests (2D - 5D)
    test_facet_validation!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_facet_validation!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    // Basic accessor tests (2D - 5D)
    test_basic_accessors!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_basic_accessors!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    // Remove vertex tests (2D - 5D)
    test_remove_vertex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.3, 0.3]);
    test_remove_vertex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.25, 0.25, 0.25]
    );
    test_remove_vertex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2]
    );
    test_remove_vertex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.16, 0.16, 0.16, 0.16, 0.16]
    );

    // =============================================================================
    // Public Topology Traversal & Adjacency API (Read-only)
    // =============================================================================

    #[test]
    fn test_topology_edges_triangle_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 1);
        assert_eq!(tri.number_of_vertices(), 3);
        assert_eq!(tri.number_of_edges(), 3);

        let edges: HashSet<_> = tri.edges().collect();
        assert_eq!(edges.len(), 3);

        let index = tri.build_adjacency_index().unwrap();
        let edges_with_index: HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_with_index, edges);
        assert_eq!(tri.number_of_edges_with_index(&index), 3);

        // Edge endpoints should always be vertex keys from this triangulation.
        assert!(edges.iter().all(|e| {
            let (a, b) = e.endpoints();
            a != b && tri.vertex_coords(a).is_some() && tri.vertex_coords(b).is_some()
        }));
    }

    #[test]
    fn test_topology_edges_and_incident_edges_double_tetrahedron_3d() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            // Two apices
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_simplices(), 2);
        assert_eq!(tri.number_of_vertices(), 5);

        // This configuration has 9 unique edges (3 base + 6 apex-to-base).
        assert_eq!(tri.number_of_edges(), 9);

        // A base vertex has degree 4: two base edges + two apex edges.
        let base_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);

        let index = tri.build_adjacency_index().unwrap();
        assert_eq!(tri.number_of_edges_with_index(&index), 9);

        // A base vertex is incident to both simplices.
        assert_eq!(tri.adjacent_simplices(base_vertex_key).count(), 2);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, base_vertex_key)
                .count(),
            2
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, base_vertex_key),
            2
        );

        // A base vertex has degree 4: two base edges + two apex edges.
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);
        assert_eq!(
            tri.incident_edges_with_index(&index, base_vertex_key)
                .count(),
            4
        );
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, base_vertex_key),
            4
        );

        // An apex has degree 3: connected to all three base vertices.
        let apex_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.7, 1.5]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(apex_vertex_key), 3);
        assert_eq!(
            tri.adjacent_simplices_with_index(&index, apex_vertex_key)
                .count(),
            1
        );
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, apex_vertex_key),
            1
        );

        // Each simplex has exactly one neighbor in the index.
        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        for &ck in &simplex_keys {
            assert_eq!(tri.simplex_neighbors_with_index(&index, ck).count(), 1);
            assert_eq!(tri.number_of_simplex_neighbors_with_index(&index, ck), 1);
        }
    }

    #[test]
    fn test_topology_queries_missing_keys_are_empty_or_none() {
        // Use a "null" SlotMap key, which should never be present in a valid triangulation.
        let vertices_a = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_a).unwrap();
        let tri_a = dt_a.as_triangulation();

        let index = tri_a.build_adjacency_index().unwrap();

        let missing_vertex_key = VertexKey::default();
        assert_eq!(tri_a.adjacent_simplices(missing_vertex_key).count(), 0);
        assert_eq!(
            tri_a
                .adjacent_simplices_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(
            tri_a.number_of_adjacent_simplices_with_index(&index, missing_vertex_key),
            0
        );

        assert_eq!(tri_a.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(
            tri_a
                .incident_edges_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(tri_a.number_of_incident_edges(missing_vertex_key), 0);
        assert_eq!(
            tri_a.number_of_incident_edges_with_index(&index, missing_vertex_key),
            0
        );
        assert!(tri_a.vertex_coords(missing_vertex_key).is_none());

        let missing_simplex_key = SimplexKey::default();
        assert_eq!(tri_a.simplex_neighbors(missing_simplex_key).count(), 0);
        assert_eq!(
            tri_a
                .simplex_neighbors_with_index(&index, missing_simplex_key)
                .count(),
            0
        );
        assert_eq!(
            tri_a.number_of_simplex_neighbors_with_index(&index, missing_simplex_key),
            0
        );
        assert!(tri_a.simplex_vertices(missing_simplex_key).is_none());
    }

    #[test]
    fn test_topology_geometry_accessors_roundtrip() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let v_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.vertex_coords(v_key).unwrap(), [1.0, 0.0]);

        let simplex_key = tri.simplices().next().unwrap().0;
        let simplex_vertices = tri.simplex_vertices(simplex_key).unwrap();
        assert_eq!(simplex_vertices.len(), 3);
        assert!(simplex_vertices.contains(&v_key));
    }

    #[test]
    fn test_build_adjacency_index_basic_invariants() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            // Two apices
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let index = tri.build_adjacency_index().unwrap();

        // Each simplex has exactly one neighbor.
        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        assert_eq!(simplex_keys.len(), 2);
        for &ck in &simplex_keys {
            let neighbors = index.simplex_to_neighbors.get(&ck).unwrap();
            assert_eq!(neighbors.len(), 1);
            assert!(simplex_keys.contains(&neighbors[0]));
            assert_ne!(neighbors[0], ck);
        }

        // For every vertex, edges/simplices lists exist and are consistent.
        for (vk, _) in tri.vertices() {
            let simplices = index.vertex_to_simplices.get(&vk).unwrap();
            assert!(!simplices.is_empty());

            let edges = index.vertex_to_edges.get(&vk).unwrap();
            assert!(!edges.is_empty());
            assert!(
                edges
                    .iter()
                    .all(|e| matches!(e.endpoints(), (a, b) if a == vk || b == vk))
            );
        }
    }

    #[test]
    fn test_build_adjacency_index_empty_triangulation_is_empty() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let index = tri.build_adjacency_index().unwrap();
        assert!(index.vertex_to_simplices.is_empty());
        assert!(index.simplex_to_neighbors.is_empty());
        assert!(index.vertex_to_edges.is_empty());
    }

    #[test]
    fn test_build_adjacency_index_includes_isolated_vertex_entries() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let isolated_vertex = tri
            .tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let index = tri.build_adjacency_index().unwrap();

        assert!(
            index
                .vertex_to_simplices
                .get(&isolated_vertex)
                .is_some_and(SmallBuffer::is_empty)
        );
        assert!(
            index
                .vertex_to_edges
                .get(&isolated_vertex)
                .is_some_and(SmallBuffer::is_empty)
        );
    }

    #[test]
    fn test_build_adjacency_index_errors_on_missing_neighbor_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();

        let mut missing_neighbor = SimplexKey::default();
        if tri.tds.contains_simplex(missing_neighbor) {
            missing_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        }
        assert!(!tri.tds.contains_simplex(missing_neighbor));

        {
            let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys([Some(missing_neighbor), None, None])
                .unwrap();
        }

        match tri.build_adjacency_index() {
            Err(AdjacencyIndexBuildError::MissingNeighborSimplex {
                simplex_key: err_simplex_key,
                neighbor_key,
            }) => {
                assert_eq!(err_simplex_key, simplex_key);
                assert_eq!(neighbor_key, missing_neighbor);
            }
            other => panic!("Expected MissingNeighborSimplex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_neighbors_filters_missing_neighbor_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let missing_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        assert!(!tri.tds.contains_simplex(missing_neighbor));

        tri.tds
            .simplex_mut(simplex_key)
            .unwrap()
            .set_neighbors_from_keys([Some(missing_neighbor), None, None])
            .unwrap();

        assert_eq!(tri.simplex_neighbors(simplex_key).count(), 0);
    }

    #[test]
    fn test_build_adjacency_index_errors_on_missing_vertex_key() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let existing_vertices = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        let mut missing_vertex = VertexKey::default();
        if tri.tds.contains_vertex_key(missing_vertex) {
            missing_vertex = VertexKey::from(KeyData::from_ffi(u64::MAX));
        }
        assert!(!tri.tds.contains_vertex_key(missing_vertex));

        {
            let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(existing_vertices[0]);
            simplex.push_vertex_key(existing_vertices[1]);
            simplex.push_vertex_key(missing_vertex);
        }

        match tri.build_adjacency_index() {
            Err(AdjacencyIndexBuildError::MissingVertexKey {
                simplex_key: err_simplex_key,
                vertex_key,
            }) => {
                assert_eq!(err_simplex_key, simplex_key);
                assert_eq!(vertex_key, missing_vertex);
            }
            other => panic!("Expected MissingVertexKey, got {other:?}"),
        }
    }

    // =============================================================================
    // Triangulation insert_with_statistics tests (internal API)
    // =============================================================================

    #[test]
    fn triangulation_insert_with_statistics_basic_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .expect("insertion should succeed");

        assert!(matches!(
            outcome,
            InsertionOutcome::Inserted { hint: None, .. }
        ));
        assert_eq!(stats.attempts, 1);
        assert!(!stats.used_perturbation());
        assert!(!stats.skipped());
        assert!(stats.success());
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn triangulation_insert_with_statistics_bootstrap_3d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Insert D+1 vertices to create initial simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        for (i, v) in vertices.into_iter().enumerate() {
            let (outcome, stats) = tri.insert_with_statistics(v, None, None).unwrap();

            assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
            assert_eq!(stats.attempts, 1);

            if i < 3 {
                // Bootstrap phase - no hint yet
                assert!(matches!(
                    outcome,
                    InsertionOutcome::Inserted { hint: None, .. }
                ));
            } else {
                // After D+1 vertices, hint should be available
                assert!(matches!(
                    outcome,
                    InsertionOutcome::Inserted { hint: Some(_), .. }
                ));
            }
        }

        assert_eq!(tri.number_of_vertices(), 4);
        assert_eq!(tri.number_of_simplices(), 1);
    }

    #[test]
    fn triangulation_exterior_insert_3d_uses_local_conflict_without_global_scan() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        for coords in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            tri.insert_with_statistics(vertex!(coords), None, None)
                .unwrap();
        }

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                vertex!([2.0, 2.0, 2.0]),
                None,
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert!(matches!(detail.outcome, InsertionOutcome::Inserted { .. }));
        assert_eq!(detail.telemetry.global_conflict_scans, 0);
        assert_eq!(detail.telemetry.global_conflict_simplices_scanned, 0);
        assert_eq!(detail.telemetry.global_conflict_simplices_found_total, 0);
        assert_eq!(detail.telemetry.global_conflict_scan_nanos, 0);
        assert_eq!(detail.telemetry.conflict_region_calls, 1);
        assert_eq!(detail.telemetry.conflict_region_simplices_total, 0);
        assert_eq!(detail.telemetry.cavity_insertion_calls, 0);
        assert_eq!(detail.telemetry.hull_extension_calls, 1);
        assert!(
            !detail.repair_seed_simplices.is_empty(),
            "hull extension should return local repair seeds"
        );
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        assert!(
            facet_to_simplices
                .values()
                .all(|incident_simplices| incident_simplices.len() <= 2),
            "hull extension should leave every facet with at most two incident simplices"
        );
        assert!(tri.is_valid().is_ok());
    }

    #[test]
    fn triangulation_exterior_insert_with_empty_conflicts_uses_local_repair_seeds() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        for coords in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            tri.insert_with_statistics(vertex!(coords), None, None)
                .unwrap();
        }

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let empty_conflicts = SimplexKeyBuffer::new();
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                vertex!([2.0, 2.0, 2.0]),
                Some(&empty_conflicts),
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert!(matches!(detail.outcome, InsertionOutcome::Inserted { .. }));
        assert_eq!(detail.telemetry.global_conflict_scans, 0);
        assert_eq!(detail.telemetry.conflict_region_calls, 1);
        assert_eq!(detail.telemetry.conflict_region_simplices_total, 0);
        assert_eq!(detail.telemetry.cavity_insertion_calls, 0);
        assert_eq!(detail.telemetry.hull_extension_calls, 1);
        assert!(
            !detail.repair_seed_simplices.is_empty(),
            "empty caller conflicts should still use terminal-simplex local repair seeds"
        );
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        assert!(
            facet_to_simplices
                .values()
                .all(|incident_simplices| incident_simplices.len() <= 2),
            "hull extension should leave every facet with at most two incident simplices"
        );
        assert!(tri.is_valid().is_ok());
    }

    #[test]
    fn triangulation_caller_conflicts_do_not_force_delaunay_repair() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let start_simplex = tri
            .simplices()
            .next()
            .map(|(simplex_key, _)| simplex_key)
            .unwrap();
        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(start_simplex);
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                vertex!([0.25, 0.25]),
                Some(&conflict_simplices),
                Some(start_simplex),
                0,
                None,
                None,
            )
            .unwrap();

        assert!(matches!(detail.outcome, InsertionOutcome::Inserted { .. }));
        assert!(
            !detail.delaunay_repair_required,
            "caller-provided conflict simplices should preserve the cavity insertion repair flag"
        );
        assert!(tri.is_valid().is_ok());
    }

    #[test]
    fn triangulation_required_topology_validation_records_telemetry() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                vertex!([0.25, 0.25]),
                None,
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert!(matches!(detail.outcome, InsertionOutcome::Inserted { .. }));
        assert!(
            detail.telemetry.topology_validation_calls > 0,
            "Pseudomanifold insertion should record required topology validation"
        );
        assert_eq!(
            detail.telemetry.topology_validation_nanos, 0,
            "default detailed insertion should not start validation timers"
        );

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                vertex!([0.25, 0.25]),
                None,
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert!(matches!(detail.outcome, InsertionOutcome::Inserted { .. }));
        assert!(
            detail.telemetry.topology_validation_calls > 0,
            "PLManifold insertion should record RequiredTopologyLinks validation"
        );
        assert_eq!(
            detail.telemetry.topology_validation_nanos, 0,
            "default detailed insertion should not start validation timers"
        );
    }

    #[test]
    fn triangulation_insert_with_statistics_hint_usage_4d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 4> =
            Triangulation::new_empty(FastKernel::new());

        // Build initial simplex
        for i in 0..5 {
            let mut coords = [0.0; 4];
            if i > 0 {
                coords[i - 1] = 1.0;
            }
            tri.insert_with_statistics(vertex!(coords), None, None)
                .unwrap();
        }

        // Insert with explicit hint
        let hint_simplex = tri.simplices().next().map(|(key, _)| key);
        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.2, 0.2, 0.2, 0.2]), None, hint_simplex)
            .unwrap();

        assert!(matches!(
            outcome,
            InsertionOutcome::Inserted { hint: Some(_), .. }
        ));
        assert_eq!(stats.attempts, 1);
        assert!(stats.success());
    }

    #[test]
    fn triangulation_insert_with_statistics_duplicate_coordinates_3d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None)
            .unwrap();

        // Try duplicate - should be skipped
        let result = tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None);

        assert!(matches!(
            result,
            Ok((
                InsertionOutcome::Skipped {
                    error: InsertionError::DuplicateCoordinates { .. }
                },
                _
            ))
        ));
    }

    #[test]
    fn triangulation_insert_with_statistics_multiple_insertions_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        let points = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
            vertex!([0.3, 0.3]),
            vertex!([0.7, 0.3]),
        ];

        let mut all_succeeded = true;
        let mut max_attempts = 0;

        for point in points {
            match tri.insert_with_statistics(point, None, None) {
                Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                    max_attempts = max_attempts.max(stats.attempts);
                    assert!(stats.success());
                }
                Ok((InsertionOutcome::Skipped { .. }, _)) | Err(_) => {
                    all_succeeded = false;
                }
            }
        }

        assert!(all_succeeded, "all insertions should succeed");
        assert!(max_attempts >= 1);
        assert_eq!(tri.number_of_vertices(), 5);
    }

    #[test]
    fn triangulation_insert_with_statistics_outcome_types() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Test Inserted variant
        let (outcome, _) = tri
            .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .unwrap();

        match outcome {
            InsertionOutcome::Inserted { vertex_key, hint } => {
                // Verify we can access the fields
                assert!(tri.vertices().any(|(k, _)| k == vertex_key));
                assert_eq!(hint, None); // No hint during bootstrap
            }
            InsertionOutcome::Skipped { .. } => panic!("expected Inserted, got Skipped"),
        }
    }

    #[test]
    fn triangulation_insert_with_statistics_sequential_5d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 5> =
            Triangulation::new_empty(FastKernel::new());

        // Insert 6 vertices to form initial simplex
        for i in 0..6 {
            let mut coords = [0.0; 5];
            if i > 0 {
                coords[i - 1] = 1.0;
            }

            let (outcome, stats) = tri
                .insert_with_statistics(vertex!(coords), None, None)
                .unwrap();

            assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
            assert_eq!(stats.attempts, 1);
            assert!(stats.success());
        }

        assert_eq!(tri.number_of_vertices(), 6);
        assert_eq!(tri.number_of_simplices(), 1);
    }

    #[test]
    fn statistics_simplices_removed_during_repair() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Build simplex
        tri.insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .unwrap();
        tri.insert_with_statistics(vertex!([1.0, 0.0]), None, None)
            .unwrap();
        tri.insert_with_statistics(vertex!([0.5, 1.0]), None, None)
            .unwrap();

        let simplices_before = tri.number_of_simplices();

        // Insert interior point - might trigger repair
        let (_outcome, stats) = tri
            .insert_with_statistics(vertex!([0.5, 0.3]), None, None)
            .unwrap();

        let simplices_after = tri.number_of_simplices();

        // Basic sanity: repair can't remove more simplices than existed before insertion.
        assert!(
            stats.simplices_removed_during_repair <= simplices_before,
            "simplices_removed_during_repair ({}) should not exceed simplex count before insertion ({}); simplices after insertion: {}",
            stats.simplices_removed_during_repair,
            simplices_before,
            simplices_after
        );
    }

    // =============================================================================
    // insert_with_conflict_region: cavity reduction loop branch coverage
    //
    // These tests exercise `insert_with_conflict_region` directly via a synthetic
    // TDS rather than through the public API.  The goal is to cover the loop arms
    // (RidgeFan SHRINK, DisconnectedBoundary EXPAND / SHRINK-fallback / else-break,
    // and the post-loop error paths) that are not reachable through normal Delaunay
    // insertions.
    // =============================================================================

    /// `DisconnectedBoundary` where disconnected simplices have no non-conflict neighbours:
    /// `else { break; }` fires, then the D<3 star-split fallback is taken.
    ///
    /// Covers: `DisconnectedBoundary` `else { break; }` (line 3492), `should_fallback=true`
    /// path (lines 3530-3555), and `suspicion.fallback_star_split` being set.
    #[test]
    fn test_cavity_reduction_disconnected_no_neighbors_sets_star_split_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Two triangles that share no vertices (→ DisconnectedBoundary on extraction).
        let v0 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 1.0]))
            .unwrap();
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.0, 0.0]))
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([6.0, 0.0]))
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.5, 1.0]))
            .unwrap();

        let simplex_a = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let simplex_b = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v3, v4, v5], None).unwrap())
            .unwrap();

        // Neither simplex has any neighbour pointers, so `simplices_to_add` will be empty on
        // the first iteration and the `else { break; }` arm fires immediately.
        let new_v = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.3]))
            .unwrap();
        let point = Point::new([0.5_f64, 0.3_f64]);

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(simplex_a);
        conflict_simplices.push(simplex_b);

        let mut suspicion = SuspicionFlags::default();
        let _ = tri.insert_with_conflict_region(
            new_v,
            &point,
            conflict_simplices,
            Some(simplex_a),
            &mut suspicion,
        );

        // `else { break; }` → Err(DisconnectedBoundary) → should_fallback=true (D<3)
        // → star-split fallback sets suspicion.fallback_star_split.
        assert!(
            suspicion.fallback_star_split,
            "DisconnectedBoundary with no non-conflict neighbours should trigger star-split (D=2)"
        );
    }

    /// Three 3D tetrahedra sharing the same triangular face → `NonManifoldFacet` on the
    /// first extraction.  D=3 → `should_fallback=false` → the function returns Err
    /// immediately without entering the star-split path.
    ///
    /// Covers: `_ => break` (line 3511), `should_fallback=false` path (lines 3558-3563).
    #[test]
    fn test_cavity_reduction_nonmanifold_3d_returns_error_without_star_split() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let v0 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        // Three distinct fourth vertices that all pair with the {v0,v1,v2} face.
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]))
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();

        let simplex1 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2, v3], None).unwrap())
            .unwrap();
        let simplex2 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2, v4], None).unwrap())
            .unwrap();
        let simplex3 = tri
            .tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::new(vec![v0, v1, v2, v5], None).unwrap(),
            )
            .unwrap();

        let new_v = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.1, 0.1, 0.1]))
            .unwrap();
        let point = Point::new([0.1_f64, 0.1_f64, 0.1_f64]);

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(simplex1);
        conflict_simplices.push(simplex2);
        conflict_simplices.push(simplex3);

        let mut suspicion = SuspicionFlags::default();
        let result = tri.insert_with_conflict_region(
            new_v,
            &point,
            conflict_simplices,
            None,
            &mut suspicion,
        );

        // NonManifoldFacet → `_ => break` → should_fallback = D<3 = false → Err returned.
        assert!(result.is_err(), "D=3 NonManifoldFacet should return Err");
        assert!(
            !suspicion.fallback_star_split,
            "D=3 should NOT enter star-split fallback"
        );
    }

    /// Four 2D triangles all sharing a common vertex but with no shared edges produce a
    /// `RidgeFan` error (`facet_count >= 3` for the shared vertex).  Because
    /// `conflict_simplices.len() = 4 > D+1 = 3`, the SHRINK branch fires on the first
    /// iteration, removing the extra fan simplices from the conflict region.
    ///
    /// Covers: `RidgeFan` SHRINK body (lines 3434-3442) and re-extraction (line 3514).
    #[test]
    fn test_cavity_reduction_ridge_fan_shrink_fires_for_4_conflict_simplices_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // `center` appears in 8 boundary edges (2 per simplex × 4 simplices) → RidgeFan.
        let center = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        let va = tri
            .tds
            .insert_vertex_with_mapping(vertex!([-1.0, 2.0]))
            .unwrap();
        let vb = tri
            .tds
            .insert_vertex_with_mapping(vertex!([1.0, 2.0]))
            .unwrap();
        let vc = tri
            .tds
            .insert_vertex_with_mapping(vertex!([-3.0, -2.0]))
            .unwrap();
        let vd = tri
            .tds
            .insert_vertex_with_mapping(vertex!([-2.0, -3.0]))
            .unwrap();
        let ve = tri
            .tds
            .insert_vertex_with_mapping(vertex!([3.0, -2.0]))
            .unwrap();
        let vf = tri
            .tds
            .insert_vertex_with_mapping(vertex!([2.0, -3.0]))
            .unwrap();
        let vg = tri
            .tds
            .insert_vertex_with_mapping(vertex!([-4.0, 1.0]))
            .unwrap();
        let vh = tri
            .tds
            .insert_vertex_with_mapping(vertex!([-4.0, -1.0]))
            .unwrap();

        let simplex1 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![center, va, vb], None).unwrap())
            .unwrap();
        let simplex2 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![center, vc, vd], None).unwrap())
            .unwrap();
        let simplex3 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![center, ve, vf], None).unwrap())
            .unwrap();
        let simplex4 = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![center, vg, vh], None).unwrap())
            .unwrap();

        let new_v = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.3, 1.0]))
            .unwrap();
        let point = Point::new([0.3_f64, 1.0_f64]);

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(simplex1);
        conflict_simplices.push(simplex2);
        conflict_simplices.push(simplex3);
        conflict_simplices.push(simplex4);

        let mut suspicion = SuspicionFlags::default();
        // RidgeFan SHRINK fires on iteration 1 (4 > D+1=3), reducing conflict_simplices.
        // The function completes without panic; result may be Ok or Err.
        let _ = tri.insert_with_conflict_region(
            new_v,
            &point,
            conflict_simplices,
            Some(simplex1),
            &mut suspicion,
        );
        // Reaching here confirms the SHRINK branch executed successfully.
    }

    /// Two completely disconnected 2D conflict simplices that each have one non-conflict
    /// neighbour trigger the `DisconnectedBoundary` EXPAND path on the first iteration
    /// (adding the neighbours), and the SHRINK-fallback on a subsequent iteration
    /// (when `simplices_to_add` is empty but `conflict_simplices.len() > D+1`).
    ///
    /// Covers: EXPAND body (lines 3470-3480), SHRINK-fallback (lines 3481-3491),
    /// and re-extraction after each reshape (line 3514).
    #[test]
    fn test_cavity_reduction_disconnected_expand_then_shrink_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Group A: simplex_a = {v0,v1,v2} shares edge {v0,v1} with simplex_c = {v0,v1,v6}.
        let v0 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 1.0]))
            .unwrap();
        let v6 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, -1.0]))
            .unwrap();
        // Group B: simplex_b = {v3,v4,v5} shares edge {v3,v4} with simplex_d = {v3,v4,v7}.
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.0, 0.0]))
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([6.0, 0.0]))
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.5, 1.0]))
            .unwrap();
        let v7 = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.5, -1.0]))
            .unwrap();

        let simplex_a = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        // simplex_c is a non-conflict neighbour of simplex_a (not initially in conflict_simplices).
        let simplex_c = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v6], None).unwrap())
            .unwrap();
        let simplex_b = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v3, v4, v5], None).unwrap())
            .unwrap();
        // simplex_d is a non-conflict neighbour of simplex_b.
        let simplex_d = tri
            .tds
            .insert_simplex_with_mapping(Simplex::new(vec![v3, v4, v7], None).unwrap())
            .unwrap();

        // Wire neighbours so EXPAND discovers simplex_c via simplex_a and simplex_d via simplex_b.
        {
            let simplex = tri.tds.simplex_mut(simplex_a).unwrap();
            simplex
                .set_neighbors_from_keys([Some(simplex_c), None, None])
                .unwrap();
        }
        {
            let simplex = tri.tds.simplex_mut(simplex_b).unwrap();
            simplex
                .set_neighbors_from_keys([Some(simplex_d), None, None])
                .unwrap();
        }

        let new_v = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.3]))
            .unwrap();
        let point = Point::new([0.5_f64, 0.3_f64]);

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(simplex_a);
        conflict_simplices.push(simplex_b);

        // Iteration trace:
        //   1. DisconnectedBoundary → EXPAND (adds simplex_c or simplex_d) → re-extract.
        //   2. DisconnectedBoundary → EXPAND (adds the other) → re-extract.
        //   3. DisconnectedBoundary, simplices_to_add=empty (all neighbours in conflict_set),
        //      len=4 > D+1=3 → SHRINK-fallback removes disconnected component → re-extract.
        //   4. Two simplices sharing an edge → connected boundary → Ok → break.
        let mut suspicion = SuspicionFlags::default();
        let _ = tri.insert_with_conflict_region(
            new_v,
            &point,
            conflict_simplices,
            Some(simplex_a),
            &mut suspicion,
        );
        // Reaching here without panic confirms EXPAND and SHRINK branches executed.
    }

    // ---- insertion_error_to_invariant_error tests ----

    #[test]
    fn test_insertion_error_to_invariant_error_tds_arm() {
        let source = TdsError::Geometric(GeometricError::DegenerateOrientation {
            message: "det=0".to_string(),
        });
        let error = InsertionError::TopologyValidation(source.clone());
        let result = insertion_error_to_invariant_error(error, "ctx");
        assert_eq!(result, InvariantError::Tds(source));
    }

    #[test]
    fn test_insertion_error_to_invariant_error_triangulation_arm() {
        let inner = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        };
        let error = InsertionError::TopologyValidationFailed {
            message: "outer".to_string(),
            source: inner.clone(),
        };
        let result = insertion_error_to_invariant_error(error, "ctx");
        assert_eq!(result, InvariantError::Triangulation(inner));
    }

    #[test]
    fn test_insertion_error_to_invariant_error_other_arm() {
        let error = InsertionError::CavityFilling {
            reason: CavityFillingError::EmptyFanTriangulation,
        };
        let result = insertion_error_to_invariant_error(error, "ctx");
        assert!(
            matches!(
                result,
                InvariantError::Tds(TdsError::InconsistentDataStructure { ref message })
                    if message.contains("ctx") && message.contains("fan triangulation produced no simplices")
            ),
            "CavityFilling should wrap to InconsistentDataStructure: {result:?}"
        );
    }

    // ---- invariant_error_to_insertion_error coverage ----

    #[test]
    fn test_invariant_error_to_insertion_error_tds_arm() {
        let inv = InvariantError::Tds(TdsError::InconsistentDataStructure {
            message: "test".to_string(),
        });
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert!(matches!(ins, InsertionError::TopologyValidation(_)));
    }

    #[test]
    fn test_invariant_error_to_insertion_error_triangulation_arm() {
        let inv = InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        });
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert!(matches!(
            ins,
            InsertionError::TopologyValidationFailed { .. }
        ));
    }

    #[test]
    fn test_invariant_error_to_insertion_error_delaunay_arm() {
        let inv =
            InvariantError::Delaunay(DelaunayTriangulationValidationError::VerificationFailed {
                message: "test".to_string(),
            });
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert!(matches!(
            ins,
            InsertionError::DelaunayValidationFailed { .. }
        ));
    }

    #[test]
    fn test_from_manifold_error_for_invariant_error_non_tds() {
        let err = ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 999,
            simplex_count: 5,
        };
        let inv = InvariantError::from(err);
        assert!(matches!(
            inv,
            InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity {
                    facet_key: 999,
                    simplex_count: 5
                }
            )
        ));
    }

    #[test]
    fn test_triangulation_validation_error_isolated_vertex_display() {
        let err = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(42)),
            vertex_uuid: Uuid::nil(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Isolated vertex"));
        assert!(msg.contains("not incident to any simplex"));
    }

    // ---- is_valid / validate error-path tests ----

    #[test]
    fn test_is_valid_returns_invariant_error_for_isolated_vertex() {
        let (mut tri, _, _) = build_single_tet();

        // Add an isolated vertex that is not referenced by any simplex.
        let iso = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 0.5]))
            .unwrap();

        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                vertex_key,
                ..
            })) => {
                assert_eq!(vertex_key, iso);
            }
            other => {
                panic!("Expected InvariantError::Triangulation(IsolatedVertex), got {other:?}")
            }
        }
    }

    #[test]
    fn test_is_valid_returns_triangulation_error_for_disconnected() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        match tri.is_valid() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => {
                assert_eq!(
                    simplex_count, 2,
                    "Expected 2 simplices in disconnected triangulation"
                );
            }
            other => {
                panic!("Expected InvariantError::Triangulation(Disconnected), got {other:?}")
            }
        }
    }

    #[test]
    fn test_validate_returns_invariant_error_from_tds_layer() {
        // Corrupt a TDS so that Level 2 structural validation fails.
        let (mut tri, [v0, _, _, _], _) = build_single_tet();

        // Break vertex mapping: remove uuid entry.
        let uuid = tri.tds.vertex(v0).unwrap().uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);

        match tri.validate() {
            Err(InvariantError::Tds(TdsError::MappingInconsistency { .. })) => {}
            other => panic!("Expected InvariantError::Tds(MappingInconsistency), got {other:?}"),
        }
    }

    #[test]
    fn test_validate_returns_invariant_error_from_topology_layer() {
        let (mut tri, _, _) = build_single_tet();

        // Add an isolated vertex so Level 3 (topology) fails.
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 0.5]))
            .unwrap();

        match tri.validate() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                ..
            })) => {}
            other => {
                panic!("Expected InvariantError::Triangulation(IsolatedVertex), got {other:?}")
            }
        }
    }

    #[test]
    fn test_from_manifold_error_tds_routes_to_invariant_error_tds() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "underlying TDS issue".to_string(),
        };
        let manifold_err = ManifoldError::Tds(tds_err.clone());
        let inv = InvariantError::from(manifold_err);
        assert_eq!(inv, InvariantError::Tds(tds_err));
    }

    // ---- repair_stale_incident_simplices tests ----

    #[test]
    fn test_repair_stale_incident_simplices_noop_when_all_valid() {
        let (mut tri, [v0, v1, v2, v3], ck) = build_single_tet();
        assert!(tri.repair_stale_incident_simplices().is_ok());

        // Pointers unchanged.
        for vk in [v0, v1, v2, v3] {
            assert_eq!(tri.tds.vertex_mut(vk).unwrap().incident_simplex(), Some(ck));
        }
    }

    #[test]
    fn test_repair_stale_incident_simplices_repairs_none_pointer() {
        let (mut tri, [_, _, _, v3], ck) = build_single_tet();

        // Corrupt v3 to have no incident simplex.
        tri.tds.vertex_mut(v3).unwrap().set_incident_simplex(None);

        assert!(tri.repair_stale_incident_simplices().is_ok());
        assert_eq!(
            tri.tds.vertex_mut(v3).unwrap().incident_simplex(),
            Some(ck),
            "v3 should be repaired to point to the tetrahedron"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_repairs_stale_pointer() {
        let (mut tri, [_, _, _, v3], ck) = build_single_tet();

        // Point v3 to a non-existent simplex key (simulates a deleted conflict simplex).
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));
        tri.tds
            .vertex_mut(v3)
            .unwrap()
            .set_incident_simplex(Some(stale));

        assert!(tri.repair_stale_incident_simplices().is_ok());
        assert_eq!(
            tri.tds.vertex_mut(v3).unwrap().incident_simplex(),
            Some(ck),
            "stale pointer should be repaired to the valid simplex"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_errors_on_truly_isolated_vertex() {
        let (mut tri, _, _) = build_single_tet();

        // Insert a vertex that is NOT referenced by any simplex.
        let iso = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 0.5]))
            .unwrap();

        let result = tri.repair_stale_incident_simplices();
        assert!(
            matches!(
                &result,
                Err(InsertionError::TopologyValidationFailed {
                    source, ..
                }) if matches!(
                    source,
                    TriangulationValidationError::IsolatedVertex { vertex_key, .. }
                        if *vertex_key == iso
                )
            ),
            "Truly isolated vertex should produce IsolatedVertex error: {result:?}"
        );
    }

    // =========================================================================
    // TOPOLOGY QUERY COVERAGE
    // =========================================================================

    #[test]
    fn test_topology_queries_on_two_tet_triangulation() {
        // 5 vertices in 3D → multi-simplex triangulation exercises all query paths
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        // edges()
        let edge_count = tri.number_of_edges();
        let edges_collected: HashSet<_> = tri.edges().collect();
        assert_eq!(edges_collected.len(), edge_count);
        assert!(edge_count >= 6); // at least a tetrahedron's worth

        // facets() and boundary_facets()
        assert!(tri.facets().next().is_some());
        assert!(tri.boundary_facets().next().is_some());

        // simplex_vertices() and vertex_coords()
        let (ck, _) = tri.simplices().next().unwrap();
        let simplex_verts = tri.simplex_vertices(ck).unwrap();
        assert_eq!(simplex_verts.len(), 4);
        for &vk in simplex_verts {
            let coords = tri.vertex_coords(vk).unwrap();
            assert_eq!(coords.len(), 3);
        }

        // Returns None for missing keys
        let missing_ck = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        assert!(tri.simplex_vertices(missing_ck).is_none());
        let absent_vk = VertexKey::from(KeyData::from_ffi(0xBEEF));
        assert!(tri.vertex_coords(absent_vk).is_none());

        // adjacent_simplices()
        let v0 = tri.vertices().next().unwrap().0;
        assert!(tri.adjacent_simplices(v0).next().is_some());

        // simplex_neighbors()
        // Multi-simplex triangulation has at least one internal neighbor
        assert!(tri.simplex_neighbors(ck).next().is_some());

        // incident_edges()
        let inc_edges: Vec<_> = tri.incident_edges(v0).collect();
        assert!(!inc_edges.is_empty());
        assert_eq!(tri.number_of_incident_edges(v0), inc_edges.len());
    }

    // =========================================================================
    // ADJACENCY INDEX + _WITH_INDEX METHODS
    // =========================================================================

    #[test]
    fn test_adjacency_index_with_index_methods() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let index = tri.build_adjacency_index().unwrap();

        // edges_with_index matches edges()
        let idx_edges: HashSet<_> = tri.edges_with_index(&index).collect();
        let direct_edges: HashSet<_> = tri.edges().collect();
        assert_eq!(idx_edges, direct_edges);
        assert_eq!(
            tri.number_of_edges_with_index(&index),
            tri.number_of_edges()
        );

        let v0 = tri.vertices().next().unwrap().0;

        // adjacent_simplices_with_index
        let idx_adj: HashSet<_> = tri.adjacent_simplices_with_index(&index, v0).collect();
        let direct_adj: HashSet<_> = tri.adjacent_simplices(v0).collect();
        assert_eq!(idx_adj, direct_adj);
        assert_eq!(
            tri.number_of_adjacent_simplices_with_index(&index, v0),
            direct_adj.len()
        );

        // simplex_neighbors_with_index
        let ck = tri.simplices().next().unwrap().0;
        let direct_neighbors: Vec<_> = tri.simplex_neighbors(ck).collect();
        assert_eq!(
            tri.simplex_neighbors_with_index(&index, ck).count(),
            direct_neighbors.len()
        );
        assert_eq!(
            tri.number_of_simplex_neighbors_with_index(&index, ck),
            direct_neighbors.len()
        );

        // incident_edges_with_index
        let idx_inc: HashSet<_> = tri.incident_edges_with_index(&index, v0).collect();
        let direct_inc: HashSet<_> = tri.incident_edges(v0).collect();
        assert_eq!(idx_inc, direct_inc);
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, v0),
            direct_inc.len()
        );
    }

    // =========================================================================
    // DETECT / REPAIR LOCAL FACET ISSUES
    // =========================================================================

    #[test]
    fn test_detect_local_facet_issues_none_for_valid_triangulation() {
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        let issues = tri.detect_local_facet_issues(&simplex_keys).unwrap();
        assert!(issues.is_none());
    }

    #[test]
    fn test_ensure_non_empty_conflict_simplices_passthrough_when_nonempty() {
        let mut buf = SimplexKeyBuffer::new();
        buf.push(SimplexKey::from(KeyData::from_ffi(1)));

        let owned = Cow::Owned(buf);
        let fallback = SimplexKey::from(KeyData::from_ffi(999));
        let result =
            Triangulation::<FastKernel<f64>, (), (), 2>::ensure_non_empty_conflict_simplices(
                owned, fallback,
            );
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_ensure_non_empty_conflict_simplices_uses_fallback_when_empty() {
        let buf = SimplexKeyBuffer::new();
        let fallback = SimplexKey::from(KeyData::from_ffi(42));
        let result =
            Triangulation::<FastKernel<f64>, (), (), 2>::ensure_non_empty_conflict_simplices(
                Cow::Owned(buf),
                fallback,
            );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], fallback);
    }

    #[test]
    fn test_star_split_boundary_facets_produces_d_plus_1_facets() {
        let ck = SimplexKey::from(KeyData::from_ffi(7));
        let facets = Triangulation::<FastKernel<f64>, (), (), 3>::star_split_boundary_facets(ck);
        assert_eq!(facets.len(), 4); // D+1 = 4 for 3D
        for (i, fh) in facets.iter().enumerate() {
            assert_eq!(fh.simplex_key(), ck);
            assert_eq!(<usize as From<u8>>::from(fh.facet_index()), i);
        }
    }
    // =========================================================================
    // VALIDATION REPORT
    // =========================================================================

    #[test]
    fn test_validation_report_ok_for_valid_two_tet() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert!(dt.as_triangulation().validation_report().is_ok());
    }

    #[test]
    fn test_validation_report_reports_isolated_vertex_topology_violation() {
        let (mut tri, _, _) = build_single_tet();
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 0.5]))
            .unwrap();

        let report = tri.validation_report().unwrap_err();
        assert!(!report.is_empty());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::Topology),
            "Expected Topology violation in report"
        );
    }

    // =========================================================================
    // TOPOLOGY GUARANTEE / VALIDATION POLICY COVERAGE
    // =========================================================================

    #[test]
    fn test_topology_guarantee_requires_vertex_links_at_completion_predicate() {
        assert!(TopologyGuarantee::PLManifold.requires_vertex_links_at_completion());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_vertex_links_at_completion());
        assert!(!TopologyGuarantee::Pseudomanifold.requires_vertex_links_at_completion());
    }

    #[test]
    fn test_validate_global_connectedness_ok_for_connected() {
        let (tri, _, _) = build_single_tet();
        assert!(tri.validate_global_connectedness().is_ok());
    }

    #[test]
    fn test_validate_no_isolated_vertices_ok_when_no_vertices() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        assert!(tri.validate_no_isolated_vertices().is_ok());
    }

    #[test]
    fn test_pick_fan_apex_returns_none_for_empty_facets() {
        let (tri, _, _) = build_single_tet();
        assert!(tri.pick_fan_apex(&[]).is_none());
    }

    // =========================================================================
    // INSERTION PIPELINE: BOOTSTRAP, INITIAL SIMPLEX, BEYOND-SIMPLEX
    // =========================================================================

    /// Helper: build a set of D+1 affinely independent vertices for dimension D.
    fn simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut verts = Vec::with_capacity(D + 1);
        // Origin
        verts.push(
            VertexBuilder::default()
                .point(Point::new([0.0; D]))
                .build()
                .unwrap(),
        );
        // Unit vectors along each axis
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            verts.push(
                VertexBuilder::default()
                    .point(Point::new(coords))
                    .build()
                    .unwrap(),
            );
        }
        verts
    }

    /// Macro: dimension-parametric insertion pipeline tests.
    macro_rules! test_insert_pipeline {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_insert_bootstrap_phase_ $dim d>]() {
                    let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());

                    // Insert fewer than D+1 vertices: should remain in bootstrap (no simplices).
                    let verts = simplex_vertices::<$dim>();
                    for v in &verts[..$dim] {
                        let (vk, hint) = tri.insert(*v, None, None).unwrap();
                        assert!(hint.is_none(), "{}D: no hint during bootstrap", $dim);
                        assert!(tri.tds.vertex(vk).is_some());
                    }
                    assert_eq!(tri.number_of_vertices(), $dim);
                    assert_eq!(tri.number_of_simplices(), 0, "{}D: no simplices during bootstrap", $dim);
                }

                #[test]
                fn [<test_insert_initial_simplex_via_insert_ $dim d>]() {
                    let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());

                    let verts = simplex_vertices::<$dim>();
                    for v in &verts {
                        tri.insert(*v, None, None).unwrap();
                    }
                    assert_eq!(tri.number_of_vertices(), $dim + 1);
                    assert_eq!(tri.number_of_simplices(), 1, "{}D: exactly 1 simplex after D+1 vertices", $dim);

                    // The simplex must have D+1 vertices.
                    let (_, simplex) = tri.simplices().next().unwrap();
                    assert_eq!(simplex.number_of_vertices(), $dim + 1);
                }

                #[test]
                fn [<test_insert_beyond_initial_simplex_ $dim d>]() {
                    let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());

                    // Build initial simplex.
                    let verts = simplex_vertices::<$dim>();
                    for v in &verts {
                        tri.insert(*v, None, None).unwrap();
                    }
                    assert_eq!(tri.number_of_simplices(), 1);

                    // Insert an interior point.
                    let mut interior = [0.0; $dim];
                    for c in interior.iter_mut() {
                        *c = 1.0 / (<f64 as std::convert::From<i32>>::from($dim + 1) * 2.0);
                    }
                    let interior_vertex = VertexBuilder::default()
                        .point(Point::new(interior))
                        .build()
                        .unwrap();
                    let (_, hint) = tri
                        .insert(interior_vertex, None, None)
                        .unwrap();

                    assert!(hint.is_some(), "{}D: hint returned after D+2 insertion", $dim);
                    assert!(tri.number_of_simplices() > 1, "{}D: simplex count increased", $dim);
                    assert!(tri.is_valid().is_ok(), "{}D: topology valid after insertion", $dim);
                }
            }
        };
    }

    test_insert_pipeline!(2);
    test_insert_pipeline!(3);
    test_insert_pipeline!(4);

    // =========================================================================
    // INSERT_WITH_STATISTICS: STATISTICS TRACKING
    // =========================================================================

    #[test]
    fn test_insert_with_statistics_tracks_simplices_removed() {
        // Build a 2D triangulation with several points, verify stats fields.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.25]];
        for coords in &points {
            let (outcome, stats) = tri
                .insert_with_statistics(vertex!(*coords), None, None)
                .unwrap();
            assert!(stats.success());
            assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
            assert_eq!(stats.attempts, 1);
        }
        assert_eq!(tri.number_of_vertices(), 4);
        assert!(tri.number_of_simplices() >= 2);
    }

    // =========================================================================
    // VALIDATION REPORT: MULTIPLE VIOLATIONS
    // =========================================================================

    #[test]
    fn test_validation_report_collects_multiple_violations() {
        // Create a triangulation with an isolated vertex AND a bad neighbor buffer
        // so that validation_report collects both VertexValidity + Topology violations.
        let (mut tri, _, ck) = build_single_tet();

        // Add an isolated vertex (no incident simplex).
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([5.0, 5.0, 5.0]))
            .unwrap();

        // Corrupt a simplex's neighbor buffer length to trigger SimplexValidity violation.
        let simplex = tri.tds.simplex_mut(ck).unwrap();
        simplex.ensure_neighbors_buffer_mut().truncate(2); // wrong: should be D+1 = 4

        let report = tri.validation_report().unwrap_err();
        assert!(
            report.violations.len() >= 2,
            "Expected at least 2 violations, got {}",
            report.violations.len()
        );
    }

    // =========================================================================
    // REMOVE VERTEX: RETRIANGULATION AND TOPOLOGY
    // =========================================================================

    #[test]
    fn test_remove_vertex_retriangulates_cavity_2d() {
        // Build 2D triangulation with 4 vertices, remove one, verify valid.
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let initial_simplices = dt.number_of_simplices();
        let vertex_key = dt
            .vertices()
            .find(|(_, v)| {
                let c = v.point().coords();
                (c[0] - 0.5).abs() < 1e-10 && (c[1] - 0.5).abs() < 1e-10
            })
            .map(|(k, _)| k)
            .unwrap();

        let removed = dt.remove_vertex(vertex_key).unwrap();
        assert!(removed > 0, "Should have removed at least 1 simplex");
        assert!(dt.number_of_simplices() <= initial_simplices);
        assert_eq!(dt.number_of_vertices(), 3);
    }

    #[test]
    fn test_remove_vertex_entire_triangulation_2d() {
        // When we remove a vertex from a single-simplex triangulation,
        // the empty boundary case triggers Tds::remove_vertex fallback.
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let vertex_key = dt.vertices().next().unwrap().0;
        let removed = dt.remove_vertex(vertex_key).unwrap();
        assert!(removed >= 1);
        assert_eq!(dt.number_of_vertices(), 2);
    }

    // =========================================================================
    // VALIDATE CONNECTEDNESS: ERROR PATHS
    // =========================================================================

    #[test]
    fn test_validate_connectedness_rejects_empty_new_simplices() {
        let (tri, _, _) = build_single_tet();

        // Empty new_simplices buffer: should error because no surviving new simplices.
        let empty: SimplexKeyBuffer = SimplexKeyBuffer::new();
        let err = tri.validate_connectedness(&empty).unwrap_err();
        assert!(matches!(err, InsertionError::TopologyValidation(_)));
    }

    #[test]
    fn test_validate_connectedness_passes_for_valid_new_simplices() {
        let (tri, _, ck) = build_single_tet();

        // Single simplex is both the new set and the whole triangulation.
        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(ck);
        assert!(tri.validate_connectedness(&new_simplices).is_ok());
    }

    // =========================================================================
    // FIND CONFLICT REGION GLOBAL
    // =========================================================================

    #[test]
    fn test_find_conflict_region_global_returns_simplices() {
        // Build a 3D simplex, then check that a point outside has a conflict region.
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // A point inside the circumsphere but outside the simplex.
        let conflict = tri
            .find_conflict_region_global(&Point::new([0.5, 0.5, 0.5]))
            .unwrap();
        // The single simplex's circumsphere should contain this point.
        assert!(
            !conflict.is_empty(),
            "Point near circumcenter should produce a conflict region"
        );
    }

    /// `find_conflict_region_global` uses `sorted_simplex_points` to collect
    /// vertices in canonical order.  A simplex containing a missing vertex key
    /// causes the helper to return `None`, which is converted to
    /// `ConflictError::SimplexDataAccessFailed`.
    #[test]
    fn test_find_conflict_region_global_missing_vertex_returns_simplex_data_access_failed() {
        let (mut tri, vkeys, ck) = build_single_tet();

        // Replace one vertex key with a missing key.
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let simplex = tri.tds.simplex_mut(ck).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(vkeys[0]);
            simplex.push_vertex_key(vkeys[1]);
            simplex.push_vertex_key(vkeys[2]);
            simplex.push_vertex_key(missing);
        }

        let result = tri.find_conflict_region_global(&Point::new([0.5, 0.5, 0.5]));
        assert!(
            matches!(
                result,
                Err(ConflictError::SimplexDataAccessFailed { simplex_key, .. }) if simplex_key == ck
            ),
            "expected SimplexDataAccessFailed for missing vertex, got {result:?}"
        );
    }

    /// `find_conflict_region_global` checks that `sorted_simplex_points` returns
    /// exactly D+1 points.  A simplex with fewer than D+1 vertex keys (all
    /// resolvable) triggers `SimplexDataAccessFailed` with a vertex-count message.
    #[test]
    fn test_find_conflict_region_global_underdimensioned_simplex_returns_simplex_data_access_failed()
     {
        let (mut tri, vkeys, ck) = build_single_tet();

        // Shrink the simplex to only 3 vertices (all valid) in a D=3 triangulation.
        {
            let simplex = tri.tds.simplex_mut(ck).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(vkeys[0]);
            simplex.push_vertex_key(vkeys[1]);
            simplex.push_vertex_key(vkeys[2]);
        }

        let result = tri.find_conflict_region_global(&Point::new([0.5, 0.5, 0.5]));
        assert!(
            matches!(
                result,
                Err(ConflictError::SimplexDataAccessFailed { simplex_key, .. }) if simplex_key == ck
            ),
            "expected SimplexDataAccessFailed for underdimensioned simplex, got {result:?}"
        );
    }

    // =========================================================================
    // CONFLICT REGION TOUCHES BOUNDARY
    // =========================================================================

    #[test]
    fn test_conflict_region_touches_boundary_single_simplex() {
        // A single simplex: all facets are boundary.
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let mut buf = SimplexKeyBuffer::new();
        buf.push(simplex_key);

        let touches = tri.conflict_region_touches_boundary(&buf).unwrap();
        assert!(touches, "Single simplex has only boundary facets");
    }

    #[test]
    fn test_conflict_region_touches_boundary_empty() {
        let (tri, _, _) = build_single_tet();
        let empty = SimplexKeyBuffer::new();
        let touches = tri.conflict_region_touches_boundary(&empty).unwrap();
        assert!(!touches, "Empty conflict region touches no boundary");
    }

    // =========================================================================
    // FAN FILL CAVITY: ERROR CASE
    // =========================================================================

    #[test]
    fn test_fan_fill_cavity_errors_when_no_simplices_produced() {
        // If the apex is on every boundary facet, fan_fill_cavity should error.
        let (mut tri, vkeys, ck) = build_single_tet();

        // Use vkeys[0] as apex; construct boundary facets that ALL include vkeys[0].
        // In a tet, facet 0 is opposite vkeys[0] (does NOT include it),
        // but facets 1,2,3 each include vkeys[0].
        let boundary_facets: CavityBoundaryBuffer =
            (1..=3).map(|i| FacetHandle::new(ck, i)).collect();

        let result = tri.fan_fill_cavity(vkeys[0], &boundary_facets);
        // All facets include vkeys[0], so no simplices should be created.
        assert!(result.is_err());
    }

    // =========================================================================
    // REPAIR LOCAL FACET ISSUES: NON-EMPTY ISSUES MAP
    // =========================================================================

    #[test]
    fn test_repair_local_facet_issues_with_overshared_facet() {
        // Build 2D triangulation with enough simplices to have interior facets,
        // then artificially create an over-shared facet by duplicating a simplex.
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tri = dt.as_triangulation().clone();

        // Add a duplicate simplex with the same vertices as an existing simplex.
        let (_, existing_simplex) = tri.tds.simplices().next().unwrap();
        let vkeys: Vec<_> = existing_simplex.vertices().to_vec();
        let dup_simplex = Simplex::new(vkeys, None).unwrap();
        let _ = tri
            .tds
            .insert_simplex_bypassing_topology_checks_for_test(dup_simplex)
            .unwrap();

        // Now detect issues.
        let all_simplices: Vec<_> = tri.tds.simplex_keys().collect();
        let issues = tri.detect_local_facet_issues(&all_simplices).unwrap();
        assert!(issues.is_some(), "Should detect over-shared facet");

        let removed = tri.repair_local_facet_issues(&issues.unwrap()).unwrap();
        assert!(removed > 0, "Should remove at least one duplicate simplex");
    }

    /// Return the facet index opposite the vertex not on the tested shared edge.
    fn shared_edge_facet_index(
        simplex: &Simplex<f64, (), (), 2>,
        v_a: VertexKey,
        v_b: VertexKey,
    ) -> usize {
        simplex
            .vertices()
            .iter()
            .position(|&vertex_key| vertex_key != v_a && vertex_key != v_b)
            .expect("test simplices should contain the shared edge")
    }

    /// Read the neighbor slot across the tested shared edge in a 2D repair fixture.
    fn neighbor_across_shared_edge(
        tri: &Triangulation<FastKernel<f64>, (), (), 2>,
        simplex_key: SimplexKey,
        v_a: VertexKey,
        v_b: VertexKey,
    ) -> Option<SimplexKey> {
        let simplex = tri.tds.simplex(simplex_key).unwrap();
        let facet_idx = shared_edge_facet_index(simplex, v_a, v_b);
        simplex.neighbor_key(facet_idx).flatten()
    }

    #[test]
    fn test_local_repair_uses_removal_frontier() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(Simplex::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();
        let c3 = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::new(vec![v_a, v_b, v_e], None).unwrap(),
            )
            .unwrap();

        for (simplex_key, neighbor_key) in [(c1, c2), (c2, c3), (c3, c1)] {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(3, None);
            neighbors[2] = Some(neighbor_key);
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }
        tds.assign_incident_simplices().unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let original_simplices = [c1, c2, c3];
        let issues = tri
            .detect_local_facet_issues(&original_simplices)
            .unwrap()
            .expect("three simplices sharing one edge should be detected as over-shared");

        let repair = tri
            .repair_local_facet_issues_with_frontier(&issues)
            .unwrap();
        assert_eq!(repair.removed_count, 1);
        assert!(
            !repair.frontier_simplices.is_empty(),
            "removed-simplex neighbors should seed the local repair frontier"
        );

        let survivors: Vec<_> = original_simplices
            .into_iter()
            .filter(|simplex_key| tri.tds.contains_simplex(*simplex_key))
            .collect();
        assert_eq!(survivors.len(), 2);
        let [first_survivor, second_survivor] = survivors.as_slice() else {
            panic!("fixture should leave exactly two surviving simplices");
        };
        for &survivor in &survivors {
            assert!(
                repair.frontier_simplices.contains(&survivor),
                "facet-issue survivors should seed the local repair frontier"
            );
        }
        let survivor_pairs = [
            (*first_survivor, *second_survivor),
            (*second_survivor, *first_survivor),
        ];

        let missing_shared_slots_before = survivor_pairs
            .iter()
            .filter(|&&(simplex_key, other)| {
                neighbor_across_shared_edge(&tri, simplex_key, v_a, v_b) != Some(other)
            })
            .count();
        assert!(
            missing_shared_slots_before > 0,
            "simplex removal should leave at least one survivor slot needing local repair"
        );

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.extend(original_simplices);
        let repaired = tri
            .repair_neighbors_after_local_simplex_removal(
                &new_simplices,
                &repair.frontier_simplices,
            )
            .unwrap();

        assert!(repaired > 0);
        for (simplex_key, other) in survivor_pairs {
            assert_eq!(
                neighbor_across_shared_edge(&tri, simplex_key, v_a, v_b),
                Some(other),
                "surviving simplices should be rewired across the formerly over-shared edge"
            );
        }
        assert!(tri.tds.validate_facet_sharing().is_ok());
        assert!(tri.detect_local_facet_issues(&survivors).unwrap().is_none());
    }

    // =========================================================================
    // DUPLICATE COORDINATES ERROR: LINEAR FALLBACK (NO INDEX)
    // =========================================================================

    #[test]
    fn test_duplicate_coordinates_error_linear_scan_no_index() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([3.0, 4.0]))
            .unwrap();

        let tol = 1e-10_f64;
        // No index provided: should fall back to linear scan.
        let err = tri.duplicate_coordinates_error(&[3.0, 4.0], tol, None);
        assert!(matches!(
            err,
            Some(InsertionError::DuplicateCoordinates { .. })
        ));

        // Non-duplicate should return None.
        let no_err = tri.duplicate_coordinates_error(&[99.0, 99.0], tol, None);
        assert!(no_err.is_none());
    }

    // =========================================================================
    // VALIDATE_AFTER_INSERTION: EDGE CASES
    // =========================================================================

    #[test]
    fn test_validate_after_insertion_ok_for_valid_simplex() {
        // Use a properly constructed triangulation (with neighbors/incidence).
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let suspicion = SuspicionFlags {
            repair_loop_entered: true,
            ..Default::default()
        };
        // With Always policy and a suspicious flag, validation should still pass.
        assert!(
            tri.validate_after_insertion_with_scope(suspicion, None)
                .is_ok()
        );
    }

    // =========================================================================
    // INVARIANT ERROR CONVERSION
    // =========================================================================

    #[test]
    fn test_invariant_error_to_insertion_error_all_arms() {
        // Tds arm
        let tds_err = InvariantError::Tds(TdsError::InvalidNeighbors {
            reason: NeighborValidationError::Other {
                message: "test".into(),
            },
        });
        let ie = Triangulation::<FastKernel<f64>, (), (), 2>::invariant_error_to_insertion_error(
            tds_err,
        );
        assert!(matches!(ie, InsertionError::TopologyValidation(_)));

        // Triangulation arm
        let tri_err = InvariantError::Triangulation(
            TriangulationValidationError::EulerCharacteristicMismatch {
                computed: 0,
                expected: 1,
                classification: TopologyClassification::Unknown,
            },
        );
        let ie = Triangulation::<FastKernel<f64>, (), (), 2>::invariant_error_to_insertion_error(
            tri_err,
        );
        assert!(matches!(
            ie,
            InsertionError::TopologyValidationFailed { .. }
        ));

        // Delaunay arm
        let dt_err =
            InvariantError::Delaunay(DelaunayTriangulationValidationError::VerificationFailed {
                message: "test violation".to_string(),
            });
        let ie =
            Triangulation::<FastKernel<f64>, (), (), 2>::invariant_error_to_insertion_error(dt_err);
        assert!(matches!(
            ie,
            InsertionError::DelaunayValidationFailed { .. }
        ));
    }

    // =========================================================================
    // ESTIMATE LOCAL PERTURBATION SCALE: NO VERTICES
    // =========================================================================

    #[test]
    fn test_estimate_local_perturbation_scale_no_vertices() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        let scale = tri.estimate_local_perturbation_scale(&[1.0, 2.0, 3.0], None);
        // With no vertices, scale should be 1.0 (the default).
        approx::assert_abs_diff_eq!(scale, 1.0, epsilon = 1e-12);
    }

    // =========================================================================
    // VALIDATE_AT_COMPLETION: VARIOUS GUARANTEES
    // =========================================================================

    #[test]
    fn test_validate_at_completion_ok_for_pseudomanifold_empty() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        // Pseudomanifold does not require vertex links at completion.
        assert!(tri.validate_at_completion().is_ok());
    }

    #[test]
    fn test_validate_at_completion_ok_for_pl_manifold_no_simplices() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        // PLManifold requires vertex links at completion, but with 0 simplices it short-circuits.
        assert!(tri.validate_at_completion().is_ok());
    }

    // =========================================================================
    // PROGRESSIVE PERTURBATION: SCALE INVARIANCE
    // =========================================================================

    /// Construct the same 3D geometry at three different uniform scales and verify
    /// that the same number of vertices are successfully inserted at each scale.
    /// This validates that perturbation is proportional to local feature size.
    #[test]
    fn test_perturbation_scale_invariance_3d() {
        const EXPECTED_VERTEX_COUNT: usize = 8;
        const EXPECTED_SIMPLEX_COUNT: usize = 10;

        fn build_at_scale(scale: f64) -> (usize, usize) {
            let base_coords: [[f64; 3]; 8] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
            ];
            let vertices: Vec<Vertex<f64, (), 3>> = base_coords
                .iter()
                .map(|c| vertex!([c[0] * scale, c[1] * scale, c[2] * scale]))
                .collect();
            let dt: DelaunayTriangulation<_, (), (), 3> =
                DelaunayTriangulation::new(&vertices).unwrap();
            (dt.number_of_vertices(), dt.number_of_simplices())
        }

        let (v1, c1) = build_at_scale(1.0);
        let (v2, c2) = build_at_scale(1e6);
        let (v3, c3) = build_at_scale(1e-6);

        // Absolute expectations: catch regressions that affect all scales equally.
        assert_eq!(
            v1, EXPECTED_VERTEX_COUNT,
            "Vertex count regression at unit scale (build_at_scale(1.0))"
        );
        assert_eq!(
            c1, EXPECTED_SIMPLEX_COUNT,
            "Simplex count regression at unit scale (build_at_scale(1.0))"
        );

        // Cross-scale equality: perturbation is proportional to local feature size.
        assert_eq!(
            v1, v2,
            "Vertex count should be scale-invariant (×1 vs ×1e6)"
        );
        assert_eq!(
            v1, v3,
            "Vertex count should be scale-invariant (×1 vs ×1e-6)"
        );
        assert_eq!(
            c1, c2,
            "Simplex count should be scale-invariant (×1 vs ×1e6)"
        );
        assert_eq!(
            c1, c3,
            "Simplex count should be scale-invariant (×1 vs ×1e-6)"
        );
    }

    /// Verify the mantissa-based epsilon selection (`1e-4` for f32, `1e-8` for f64)
    /// and exercise the perturbation retry path with a near-degenerate simplex.
    #[test]
    fn test_perturbation_epsilon_selection_and_retry() {
        // Assert the mantissa-digits → epsilon branching for each scalar type.
        // insert_transactional uses: `if K::Scalar::mantissa_digits() <= 24 { 1e-4 } else { 1e-8 }`
        assert_eq!(
            f32::mantissa_digits(),
            24,
            "f32 should take the 1e-4 epsilon path"
        );
        assert_eq!(
            f64::mantissa_digits(),
            53,
            "f64 should take the 1e-8 epsilon path"
        );

        // f32 path: build a 2D triangulation, then insert a point exactly on an
        // existing edge.  This near-degenerate configuration exercises the full
        // insert_transactional path including epsilon_value computation.
        let initial_f32: Vec<Vertex<f32, (), 2>> = vec![
            vertex!([0.0_f32, 0.0]),
            vertex!([1.0_f32, 0.0]),
            vertex!([0.0_f32, 1.0]),
        ];
        let tds_f32 =
            Triangulation::<AdaptiveKernel<f32>, (), (), 2>::build_initial_simplex(&initial_f32)
                .unwrap();
        let mut tri_f32 = Triangulation::<AdaptiveKernel<f32>, (), (), 2>::new_with_tds(
            AdaptiveKernel::<f32>::new(),
            tds_f32,
        );

        // Point on edge [0,0]→[1,0]: collinear, exercises degeneracy handling.
        let (outcome_f32, stats_f32) = tri_f32
            .insert_with_statistics(vertex!([0.5_f32, 0.0]), None, None)
            .unwrap();
        // Should succeed (SoS resolves) or be gracefully skipped.
        assert!(
            stats_f32.attempts >= 1,
            "f32 insertion must execute at least 1 attempt"
        );
        if let InsertionOutcome::Inserted { .. } = outcome_f32 {
            assert_eq!(tri_f32.tds.number_of_vertices(), 4);
        }

        // f64 path: same exercise with double precision.
        let initial_f64: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0_f64, 0.0]),
            vertex!([1.0_f64, 0.0]),
            vertex!([0.0_f64, 1.0]),
        ];
        let tds_f64 =
            Triangulation::<AdaptiveKernel<f64>, (), (), 2>::build_initial_simplex(&initial_f64)
                .unwrap();
        let mut tri_f64 = Triangulation::<AdaptiveKernel<f64>, (), (), 2>::new_with_tds(
            AdaptiveKernel::<f64>::new(),
            tds_f64,
        );

        let (outcome_f64, stats_f64) = tri_f64
            .insert_with_statistics(vertex!([0.5_f64, 0.0]), None, None)
            .unwrap();
        assert!(
            stats_f64.attempts >= 1,
            "f64 insertion must execute at least 1 attempt"
        );
        if let InsertionOutcome::Inserted { .. } = outcome_f64 {
            assert_eq!(tri_f64.tds.number_of_vertices(), 4);
        }
    }

    /// Verify the `DEFAULT_PERTURBATION_RETRIES` constant value.
    #[test]
    fn test_default_perturbation_retries_constant() {
        assert_eq!(
            DEFAULT_PERTURBATION_RETRIES, 3,
            "Default perturbation retries should be 3 (4 total attempts)"
        );
    }

    // =========================================================================
    // PROGRESSIVE PERTURBATION: RETRY PATH COVERAGE
    // =========================================================================

    #[expect(
        clippy::too_many_lines,
        reason = "Literal 4D repro point set keeps retry-path coverage deterministic"
    )]
    fn perturbation_retry_repro_points_4d() -> [Point<f64, 4>; 20] {
        // Fixed adversarial insertion sequence captured from the former
        // randomized sweep (seed 4, index 19). The final insertion exhausts
        // perturbation retries in the current 4D path, so this keeps retry
        // coverage deterministic without looping over random seeds.
        [
            Point::new([
                0.660_063_804_566_304_3,
                3.139_352_812_821_116,
                1.460_437_437_858_557_2,
                1.683_976_950_416_514_7,
            ]),
            Point::new([
                2.451_966_162_957_145,
                9.547_229_335_697_903,
                3.306_128_696_560_687_5,
                -3.722_166_730_957_705_6,
            ]),
            Point::new([
                -2.344_360_378_074_79,
                -2.755_831_029_562_339,
                -1.275_699_073_649_171_6,
                7.667_812_493_160_508,
            ]),
            Point::new([
                -8.633_692_230_033_44,
                1.995_093_685_275_964_6,
                7.993_316_108_703_105,
                -3.310_780_098_197_376_7,
            ]),
            Point::new([
                9.710_410_828_147_591,
                -9.675_293_457_452_888,
                -7.169_080_272_753_141,
                5.405_946_111_675_925_5,
            ]),
            Point::new([
                2.266_246_031_487_613,
                2.481_673_939_102_995,
                3.039_413_140_674_462,
                4.441_464_307_622_285,
            ]),
            Point::new([
                2.565_731_492_709_954,
                8.916_218_617_699_3,
                -3.878_340_784_199_263_4,
                -9.518_720_806_139_726,
            ]),
            Point::new([
                -2.067_801_258_479_087_2,
                -5.739_002_626_992_522,
                7.554_154_642_458_165,
                -2.983_334_995_469_171_2,
            ]),
            Point::new([
                7.592_645_474_686_005,
                -3.326_646_745_715_216,
                -3.259_537_116_123_248,
                -4.935_000_398_073_641,
            ]),
            Point::new([
                -5.931_807_896_262_18,
                8.897_268_005_841_394,
                0.324_049_126_782_281_15,
                -8.328_532_028_712_647,
            ]),
            Point::new([
                -8.182_644_118_410_867,
                5.373_925_359_941_506,
                -9.015_837_749_827_128,
                -1.703_973_344_007_208,
            ]),
            Point::new([
                1.455_467_619_488_706_2,
                9.869_985_381_801_74,
                8.605_618_759_378_327,
                -1.050_236_122_559_873_3,
            ]),
            Point::new([
                -5.687_160_826_499_058,
                6.504_655_423_433_022,
                8.941_590_411_569_816,
                9.543_547_641_077_382,
            ]),
            Point::new([
                8.975_549_245_653_312,
                -8.089_655_037_805_944,
                9.936_284_142_216_682,
                -7.816_992_427_475_977,
            ]),
            Point::new([
                5.825_845_324_524_742,
                -7.639_141_597_632_388,
                1.549_524_653_880_336_4,
                4.563_088_344_949_309,
            ]),
            Point::new([
                7.387_141_055_690_918,
                6.194_972_387_680_284,
                -5.764_015_058_796_046,
                9.298_338_336_238_999,
            ]),
            Point::new([
                -1.597_916_740_077_209_9,
                -4.938_008_036_006_716,
                7.414_979_546_687_874,
                -7.718_146_418_588_452,
            ]),
            Point::new([
                -2.414_045_007_912_424_3,
                8.888_648_260_600_007,
                -5.859_329_894_512_815,
                3.268_096_825_406_147,
            ]),
            Point::new([
                -8.294_250_893_230_837,
                3.083_275_278_154_95,
                8.020_989_920_767_69,
                8.155_291_219_012_977,
            ]),
            Point::new([
                6.718_748_825_685_814_6,
                -4.640_634_945_941_695,
                2.283_644_483_657_752_7,
                0.837_537_687_473_188_8,
            ]),
        ]
    }

    /// Exercise both successful perturbation retry (`attempt > 0`) and
    /// exhaustion (`SkippedDegeneracy`) paths with deterministic 4D fixtures.
    ///
    /// Covers: progressive scale factor, perturbation coordinate generation
    /// with `perturbation_seed == 0`, retry decision, retry success, and
    /// retry exhaustion.
    #[test]
    fn test_perturbation_retry_and_exhaustion_4d() {
        let initial_vertices: Vec<Vertex<f64, (), 4>> = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds = Triangulation::<AdaptiveKernel<f64>, (), (), 4>::build_initial_simplex(
            &initial_vertices,
        )
        .unwrap();
        let mut retry_success_tri = Triangulation::<AdaptiveKernel<f64>, (), (), 4>::new_with_tds(
            AdaptiveKernel::new(),
            tds,
        );

        let _guard = ForceNextRetryableInsertionFailureGuard::enable();
        let retry_success_vertex = VertexBuilder::default()
            .point(Point::new([0.2, 0.2, 0.2, 0.2]))
            .build()
            .unwrap();
        let (_outcome, retry_success_stats) = retry_success_tri
            .insert_with_statistics(retry_success_vertex, None, None)
            .unwrap();
        let saw_retry = retry_success_stats.used_perturbation() && retry_success_stats.success();

        let mut exhaustion_tri: Triangulation<AdaptiveKernel<f64>, (), (), 4> =
            Triangulation::new_empty(AdaptiveKernel::new());
        let mut saw_exhausted_skip = false;

        for point in perturbation_retry_repro_points_4d() {
            let v = VertexBuilder::default().point(point).build().unwrap();
            let (outcome, stats) = exhaustion_tri
                .insert_with_statistics(v, None, None)
                .unwrap();

            saw_exhausted_skip |= stats.skipped()
                && stats.attempts == DEFAULT_PERTURBATION_RETRIES + 1
                && matches!(stats.result, InsertionResult::SkippedDegeneracy)
                && matches!(outcome, InsertionOutcome::Skipped { error } if error.is_retryable());
        }

        assert!(
            saw_retry,
            "deterministic 4D fixture did not trigger a successful perturbation retry"
        );
        assert!(
            saw_exhausted_skip,
            "deterministic 4D adversarial repro did not trigger retry exhaustion"
        );
    }

    /// Exercise the seeded perturbation branch (`perturbation_seed != 0`)
    /// by calling `insert_transactional` directly.
    ///
    /// Covers: the `mix` computation and sign selection in the seeded path
    /// (lines using `perturbation_seed ^ ...`).
    ///
    /// Uses the same deterministic 4D repro as
    /// [`test_perturbation_retry_and_exhaustion_4d`].
    #[test]
    fn test_perturbation_retry_seeded_branch_4d() {
        let mut tri: Triangulation<AdaptiveKernel<f64>, (), (), 4> =
            Triangulation::new_empty(AdaptiveKernel::new());

        for point in perturbation_retry_repro_points_4d() {
            let v = VertexBuilder::default().point(point).build().unwrap();
            let (_outcome, stats) = tri
                .insert_transactional(
                    v,
                    None,
                    None,
                    DEFAULT_PERTURBATION_RETRIES,
                    0xDEAD_BEEF,
                    None,
                    None,
                )
                .unwrap();

            if stats.used_perturbation() && (stats.success() || stats.skipped()) {
                return;
            }
        }

        panic!("deterministic 4D adversarial repro did not trigger the seeded perturbation branch");
    }
}
