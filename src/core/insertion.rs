//! Incremental insertion for generic triangulations.
//!
//! This module owns transactional vertex insertion, duplicate-coordinate
//! detection, perturbation retry, conflict-region shaping, cavity insertion,
//! and insertion telemetry for [`Triangulation`](crate::prelude::triangulation::Triangulation).

#![forbid(unsafe_code)]

use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, CavityRepairStage, HullExtensionReason, InsertionError, extend_hull,
    external_facets_for_boundary, fill_cavity_replacing_simplices, wire_cavity_neighbors,
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
    CLEANUP_OPERATION_BUFFER_SIZE, CavityBoundaryBuffer, SimplexKeyBuffer, SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::operations::{
    InsertionOutcome, InsertionResult, InsertionStatistics, InsertionTelemetry,
    InsertionTelemetryMode, SuspicionFlags,
};
use crate::core::rollback::TriangulationRollbackTransaction;
#[cfg(debug_assertions)]
use crate::core::simplex::Simplex;
use crate::core::tds::{InvariantError, SimplexKey, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateValues, DEFAULT_TOLERANCE_F64};
use crate::locality::{
    append_live_unique_simplex_seeds, collect_local_exterior_conflict_seed_simplices,
    replace_simplices_and_record_removed, retain_simplices_and_record_removed,
};
#[cfg(debug_assertions)]
use crate::topology::manifold::validate_ridge_links;
use std::borrow::Cow;
use std::env;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
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
static CAVITY_REDUCTION_TRACE_EMITTED: AtomicBool = AtomicBool::new(false);

fn duplicate_detection_metrics_enabled() -> bool {
    #[cfg(test)]
    if tests::duplicate_detection_force_enabled() {
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
        ConflictError::InvalidSimplexArity {
            simplex_key,
            expected,
            found,
        } => {
            format!(
                "invalid_simplex_arity simplex_key={simplex_key:?} expected={expected} \
                 found={found}"
            )
        }
        ConflictError::MissingSimplexVertex {
            simplex_key,
            vertex_key,
        } => {
            format!("missing_simplex_vertex simplex_key={simplex_key:?} vertex_key={vertex_key:?}")
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

    let conflict_preview: SimplexKeyBuffer = conflict_simplices.iter().copied().take(12).collect();
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

// =============================================================================
// Geometric Operations (Requires Extra Numeric Conversion Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
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
    /// use delaunay::prelude::Triangulation;
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
    /// use delaunay::prelude::{DuplicateDetectionMetrics, Triangulation};
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

    /// Insert a vertex with statistics, using a custom perturbation seed and an optional
    /// spatial hash-grid index, and also return the simplices that cavity reduction touched
    /// and left in place.
    ///
    /// The extra seed set stays internal so bulk construction and debug rebuilds can widen
    /// their local repair frontier without changing the public insertion API.
    pub(crate) fn insert_with_statistics_seeded_indexed_detailed(
        &mut self,
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<D>>,
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
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<D>>,
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
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        max_perturbation_attempts: usize,
        perturbation_seed: u64,
        mut index: Option<&mut HashGridIndex<D>>,
        bulk_index: Option<usize>,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<DetailedInsertionResult, InsertionError> {
        let mut stats = InsertionStatistics::default();
        let mut telemetry = InsertionTelemetry::default();
        let original_coords = *vertex.point().coords();
        let original_uuid = vertex.uuid();
        let mut current_vertex = vertex;
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

        // Base perturbation epsilon: ≈ √machine_epsilon for f64.
        let epsilon_value: f64 = 1e-8;

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
                let epsilon = epsilon_value * scale_factor;

                let perturbation_scale = epsilon * local_scale;
                for (idx, coord) in perturbed_coords.iter_mut().enumerate() {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "D is a const-generic triangulation dimension and practical dimensions are tiny"
                    )]
                    let coord_scale = (idx + 1) as f64;
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
                let perturbed_point = Point::try_new(perturbed_coords)
                    .map_err(|source| InsertionError::PerturbedCoordinateInvalid { source })?;
                current_vertex = Vertex::from_validated_point_with_uuid(
                    perturbed_point,
                    original_uuid,
                    current_vertex.data,
                );
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

            let mut transaction = TriangulationRollbackTransaction::begin(self);

            // Try insertion.
            //
            // Topology safety net: ensure we don't commit an insertion that breaks Level 3 topology.
            // If the cavity-based insertion produces an Euler/topology mismatch, roll back and retry a
            // conservative fallback (star-split of the containing simplex) within the same transactional attempt.
            #[cfg(test)]
            // Test-only hook for deterministic coverage of the rollback + perturbation retry
            // success path, which is otherwise rare under the adaptive SoS predicates.
            let result = if tests::take_force_next_insertion_retryable_failure() {
                Err(InsertionError::NonManifoldTopology {
                    facet_hash: 0x000F_0CED,
                    simplex_count: 3,
                })
            } else {
                Self::try_insert_with_topology_safety_net(
                    &mut transaction,
                    current_vertex,
                    conflict_simplices,
                    hint,
                    attempt,
                    &mut telemetry,
                    telemetry_mode,
                )
            };
            #[cfg(not(test))]
            let result = Self::try_insert_with_topology_safety_net(
                &mut transaction,
                current_vertex,
                conflict_simplices,
                hint,
                attempt,
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
                    transaction.commit();
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
                    transaction.rollback();

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
        coords: &[f64; D],
        index: &HashGridIndex<D>,
    ) -> Option<SimplexKey> {
        let mut best: Option<(f64, SimplexKey)> = None;

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
            let mut dist_sq = 0.0;
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq = diff.mul_add(diff, dist_sq);
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

    /// Returns the f64 relative tolerance used for duplicate-coordinate detection.
    const fn duplicate_relative_tolerance() -> f64 {
        1e-10_f64
    }

    /// Keeps duplicate-scale estimates tied to existing geometry rather than
    /// hard-coding a scalar-unit epsilon.
    fn include_duplicate_scale_reference(
        point_coords: &[f64; D],
        axis_min: &mut [f64; D],
        axis_max: &mut [f64; D],
        magnitude_scale: &mut f64,
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
        coords: &[f64; D],
        hint: Option<SimplexKey>,
    ) -> f64 {
        let mut axis_min = *coords;
        let mut axis_max = *coords;
        let mut magnitude_scale = 0.0;
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
            if local_scale.is_finite() && local_scale > 0.0 {
                if local_scale > magnitude_scale {
                    magnitude_scale = local_scale;
                }
                local_feature_scale = Some(local_scale);
            }
        }

        let feature_scale = local_feature_scale.unwrap_or_else(|| {
            let mut span_sq = 0.0;
            for i in 0..D {
                let span = axis_max[i] - axis_min[i];
                span_sq = span.mul_add(span, span_sq);
            }
            span_sq.sqrt()
        });
        let relative_tolerance = Self::duplicate_relative_tolerance() * feature_scale;
        let ulp_tolerance = f64::EPSILON * 16.0 * magnitude_scale;
        let mut tolerance = if relative_tolerance > ulp_tolerance {
            relative_tolerance
        } else {
            ulp_tolerance
        };

        if !tolerance.is_finite() || tolerance <= 0.0 {
            tolerance = Self::duplicate_relative_tolerance();
        }

        tolerance
    }

    /// Rebuilds the duplicate index when a scale-aware tolerance grows beyond
    /// the current grid cell size, preserving complete candidate coverage.
    fn ensure_duplicate_index_cell_size(
        &self,
        index: Option<&mut HashGridIndex<D>>,
        tolerance: f64,
    ) {
        let Some(index) = index else {
            return;
        };
        if !HashGridIndex::<D>::supports_dimension() || !tolerance.is_finite() || tolerance <= 0.0 {
            return;
        }
        if index.cell_size() >= tolerance {
            return;
        }

        let Ok(mut rebuilt) = HashGridIndex::try_new(tolerance) else {
            return;
        };
        for (vkey, vertex) in self.tds.vertices() {
            rebuilt.insert_vertex(vkey, vertex.point().coords());
        }
        *index = rebuilt;
    }

    /// Compares a squared distance against the duplicate tolerance without
    /// overflowing the tolerance square on extreme coordinate scales.
    fn duplicate_distance_within_tolerance(dist_sq: f64, tolerance: f64) -> bool {
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
        coords: &[f64; D],
        tolerance: f64,
        index: Option<&HashGridIndex<D>>,
    ) -> Option<InsertionError> {
        let mut duplicate_found = false;
        let make_duplicate_error = || {
            let coordinates = CoordinateValues::from_numeric_slice(coords);
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
                let mut dist_sq = 0.0;
                for i in 0..D {
                    let diff = vcoords[i] - coords[i];
                    dist_sq = diff.mul_add(diff, dist_sq);
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
            let mut dist_sq = 0.0;
            for i in 0..D {
                let diff = coords[i] - existing_coords[i];
                dist_sq = diff.mul_add(diff, dist_sq);
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
        coords: &[f64; D],
        hint: Option<SimplexKey>,
    ) -> f64 {
        let mut min_dist_sq: Option<f64> = None;

        let consider_vertex = |vertex: &Vertex<U, D>, min_dist_sq: &mut Option<f64>| {
            let vcoords = vertex.point().coords();
            let mut dist_sq = 0.0;
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq = diff.mul_add(diff, dist_sq);
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

        let mut scale = min_dist_sq.map_or(1.0, f64::sqrt);

        let min_scale = DEFAULT_TOLERANCE_F64;
        if scale < min_scale {
            scale = min_scale;
        }

        scale
    }

    /// Attempt an insertion, and if Level 3 validation fails, roll back and try a
    /// conservative star-split fallback of the containing simplex.
    fn try_insert_with_topology_safety_net(
        transaction: &mut TriangulationRollbackTransaction<'_, K, U, V, D>,
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        attempt: usize,
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<TryInsertImplOk, InsertionError> {
        let mut insert_ok = transaction.triangulation_mut().try_insert_impl(
            vertex,
            conflict_simplices,
            hint,
            telemetry,
            telemetry_mode,
        )?;

        if attempt > 0 {
            insert_ok.suspicion.perturbation_used = true;
        }
        if insert_ok.suspicion.is_suspicious() {
            insert_ok.delaunay_repair_required = true;
        }

        // Skip Level 3 validation during bootstrap (vertices but no simplices yet).
        if transaction.triangulation_mut().tds.number_of_simplices() == 0 {
            return Ok(insert_ok);
        }

        let validation_result = transaction
            .triangulation_mut()
            .validate_after_insertion_and_record_telemetry(
                insert_ok.suspicion,
                &insert_ok.repair_seed_simplices,
                telemetry,
                telemetry_mode,
            );
        if let Err(validation_err) = validation_result {
            // Roll back to snapshot and attempt a star-split fallback for interior points.
            transaction.restore();
            return transaction
                .triangulation_mut()
                .try_star_split_fallback_after_topology_failure(
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
        vertex: Vertex<U, D>,
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
                FacetHandle::from_validated(
                    start_simplex,
                    u8::try_from(i).expect("facet index must fit in u8"),
                )
            })
            .collect()
    }

    /// Reshape a conflict region until it yields a valid cavity boundary.
    ///
    /// Iteratively resolves cavity-boundary errors rather than immediately
    /// falling back to a star-split. Star-splits create non-Delaunay
    /// configurations that global flip repair must fix; in high dimensions this
    /// is extremely slow. The reduction rules are:
    ///
    /// - `RidgeFan`: shrink by removing extra fan simplices.
    /// - `DisconnectedBoundary`: expand through non-conflict neighbors, or
    ///   shrink if expansion cannot make progress.
    /// - `OpenBoundary`: shrink by removing the simplex with the dangling facet.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep the cavity-reduction state machine together so reshape rules share one iteration budget"
    )]
    fn reduce_conflict_region_to_cavity_boundary(
        &self,
        conflict_simplices: &mut SimplexKeyBuffer,
        repair_seed_simplices: &mut SimplexKeyBuffer,
        delaunay_repair_required: &mut bool,
    ) -> Result<CavityBoundaryBuffer, ConflictError> {
        const MAX_CAVITY_ITERATIONS: usize = 32;

        let mut extraction_result = extract_cavity_boundary(&self.tds, conflict_simplices);
        let mut iterations: usize = 0;
        let trace_enabled = cavity_reduction_trace_enabled();
        let mut trace_cavity_reduction = false;
        let mut saw_ridge_fan_shrink = false;

        match &extraction_result {
            Ok(boundary) => {
                log_cavity_reduction_event(
                    trace_cavity_reduction,
                    iterations,
                    conflict_simplices,
                    || format!("initial_ok boundary_facets={}", boundary.len()),
                );
            }
            Err(err) => {
                trace_cavity_reduction =
                    trace_enabled && !CAVITY_REDUCTION_TRACE_EMITTED.swap(true, Ordering::Relaxed);
                log_cavity_reduction_event(
                    trace_cavity_reduction,
                    iterations,
                    conflict_simplices,
                    || format!("initial_err {}", cavity_conflict_error_summary(err)),
                );
            }
        }

        loop {
            if iterations >= MAX_CAVITY_ITERATIONS {
                log_cavity_reduction_event(
                    trace_cavity_reduction,
                    iterations,
                    conflict_simplices,
                    || "budget_exhausted".to_string(),
                );
                break;
            }
            iterations += 1;

            match &extraction_result {
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
                        conflict_simplices,
                        || format!("ridge_fan_shrink remove_simplices={extra_simplices:?}"),
                    );
                    saw_ridge_fan_shrink = true;
                    *delaunay_repair_required = true;
                    retain_simplices_and_record_removed(
                        conflict_simplices,
                        repair_seed_simplices,
                        |simplex_key| !extra_simplices.contains(&simplex_key),
                    );
                }
                Err(ConflictError::DisconnectedBoundary {
                    disconnected_simplices,
                    ..
                }) if !disconnected_simplices.is_empty() => {
                    let mut simplices_to_add = SimplexKeyBuffer::new();
                    if !saw_ridge_fan_shrink {
                        for &dc in disconnected_simplices {
                            if let Some(simplex) = self.tds.simplex(dc)
                                && let Some(neighbors) = simplex.neighbor_keys()
                            {
                                for neighbor_opt in neighbors {
                                    if let Some(nk) = neighbor_opt
                                        && !conflict_simplices.contains(&nk)
                                        && !simplices_to_add.contains(&nk)
                                    {
                                        simplices_to_add.push(nk);
                                    }
                                }
                            }
                        }
                    }

                    if !simplices_to_add.is_empty() {
                        *delaunay_repair_required = true;
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            add_count = simplices_to_add.len(),
                            conflict_simplices_before = conflict_simplices.len(),
                            "D={D}: cavity expansion (DisconnectedBoundary hole-fill)"
                        );
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            conflict_simplices,
                            || {
                                format!(
                                    "disconnected_boundary_expand add_simplices={simplices_to_add:?}"
                                )
                            },
                        );
                        conflict_simplices.extend(simplices_to_add);
                    } else if conflict_simplices.len() > D + 1 {
                        *delaunay_repair_required = true;
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            remove_count = disconnected_simplices.len(),
                            conflict_simplices_before = conflict_simplices.len(),
                            "D={D}: cavity reduction (DisconnectedBoundary shrink fallback)"
                        );
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            conflict_simplices,
                            || {
                                format!(
                                    "disconnected_boundary_shrink remove_simplices={disconnected_simplices:?}"
                                )
                            },
                        );
                        retain_simplices_and_record_removed(
                            conflict_simplices,
                            repair_seed_simplices,
                            |simplex_key| !disconnected_simplices.contains(&simplex_key),
                        );
                    } else {
                        log_cavity_reduction_event(
                            trace_cavity_reduction,
                            iterations,
                            conflict_simplices,
                            || "disconnected_boundary_no_progress".to_string(),
                        );
                        break;
                    }
                }
                Err(ConflictError::OpenBoundary { open_simplex, .. })
                    if conflict_simplices.len() > D + 1 =>
                {
                    *delaunay_repair_required = true;
                    #[cfg(debug_assertions)]
                    tracing::debug!(
                        ?open_simplex,
                        conflict_simplices_before = conflict_simplices.len(),
                        "D={D}: cavity reduction (OpenBoundary shrink)"
                    );
                    log_cavity_reduction_event(
                        trace_cavity_reduction,
                        iterations,
                        conflict_simplices,
                        || format!("open_boundary_shrink open_simplex={open_simplex:?}"),
                    );
                    let open = *open_simplex;
                    retain_simplices_and_record_removed(
                        conflict_simplices,
                        repair_seed_simplices,
                        |simplex_key| simplex_key != open,
                    );
                }
                _ => {
                    log_cavity_reduction_event(
                        trace_cavity_reduction,
                        iterations,
                        conflict_simplices,
                        || "no_reduction_rule_matched".to_string(),
                    );
                    break;
                }
            }

            extraction_result = extract_cavity_boundary(&self.tds, conflict_simplices);
            match &extraction_result {
                Ok(boundary) => {
                    log_cavity_reduction_event(
                        trace_cavity_reduction,
                        iterations,
                        conflict_simplices,
                        || format!("reextract_ok boundary_facets={}", boundary.len()),
                    );
                }
                Err(err) => {
                    log_cavity_reduction_event(
                        trace_cavity_reduction,
                        iterations,
                        conflict_simplices,
                        || format!("reextract_err {}", cavity_conflict_error_summary(err)),
                    );
                }
            }
        }

        extraction_result
    }

    /// Perform cavity insertion given an explicit conflict region.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep cavity insertion and repair logic together for clarity"
    )]
    fn insert_with_conflict_region(
        &mut self,
        v_key: VertexKey,
        point: &Point<D>,
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

        let mut boundary_facets = match self.reduce_conflict_region_to_cavity_boundary(
            &mut conflict_simplices,
            &mut repair_seed_simplices,
            &mut delaunay_repair_required,
        ) {
            Ok(boundary) => boundary,
            Err(err) => {
                // For D=3 and D>=4: do NOT fall back to star-split once cavity reduction
                // is exhausted. Star-splits create heavily non-Delaunay configurations
                // whose isolated violations may not be connected to the star-split star
                // through any violation chain. Return a retryable error instead so
                // insert_transactional can retry with a perturbed vertex and, after all
                // retries, skip the vertex.
                //
                // For D=2: star-split is used as a last resort. The 2D flip repair
                // guarantees convergence from star-split configurations and the extra
                // simplices are quickly handled by the k=2 repair loop.
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
        if env::var_os("DELAUNAY_DEBUG_ORIENTATION").is_some() {
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
        repair_seed_simplices.retain(|ck| !conflict_simplices.contains(ck));
        let mut seen_repair_seed_simplices = SimplexKeyBuffer::new();
        repair_seed_simplices.retain(|ck| {
            if seen_repair_seed_simplices.contains(ck) {
                false
            } else {
                seen_repair_seed_simplices.push(*ck);
                true
            }
        });

        let mut incident_repair_vertices =
            SmallBuffer::<VertexKey, CLEANUP_OPERATION_BUFFER_SIZE>::new();
        Self::push_incident_repair_vertex(&mut incident_repair_vertices, v_key);
        self.extend_incident_repair_vertices_from_simplices(
            &conflict_simplices,
            &mut incident_repair_vertices,
        );
        self.extend_incident_repair_vertices_from_simplices(
            &new_simplices,
            &mut incident_repair_vertices,
        );

        // Remove conflict simplices (now that new simplices are wired up)
        let _removed_count = self
            .tds
            .remove_simplices_by_keys(&conflict_simplices)
            .map_err(|e| InsertionError::TopologyValidation(e.into_inner()))?;

        // Iteratively repair non-manifold topology until facet sharing is valid
        let mut total_removed = 0;
        let max_repair_simplices_removed = new_simplices.len();
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

                let repair_budget_remaining =
                    max_repair_simplices_removed.saturating_sub(total_removed);
                let repair =
                    self.repair_local_facet_issues_with_frontier(&issues, repair_budget_remaining)?;
                let removed = repair.removed_count;
                for &vertex_key in &repair.affected_vertices {
                    Self::push_incident_repair_vertex(&mut incident_repair_vertices, vertex_key);
                }

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
        self.extend_incident_repair_vertices_from_simplices(
            &neighbor_repair_frontier,
            &mut incident_repair_vertices,
        );

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
        if env::var_os("DELAUNAY_DEBUG_VALIDATE_LOCATE").is_some() {
            let _ = locate(&self.tds, &self.kernel, point, None)?;
        }

        #[cfg(debug_assertions)]
        if env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
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
        self.repair_stale_incident_simplices(&incident_repair_vertices)?;

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
        point: &Point<D>,
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
        vertex: Vertex<U, D>,
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
        if env::var_os("DELAUNAY_DEBUG_HULL").is_some()
            || env::var_os("DELAUNAY_DEBUG_LOCATE").is_some()
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
                if env::var_os("DELAUNAY_DEBUG_CONFLICT_VERIFY").is_some() {
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
                    if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                    if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                                if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                                if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                        if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
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
                let mut incident_repair_vertices =
                    SmallBuffer::<VertexKey, CLEANUP_OPERATION_BUFFER_SIZE>::new();
                Self::push_incident_repair_vertex(&mut incident_repair_vertices, v_key);
                self.extend_incident_repair_vertices_from_simplices(
                    &new_simplices,
                    &mut incident_repair_vertices,
                );
                #[cfg(debug_assertions)]
                if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        new_simplices = new_simplices.len(),
                        "Outside insertion: hull extension succeeded"
                    );
                }

                #[cfg(debug_assertions)]
                if env::var_os("DELAUNAY_DEBUG_NEIGHBORS").is_some() {
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
                let max_repair_simplices_removed = new_simplices.len();
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

                        let repair_budget_remaining =
                            max_repair_simplices_removed.saturating_sub(total_removed);
                        let repair = self.repair_local_facet_issues_with_frontier(
                            &issues,
                            repair_budget_remaining,
                        )?;
                        let removed = repair.removed_count;
                        for &vertex_key in &repair.affected_vertices {
                            Self::push_incident_repair_vertex(
                                &mut incident_repair_vertices,
                                vertex_key,
                            );
                        }

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
                self.extend_incident_repair_vertices_from_simplices(
                    &neighbor_repair_frontier,
                    &mut incident_repair_vertices,
                );

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
                if env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
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
                self.repair_stale_incident_simplices(&incident_repair_vertices)?;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::locate::InternalInconsistencySite;
    use crate::core::collections::spatial_hash_grid::HashGridIndex;
    use crate::core::simplex::Simplex;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{
        CoordinateConversionError, CoordinateConversionValue, DEFAULT_TOLERANCE_F64,
        F64_MANTISSA_DIGITS,
    };
    use crate::triangulation::DelaunayTriangulation;
    use std::assert_matches;

    use slotmap::KeyData;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    static DUPLICATE_DETECTION_FORCE_ENABLED: AtomicBool = AtomicBool::new(false);

    thread_local! {
        static FORCE_NEXT_INSERTION_RETRYABLE_FAILURE: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn duplicate_detection_force_enabled() -> bool {
        DUPLICATE_DETECTION_FORCE_ENABLED.load(AtomicOrdering::Relaxed)
    }

    fn set_duplicate_detection_force_enabled(enabled: bool) {
        DUPLICATE_DETECTION_FORCE_ENABLED.store(enabled, AtomicOrdering::Relaxed);
    }

    pub(super) fn take_force_next_insertion_retryable_failure() -> bool {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.replace(false)
    }

    fn set_force_next_insertion_retryable_failure(enabled: bool) -> bool {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.replace(enabled)
    }

    fn restore_force_next_insertion_retryable_failure(prior: bool) {
        FORCE_NEXT_INSERTION_RETRYABLE_FAILURE.set(prior);
    }

    fn insert<K, U, V, const D: usize>(
        tri: &mut Triangulation<K, U, V, D>,
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
    ) -> Result<(VertexKey, Option<SimplexKey>), InsertionError>
    where
        K: Kernel<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        let (outcome, _stats) = insert_transactional(
            tri,
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

    fn insert_with_statistics<K, U, V, const D: usize>(
        tri: &mut Triangulation<K, U, V, D>,
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K: Kernel<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        insert_transactional(
            tri,
            vertex,
            conflict_simplices,
            hint,
            DEFAULT_PERTURBATION_RETRIES,
            0,
            None,
            None,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Test helper mirrors the detailed transactional insertion signature"
    )]
    fn insert_transactional<K, U, V, const D: usize>(
        tri: &mut Triangulation<K, U, V, D>,
        vertex: Vertex<U, D>,
        conflict_simplices: Option<&SimplexKeyBuffer>,
        hint: Option<SimplexKey>,
        max_perturbation_attempts: usize,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<D>>,
        bulk_index: Option<usize>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K: Kernel<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        let detail = tri.insert_transactional_detailed(
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

    struct ForceNextRetryableInsertionFailureGuard {
        prior: bool,
    }

    impl ForceNextRetryableInsertionFailureGuard {
        fn enable() -> Self {
            let prior = set_force_next_insertion_retryable_failure(true);
            Self { prior }
        }
    }

    impl Drop for ForceNextRetryableInsertionFailureGuard {
        fn drop(&mut self) {
            restore_force_next_insertion_retryable_failure(self.prior);
        }
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
                coordinate_value: CoordinateConversionValue::from_f64(f64::NAN),
                from_type: "f64",
                to_type: "f64",
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
    fn test_duplicate_detection_metrics_force_enable() {
        struct DuplicateDetectionGuard;

        impl Drop for DuplicateDetectionGuard {
            fn drop(&mut self) {
                set_duplicate_detection_force_enabled(false);
            }
        }

        let _guard = DuplicateDetectionGuard;
        set_duplicate_detection_force_enabled(true);

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

    fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(crate::core::vertex::Vertex::<(), _>::try_new([0.0_f64; D]).unwrap());
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            vertices.push(crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap());
        }
        vertices
    }

    /// Build a simplex whose feature length is controlled by one shared axis scale.
    fn axis_scaled_simplex_vertices<const D: usize>(scale: f64) -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(crate::core::vertex::Vertex::<(), _>::try_new([0.0_f64; D]).unwrap());
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = scale;
            vertices.push(crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap());
        }
        vertices
    }

    /// Build coordinates with only the first component set for tolerance-scale tests.
    fn coords_with_first<const D: usize>(first: f64) -> [f64; D] {
        let mut coords = [0.0_f64; D];
        coords[0] = first;
        coords
    }

    #[test]
    fn test_select_locate_hint_from_hash_grid_returns_incident_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();

        let mut index: HashGridIndex<2> = HashGridIndex::try_new(1.0).unwrap();
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        {
            let vertex = tri.tds.vertex_mut(vkey).unwrap();
            vertex.set_incident_simplex(Some(SimplexKey::default()));
        }

        let mut index: HashGridIndex<2> = HashGridIndex::try_new(1.0).unwrap();
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let mut index: HashGridIndex<2> = HashGridIndex::try_new(1.0).unwrap();
        index.insert_vertex(vkey, &[0.0, 0.0]);

        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol, Some(&index));
        assert_matches!(err, Some(InsertionError::DuplicateCoordinates { .. }));
    }

    #[test]
    fn test_duplicate_coordinates_error_falls_back_when_index_unusable() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let mut index: HashGridIndex<2> = HashGridIndex::try_new(1.0).unwrap();
        index.remove_vertex(&VertexKey::default(), &[f64::NAN, 0.0]);
        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol, Some(&index));
        assert_matches!(err, Some(InsertionError::DuplicateCoordinates { .. }));
    }

    fn duplicate_coordinate_tolerance_scales_down_for_small_features<const D: usize>() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), D> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new(coords_with_first::<D>(1.0e-6))
                    .unwrap(),
            )
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
        assert_matches!(
            tri.duplicate_coordinates_error(&candidate, tolerance, None),
            Some(InsertionError::DuplicateCoordinates { .. })
        );
    }

    fn duplicate_index_rebuilds_when_tolerance_exceeds_cell_size<const D: usize>() {
        let vertices = axis_scaled_simplex_vertices::<D>(1.0e6);
        let tds =
            Triangulation::<FastKernel<f64>, (), (), D>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), D>::new_with_tds(FastKernel::new(), tds);
        let hint = tri.tds.simplex_keys().next();
        let candidate = [1.0_f64; D];
        let tolerance = tri.estimate_duplicate_coordinate_tolerance(&candidate, hint);
        let mut index: HashGridIndex<D> = HashGridIndex::try_new(1.0e-10).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let scale = tri.estimate_local_perturbation_scale(&[0.0, 0.0], None);
        let min_scale = DEFAULT_TOLERANCE_F64;
        approx::assert_abs_diff_eq!(scale, min_scale, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_insert_duplicate_coordinates_skips_with_statistics_and_errors_without() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // First insertion succeeds.
        insert(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        // Second insertion at same coordinates: insert() returns Err; the internal
        // statistics helper reports Skipped so telemetry can classify the no-op.
        let err = insert(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .unwrap_err();
        assert_matches!(err, InsertionError::DuplicateCoordinates { .. });

        let (outcome, stats) = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .unwrap();
        assert!(stats.skipped());
        assert_matches!(outcome, InsertionOutcome::Skipped { .. });

        // No new vertex should have been inserted.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn test_insert_duplicate_uuid_is_non_retryable_and_rolls_back() {
        // Insert a then attempt to insert another vertex with the same UUID.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        insert(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        let existing_uuid = tri.tds.vertices().next().unwrap().1.uuid();
        let dup = Vertex::try_new_with_uuid(
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            existing_uuid,
            None,
        )
        .unwrap();

        let err = insert(&mut tri, dup, None, None).unwrap_err();
        assert!(
            !err.is_retryable(),
            "Duplicate UUID should be non-retryable"
        );

        // Ensure rollback: vertex count unchanged.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    // =============================================================================
    // Triangulation insert_with_statistics tests (internal API)
    // =============================================================================

    #[test]
    fn triangulation_insert_with_statistics_basic_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        let (outcome, stats) = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .expect("insertion should succeed");

        assert_matches!(outcome, InsertionOutcome::Inserted { hint: None, .. });
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        for (i, v) in vertices.into_iter().enumerate() {
            let (outcome, stats) = insert_with_statistics(&mut tri, v, None, None).unwrap();

            assert_matches!(outcome, InsertionOutcome::Inserted { .. });
            assert_eq!(stats.attempts, 1);

            if i < 3 {
                // Bootstrap phase - no hint yet
                assert_matches!(outcome, InsertionOutcome::Inserted { hint: None, .. });
            } else {
                // After D+1 vertices, hint should be available
                assert_matches!(outcome, InsertionOutcome::Inserted { hint: Some(_), .. });
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
            insert_with_statistics(
                &mut tri,
                crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                None,
                None,
            )
            .unwrap();
        }

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0, 2.0]).unwrap(),
                None,
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert_matches!(detail.outcome, InsertionOutcome::Inserted { .. });
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
            insert_with_statistics(
                &mut tri,
                crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                None,
                None,
            )
            .unwrap();
        }

        let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
        let empty_conflicts = SimplexKeyBuffer::new();
        let detail = tri
            .insert_with_statistics_seeded_indexed_detailed(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0, 2.0]).unwrap(),
                Some(&empty_conflicts),
                hint,
                0,
                None,
                None,
            )
            .unwrap();

        assert_matches!(detail.outcome, InsertionOutcome::Inserted { .. });
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
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
                crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25]).unwrap(),
                Some(&conflict_simplices),
                Some(start_simplex),
                0,
                None,
                None,
            )
            .unwrap();

        assert_matches!(detail.outcome, InsertionOutcome::Inserted { .. });
        assert!(
            !detail.delaunay_repair_required,
            "caller-provided conflict simplices should preserve the cavity insertion repair flag"
        );
        assert!(tri.is_valid().is_ok());
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
            insert_with_statistics(
                &mut tri,
                crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                None,
                None,
            )
            .unwrap();
        }

        // Insert with explicit hint
        let hint_simplex = tri.simplices().next().map(|(key, _)| key);
        let (outcome, stats) = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
            None,
            hint_simplex,
        )
        .unwrap();

        assert_matches!(outcome, InsertionOutcome::Inserted { hint: Some(_), .. });
        assert_eq!(stats.attempts, 1);
        assert!(stats.success());
    }

    #[test]
    fn triangulation_insert_with_statistics_duplicate_coordinates_3d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap(),
            None,
            None,
        )
        .unwrap();

        // Try duplicate - should be skipped
        let result = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap(),
            None,
            None,
        );

        assert_matches!(
            result,
            Ok((
                InsertionOutcome::Skipped {
                    error: InsertionError::DuplicateCoordinates { .. }
                },
                _
            ))
        );
    }

    #[test]
    fn triangulation_insert_with_statistics_multiple_insertions_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        let points = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.7, 0.3]).unwrap(),
        ];

        let mut all_succeeded = true;
        let mut max_attempts = 0;

        for point in points {
            match insert_with_statistics(&mut tri, point, None, None) {
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
        let (outcome, _) = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            None,
            None,
        )
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

            let (outcome, stats) = insert_with_statistics(
                &mut tri,
                crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                None,
                None,
            )
            .unwrap();

            assert_matches!(outcome, InsertionOutcome::Inserted { .. });
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
        insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            None,
            None,
        )
        .unwrap();
        insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            None,
            None,
        )
        .unwrap();
        insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            None,
            None,
        )
        .unwrap();

        let simplices_before = tri.number_of_simplices();

        // Insert interior point - might trigger repair
        let (_outcome, stats) = insert_with_statistics(
            &mut tri,
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.3]).unwrap(),
            None,
            None,
        )
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([5.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([6.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([5.5, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_a = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let simplex_b = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v3, v4, v5], None).unwrap(),
            )
            .unwrap();

        // Neither simplex has any neighbour pointers, so `simplices_to_add` will be empty on
        // the first iteration and the `else { break; }` arm fires immediately.
        let new_v = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.3]).unwrap(),
            )
            .unwrap();
        let point = Point::try_new([0.5_f64, 0.3_f64]).expect("finite point coordinates");

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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        // Three distinct fourth vertices that all pair with the {v0,v1,v2} face.
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 2.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();
        let simplex3 = tri
            .tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v0, v1, v2, v5], None).unwrap(),
            )
            .unwrap();

        let new_v = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.1, 0.1, 0.1]).unwrap(),
            )
            .unwrap();
        let point = Point::try_new([0.1_f64, 0.1_f64, 0.1_f64]).expect("finite point coordinates");

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
    #[expect(
        clippy::too_many_lines,
        reason = "regression test keeps the ridge-fan shrink fixture explicit"
    )]
    fn test_cavity_reduction_ridge_fan_shrink_fires_for_4_conflict_simplices_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // `center` appears in 8 boundary edges (2 per simplex × 4 simplices) → RidgeFan.
        let center = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let va = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 2.0]).unwrap(),
            )
            .unwrap();
        let vb = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0]).unwrap(),
            )
            .unwrap();
        let vc = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-3.0, -2.0]).unwrap(),
            )
            .unwrap();
        let vd = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-2.0, -3.0]).unwrap(),
            )
            .unwrap();
        let ve = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, -2.0]).unwrap(),
            )
            .unwrap();
        let vf = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, -3.0]).unwrap(),
            )
            .unwrap();
        let vg = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-4.0, 1.0]).unwrap(),
            )
            .unwrap();
        let vh = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-4.0, -1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, va, vb], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, vc, vd], None).unwrap(),
            )
            .unwrap();
        let simplex3 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, ve, vf], None).unwrap(),
            )
            .unwrap();
        let simplex4 = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, vg, vh], None).unwrap(),
            )
            .unwrap();

        let new_v = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.3, 1.0]).unwrap(),
            )
            .unwrap();
        let point = Point::try_new([0.3_f64, 1.0_f64]).expect("finite point coordinates");

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(simplex1);
        conflict_simplices.push(simplex2);
        conflict_simplices.push(simplex3);
        conflict_simplices.push(simplex4);

        let mut reduced_conflict_simplices = conflict_simplices.clone();
        let mut repair_seed_simplices = SimplexKeyBuffer::new();
        let mut delaunay_repair_required = false;
        let _ = tri.reduce_conflict_region_to_cavity_boundary(
            &mut reduced_conflict_simplices,
            &mut repair_seed_simplices,
            &mut delaunay_repair_required,
        );
        assert!(
            reduced_conflict_simplices.len() < conflict_simplices.len(),
            "ridge-fan reduction should remove at least one extra simplex"
        );
        assert!(
            !repair_seed_simplices.is_empty(),
            "removed ridge-fan simplices should seed later local repair"
        );
        assert!(
            delaunay_repair_required,
            "shrinking an invalid cavity should force local Delaunay repair"
        );

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
    #[expect(
        clippy::too_many_lines,
        reason = "regression test keeps the disconnected cavity fixture explicit"
    )]
    fn test_cavity_reduction_disconnected_expand_then_shrink_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Group A: simplex_a = {v0,v1,v2} shares edge {v0,v1} with simplex_c = {v0,v1,v6}.
        let v0 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            )
            .unwrap();
        let v6 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, -1.0]).unwrap(),
            )
            .unwrap();
        // Group B: simplex_b = {v3,v4,v5} shares edge {v3,v4} with simplex_d = {v3,v4,v7}.
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([5.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v4 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([6.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v5 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([5.5, 1.0]).unwrap(),
            )
            .unwrap();
        let v7 = tri
            .tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([5.5, -1.0]).unwrap(),
            )
            .unwrap();

        let simplex_a = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        // simplex_c is a non-conflict neighbour of simplex_a (not initially in conflict_simplices).
        let simplex_c = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v6], None).unwrap(),
            )
            .unwrap();
        let simplex_b = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v3, v4, v5], None).unwrap(),
            )
            .unwrap();
        // simplex_d is a non-conflict neighbour of simplex_b.
        let simplex_d = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v3, v4, v7], None).unwrap(),
            )
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.3]).unwrap(),
            )
            .unwrap();
        let point = Point::try_new([0.5_f64, 0.3_f64]).expect("finite point coordinates");

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
    // INSERTION PIPELINE: BOOTSTRAP, INITIAL SIMPLEX, BEYOND-SIMPLEX
    // =========================================================================

    /// Helper: build a set of D+1 affinely independent vertices for dimension D.
    fn simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut verts = Vec::with_capacity(D + 1);
        // Origin
        verts.push(Vertex::try_new([0.0; D]).unwrap());
        // Unit vectors along each axis
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            verts.push(Vertex::try_new(coords).unwrap());
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
                        let (vk, hint) = insert(&mut tri, *v, None, None).unwrap();
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
                        insert(&mut tri, *v, None, None).unwrap();
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
                        insert(&mut tri, *v, None, None).unwrap();
                    }
                    assert_eq!(tri.number_of_simplices(), 1);

                    // Insert an interior point.
                    let mut interior = [0.0; $dim];
                    for c in interior.iter_mut() {
                        *c = 1.0 / (<f64 as std::convert::From<i32>>::from($dim + 1) * 2.0);
                    }
                    let interior_vertex = Vertex::try_new(interior).unwrap();
                    let (_, hint) = insert(&mut tri, interior_vertex, None, None).unwrap();

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
            let (outcome, stats) = insert_with_statistics(
                &mut tri,
                crate::core::vertex::Vertex::<(), _>::try_new(*coords).unwrap(),
                None,
                None,
            )
            .unwrap();
            assert!(stats.success());
            assert_matches!(outcome, InsertionOutcome::Inserted { .. });
            assert_eq!(stats.attempts, 1);
        }
        assert_eq!(tri.number_of_vertices(), 4);
        assert!(tri.number_of_simplices() >= 2);
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, 4.0]).unwrap(),
            )
            .unwrap();

        let tol = 1e-10_f64;
        // No index provided: should fall back to linear scan.
        let err = tri.duplicate_coordinates_error(&[3.0, 4.0], tol, None);
        assert_matches!(err, Some(InsertionError::DuplicateCoordinates { .. }));

        // Non-duplicate should return None.
        let no_err = tri.duplicate_coordinates_error(&[99.0, 99.0], tol, None);
        assert!(no_err.is_none());
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
            let vertices: Vec<Vertex<(), 3>> = base_coords
                .iter()
                .map(|c| {
                    crate::core::vertex::Vertex::<(), _>::try_new([
                        c[0] * scale,
                        c[1] * scale,
                        c[2] * scale,
                    ])
                    .unwrap()
                })
                .collect();
            let dt: DelaunayTriangulation<_, (), (), 3> =
                DelaunayTriangulation::try_new(&vertices).unwrap();
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

    /// Verify the mantissa-based epsilon selection and exercise the perturbation retry path
    /// with a near-degenerate simplex.
    #[test]
    fn test_perturbation_epsilon_selection_and_retry() {
        // Assert the supported scalar's mantissa-digits branch.
        assert_eq!(
            F64_MANTISSA_DIGITS, 53,
            "f64 should take the 1e-8 epsilon path"
        );

        // Build a 2D triangulation, then insert a point exactly on an existing edge.
        // This near-degenerate configuration exercises the full insert_transactional path
        // including epsilon_value computation.
        let initial_f64: Vec<Vertex<(), 2>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0_f64, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0_f64, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0_f64, 1.0]).unwrap(),
        ];
        let tds_f64 =
            Triangulation::<AdaptiveKernel<f64>, (), (), 2>::build_initial_simplex(&initial_f64)
                .unwrap();
        let mut tri_f64 = Triangulation::<AdaptiveKernel<f64>, (), (), 2>::new_with_tds(
            AdaptiveKernel::<f64>::new(),
            tds_f64,
        );

        let (outcome_f64, stats_f64) = insert_with_statistics(
            &mut tri_f64,
            crate::core::vertex::Vertex::<(), _>::try_new([0.5_f64, 0.0]).unwrap(),
            None,
            None,
        )
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
    fn perturbation_retry_repro_points_4d() -> [Point<4>; 20] {
        // Fixed adversarial insertion sequence captured from the former
        // randomized sweep (seed 4, index 19). The final insertion exhausts
        // perturbation retries in the current 4D path, so this keeps retry
        // coverage deterministic without looping over random seeds.
        [
            Point::try_new([
                0.660_063_804_566_304_3,
                3.139_352_812_821_116,
                1.460_437_437_858_557_2,
                1.683_976_950_416_514_7,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                2.451_966_162_957_145,
                9.547_229_335_697_903,
                3.306_128_696_560_687_5,
                -3.722_166_730_957_705_6,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -2.344_360_378_074_79,
                -2.755_831_029_562_339,
                -1.275_699_073_649_171_6,
                7.667_812_493_160_508,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -8.633_692_230_033_44,
                1.995_093_685_275_964_6,
                7.993_316_108_703_105,
                -3.310_780_098_197_376_7,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                9.710_410_828_147_591,
                -9.675_293_457_452_888,
                -7.169_080_272_753_141,
                5.405_946_111_675_925_5,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                2.266_246_031_487_613,
                2.481_673_939_102_995,
                3.039_413_140_674_462,
                4.441_464_307_622_285,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                2.565_731_492_709_954,
                8.916_218_617_699_3,
                -3.878_340_784_199_263_4,
                -9.518_720_806_139_726,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -2.067_801_258_479_087_2,
                -5.739_002_626_992_522,
                7.554_154_642_458_165,
                -2.983_334_995_469_171_2,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                7.592_645_474_686_005,
                -3.326_646_745_715_216,
                -3.259_537_116_123_248,
                -4.935_000_398_073_641,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -5.931_807_896_262_18,
                8.897_268_005_841_394,
                0.324_049_126_782_281_15,
                -8.328_532_028_712_647,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -8.182_644_118_410_867,
                5.373_925_359_941_506,
                -9.015_837_749_827_128,
                -1.703_973_344_007_208,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                1.455_467_619_488_706_2,
                9.869_985_381_801_74,
                8.605_618_759_378_327,
                -1.050_236_122_559_873_3,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -5.687_160_826_499_058,
                6.504_655_423_433_022,
                8.941_590_411_569_816,
                9.543_547_641_077_382,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                8.975_549_245_653_312,
                -8.089_655_037_805_944,
                9.936_284_142_216_682,
                -7.816_992_427_475_977,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                5.825_845_324_524_742,
                -7.639_141_597_632_388,
                1.549_524_653_880_336_4,
                4.563_088_344_949_309,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                7.387_141_055_690_918,
                6.194_972_387_680_284,
                -5.764_015_058_796_046,
                9.298_338_336_238_999,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -1.597_916_740_077_209_9,
                -4.938_008_036_006_716,
                7.414_979_546_687_874,
                -7.718_146_418_588_452,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -2.414_045_007_912_424_3,
                8.888_648_260_600_007,
                -5.859_329_894_512_815,
                3.268_096_825_406_147,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                -8.294_250_893_230_837,
                3.083_275_278_154_95,
                8.020_989_920_767_69,
                8.155_291_219_012_977,
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                6.718_748_825_685_814_6,
                -4.640_634_945_941_695,
                2.283_644_483_657_752_7,
                0.837_537_687_473_188_8,
            ])
            .expect("finite point coordinates"),
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
        let initial_vertices: Vec<Vertex<(), 4>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
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
        let retry_success_vertex = Vertex::try_new([0.2, 0.2, 0.2, 0.2]).unwrap();
        let (_outcome, retry_success_stats) =
            insert_with_statistics(&mut retry_success_tri, retry_success_vertex, None, None)
                .unwrap();
        let saw_retry = retry_success_stats.used_perturbation() && retry_success_stats.success();

        let mut exhaustion_tri: Triangulation<AdaptiveKernel<f64>, (), (), 4> =
            Triangulation::new_empty(AdaptiveKernel::new());
        let mut saw_exhausted_skip = false;

        for point in perturbation_retry_repro_points_4d() {
            let v = Vertex::from_validated_point(point, None);
            let (outcome, stats) =
                insert_with_statistics(&mut exhaustion_tri, v, None, None).unwrap();

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
            let v = Vertex::from_validated_point(point, None);
            let (_outcome, stats) = insert_transactional(
                &mut tri,
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
