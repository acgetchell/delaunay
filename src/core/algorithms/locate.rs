//! Point location algorithms for triangulations.
//!
//! Implements facet-walking point location for finding the cell containing
//! a query point in O(√n) to O(n^(1/D)) expected time.
//!
//! # Algorithm
//!
//! The facet walking algorithm starts from a hint cell (or arbitrary cell)
//! and walks toward the query point by repeatedly:
//! 1. Testing orientation of query point relative to each facet
//! 2. Crossing to the neighbor on the side containing the query point
//! 3. Repeating until the query point is inside the current cell
//!
//! # References
//!
//! - O. Devillers, S. Pion, and M. Teillaud, "Walking in a Triangulation",
//!   International Journal of Foundations of Computer Science, 2001.
//! - CGAL Triangulation_3 documentation

use crate::core::collections::{
    CavityBoundaryBuffer, CellKeyBuffer, CellSecondaryMap, FacetToCellsMap, FastHashMap,
    FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::tds::{CellKey, Tds, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::util::canonical_points::{sorted_cell_points, sorted_facet_points_with_extra};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use std::env;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Copy)]
struct ConflictDebugConfig {
    log_conflict: bool,
    progress_enabled: bool,
    progress_every: usize,
}

#[cfg(debug_assertions)]
fn conflict_debug_config() -> &'static ConflictDebugConfig {
    static CONFIG: OnceLock<ConflictDebugConfig> = OnceLock::new();

    CONFIG.get_or_init(|| ConflictDebugConfig {
        log_conflict: env::var_os("DELAUNAY_DEBUG_CONFLICT").is_some(),
        progress_enabled: env::var_os("DELAUNAY_DEBUG_CONFLICT_PROGRESS").is_some(),
        progress_every: env::var("DELAUNAY_DEBUG_CONFLICT_PROGRESS_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(5000),
    })
}

static RIDGE_FAN_DUMP_ENABLED: OnceLock<bool> = OnceLock::new();
static RIDGE_FAN_DUMP_EMITTED: AtomicBool = AtomicBool::new(false);

/// Returns whether a one-shot release-visible ridge-fan dump is enabled.
fn ridge_fan_dump_enabled() -> bool {
    *RIDGE_FAN_DUMP_ENABLED.get_or_init(|| env::var_os("DELAUNAY_DEBUG_RIDGE_FAN_ONCE").is_some())
}

/// Result of point location query.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::LocateResult;
/// use delaunay::core::tds::VertexKey;
/// use slotmap::KeyData;
///
/// let vertex = VertexKey::from(KeyData::from_ffi(2));
/// let result = LocateResult::OnVertex(vertex);
/// assert!(matches!(result, LocateResult::OnVertex(_)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocateResult {
    /// Point is strictly inside the cell
    InsideCell(CellKey),
    /// Point is on a facet between two cells
    OnFacet(CellKey, u8), // cell_key, facet_index
    /// Point is on an edge (lower-dimensional simplex)
    OnEdge(CellKey),
    /// Point is on a vertex
    OnVertex(VertexKey),
    /// Point is outside the convex hull
    Outside,
}

/// Error during point location.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::LocateError;
///
/// let err = LocateError::EmptyTriangulation;
/// assert!(matches!(err, LocateError::EmptyTriangulation));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum LocateError {
    /// Triangulation has no cells.
    #[error("Cannot locate in empty triangulation")]
    EmptyTriangulation,

    /// Cell reference is invalid.
    #[error("Invalid cell reference: {cell_key:?}")]
    InvalidCell {
        /// The invalid cell key.
        cell_key: CellKey,
    },

    /// Geometric predicate failed.
    #[error("Predicate error: {source}")]
    PredicateError {
        #[from]
        /// The underlying coordinate conversion error.
        source: CoordinateConversionError,
    },
}

/// Error during conflict region finding.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::ConflictError;
/// use delaunay::core::tds::CellKey;
/// use slotmap::KeyData;
///
/// let cell_key = CellKey::from(KeyData::from_ffi(5));
/// let err = ConflictError::InvalidStartCell { cell_key };
/// assert!(matches!(err, ConflictError::InvalidStartCell { .. }));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ConflictError {
    /// Starting cell is invalid
    #[error("Invalid starting cell: {cell_key:?}")]
    InvalidStartCell {
        /// The invalid cell key
        cell_key: CellKey,
    },

    /// Geometric predicate failed
    #[error("Predicate error: {source}")]
    PredicateError {
        #[from]
        /// The underlying coordinate conversion error
        source: CoordinateConversionError,
    },

    /// Failed to access required cell data (e.g., vertices) or build facet identifiers.
    ///
    /// This represents a *data-sourcing* failure attributable to a specific cell key:
    /// the key resolved but its vertex list, facet index, or derived identifier could
    /// not be produced. For invariant violations that are *not* about a specific cell
    /// (e.g., a `boundary_facets` index that must be in range by construction), use
    /// [`ConflictError::InternalInconsistency`] instead of fabricating a cell key.
    #[error("Failed to access required data for cell {cell_key:?}: {message}")]
    CellDataAccessFailed {
        /// The cell key for which required data could not be accessed.
        cell_key: CellKey,
        /// Human-readable details about what data could not be accessed.
        message: String,
    },

    /// Internal invariant violation during cavity-boundary extraction.
    ///
    /// This is raised when an invariant that must hold by construction does not —
    /// typically a `boundary_facets` or `RidgeInfo` index that is unconditionally
    /// valid in correct code. Debug builds catch these with `debug_assert!` so the
    /// error path is only reachable in release mode; returning it rather than
    /// panicking preserves the caller's transactional rollback guarantees.
    ///
    /// Orthogonality: this variant is distinct from
    /// [`ConflictError::CellDataAccessFailed`]. Use `CellDataAccessFailed` when
    /// a specific, real cell key is the subject of the failure; use
    /// `InternalInconsistency` when the failure is structural and has no such key.
    /// Treated as non-retryable by [`InsertionError::is_retryable`] because
    /// perturbing coordinates cannot resolve a logic error.
    ///
    /// The specific violation site is carried in [`InternalInconsistencySite`]
    /// as a typed payload so callers can pattern-match without parsing strings.
    ///
    /// [`InsertionError::is_retryable`]:
    ///     crate::core::algorithms::incremental_insertion::InsertionError::is_retryable
    #[error("Internal cavity-boundary inconsistency: {site}")]
    InternalInconsistency {
        /// Structured, typed description of the violated invariant — the index,
        /// counts, and slice lengths that exposed the failure.
        site: InternalInconsistencySite,
    },

    /// Non-manifold facet detected (facet shared by more than 2 conflict cells).
    #[error(
        "Non-manifold facet detected: facet {facet_hash:#x} shared by {cell_count} conflict cells (expected ≤2)"
    )]
    NonManifoldFacet {
        /// Hash of the facet's canonical vertex keys (sorted).
        facet_hash: u64,
        /// Number of conflict cells incident to this facet.
        cell_count: usize,
    },

    /// Ridge fan detected (many facets sharing same (D-2)-simplex).
    ///
    /// When a single conflict region contains multiple ridge fans,
    /// [`extract_cavity_boundary`] accumulates the removal candidates from every
    /// fan into `extra_cells` before returning, so a single cavity-reduction step
    /// can shrink all of them at once. In that case:
    ///
    /// - `facet_count` and `ridge_vertex_count` describe the **first** fan that
    ///   the boundary walk observed (a representative example, not an aggregate).
    /// - `extra_cells` contains the **union** of extra-cell candidates across all
    ///   detected fans in the conflict region (deduplicated).
    ///
    /// The error message reports the representative scalars; consult
    /// `extra_cells.len()` in traces when the conflict region is large enough to
    /// host several fans.
    #[error(
        "Ridge fan detected: {facet_count} facets share ridge with {ridge_vertex_count} vertices (indicates degenerate geometry requiring perturbation)"
    )]
    RidgeFan {
        /// Number of facets in the *first* fan encountered during the boundary
        /// walk. When several ridge fans are present in the same conflict region,
        /// this is a representative value, not the maximum or sum.
        facet_count: usize,
        /// Number of vertices in the shared ridge for the first fan encountered.
        ridge_vertex_count: usize,
        /// Deduplicated cell keys that contribute the *extra* (3rd, 4th, …)
        /// facets to one or more ridge fans in the conflict region. Removing
        /// these cells from the conflict region eliminates every currently
        /// detected ridge fan at once, enabling cavity insertion to proceed at
        /// the cost of leaving those cells temporarily non-Delaunay (the
        /// subsequent flip-repair pass restores the Delaunay property).
        extra_cells: Vec<CellKey>,
    },

    /// Cavity boundary is disconnected (multiple components).
    ///
    /// This indicates the conflict region is not a topological ball, which can happen
    /// in degenerate/co-spherical configurations when strict in-sphere classification
    /// produces a non-simply-connected conflict set. Treat as retryable degeneracy.
    #[error(
        "Disconnected cavity boundary: visited {visited} of {total} boundary facets (indicates degenerate geometry requiring perturbation)"
    )]
    DisconnectedBoundary {
        /// Number of boundary facets reachable from an arbitrary start facet.
        visited: usize,
        /// Total number of boundary facets.
        total: usize,
        /// Cell keys from the disconnected (unreachable) boundary component.
        /// Removing these cells from the conflict region makes the cavity boundary
        /// connected, enabling insertion to proceed (the cells are left temporarily
        /// non-Delaunay and fixed by the subsequent flip-repair pass).
        disconnected_cells: Vec<CellKey>,
    },

    /// Cavity boundary is not closed (a ridge is incident to only one boundary facet).
    ///
    /// For a valid cavity boundary (a closed (D-1)-manifold), each (D-2)-ridge must be
    /// shared by exactly 2 boundary facets. An incidence of 1 indicates an "open" boundary
    /// surface and is treated as retryable degeneracy.
    #[error(
        "Open cavity boundary: ridge with {ridge_vertex_count} vertices is incident to {facet_count} boundary facets (expected 2; indicates degenerate geometry requiring perturbation)"
    )]
    OpenBoundary {
        /// Number of boundary facets incident to the ridge.
        facet_count: usize,
        /// Number of vertices in the ridge.
        ridge_vertex_count: usize,
        /// The conflict-region cell that contributes the dangling (open) boundary facet.
        /// Removing this cell from the conflict region closes the open ridge.
        open_cell: CellKey,
    },
}

/// Typed site of a [`ConflictError::InternalInconsistency`] violation.
///
/// Each variant describes one specific invariant that `extract_cavity_boundary`
/// maintains by construction. The fields carry the indices, counts, and slice
/// lengths that would normally appear in a `format!(...)` context string, but
/// keep them as typed data so callers can `matches!` / `assert_eq!` on them and
/// so future localized formatting does not need to reparse prose.
///
/// These paths are unreachable in debug builds — the corresponding
/// `debug_assert!` invariants fire there — and are guarded only to preserve
/// transactional-rollback semantics in release builds.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::{ConflictError, InternalInconsistencySite};
///
/// let site = InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
///     index: 7,
///     boundary_facets_len: 5,
///     extra_facets_len: 3,
/// };
/// let err = ConflictError::InternalInconsistency { site: site.clone() };
/// assert!(matches!(
///     err,
///     ConflictError::InternalInconsistency {
///         site: InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds { .. }
///     }
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InternalInconsistencySite {
    /// A `RidgeFan` `extra_facets` entry references an index outside the
    /// `boundary_facets` slice that populated it during the same traversal.
    RidgeFanExtraFacetOutOfBounds {
        /// Offending `extra_facets` value that indexed outside `boundary_facets`.
        index: usize,
        /// Length of the boundary-facet slice at the time of the violation.
        boundary_facets_len: usize,
        /// Total number of entries in the offending `extra_facets` list.
        extra_facets_len: usize,
    },

    /// An `OpenBoundary` `first_facet` index is out of range for
    /// `boundary_facets` even though the two are written together.
    OpenBoundaryMissingFirstFacet {
        /// Out-of-range `first_facet` index that should have resolved to a boundary facet.
        first_facet: usize,
        /// Length of the boundary-facet slice at the time of the violation.
        boundary_facets_len: usize,
        /// Observed `facet_count` for the violating ridge.
        facet_count: usize,
        /// Observed ridge-vertex count for the violating ridge.
        ridge_vertex_count: usize,
    },

    /// `RidgeInfo::second_facet` is `None` while `facet_count == 2`, even
    /// though the two fields are written together when a second incident
    /// facet is added.
    RidgeInfoMissingSecondFacet {
        /// `first_facet` index that was recorded alongside the missing `second_facet`.
        first_facet: usize,
        /// Length of the boundary-facet slice at the time of the violation.
        boundary_facets_len: usize,
        /// Observed ridge-vertex count for the violating ridge.
        ridge_vertex_count: usize,
    },
}

impl fmt::Display for InternalInconsistencySite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RidgeFanExtraFacetOutOfBounds {
                index,
                boundary_facets_len,
                extra_facets_len,
            } => write!(
                f,
                "RidgeFan extra_facets index {index} out of bounds \
                 (boundary_facets.len()={boundary_facets_len}, extra_facets_len={extra_facets_len})"
            ),
            Self::OpenBoundaryMissingFirstFacet {
                first_facet,
                boundary_facets_len,
                facet_count,
                ridge_vertex_count,
            } => write!(
                f,
                "OpenBoundary missing first_facet index {first_facet} \
                 (boundary_facets.len()={boundary_facets_len}, facet_count={facet_count}, \
                 ridge_vertex_count={ridge_vertex_count})"
            ),
            Self::RidgeInfoMissingSecondFacet {
                first_facet,
                boundary_facets_len,
                ridge_vertex_count,
            } => write!(
                f,
                "RidgeInfo missing second_facet when facet_count == 2 \
                 (first_facet={first_facet}, boundary_facets_len={boundary_facets_len}, \
                 ridge_vertex_count={ridge_vertex_count})"
            ),
        }
    }
}

/// Ridge incidence information used for cavity-boundary validation.
#[derive(Debug, Clone)]
struct RidgeInfo {
    ridge_vertex_count: usize,
    /// Canonical vertex keys for the shared ridge.
    ridge_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    facet_count: usize,
    first_facet: usize,
    second_facet: Option<usize>,
    /// Indices (into `boundary_facets`) of the 3rd, 4th, … facets in the fan.
    /// Populated only when `facet_count >= 3`.
    extra_facets: Vec<usize>,
}

fn format_vertex_refs<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    vertex_keys: &[VertexKey],
) -> String
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    vertex_keys
        .iter()
        .map(|&vertex_key| {
            let uuid = tds.get_vertex_by_key(vertex_key).map_or_else(
                || String::from("missing"),
                |vertex| vertex.uuid().to_string(),
            );
            format!("{vertex_key:?}/{uuid}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_facet_vertices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    handle: FacetHandle,
) -> String
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(handle.cell_key()) else {
        return String::from("<missing-cell>");
    };

    let facet_index = usize::from(handle.facet_index());
    let vertex_keys: Vec<VertexKey> = cell
        .vertices()
        .iter()
        .enumerate()
        .filter_map(|(idx, &vertex_key)| (idx != facet_index).then_some(vertex_key))
        .collect();
    format_vertex_refs(tds, &vertex_keys)
}

fn format_cell_vertices<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>, cell_key: CellKey) -> String
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(cell_key) else {
        return String::from("<missing-cell>");
    };
    format_vertex_refs(tds, cell.vertices())
}

/// Emits a compact one-shot snapshot of the first detected ridge fan in a run.
///
/// Enabled via `DELAUNAY_DEBUG_RIDGE_FAN_ONCE`. Output is routed through
/// `tracing::debug!` so it respects the configured tracing subscriber;
/// callers that want these lines during a release-mode run should set
/// `RUST_LOG=debug` (or the matching filter in the large-scale debug harness).
///
/// The snapshot captures the shared ridge vertices, the participating boundary
/// facets, and the extra cells that cavity reduction would remove.
fn log_first_ridge_fan_dump<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    conflict_cells: &CellKeyBuffer,
    boundary_facets: &CavityBoundaryBuffer,
    info: &RidgeInfo,
    extra_cells: &[CellKey],
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if !ridge_fan_dump_enabled() || RIDGE_FAN_DUMP_EMITTED.swap(true, Ordering::Relaxed) {
        return;
    }

    let mut participating_indices = Vec::with_capacity(2 + info.extra_facets.len());
    participating_indices.push(info.first_facet);
    if let Some(second_facet) = info.second_facet {
        participating_indices.push(second_facet);
    }
    participating_indices.extend(info.extra_facets.iter().copied());

    let conflict_preview: Vec<CellKey> = conflict_cells.iter().copied().take(16).collect();
    let ridge_vertices = format_vertex_refs(tds, info.ridge_vertices.as_slice());

    let participating_facets: Vec<String> = participating_indices
        .iter()
        .copied()
        .map(|boundary_index| {
            boundary_facets.get(boundary_index).copied().map_or_else(
                || format!("boundary_idx={boundary_index} <missing-boundary-facet>"),
                |handle| {
                    format!(
                        "boundary_idx={} cell={:?} facet_index={} vertices=[{}]",
                        boundary_index,
                        handle.cell_key(),
                        handle.facet_index(),
                        format_facet_vertices(tds, handle),
                    )
                },
            )
        })
        .collect();

    let extra_cell_details: Vec<String> = extra_cells
        .iter()
        .copied()
        .map(|cell_key| {
            format!(
                "cell={cell_key:?} vertices=[{}]",
                format_cell_vertices(tds, cell_key)
            )
        })
        .collect();

    tracing::debug!(
        target: "delaunay::ridge_fan_dump",
        D,
        conflict_cells = conflict_cells.len(),
        boundary_facets = boundary_facets.len(),
        facet_count = info.facet_count,
        ridge_vertex_count = info.ridge_vertex_count,
        extra_cells = ?extra_cells,
        conflict_preview = ?conflict_preview,
        ridge_vertices = %ridge_vertices,
        participating_boundary_indices = ?participating_indices,
        participating_facets = ?participating_facets,
        extra_cell_details = ?extra_cell_details,
        "ridge-fan-dump: first detected ridge fan"
    );
}

fn collect_ridge_fan_extra_cells(
    boundary_facets: &CavityBoundaryBuffer,
    info: &RidgeInfo,
) -> Result<Vec<CellKey>, ConflictError> {
    debug_assert!(
        info.extra_facets
            .iter()
            .all(|&fi| fi < boundary_facets.len()),
        "RidgeFan extra_facets index out of bounds: extra_facets={:?}, boundary_facets.len()={}",
        info.extra_facets,
        boundary_facets.len(),
    );

    // Deduplicate: multiple extra facets can come from the same cell. Downstream code
    // expects unique cell keys when shrinking the conflict region.
    let mut seen = FastHashSet::<CellKey>::default();
    let mut extra_cells: Vec<CellKey> = Vec::new();
    for &fi in &info.extra_facets {
        // Every entry in `info.extra_facets` is a `boundary_facets` index written by the
        // same traversal that populated `boundary_facets`, so any out-of-range value
        // represents an internal invariant violation rather than a data-access failure
        // attributable to a real cell. Report it as such so the error message is truthful
        // (no fabricated `CellKey::default()` placeholder) and stays non-retryable.
        let ck = boundary_facets
            .get(fi)
            .ok_or_else(|| ConflictError::InternalInconsistency {
                site: InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
                    index: fi,
                    boundary_facets_len: boundary_facets.len(),
                    extra_facets_len: info.extra_facets.len(),
                },
            })?
            .cell_key();
        if seen.insert(ck) {
            extra_cells.push(ck);
        }
    }
    Ok(extra_cells)
}

/// Indicates why facet-walking fell back to a brute-force scan.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::LocateFallbackReason;
///
/// let reason = LocateFallbackReason::StepLimit;
/// assert_eq!(reason, LocateFallbackReason::StepLimit);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocateFallbackReason {
    /// The facet-walking traversal revisited a previously seen cell.
    CycleDetected,
    /// The facet-walking traversal exceeded the maximum step budget.
    StepLimit,
}

/// Information about a facet-walking fallback.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::{LocateFallback, LocateFallbackReason};
///
/// let fallback = LocateFallback {
///     reason: LocateFallbackReason::CycleDetected,
///     steps: 12,
/// };
/// assert_eq!(fallback.steps, 12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateFallback {
    /// Why the fallback was triggered.
    pub reason: LocateFallbackReason,
    /// Number of facet-walking steps taken before falling back.
    pub steps: usize,
}

/// Statistics describing how point location was performed.
///
/// The primary purpose of this type is **observability**: callers can detect when the fast
/// facet-walking path failed to make progress (cycle/step-limit) and a scan fallback was used.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::LocateStats;
/// use delaunay::core::tds::CellKey;
/// use slotmap::KeyData;
///
/// let stats = LocateStats {
///     start_cell: CellKey::from(KeyData::from_ffi(9)),
///     used_hint: false,
///     walk_steps: 0,
///     fallback: None,
/// };
/// assert!(!stats.fell_back_to_scan());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateStats {
    /// The start cell used for the facet-walking phase.
    pub start_cell: CellKey,
    /// Whether the caller-provided hint was used as the start cell.
    pub used_hint: bool,
    /// Number of facet-walking steps taken.
    pub walk_steps: usize,
    /// Fallback information, if facet walking did not converge and a scan was used.
    pub fallback: Option<LocateFallback>,
}

impl LocateStats {
    /// Returns `true` if point location fell back to a brute-force scan.
    #[must_use]
    pub const fn fell_back_to_scan(&self) -> bool {
        self.fallback.is_some()
    }
}

/// Locate a point in the triangulation using facet walking (correctness-first).
///
/// This function attempts a fast facet-walking traversal starting from `hint` (when provided).
/// If a cycle or step limit is detected, it falls back to a brute-force scan to guarantee
/// termination.
///
/// If you need to detect when the fast path did not behave (cycle/step limit), use
/// [`locate_with_stats`].
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `kernel` - Geometric kernel for orientation tests
/// * `point` - Query point to locate
/// * `hint` - Optional starting cell (uses arbitrary cell if None)
///
/// # Returns
///
/// Returns `LocateResult` indicating where the point is located.
///
/// # Errors
///
/// Returns `LocateError` if:
/// - The triangulation is empty
/// - Cell references are invalid
/// - Geometric predicates fail
///
/// # Examples
///
/// Basic point location in a 4D simplex:
///
/// ```rust
/// use delaunay::prelude::algorithms::*;
/// use delaunay::prelude::query::*;
///
/// // Create a 4D simplex (5 vertices)
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let kernel = FastKernel::<f64>::new();
///
/// // Point inside the 4-simplex
/// let inside_point = Point::new([0.2, 0.2, 0.2, 0.2]);
/// match locate(dt.tds(), &kernel, &inside_point, None) {
///     Ok(LocateResult::InsideCell(cell_key)) => {
///         assert!(dt.tds().contains_cell(cell_key));
///     }
///     _ => panic!("Expected point to be inside a cell"),
/// }
///
/// // Point outside the convex hull
/// let outside_point = Point::new([2.0, 2.0, 2.0, 2.0]);
/// match locate(dt.tds(), &kernel, &outside_point, None) {
///     Ok(LocateResult::Outside) => { /* Expected */ }
///     _ => panic!("Expected point to be outside convex hull"),
/// }
/// ```
///
/// Using a hint cell for faster location:
///
/// ```rust
/// use delaunay::geometry::kernel::RobustKernel;
/// use delaunay::prelude::algorithms::*;
/// use delaunay::prelude::query::*;
///
/// // Create a 4D simplex
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let kernel = RobustKernel::<f64>::default();
///
/// // Get a cell to use as hint (spatially close to query point)
/// let hint_cell = dt.tds().cell_keys().next().unwrap();
/// let query_point = Point::new([0.15, 0.15, 0.15, 0.15]);
///
/// match locate(dt.tds(), &kernel, &query_point, Some(hint_cell)) {
///     Ok(LocateResult::InsideCell(_)) => { /* Success */ }
///     _ => panic!("Expected to find cell"),
/// }
/// ```
pub fn locate<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    hint: Option<CellKey>,
) -> Result<LocateResult, LocateError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    locate_with_stats(tds, kernel, point, hint).map(|(result, _stats)| result)
}

/// Locate a point and return diagnostics about the facet-walking traversal.
///
/// This performs the same algorithm as [`locate`], but returns a [`LocateStats`] struct
/// that indicates whether the caller-provided hint was used and whether a scan fallback was
/// required (cycle detected / step limit).
///
/// # Errors
///
/// Returns `LocateError` only for structural/predicate failures. Cycles and step limits are not
/// treated as errors; they trigger the scan fallback and are recorded in the returned stats.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::*;
/// use delaunay::prelude::query::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
/// let kernel = FastKernel::<f64>::new();
///
/// let query_point = Point::new([0.3, 0.3]);
/// let (_result, stats) = locate_with_stats(dt.tds(), &kernel, &query_point, None).unwrap();
///
/// // In well-conditioned cases, the facet-walk should converge without falling back.
/// assert!(!stats.fell_back_to_scan());
/// ```
pub fn locate_with_stats<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    hint: Option<CellKey>,
) -> Result<(LocateResult, LocateStats), LocateError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    const MAX_STEPS: usize = 10000;

    if tds.number_of_cells() == 0 {
        return Err(LocateError::EmptyTriangulation);
    }

    let (start_cell, used_hint) = match hint {
        Some(key) if tds.contains_cell(key) => (key, true),
        _ => (
            tds.cell_keys()
                .next()
                .ok_or(LocateError::EmptyTriangulation)?,
            false,
        ),
    };

    let mut stats = LocateStats {
        start_cell,
        used_hint,
        walk_steps: 0,
        fallback: None,
    };

    let mut current_cell = start_cell;
    let mut visited: FastHashSet<CellKey> = FastHashSet::default();

    for step in 0..MAX_STEPS {
        stats.walk_steps = step + 1;

        if !visited.insert(current_cell) {
            stats.fallback = Some(LocateFallback {
                reason: LocateFallbackReason::CycleDetected,
                steps: stats.walk_steps,
            });
            let result = locate_by_scan(tds, kernel, point)?;
            return Ok((result, stats));
        }

        let cell = tds.get_cell(current_cell).ok_or(LocateError::InvalidCell {
            cell_key: current_cell,
        })?;

        let facet_count = cell.number_of_vertices();
        let mut found_outside_facet = false;

        for facet_idx in 0..facet_count {
            if is_point_outside_facet(tds, kernel, current_cell, facet_idx, point)? == Some(true) {
                if let Some(neighbor_key) = cell
                    .neighbors()
                    .and_then(|neighbors| neighbors.get(facet_idx))
                    .and_then(|&opt_key| opt_key)
                {
                    current_cell = neighbor_key;
                    found_outside_facet = true;
                    break;
                }
                return Ok((LocateResult::Outside, stats));
            }
        }

        if !found_outside_facet {
            return Ok((LocateResult::InsideCell(current_cell), stats));
        }
    }

    stats.fallback = Some(LocateFallback {
        reason: LocateFallbackReason::StepLimit,
        steps: stats.walk_steps,
    });
    let result = locate_by_scan(tds, kernel, point)?;
    Ok((result, stats))
}

pub(crate) fn locate_by_scan<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
) -> Result<LocateResult, LocateError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    for (cell_key, cell) in tds.cells() {
        let mut found_outside_facet = false;
        let facet_count = cell.number_of_vertices();

        for facet_idx in 0..facet_count {
            if is_point_outside_facet(tds, kernel, cell_key, facet_idx, point)? == Some(true) {
                found_outside_facet = true;
                break;
            }
        }

        if !found_outside_facet {
            return Ok(LocateResult::InsideCell(cell_key));
        }
    }

    Ok(LocateResult::Outside)
}

/// Test if a point is on the outside of a cell's facet.
///
/// A point is "outside" a facet if walking through that facet moves us closer
/// to the query point. This is determined by comparing orientations with a
/// consistent vertex ordering.
///
/// # Invariant Dependency
///
/// **CRITICAL**: This function relies on the triangulation's topological invariant:
/// - `facet_idx` refers to both the facet AND the vertex opposite to that facet
/// - `cell.vertices()[facet_idx]` is the vertex opposite the facet
/// - The facet consists of all vertices EXCEPT `vertices[facet_idx]`
/// - This invariant is documented in [`Cell`](crate::core::cell::Cell) and enforced by
///   [`Tds::assign_neighbors`](crate::core::tds::Tds::assign_neighbors).
///
/// It is validated as part of Level 2 structural validation via
/// [`Tds::is_valid`](crate::core::tds::Tds::is_valid)
/// (or cumulatively via [`Tds::validate`](crate::core::tds::Tds::validate)).
///
/// This correspondence is essential for the canonical ordering used in orientation tests.
/// If this invariant is violated, point location will produce incorrect results.
///
/// Returns:
/// - `Some(true)` if point is outside (should cross facet)
/// - `Some(false)` if point is inside (should not cross facet)  
/// - `None` for degenerate cases
fn is_point_outside_facet<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    cell_key: CellKey,
    facet_idx: usize,
    query_point: &Point<K::Scalar, D>,
) -> Result<Option<bool>, LocateError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    let cell = tds
        .get_cell(cell_key)
        .ok_or(LocateError::InvalidCell { cell_key })?;

    let cell_vertex_keys = cell.vertices();
    if cell_vertex_keys.len() != D + 1 {
        return Ok(None); // Degenerate cell
    }

    if facet_idx > D {
        return Ok(None); // Out-of-range facet index
    }

    // The vertex at facet_idx is opposite the facet
    let opposite_key = cell_vertex_keys[facet_idx];
    let Some(opposite_point) = tds.get_vertex_by_key(opposite_key).map(|v| *v.point()) else {
        return Ok(None); // Unresolvable vertex → degenerate cell
    };

    // Facet keys: all vertex keys except the one at facet_idx
    let facet_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = cell_vertex_keys
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != facet_idx)
        .map(|(_, &vk)| vk)
        .collect();

    // Build facet simplex + opposite vertex in canonical key order.
    // If any facet vertex is unresolvable, treat as degenerate.
    let Some(canonical_cell) = sorted_facet_points_with_extra(tds, &facet_keys, opposite_point)
    else {
        return Ok(None);
    };

    let cell_orientation = kernel.orientation(&canonical_cell)?;

    // Build query simplex by reusing the canonical facet ordering:
    // replace the last element (opposite → query point).
    let mut query_simplex = canonical_cell;
    let last = query_simplex.len() - 1;
    query_simplex[last] = *query_point;

    let query_orientation = kernel.orientation(&query_simplex)?;

    // If orientations differ, query point and opposite vertex are on
    // opposite sides of the facet → point is "outside" (should cross)
    // If orientations match, they're on the same side → point is "inside" (should not cross)
    Ok(Some(cell_orientation * query_orientation < 0))
}

/// Find all cells whose circumspheres contain the query point (conflict region).
///
/// Uses BFS traversal starting from a located cell to find all cells in conflict.
/// A cell is in conflict if the query point lies inside **or on** its circumsphere.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `kernel` - Geometric kernel for `in_sphere` tests
/// * `point` - Query point to test
/// * `start_cell` - Starting cell (typically from `locate()`)
///
/// # Returns
///
/// Returns a buffer of all `CellKey`s whose circumspheres contain the point.
///
/// # Errors
///
/// Returns `ConflictError` if:
/// - The starting cell is invalid
/// - Geometric predicates fail
/// - Cannot retrieve cell vertices
///
/// # Algorithm
///
/// 1. Start BFS from the located cell
/// 2. For each cell, test `kernel.in_sphere()`
/// 3. If point is inside or on circumsphere (sign >= 0), add to conflict region
/// 4. Expand search to neighbors of conflicting cells
/// 5. Track visited cells with `CellSecondaryMap` for O(1) lookups
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::{locate, find_conflict_region, LocateResult};
/// use delaunay::prelude::DelaunayTriangulation;
/// use delaunay::geometry::kernel::FastKernel;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::vertex;
///
/// // Create a 4D simplex (5 vertices forming a 4-simplex)
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let kernel = FastKernel::<f64>::new();
/// // Point inside the 4-simplex
/// let query_point = Point::new([0.2, 0.2, 0.2, 0.2]);
///
/// // First locate the point
/// let location = locate(dt.tds(), &kernel, &query_point, None).unwrap();
/// if let LocateResult::InsideCell(cell_key) = location {
///     // Find all cells whose circumspheres contain the point
///     let conflict_cells = find_conflict_region(dt.tds(), &kernel, &query_point, cell_key).unwrap();
///     assert_eq!(conflict_cells.len(), 1); // Single 4-simplex contains the point
/// }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "function is long due to complex locate logic and should be split when refactoring"
)]
pub fn find_conflict_region<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    start_cell: CellKey,
) -> Result<CellKeyBuffer, ConflictError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    #[cfg(debug_assertions)]
    let debug_config = conflict_debug_config();
    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();
    #[cfg(debug_assertions)]
    let mut visited_count = 0usize;
    #[cfg(debug_assertions)]
    let mut conflict_count = 0usize;
    #[cfg(debug_assertions)]
    let mut neighbor_enqueued = 0usize;

    // Validate start cell exists
    if !tds.contains_cell(start_cell) {
        return Err(ConflictError::InvalidStartCell {
            cell_key: start_cell,
        });
    }

    // Result buffer for conflicting cells
    let mut conflict_cells = CellKeyBuffer::new();

    // BFS work queue
    let mut queue = CellKeyBuffer::new();
    queue.push(start_cell);

    // Track visited cells with SparseSecondaryMap (idiomatic for SlotMap)
    let mut visited = CellSecondaryMap::new();

    while let Some(cell_key) = queue.pop() {
        // Skip if already visited
        if visited.contains_key(cell_key) {
            continue;
        }
        visited.insert(cell_key, ());

        #[cfg(debug_assertions)]
        {
            visited_count = visited_count.saturating_add(1);
            if debug_config.progress_enabled
                && visited_count.is_multiple_of(debug_config.progress_every)
            {
                tracing::debug!(
                    visited_count,
                    conflict_count,
                    queue_len = queue.len(),
                    neighbor_enqueued,
                    elapsed = ?start_time.elapsed(),
                    "find_conflict_region: progress"
                );
            }
        }

        // Get cell vertices for in_sphere test
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| ConflictError::CellDataAccessFailed {
                cell_key,
                message: "Cell vanished during BFS traversal".to_string(),
            })?;

        // Collect cell vertex points in canonical VertexKey order for consistent
        // SoS perturbation priority.
        let simplex_points =
            sorted_cell_points(tds, cell).ok_or_else(|| ConflictError::CellDataAccessFailed {
                cell_key,
                message: format!("Failed to resolve all {} cell vertices", D + 1),
            })?;

        if simplex_points.len() != D + 1 {
            return Err(ConflictError::CellDataAccessFailed {
                cell_key,
                message: format!("Expected {} vertices, got {}", D + 1, simplex_points.len()),
            });
        }

        #[cfg(debug_assertions)]
        if debug_config.log_conflict {
            tracing::debug!(
                cell_key = ?cell_key,
                vertex_keys = ?cell.vertices(),
                simplex_len = simplex_points.len(),
                query_point = ?point,
                "find_conflict_region: in_sphere input"
            );
        }

        // Test if point is inside/on circumsphere
        let sign = match kernel.in_sphere(&simplex_points, point) {
            Ok(value) => value,
            Err(err) => {
                #[cfg(debug_assertions)]
                if debug_config.log_conflict {
                    tracing::debug!(
                        cell_key = ?cell_key,
                        vertex_keys = ?cell.vertices(),
                        query_point = ?point,
                        error = ?err,
                        "find_conflict_region: in_sphere failed"
                    );
                }
                return Err(err.into());
            }
        };

        #[cfg(debug_assertions)]
        if debug_config.log_conflict {
            tracing::debug!(
                cell_key = ?cell_key,
                sign,
                in_conflict = sign >= 0,
                "find_conflict_region: in_sphere classification"
            );
        }

        if sign >= 0 {
            // Point is inside or on circumsphere - cell is in conflict
            conflict_cells.push(cell_key);

            #[cfg(debug_assertions)]
            {
                conflict_count = conflict_count.saturating_add(1);
            }

            // Add neighbors to queue for exploration
            if let Some(neighbors) = cell.neighbors() {
                for &neighbor_opt in neighbors {
                    if let Some(neighbor_key) = neighbor_opt
                        && !visited.contains_key(neighbor_key)
                    {
                        queue.push(neighbor_key);
                        #[cfg(debug_assertions)]
                        {
                            neighbor_enqueued = neighbor_enqueued.saturating_add(1);
                        }
                    }
                }
            }
        } else {
            // Cell is NOT in conflict (sign < 0): BFS boundary.
            // Log boundary cells so investigators can see exactly where
            // and why the BFS stopped expanding.
            #[cfg(debug_assertions)]
            if debug_config.log_conflict {
                let neighbor_keys: SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    cell.neighbors()
                        .map(|ns| ns.iter().copied().collect())
                        .unwrap_or_default();
                tracing::debug!(
                    cell_key = ?cell_key,
                    vertex_keys = ?cell.vertices(),
                    sign,
                    neighbors = ?neighbor_keys,
                    "find_conflict_region: BFS boundary (non-conflict)"
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    if debug_config.progress_enabled || debug_config.log_conflict {
        tracing::debug!(
            visited_count,
            conflict_cells = conflict_cells.len(),
            neighbor_enqueued,
            elapsed = ?start_time.elapsed(),
            "find_conflict_region: summary"
        );
    }

    Ok(conflict_cells)
}

/// Verify that a BFS-found conflict region is complete by brute-force scanning all cells.
///
/// This debug-only function compares the conflict region found by BFS traversal against
/// a full scan of every cell in the TDS using insphere tests. Any cell that the BFS missed
/// (i.e., the point is inside its circumsphere but the cell was not found by BFS) is logged
/// as a "missed" cell.
///
/// Activated by setting `DELAUNAY_DEBUG_CONFLICT_VERIFY=1`.
///
/// Returns the number of missed cells (0 means the BFS result is complete).
#[cfg(debug_assertions)]
pub fn verify_conflict_region_completeness<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    bfs_conflict_cells: &CellKeyBuffer,
) -> usize
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    let bfs_set: FastHashSet<CellKey> = bfs_conflict_cells.iter().copied().collect();
    let mut missed_count = 0usize;
    let mut brute_force_count = 0usize;
    let mut malformed_cells = 0usize;
    let mut predicate_errors = 0usize;

    for (cell_key, cell) in tds.cells() {
        let Some(simplex_points) = sorted_cell_points(tds, cell) else {
            malformed_cells += 1;
            tracing::debug!(
                cell_key = ?cell_key,
                vertex_keys = ?cell.vertices(),
                "verify_conflict_region: skipping malformed cell (sorted_cell_points returned None)"
            );
            continue;
        };
        if simplex_points.len() != D + 1 {
            malformed_cells += 1;
            continue;
        }

        let Ok(sign) = kernel.in_sphere(&simplex_points, point) else {
            predicate_errors += 1;
            tracing::debug!(
                cell_key = ?cell_key,
                vertex_keys = ?cell.vertices(),
                "verify_conflict_region: in_sphere predicate failed"
            );
            continue;
        };

        if sign >= 0 {
            brute_force_count += 1;
            if !bfs_set.contains(&cell_key) {
                missed_count += 1;

                // Reachability analysis: determine WHY BFS missed this cell.
                // Check if any TDS neighbor of the missed cell is in the BFS
                // conflict set.  This distinguishes two root causes:
                //   - Reachable: a neighbor IS in bfs_set, so BFS reached a
                //     neighbor but an intermediate insphere test rejected it
                //   - Unreachable: NO neighbors are in bfs_set, indicating
                //     broken neighbor pointers or a disconnected pocket
                let (neighbor_in_bfs, neighbor_total, neighbor_none) =
                    cell.neighbors().map_or((0, 0, 0), |neighbors| {
                        let total = neighbors.len();
                        let none_count = neighbors.iter().filter(|n| n.is_none()).count();
                        let in_bfs = neighbors
                            .iter()
                            .filter_map(|n| *n)
                            .filter(|nk| bfs_set.contains(nk))
                            .count();
                        (in_bfs, total, none_count)
                    });

                let reachability = if neighbor_in_bfs > 0 {
                    "REACHABLE_BUT_REJECTED"
                } else {
                    "UNREACHABLE"
                };

                tracing::warn!(
                    cell_key = ?cell_key,
                    vertex_keys = ?cell.vertices(),
                    sign,
                    reachability,
                    neighbor_in_bfs,
                    neighbor_total,
                    neighbor_none,
                    bfs_conflict_len = bfs_conflict_cells.len(),
                    brute_force_conflict_so_far = brute_force_count,
                    "verify_conflict_region: BFS MISSED conflicting cell"
                );
            }
        }
    }

    if missed_count > 0 || malformed_cells > 0 || predicate_errors > 0 {
        tracing::warn!(
            bfs_conflict = bfs_conflict_cells.len(),
            brute_force_conflict = brute_force_count,
            missed = missed_count,
            malformed_cells,
            predicate_errors,
            query_point = ?point,
            "verify_conflict_region: INCOMPLETE — missed cells or evaluation failures"
        );
    } else {
        tracing::debug!(
            bfs_conflict = bfs_conflict_cells.len(),
            brute_force_conflict = brute_force_count,
            "verify_conflict_region: conflict region is COMPLETE"
        );
    }

    missed_count
}

/// Extract boundary facets of a conflict region (cavity).
///
/// Finds all facets where exactly one adjacent cell is in the conflict region.
/// These boundary facets form the surface that will be connected to the new point.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `conflict_cells` - Buffer of cell keys in the conflict region
///
/// # Returns
///
/// Returns a buffer of `FacetHandle`s representing the cavity boundary.
/// Each facet is oriented such that its `cell_key` is in the conflict region.
///
/// # Errors
///
/// Returns `ConflictError` if:
/// - A conflict cell cannot be retrieved from the TDS
/// - Cell neighbor data is inconsistent
///
/// # Algorithm
///
/// 1. Convert `conflict_cells` to a `FastHashSet` for O(1) lookups
/// 2. For each cell in the conflict region:
///    - Iterate through all facets (opposite each vertex)
///    - Check if the neighbor across that facet is also in conflict
///    - If neighbor is NOT in conflict (or is None/hull), it's a boundary facet
/// 3. Return all boundary facets as `FacetHandle`s
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::extract_cavity_boundary;
/// use delaunay::core::collections::CellKeyBuffer;
/// use delaunay::core::tds::Tds;
///
/// let tds: Tds<f64, (), (), 3> = Tds::empty();
/// let boundary = extract_cavity_boundary(&tds, &CellKeyBuffer::new()).unwrap();
/// assert!(boundary.is_empty());
/// ```
///
///
/// ```rust
/// use delaunay::core::algorithms::locate::{locate, find_conflict_region, extract_cavity_boundary, LocateResult};
/// use delaunay::prelude::DelaunayTriangulation;
/// use delaunay::geometry::kernel::FastKernel;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::vertex;
///
/// // Create a 4D simplex
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let kernel = FastKernel::<f64>::new();
/// let query_point = Point::new([0.2, 0.2, 0.2, 0.2]);
///
/// // Locate and find conflict region
/// let location = locate(dt.tds(), &kernel, &query_point, None).unwrap();
/// if let LocateResult::InsideCell(cell_key) = location {
///     let conflict_cells = find_conflict_region(dt.tds(), &kernel, &query_point, cell_key).unwrap();
///     
///     // Extract cavity boundary
///     let boundary_facets = extract_cavity_boundary(dt.tds(), &conflict_cells).unwrap();
///     
///     // For a single 4-simplex, all 5 facets are on the boundary (convex hull)
///     assert_eq!(boundary_facets.len(), 5);
/// }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Long function; keep boundary extraction logic in one place for clarity"
)]
pub fn extract_cavity_boundary<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    conflict_cells: &CellKeyBuffer,
) -> Result<CavityBoundaryBuffer, ConflictError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Empty conflict region => empty boundary
    if conflict_cells.is_empty() {
        return Ok(CavityBoundaryBuffer::new());
    }

    #[cfg(debug_assertions)]
    let detail_enabled = env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();
    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();
    #[cfg(debug_assertions)]
    let mut boundary_facet_count = 0usize;
    #[cfg(debug_assertions)]
    let mut internal_facet_count = 0usize;

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            conflict_cells = conflict_cells.len(),
            "extract_cavity_boundary: start"
        );
    }

    // IMPORTANT:
    // We intentionally do NOT rely on neighbor pointers to classify boundary facets here.
    //
    // Neighbor pointers can be temporarily incomplete during incremental updates (e.g., after
    // cell removal or before a full neighbor repair). If we rely on `cell.neighbors()` and a
    // shared facet between two conflict cells is missing a neighbor pointer, that facet will be
    // misclassified as a boundary facet. This can introduce internal boundary components and
    // break Level-3 Euler/topology validation (observed as χ=2 for Ball(3)).
    //
    // Instead, classify facets purely by facet incidence *within the conflict region*:
    // - A facet is on the cavity boundary iff it is incident to exactly 1 conflict cell.
    let conflict_set: FastHashSet<CellKey> = conflict_cells.iter().copied().collect();

    // facet_hash -> all facets in the conflict region that contain the facet
    let mut facet_to_conflict: FacetToCellsMap = FacetToCellsMap::default();

    // facet_hash -> canonical vertex keys (sorted, excluding the opposite vertex)
    // Cached so ridge analysis doesn't have to rebuild facet vertex sets from cells.
    let mut facet_hash_to_vkeys: FastHashMap<
        u64,
        SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    > = FastHashMap::default();

    for &cell_key in &conflict_set {
        let cell = tds
            .get_cell(cell_key)
            .ok_or(ConflictError::InvalidStartCell { cell_key })?;

        let facet_count = cell.number_of_vertices(); // D+1 facets
        for facet_idx in 0..facet_count {
            // Compute canonical facet hash: sorted vertex keys excluding the opposite vertex.
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();

            let mut hasher = FastHasher::default();
            for &vkey in &facet_vkeys {
                vkey.hash(&mut hasher);
            }
            let facet_hash = hasher.finish();

            // Stash canonical vertex keys so we can reuse them for ridge analysis later.
            facet_hash_to_vkeys.entry(facet_hash).or_insert(facet_vkeys);

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| ConflictError::CellDataAccessFailed {
                    cell_key,
                    message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                })?;

            facet_to_conflict
                .entry(facet_hash)
                .or_default()
                .push(FacetHandle::new(cell_key, facet_idx_u8));
        }
    }

    let mut boundary_facets = CavityBoundaryBuffer::new();

    // Track ridge incidence for detecting ridge fans and validating boundary connectivity.
    //
    // A ridge is a (D-2)-simplex. For a valid *closed* cavity boundary (a (D-1)-manifold), each
    // ridge must be incident to exactly 2 boundary facets.

    // Map: ridge_hash -> RidgeInfo
    let mut ridge_map: FastHashMap<u64, RidgeInfo> = FastHashMap::default();

    for (facet_hash, cell_facet_pairs) in &facet_to_conflict {
        match cell_facet_pairs.as_slice() {
            // Exactly one conflict cell owns this facet => boundary facet
            [handle] => {
                let cell_key = handle.cell_key();
                let facet_idx_u8 = handle.facet_index();

                let boundary_facet_idx = boundary_facets.len();
                boundary_facets.push(FacetHandle::new(cell_key, facet_idx_u8));
                #[cfg(debug_assertions)]
                {
                    boundary_facet_count = boundary_facet_count.saturating_add(1);
                }

                // Use the cached canonical facet vertex keys for ridge analysis.
                let facet_vkeys = facet_hash_to_vkeys.get(facet_hash).ok_or_else(|| {
                    ConflictError::CellDataAccessFailed {
                        cell_key,
                        message: format!(
                            "Missing canonical vertex keys for facet hash {:#x}",
                            *facet_hash
                        ),
                    }
                })?;

                // A ridge is a (D-2)-simplex: remove one more vertex from this (D-1)-facet.
                if facet_vkeys.len() >= 2 {
                    let ridge_vertex_count = facet_vkeys.len() - 1;

                    for ridge_idx in 0..facet_vkeys.len() {
                        let mut ridge_vertices =
                            SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                        let mut ridge_hasher = FastHasher::default();
                        for (i, &vkey) in facet_vkeys.iter().enumerate() {
                            if i != ridge_idx {
                                ridge_vertices.push(vkey);
                                vkey.hash(&mut ridge_hasher);
                            }
                        }
                        let ridge_hash = ridge_hasher.finish();

                        ridge_map
                            .entry(ridge_hash)
                            .and_modify(|info| {
                                info.facet_count += 1;
                                if info.second_facet.is_none() {
                                    info.second_facet = Some(boundary_facet_idx);
                                } else {
                                    // 3rd+ facet: record for fan-cell identification.
                                    info.extra_facets.push(boundary_facet_idx);
                                }
                            })
                            .or_insert(RidgeInfo {
                                ridge_vertex_count,
                                ridge_vertices,
                                facet_count: 1,
                                first_facet: boundary_facet_idx,
                                second_facet: None,
                                extra_facets: Vec::new(),
                            });
                    }
                }
            }

            // Two conflict cells share this facet => internal facet (not on boundary)
            [_, _] => {
                #[cfg(debug_assertions)]
                {
                    internal_facet_count = internal_facet_count.saturating_add(1);
                }
            }

            // >2 conflict cells share this facet => non-manifold (should be impossible in valid TDS)
            // Treat as a retryable degeneracy.
            many => {
                #[cfg(debug_assertions)]
                if detail_enabled {
                    tracing::debug!(
                        facet_hash = *facet_hash,
                        cell_count = many.len(),
                        conflict_cells = conflict_cells.len(),
                        boundary_facet_count,
                        internal_facet_count,
                        elapsed = ?start_time.elapsed(),
                        "extract_cavity_boundary: non-manifold facet"
                    );
                }
                return Err(ConflictError::NonManifoldFacet {
                    facet_hash: *facet_hash,
                    cell_count: many.len(),
                });
            }
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            conflict_cells = conflict_cells.len(),
            facet_entries = facet_to_conflict.len(),
            boundary_facets = boundary_facets.len(),
            internal_facets = internal_facet_count,
            elapsed = ?start_time.elapsed(),
            "extract_cavity_boundary: facet classification summary"
        );
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        let mut ridge_facet_one = 0usize;
        let mut ridge_facet_two = 0usize;
        let mut ridge_facet_many = 0usize;
        let mut ridge_vertex_count_min = usize::MAX;
        let mut ridge_vertex_count_max = 0usize;
        for info in ridge_map.values() {
            ridge_vertex_count_min = ridge_vertex_count_min.min(info.ridge_vertex_count);
            ridge_vertex_count_max = ridge_vertex_count_max.max(info.ridge_vertex_count);
            match info.facet_count {
                1 => ridge_facet_one += 1,
                2 => ridge_facet_two += 1,
                _ => ridge_facet_many += 1,
            }
        }
        if ridge_map.is_empty() {
            ridge_vertex_count_min = 0;
        }
        tracing::debug!(
            conflict_cells = conflict_cells.len(),
            boundary_facets = boundary_facets.len(),
            ridge_count = ridge_map.len(),
            ridge_facet_one,
            ridge_facet_two,
            ridge_facet_many,
            ridge_vertex_count_min,
            ridge_vertex_count_max,
            "extract_cavity_boundary: ridge summary"
        );
    }

    // Build boundary facet adjacency via ridges and validate manifold properties of the boundary.
    if !boundary_facets.is_empty() {
        let boundary_len = boundary_facets.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); boundary_len];
        let mut first_ridge_fan: Option<(usize, usize)> = None;
        let mut ridge_fan_extra_cells: Vec<CellKey> = Vec::new();
        let mut ridge_fan_seen_cells = FastHashSet::<CellKey>::default();

        for info in ridge_map.values() {
            // Closed manifold boundary requires exactly 2 incident facets per ridge.
            if info.facet_count == 1 {
                #[cfg(debug_assertions)]
                if detail_enabled {
                    tracing::debug!(
                        facet_count = info.facet_count,
                        ridge_vertex_count = info.ridge_vertex_count,
                        boundary_facets = boundary_facets.len(),
                        ridge_count = ridge_map.len(),
                        elapsed = ?start_time.elapsed(),
                        "extract_cavity_boundary: open boundary ridge"
                    );
                }
                // The open facet's cell is the cell to remove to close the boundary.
                // `first_facet` is always a valid `boundary_facets` index by construction
                // (it is set during the same boundary-building traversal), so a missing
                // entry is an internal invariant violation rather than a cell-data-access
                // failure attributable to a real cell.
                let open_cell = boundary_facets
                    .get(info.first_facet)
                    .ok_or_else(|| ConflictError::InternalInconsistency {
                        site: InternalInconsistencySite::OpenBoundaryMissingFirstFacet {
                            first_facet: info.first_facet,
                            boundary_facets_len: boundary_facets.len(),
                            facet_count: info.facet_count,
                            ridge_vertex_count: info.ridge_vertex_count,
                        },
                    })
                    .map(FacetHandle::cell_key)?;
                return Err(ConflictError::OpenBoundary {
                    facet_count: info.facet_count,
                    ridge_vertex_count: info.ridge_vertex_count,
                    open_cell,
                });
            }

            if info.facet_count >= 3 {
                #[cfg(debug_assertions)]
                if detail_enabled {
                    tracing::debug!(
                        facet_count = info.facet_count,
                        ridge_vertex_count = info.ridge_vertex_count,
                        extra_facets = info.extra_facets.len(),
                        boundary_facets = boundary_facets.len(),
                        ridge_count = ridge_map.len(),
                        elapsed = ?start_time.elapsed(),
                        "extract_cavity_boundary: ridge fan"
                    );
                }
                // Collect the extra cells for this fan, but keep scanning so we can shrink
                // all currently-detected ridge fans in one reduction step instead of peeling
                // them one hash-map iteration at a time.
                let extra_cells = collect_ridge_fan_extra_cells(&boundary_facets, info)?;
                log_first_ridge_fan_dump(tds, conflict_cells, &boundary_facets, info, &extra_cells);
                first_ridge_fan.get_or_insert((info.facet_count, info.ridge_vertex_count));
                for cell_key in extra_cells {
                    if ridge_fan_seen_cells.insert(cell_key) {
                        ridge_fan_extra_cells.push(cell_key);
                    }
                }
                continue;
            }

            // facet_count == 2
            let a = info.first_facet;
            // `second_facet` is populated by the same ridge-map update that increments
            // `facet_count` to 2, so a `None` here is an internal invariant violation.
            // Report it as such instead of fabricating a cell key.
            let b = info
                .second_facet
                .ok_or_else(|| ConflictError::InternalInconsistency {
                    site: InternalInconsistencySite::RidgeInfoMissingSecondFacet {
                        first_facet: a,
                        boundary_facets_len: boundary_facets.len(),
                        ridge_vertex_count: info.ridge_vertex_count,
                    },
                })?;
            adjacency[a].push(b);
            adjacency[b].push(a);
        }

        if let Some((facet_count, ridge_vertex_count)) = first_ridge_fan {
            return Err(ConflictError::RidgeFan {
                facet_count,
                ridge_vertex_count,
                extra_cells: ridge_fan_extra_cells,
            });
        }

        // Connectedness: the cavity boundary must be a single component.
        // A disconnected boundary indicates a non-ball conflict region (e.g., shell), which
        // can lead to Euler characteristic violations if we proceed.
        let mut visited = vec![false; boundary_len];
        let mut stack = vec![0usize];
        visited[0] = true;
        let mut visited_count = 1usize;

        while let Some(cur) = stack.pop() {
            for &n in &adjacency[cur] {
                if !visited[n] {
                    visited[n] = true;
                    visited_count += 1;
                    stack.push(n);
                }
            }
        }

        if visited_count != boundary_len {
            #[cfg(debug_assertions)]
            if detail_enabled {
                tracing::debug!(
                    visited = visited_count,
                    total = boundary_len,
                    boundary_facets = boundary_facets.len(),
                    elapsed = ?start_time.elapsed(),
                    "extract_cavity_boundary: disconnected boundary"
                );
            }
            // Collect de-duplicated cell keys from the unreachable (disconnected) component
            // so callers can reduce the conflict region to eliminate the disconnection.
            let mut seen = FastHashSet::<CellKey>::default();
            let disconnected_cells: Vec<CellKey> = boundary_facets
                .iter()
                .enumerate()
                .filter(|(i, _)| !visited[*i])
                .map(|(_, fh)| fh.cell_key())
                .filter(|ck| seen.insert(*ck))
                .collect();
            return Err(ConflictError::DisconnectedBoundary {
                visited: visited_count,
                total: boundary_len,
                disconnected_cells,
            });
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            conflict_cells = conflict_cells.len(),
            boundary_facets = boundary_facets.len(),
            internal_facets = internal_facet_count,
            ridge_count = ridge_map.len(),
            elapsed = ?start_time.elapsed(),
            "extract_cavity_boundary: boundary connectivity validated"
        );
    }

    Ok(boundary_facets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cell::Cell;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::prelude::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

    #[test]
    fn test_internal_inconsistency_site_display_variants() {
        let ridge_fan = InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
            index: 7,
            boundary_facets_len: 5,
            extra_facets_len: 3,
        };
        assert_eq!(
            ridge_fan.to_string(),
            "RidgeFan extra_facets index 7 out of bounds \
             (boundary_facets.len()=5, extra_facets_len=3)"
        );

        let open_boundary = InternalInconsistencySite::OpenBoundaryMissingFirstFacet {
            first_facet: 11,
            boundary_facets_len: 9,
            facet_count: 1,
            ridge_vertex_count: 2,
        };
        assert_eq!(
            open_boundary.to_string(),
            "OpenBoundary missing first_facet index 11 \
             (boundary_facets.len()=9, facet_count=1, ridge_vertex_count=2)"
        );

        let missing_second = InternalInconsistencySite::RidgeInfoMissingSecondFacet {
            first_facet: 4,
            boundary_facets_len: 6,
            ridge_vertex_count: 3,
        };
        assert_eq!(
            missing_second.to_string(),
            "RidgeInfo missing second_facet when facet_count == 2 \
             (first_facet=4, boundary_facets_len=6, ridge_vertex_count=3)"
        );
    }

    #[test]
    fn test_format_vertex_and_cell_references_include_missing_markers() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let tds = dt.tds();
        let cell_key = tds.cell_keys().next().unwrap();
        let cell = tds.get_cell(cell_key).unwrap();

        let formatted_vertices = format_vertex_refs(tds, cell.vertices());
        assert!(formatted_vertices.contains("VertexKey"));
        assert!(!formatted_vertices.contains("missing"));

        let missing_vertex = VertexKey::from(KeyData::from_ffi(999_999));
        let formatted_missing = format_vertex_refs(tds, &[missing_vertex]);
        assert!(formatted_missing.contains("missing"));

        let facet = FacetHandle::new(cell_key, 0);
        let formatted_facet = format_facet_vertices(tds, facet);
        assert!(formatted_facet.contains("VertexKey"));

        let formatted_cell = format_cell_vertices(tds, cell_key);
        assert!(formatted_cell.contains("VertexKey"));

        let missing_cell = CellKey::from(KeyData::from_ffi(999_999));
        assert_eq!(
            format_facet_vertices(tds, FacetHandle::new(missing_cell, 0)),
            "<missing-cell>"
        );
        assert_eq!(format_cell_vertices(tds, missing_cell), "<missing-cell>");
    }

    #[test]
    fn test_collect_ridge_fan_extra_cells_deduplicates_cells() {
        let cell_a = CellKey::from(KeyData::from_ffi(1));
        let cell_b = CellKey::from(KeyData::from_ffi(2));
        let cell_c = CellKey::from(KeyData::from_ffi(3));
        let cell_d = CellKey::from(KeyData::from_ffi(4));
        let boundary_facets: CavityBoundaryBuffer = [
            FacetHandle::new(cell_a, 0),
            FacetHandle::new(cell_b, 1),
            FacetHandle::new(cell_c, 2),
            FacetHandle::new(cell_c, 3),
            FacetHandle::new(cell_d, 0),
        ]
        .into_iter()
        .collect();

        let info = RidgeInfo {
            ridge_vertex_count: 2,
            ridge_vertices: SmallBuffer::new(),
            facet_count: 5,
            first_facet: 0,
            second_facet: Some(1),
            extra_facets: vec![2, 3, 4],
        };

        let extra_cells = collect_ridge_fan_extra_cells(&boundary_facets, &info).unwrap();
        assert_eq!(extra_cells, vec![cell_c, cell_d]);
    }

    #[test]
    fn test_orientation_logic_manual() {
        // Manual test of orientation logic for 2D triangle
        // Triangle: (0,0), (1,0), (0,1)
        // Point inside: (0.3, 0.3)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get the single cell
        let cell_key = dt.tds().cell_keys().next().unwrap();
        let cell = dt.tds().get_cell(cell_key).unwrap();

        // Get cell vertices in order
        let cell_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|&vkey| *dt.tds().get_vertex_by_key(vkey).unwrap().point())
            .collect();

        println!("Cell vertices: {cell_points:?}");

        // Test orientation of full cell
        let cell_orientation = kernel.orientation(&cell_points).unwrap();
        println!("Cell orientation: {cell_orientation}");

        // Test query point inside
        let query_inside = Point::new([0.3, 0.3]);

        // For each facet, test if point is outside using the actual function
        for facet_idx in 0..3 {
            let result =
                is_point_outside_facet(dt.tds(), &kernel, cell_key, facet_idx, &query_inside);
            let is_outside = result.unwrap() == Some(true);

            println!("Facet {facet_idx} (opposite to vertex {facet_idx}): is_outside={is_outside}");

            // Point inside should NOT be outside any facet
            assert!(
                !is_outside,
                "Point inside triangle should not be outside facet {facet_idx}"
            );
        }

        // Test query point outside
        let query_outside = Point::new([2.0, 2.0]);
        let mut found_outside_facet = false;

        for facet_idx in 0..3 {
            let result =
                is_point_outside_facet(dt.tds(), &kernel, cell_key, facet_idx, &query_outside);
            let is_outside = result.unwrap() == Some(true);

            println!("Outside point - Facet {facet_idx}: is_outside={is_outside}");

            if is_outside {
                found_outside_facet = true;
            }
        }

        // Point outside should be outside at least one facet
        assert!(
            found_outside_facet,
            "Point outside triangle should be outside at least one facet"
        );
    }

    #[test]
    fn test_locate_empty_triangulation() {
        let tds: Tds<f64, (), (), 3> = Tds::empty();
        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.0, 0.0, 0.0]);

        let result = locate(&tds, &kernel, &point, None);
        assert!(matches!(result, Err(LocateError::EmptyTriangulation)));
    }

    #[test]
    fn test_locate_point_inside_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the triangle
        let point = Point::new([0.3, 0.3]);
        let result = locate(dt.tds(), &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideCell(cell_key)) => {
                assert!(dt.tds().contains_cell(cell_key));
            }
            _ => panic!("Expected point to be inside a cell, got {result:?}"),
        }
    }

    #[test]
    fn test_locate_point_inside_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the tetrahedron
        let point = Point::new([0.25, 0.25, 0.25]);
        let result = locate(dt.tds(), &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideCell(cell_key)) => {
                assert!(dt.tds().contains_cell(cell_key));
            }
            _ => panic!("Expected point to be inside a cell, got {result:?}"),
        }
    }

    #[test]
    fn test_locate_point_outside_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the triangle
        let point = Point::new([10.0, 10.0]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert!(matches!(result, Ok(LocateResult::Outside)));
    }

    #[test]
    fn test_locate_point_outside_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the tetrahedron
        let point = Point::new([2.0, 2.0, 2.0]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert!(matches!(result, Ok(LocateResult::Outside)));
    }

    #[test]
    fn test_locate_with_hint_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get a valid cell as hint
        let hint_cell = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([0.25, 0.25, 0.25]);

        let result = locate(dt.tds(), &kernel, &point, Some(hint_cell));
        assert!(matches!(result, Ok(LocateResult::InsideCell(_))));
    }

    #[test]
    fn test_locate_with_robust_kernel() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        let point = Point::new([0.3, 0.3]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert!(matches!(result, Ok(LocateResult::InsideCell(_))));
    }

    #[test]
    fn test_locate_with_stats_falls_back_on_cycle() {
        // Construct a valid single-cell triangulation, then intentionally corrupt the neighbor
        // pointers to create a self-loop. This forces facet walking to revisit a cell, exercising
        // the cycle-detection fallback path deterministically.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let cell_key = dt.tds().cell_keys().next().unwrap();

        // ⚠️ Dangerous test-only mutation: create a neighbor self-loop on every facet.
        let cell = dt.tds_mut().get_cell_by_key_mut(cell_key).unwrap();
        let mut neighbors = crate::core::collections::NeighborBuffer::<Option<CellKey>>::new();
        neighbors.resize(3, Some(cell_key));
        cell.neighbors = Some(neighbors);

        // Point outside the simplex: walking will attempt to cross a facet, hit the self-loop,
        // detect a cycle, and fall back to scan.
        let point = Point::new([10.0, 10.0]);
        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, None).unwrap();

        assert!(matches!(result, LocateResult::Outside));
        assert!(stats.fell_back_to_scan());
        assert!(!stats.used_hint);
        assert_eq!(stats.start_cell, cell_key);
        assert_eq!(stats.walk_steps, 2);
        assert!(matches!(
            stats.fallback,
            Some(LocateFallback {
                reason: LocateFallbackReason::CycleDetected,
                steps: 2,
            })
        ));
    }

    #[test]
    fn test_is_point_outside_facet_inside() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let cell_key = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([0.25, 0.25, 0.25]); // Inside tetrahedron

        // Test all facets - point should not be outside any of them
        for facet_idx in 0..4 {
            let result = is_point_outside_facet(dt.tds(), &kernel, cell_key, facet_idx, &point);
            assert!(matches!(result, Ok(Some(false) | None)));
        }
    }

    #[test]
    fn test_is_point_outside_facet_outside() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let cell_key = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([2.0, 2.0, 2.0]); // Outside tetrahedron

        // At least one facet should show the point as outside
        let mut found_outside = false;
        for facet_idx in 0..4 {
            if matches!(
                is_point_outside_facet(dt.tds(), &kernel, cell_key, facet_idx, &point),
                Ok(Some(true))
            ) {
                found_outside = true;
                break;
            }
        }
        assert!(
            found_outside,
            "Expected point to be outside at least one facet"
        );
    }

    #[test]
    fn test_locate_near_boundary() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        // Point very close to an edge but still inside
        let point = Point::new([0.01, 0.01]);
        let result = locate(dt.tds(), &kernel, &point, None);

        // Should either be inside or on the edge, not outside
        match result {
            Ok(LocateResult::InsideCell(_) | LocateResult::OnEdge(_)) => { /* OK */ }
            other => panic!("Expected inside or on edge, got {other:?}"),
        }
    }

    // Macro to test locate across dimensions
    macro_rules! test_locate_dimension {
        ($dim:literal, $inside_point:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(vertex!($coords)),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let point = Point::new($inside_point);
            let result = locate(dt.tds(), &kernel, &point, None);

            assert!(
                matches!(result, Ok(LocateResult::InsideCell(_))),
                "Expected point to be inside a cell in {}-simplex",
                $dim
            );
        }};
    }

    #[test]
    fn test_locate_2d() {
        test_locate_dimension!(2, [0.3, 0.3], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],);
    }

    #[test]
    fn test_locate_3d() {
        test_locate_dimension!(
            3,
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_locate_4d() {
        test_locate_dimension!(
            4,
            [0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_locate_5d() {
        test_locate_dimension!(
            5,
            [0.15, 0.15, 0.15, 0.15, 0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    // Macro to test find_conflict_region across dimensions
    macro_rules! test_find_conflict_region_dimension {
        ($dim:literal, $inside_point:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(vertex!($coords)),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let start_cell = dt.tds().cell_keys().next().unwrap();
            let point = Point::new($inside_point);

            let conflict_cells = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();

            assert_eq!(
                conflict_cells.len(),
                1,
                "Expected 1 cell in conflict for point inside single {}-simplex",
                $dim
            );
        }};
    }

    #[test]
    fn test_find_conflict_region_2d() {
        test_find_conflict_region_dimension!(2, [0.3, 0.3], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],);
    }

    #[test]
    fn test_find_conflict_region_3d() {
        test_find_conflict_region_dimension!(
            3,
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_find_conflict_region_4d() {
        test_find_conflict_region_dimension!(
            4,
            [0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_find_conflict_region_5d() {
        test_find_conflict_region_dimension!(
            5,
            [0.15, 0.15, 0.15, 0.15, 0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_find_conflict_region_outside_point() {
        // Point outside - should find zero cells in conflict
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_cell = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([10.0, 10.0, 10.0]); // Far outside

        let conflict_cells = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();

        // Should find zero cells in conflict
        assert_eq!(
            conflict_cells.len(),
            0,
            "Expected 0 cells in conflict for point far outside"
        );
    }

    #[test]
    fn test_find_conflict_region_invalid_start_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Create invalid cell key
        let invalid_cell = CellKey::from(KeyData::from_ffi(999_999));
        let point = Point::new([0.3, 0.3]);

        let result = find_conflict_region(dt.tds(), &kernel, &point, invalid_cell);

        assert!(
            matches!(result, Err(ConflictError::InvalidStartCell { .. })),
            "Expected InvalidStartCell error"
        );
    }

    #[test]
    fn test_find_conflict_region_with_robust_kernel() {
        // Test with robust kernel
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        let start_cell = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([0.3, 0.3]);

        let conflict_cells = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();

        assert_eq!(
            conflict_cells.len(),
            1,
            "Robust kernel should also find 1 cell in conflict"
        );
    }

    // Macro to test cavity boundary extraction across dimensions
    macro_rules! test_cavity_boundary_dimension {
        ($dim:literal, $expected_facets:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(vertex!($coords)),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            let start_cell = dt.tds().cell_keys().next().unwrap();
            let mut conflict_cells = CellKeyBuffer::new();
            conflict_cells.push(start_cell);

            let boundary = extract_cavity_boundary(dt.tds(), &conflict_cells).unwrap();

            assert_eq!(
                boundary.len(),
                $expected_facets,
                "Expected {} boundary facets for single {}-simplex",
                $expected_facets,
                $dim
            );
        }};
    }

    #[test]
    fn test_extract_cavity_boundary_2d() {
        // 2D triangle - single simplex has 3 edges (facets) on boundary
        test_cavity_boundary_dimension!(2, 3, [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],);
    }

    #[test]
    fn test_extract_cavity_boundary_3d() {
        // 3D tetrahedron - single simplex has 4 triangular facets on boundary
        test_cavity_boundary_dimension!(
            3,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_extract_cavity_boundary_4d() {
        // 4D simplex - single simplex has 5 tetrahedral facets on boundary
        test_cavity_boundary_dimension!(
            4,
            5,
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_extract_cavity_boundary_5d() {
        // 5D simplex - single simplex has 6 4-simplicial facets on boundary
        test_cavity_boundary_dimension!(
            5,
            6,
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_extract_cavity_boundary_empty_conflict() {
        // Empty conflict region should produce empty boundary
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let conflict_cells = CellKeyBuffer::new(); // Empty

        let boundary = extract_cavity_boundary(dt.tds(), &conflict_cells).unwrap();

        assert_eq!(
            boundary.len(),
            0,
            "Expected 0 boundary facets for empty conflict region"
        );
    }

    #[test]
    fn test_locate_with_stats_invalid_hint_uses_arbitrary_start_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.25, 0.25]);

        let invalid_hint = CellKey::from(KeyData::from_ffi(999_999));
        let expected_start = dt.tds().cell_keys().next().unwrap();
        let (result, stats) =
            locate_with_stats(dt.tds(), &kernel, &point, Some(invalid_hint)).unwrap();

        assert!(matches!(result, LocateResult::InsideCell(_)));
        assert_eq!(stats.start_cell, expected_start);
        assert!(!stats.used_hint);
        assert!(!stats.fell_back_to_scan());
    }

    #[test]
    fn test_locate_by_scan_inside_and_outside() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let expected_cell = dt.tds().cell_keys().next().unwrap();

        let inside = Point::new([0.2, 0.2]);
        let outside = Point::new([3.0, 3.0]);

        assert_eq!(
            locate_by_scan(dt.tds(), &kernel, &inside).unwrap(),
            LocateResult::InsideCell(expected_cell)
        );
        assert_eq!(
            locate_by_scan(dt.tds(), &kernel, &outside).unwrap(),
            LocateResult::Outside
        );
    }

    #[test]
    fn test_extract_cavity_boundary_rejects_nonmanifold_facet_2d() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let origin = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let x_axis = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let y_axis = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let upper_right = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();
        let top_apex = tds.insert_vertex_with_mapping(vertex!([0.5, 1.5])).unwrap();

        let first_cell = tds
            .insert_cell_with_mapping(Cell::new(vec![origin, x_axis, y_axis], None).unwrap())
            .unwrap();
        let second_cell = tds
            .insert_cell_with_mapping(Cell::new(vec![origin, x_axis, upper_right], None).unwrap())
            .unwrap();
        let third_cell = tds
            .insert_cell_with_mapping(Cell::new(vec![origin, x_axis, top_apex], None).unwrap())
            .unwrap();

        let mut conflict_cells = CellKeyBuffer::new();
        conflict_cells.push(first_cell);
        conflict_cells.push(second_cell);
        conflict_cells.push(third_cell);

        let err = extract_cavity_boundary(&tds, &conflict_cells).unwrap_err();
        assert!(matches!(
            err,
            ConflictError::NonManifoldFacet { cell_count: 3, .. }
        ));
    }

    #[test]
    fn test_extract_cavity_boundary_detects_disconnected_boundary_2d() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let left_origin = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let left_x = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let left_y = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let right_origin = tds.insert_vertex_with_mapping(vertex!([3.0, 0.0])).unwrap();
        let right_x = tds.insert_vertex_with_mapping(vertex!([4.0, 0.0])).unwrap();
        let right_y = tds.insert_vertex_with_mapping(vertex!([3.0, 1.0])).unwrap();

        let left = tds
            .insert_cell_with_mapping(Cell::new(vec![left_origin, left_x, left_y], None).unwrap())
            .unwrap();
        let right = tds
            .insert_cell_with_mapping(
                Cell::new(vec![right_origin, right_x, right_y], None).unwrap(),
            )
            .unwrap();

        let mut conflict_cells = CellKeyBuffer::new();
        conflict_cells.push(left);
        conflict_cells.push(right);

        let err = extract_cavity_boundary(&tds, &conflict_cells).unwrap_err();
        match err {
            ConflictError::DisconnectedBoundary {
                visited,
                total,
                disconnected_cells,
            } => {
                assert!(visited < total);
                assert_eq!(total, 6);
                assert!(
                    !disconnected_cells.is_empty(),
                    "disconnected_cells should be non-empty"
                );
                for ck in &disconnected_cells {
                    assert!(
                        tds.contains_cell(*ck),
                        "disconnected cell key {ck:?} should be present in the TDS"
                    );
                }
            }
            other => panic!("Expected DisconnectedBoundary, got {other:?}"),
        }
    }

    #[test]
    fn test_locate_with_stats_valid_hint_marks_used_hint() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.2, 0.2]);
        let hint = dt.tds().cell_keys().next().unwrap();

        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, Some(hint)).unwrap();

        assert!(matches!(result, LocateResult::InsideCell(_)));
        assert_eq!(stats.start_cell, hint);
        assert!(stats.used_hint);
        assert!(!stats.fell_back_to_scan());
    }

    #[test]
    fn test_locate_by_scan_empty_returns_outside() {
        let tds: Tds<f64, (), (), 2> = Tds::empty();
        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.0, 0.0]);

        assert_eq!(
            locate_by_scan(&tds, &kernel, &point).unwrap(),
            LocateResult::Outside
        );
    }

    #[test]
    fn test_extract_cavity_boundary_invalid_conflict_cell_key() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let invalid = CellKey::from(KeyData::from_ffi(424_242));
        let mut conflict_cells = CellKeyBuffer::new();
        conflict_cells.push(invalid);

        let err = extract_cavity_boundary(dt.tds(), &conflict_cells).unwrap_err();
        assert!(matches!(
            err,
            ConflictError::InvalidStartCell { cell_key } if cell_key == invalid
        ));
    }

    /// A cell with fewer than D+1 vertex keys is detected early by the
    /// `cell_vertex_keys.len() != D + 1` guard and returns `Ok(None)`.
    #[test]
    fn test_is_point_outside_facet_underdimensioned_cell_returns_none() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Shrink cell to only 2 vertices (D+1 = 3 required for D=2).
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            cell.push_vertex_key(v0);
            cell.push_vertex_key(v1);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.3, 0.3]);
        let result = is_point_outside_facet(&tds, &kernel, cell_key, 0, &point);
        assert!(
            matches!(result, Ok(None)),
            "underdimensioned cell should return Ok(None), got {result:?}"
        );
    }

    /// When the vertex at `facet_idx` (the opposite vertex) is unresolvable,
    /// `is_point_outside_facet` returns `Ok(None)` before reaching the
    /// canonical ordering helpers.
    #[test]
    fn test_is_point_outside_facet_unresolvable_opposite_vertex_returns_none() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Replace vertex at index 0 (the opposite vertex for facet_idx=0)
        // with a missing key, keeping the cell at D+1 vertex keys.
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            cell.push_vertex_key(missing); // index 0 = opposite vertex for facet_idx=0
            cell.push_vertex_key(v1);
            cell.push_vertex_key(v2);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.3, 0.3]);
        // facet_idx=0 → opposite_key = missing → unresolvable → Ok(None)
        let result = is_point_outside_facet(&tds, &kernel, cell_key, 0, &point);
        assert!(
            matches!(result, Ok(None)),
            "unresolvable opposite vertex should return Ok(None), got {result:?}"
        );
    }

    /// `is_point_outside_facet` resolves vertex points via the canonical ordering
    /// helper `sorted_facet_points_with_extra`.  A cell whose vertex-key list
    /// contains a key absent from the TDS causes the helper to return `None`,
    /// which the function converts to `Ok(None)` (degenerate/unresolvable cell).
    #[test]
    fn test_is_point_outside_facet_degenerate_cell_missing_vertex_returns_none() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        // Build a valid cell first, then mutate its vertex list to include a missing key.
        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let existing_vertices = tds.get_cell(cell_key).unwrap().vertices().to_vec();
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            cell.push_vertex_key(existing_vertices[0]);
            cell.push_vertex_key(existing_vertices[1]);
            cell.push_vertex_key(missing);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.3_f64, 0.3_f64]);

        // Only 2 of the 3 vertices exist → cell_vertices.len() (2) != D+1 (3) → Ok(None).
        let result = is_point_outside_facet(&tds, &kernel, cell_key, 0, &point);
        assert!(
            matches!(result, Ok(None)),
            "degenerate cell with missing vertex should return Ok(None), got {result:?}"
        );
    }

    /// When a conflict cell's neighbor list references a non-existent cell key,
    /// the BFS in `find_conflict_region` pops that key and fails to retrieve the
    /// cell, returning `CellDataAccessFailed` with a "vanished" message.
    #[test]
    fn test_find_conflict_region_vanished_neighbor_returns_cell_data_access_failed() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Wire a neighbor that doesn't exist in the TDS.
        let ghost = CellKey::from(KeyData::from_ffi(777_777));
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            let buf = cell.ensure_neighbors_buffer_mut();
            buf[0] = Some(ghost);
        }

        let kernel = FastKernel::<f64>::new();
        // Point inside the circumcircle so the start cell is in conflict
        // and BFS tries to visit the ghost neighbor.
        let point = Point::new([0.2, 0.2]);
        let result = find_conflict_region(&tds, &kernel, &point, cell_key);
        assert!(
            matches!(
                result,
                Err(ConflictError::CellDataAccessFailed { cell_key: ck, .. }) if ck == ghost
            ),
            "expected CellDataAccessFailed for vanished neighbor, got {result:?}"
        );
    }

    /// When a cell in the BFS has fewer than D+1 vertex keys (all resolvable),
    /// `sorted_cell_points` returns a short buffer and the vertex-count guard
    /// fires `CellDataAccessFailed`.
    #[test]
    fn test_find_conflict_region_underdimensioned_cell_returns_cell_data_access_failed() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Shrink cell to only 2 vertices (both valid).
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            cell.push_vertex_key(v0);
            cell.push_vertex_key(v1);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.3, 0.3]);
        let result = find_conflict_region(&tds, &kernel, &point, cell_key);
        assert!(
            matches!(
                result,
                Err(ConflictError::CellDataAccessFailed { cell_key: ck, .. }) if ck == cell_key
            ),
            "expected CellDataAccessFailed for underdimensioned cell, got {result:?}"
        );
    }

    /// `find_conflict_region` uses `sorted_cell_points` in the BFS loop;
    /// a conflict cell whose vertex-key list contains a key absent from the TDS
    /// causes the helper to return `None`, yielding `Err(CellDataAccessFailed)`.
    #[test]
    fn test_find_conflict_region_degenerate_cell_returns_cell_data_access_failed() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        // Build valid cell then mutate one vertex to a missing key.
        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let existing_vertices = tds.get_cell(cell_key).unwrap().vertices().to_vec();
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            cell.push_vertex_key(existing_vertices[0]);
            cell.push_vertex_key(existing_vertices[1]);
            cell.push_vertex_key(missing);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.3_f64, 0.3_f64]);

        // BFS visits the cell; simplex_points.len() == 2 != D+1 == 3 → CellDataAccessFailed.
        let result = find_conflict_region(&tds, &kernel, &point, cell_key);
        assert!(
            matches!(result, Err(ConflictError::CellDataAccessFailed { cell_key: ck, .. }) if ck == cell_key),
            "expected CellDataAccessFailed for degenerate cell, got {result:?}"
        );
    }

    /// Calling `locate_with_stats` with `hint = None` exercises the `_ =>` fallback arm
    /// of the hint-match, which picks an arbitrary start cell and records `used_hint = false`.
    #[test]
    fn test_locate_with_stats_none_hint_picks_arbitrary_start_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::new([0.25_f64, 0.25_f64]);

        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, None).unwrap();

        assert!(matches!(result, LocateResult::InsideCell(_)));
        assert!(!stats.used_hint, "None hint should set used_hint = false");
        assert!(!stats.fell_back_to_scan());
    }

    // =============================================================================
    // VERIFY CONFLICT REGION COMPLETENESS TESTS
    // =============================================================================
    // The function under test is #[cfg(debug_assertions)], so these tests are
    // also gated to avoid compilation errors in release/bench builds.

    #[cfg(debug_assertions)]
    /// Macro to test `verify_conflict_region_completeness` across dimensions.
    /// Builds a single-simplex Delaunay triangulation, finds the conflict region
    /// for an interior point via BFS, then verifies the brute-force check agrees.
    macro_rules! test_verify_conflict_region_complete_dimension {
        ($dim:literal, $inside_point:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(vertex!($coords)),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let start_cell = dt.tds().cell_keys().next().unwrap();
            let point = Point::new($inside_point);

            let conflict_cells = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();
            assert!(
                !conflict_cells.is_empty(),
                "BFS should find at least 1 conflict cell for interior point in {}D",
                $dim
            );

            let missed = verify_conflict_region_completeness(
                dt.tds(),
                &kernel,
                &point,
                &conflict_cells,
            );
            assert_eq!(
                missed, 0,
                "BFS conflict region should be complete for interior point in {}D",
                $dim
            );
        }};
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_2d() {
        test_verify_conflict_region_complete_dimension!(
            2,
            [0.3, 0.3],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_3d() {
        test_verify_conflict_region_complete_dimension!(
            3,
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_4d() {
        test_verify_conflict_region_complete_dimension!(
            4,
            [0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_5d() {
        test_verify_conflict_region_complete_dimension!(
            5,
            [0.15, 0.15, 0.15, 0.15, 0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    /// An empty BFS result should detect all conflict cells as missed.
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_empty_bfs_detects_missed() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the circumcircle — the single cell should be in conflict.
        let point = Point::new([0.3, 0.3]);
        let empty_bfs = CellKeyBuffer::new();

        let missed = verify_conflict_region_completeness(dt.tds(), &kernel, &point, &empty_bfs);
        assert!(
            missed > 0,
            "Empty BFS result should detect missed conflict cells"
        );
    }

    /// Point far outside produces no conflict cells; verify returns 0 missed.
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_outside_point_zero_missed() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_cell = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([10.0, 10.0, 10.0]);

        let conflict_cells = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();
        assert!(conflict_cells.is_empty());

        let missed =
            verify_conflict_region_completeness(dt.tds(), &kernel, &point, &conflict_cells);
        assert_eq!(missed, 0, "Outside point should produce zero missed cells");
    }

    /// Truncated multi-cell BFS result detects missed conflict cells.
    ///
    /// Builds a 2D triangulation with 4 vertices (2 triangles sharing an edge),
    /// finds a multi-cell conflict region, then drops one cell from the BFS
    /// result and verifies that `verify_conflict_region_completeness` catches
    /// the omission.  Because the two triangles are adjacent, the dropped cell
    /// has a neighbor still in the truncated BFS set, so the internal
    /// classification logs `REACHABLE_BUT_REJECTED` (observable via tracing).
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_truncated_multi_cell_detects_missed() {
        // Four corners of a rectangle — DT produces 2 triangles sharing a diagonal.
        // All 4 points are co-circular, so a center-ish query point is strictly
        // inside both circumcircles → conflict region has 2 cells.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([4.0, 0.0]),
            vertex!([4.0, 3.0]),
            vertex!([0.0, 3.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_cell = dt.tds().cell_keys().next().unwrap();
        let point = Point::new([2.0, 1.5]);

        let full_conflict = find_conflict_region(dt.tds(), &kernel, &point, start_cell).unwrap();
        assert!(
            full_conflict.len() >= 2,
            "Expected ≥2 conflict cells for center query in 2-triangle mesh, got {}",
            full_conflict.len()
        );

        // Truncate: keep only the first cell, drop the rest.
        let mut truncated = CellKeyBuffer::new();
        truncated.push(full_conflict[0]);

        let missed = verify_conflict_region_completeness(dt.tds(), &kernel, &point, &truncated);
        assert!(
            missed >= 1,
            "Truncated BFS should detect at least 1 missed conflict cell, got {missed}"
        );
    }

    #[test]
    fn test_extract_cavity_boundary_rejects_ridge_fan_2d() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let center = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let axis_x = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let axis_y = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let axis_neg_x = tds
            .insert_vertex_with_mapping(vertex!([-1.0, 0.0]))
            .unwrap();
        let axis_neg_y = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let far_x = tds.insert_vertex_with_mapping(vertex!([2.0, 0.0])).unwrap();
        let far_y = tds.insert_vertex_with_mapping(vertex!([0.0, 2.0])).unwrap();

        let first_cell = tds
            .insert_cell_with_mapping(Cell::new(vec![center, axis_x, axis_y], None).unwrap())
            .unwrap();
        let second_cell = tds
            .insert_cell_with_mapping(
                Cell::new(vec![center, axis_neg_x, axis_neg_y], None).unwrap(),
            )
            .unwrap();
        let third_cell = tds
            .insert_cell_with_mapping(Cell::new(vec![center, far_x, far_y], None).unwrap())
            .unwrap();

        let mut conflict_cells = CellKeyBuffer::new();
        conflict_cells.push(first_cell);
        conflict_cells.push(second_cell);
        conflict_cells.push(third_cell);

        match extract_cavity_boundary(&tds, &conflict_cells).unwrap_err() {
            ConflictError::RidgeFan {
                facet_count,
                ridge_vertex_count,
                extra_cells,
            } => {
                assert!(facet_count >= 3);
                assert_eq!(ridge_vertex_count, 1);
                // After deduplication, extra_cells contains unique cell keys contributing
                // the 3rd, 4th, … facets. Its length is ≤ facet_count - 2 and ≥ 1 here.
                assert!(
                    !extra_cells.is_empty() && extra_cells.len() <= facet_count - 2,
                    "deduped extra_cells should be non-empty and not exceed facet_count - 2; got {} vs {}",
                    extra_cells.len(),
                    facet_count - 2
                );
                // All entries must be valid keys from the TDS and unique.
                let mut seen = FastHashSet::default();
                for ck in &extra_cells {
                    assert!(
                        tds.contains_cell(*ck),
                        "extra cell key {ck:?} should be present in the TDS"
                    );
                    assert!(seen.insert(*ck), "duplicate key {ck:?} in extra_cells");
                }
            }
            other => panic!("Expected RidgeFan, got {other:?}"),
        }
    }
}
