#![forbid(unsafe_code)]

//! Point location algorithms for triangulations.
//!
//! Implements facet-walking point location for finding the simplex containing
//! a query point in O(√n) to O(n^(1/D)) expected time.
//!
//! # Algorithm
//!
//! The facet walking algorithm starts from a hint simplex (or arbitrary simplex)
//! and walks toward the query point by repeatedly:
//! 1. Testing orientation of query point relative to each facet
//! 2. Crossing to the neighbor on the side containing the query point
//! 3. Repeating until the query point is inside the current simplex
//!
//! # References
//!
//! - O. Devillers, S. Pion, and M. Teillaud, "Walking in a Triangulation",
//!   International Journal of Foundations of Computer Science, 2001.
//! - CGAL Triangulation_3 documentation

use crate::core::collections::{
    CavityBoundaryBuffer, FacetToSimplicesMap, FastHashMap, FastHashSet, FastHasher,
    MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SimplexSecondaryMap, SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::util::canonical_points::{
    CanonicalFacetPointError, CanonicalSimplexPointError, sorted_facet_points_with_extra,
    sorted_simplex_points,
};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateConversionError;
use std::env;
use std::fmt::{self, Write as _};
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
/// use delaunay::prelude::algorithms::LocateResult;
/// use delaunay::prelude::tds::VertexKey;
/// use slotmap::KeyData;
///
/// let vertex = VertexKey::from(KeyData::from_ffi(2));
/// let result = LocateResult::OnVertex(vertex);
/// std::assert_matches!(result, LocateResult::OnVertex(_));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocateResult {
    /// Point is strictly inside the simplex
    InsideSimplex(SimplexKey),
    /// Point is on a facet between two simplices
    OnFacet(SimplexKey, u8), // simplex_key, facet_index
    /// Point is on an edge (lower-dimensional simplex)
    OnEdge(SimplexKey),
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
/// use delaunay::prelude::algorithms::LocateError;
///
/// let err = LocateError::EmptyTriangulation;
/// std::assert_matches!(err, LocateError::EmptyTriangulation);
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum LocateError {
    /// Triangulation has no simplices.
    #[error("Cannot locate in empty triangulation")]
    EmptyTriangulation,

    /// Simplex reference is invalid.
    #[error("Invalid simplex reference: {simplex_key:?}")]
    InvalidSimplex {
        /// The invalid simplex key.
        simplex_key: SimplexKey,
    },

    /// A simplex has the wrong number of vertices for this dimension.
    #[error("simplex {simplex_key:?} has {found} vertices for point location; expected {expected}")]
    InvalidSimplexArity {
        /// Simplex with invalid arity.
        simplex_key: SimplexKey,
        /// Expected simplex vertex count.
        expected: usize,
        /// Observed simplex vertex count.
        found: usize,
    },

    /// A facet index is outside the simplex's facet range.
    #[error(
        "facet index {facet_index} is invalid for simplex {simplex_key:?} with {facet_count} facets"
    )]
    InvalidFacetIndex {
        /// Simplex containing the requested facet.
        simplex_key: SimplexKey,
        /// Requested facet index.
        facet_index: usize,
        /// Number of facets available on the simplex.
        facet_count: usize,
    },

    /// A simplex references a vertex missing from the TDS.
    #[error("simplex {simplex_key:?} references missing vertex {vertex_key:?} for point location")]
    MissingSimplexVertex {
        /// Simplex containing the missing vertex reference.
        simplex_key: SimplexKey,
        /// Missing vertex key.
        vertex_key: VertexKey,
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
/// use delaunay::prelude::algorithms::ConflictError;
/// use delaunay::prelude::tds::SimplexKey;
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(5));
/// let err = ConflictError::InvalidStartSimplex { simplex_key };
/// std::assert_matches!(err, ConflictError::InvalidStartSimplex { .. });
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum ConflictError {
    /// Starting simplex is invalid
    #[error("Invalid starting simplex: {simplex_key:?}")]
    InvalidStartSimplex {
        /// The invalid simplex key
        simplex_key: SimplexKey,
    },

    /// Geometric predicate failed
    #[error("Predicate error: {source}")]
    PredicateError {
        #[from]
        /// The underlying coordinate conversion error
        source: CoordinateConversionError,
    },

    /// Failed to access required simplex data (e.g., vertices) or build facet identifiers.
    ///
    /// This represents a *data-sourcing* failure attributable to a specific simplex key:
    /// the key resolved but its vertex list, facet index, or derived identifier could
    /// not be produced. For invariant violations that are *not* about a specific simplex
    /// (e.g., a `boundary_facets` index that must be in range by construction), use
    /// [`ConflictError::InternalInconsistency`] instead of fabricating a simplex key.
    #[error("Failed to access required data for simplex {simplex_key:?}: {message}")]
    SimplexDataAccessFailed {
        /// The simplex key for which required data could not be accessed.
        simplex_key: SimplexKey,
        /// Human-readable details about what data could not be accessed.
        message: String,
    },

    /// A conflict-region simplex has the wrong number of vertices for this dimension.
    #[error(
        "simplex {simplex_key:?} has {found} vertices for conflict-region predicates; expected {expected}"
    )]
    InvalidSimplexArity {
        /// Simplex with invalid arity.
        simplex_key: SimplexKey,
        /// Expected simplex vertex count.
        expected: usize,
        /// Observed simplex vertex count.
        found: usize,
    },

    /// A conflict-region simplex references a vertex missing from the TDS.
    #[error(
        "simplex {simplex_key:?} references missing vertex {vertex_key:?} for conflict-region predicates"
    )]
    MissingSimplexVertex {
        /// Simplex containing the missing vertex reference.
        simplex_key: SimplexKey,
        /// Missing vertex key.
        vertex_key: VertexKey,
    },

    /// Internal invariant violation during cavity-boundary extraction.
    ///
    /// This is raised when an invariant that must hold by construction does not —
    /// typically a `boundary_facets` or `RidgeInfo` index that is unconditionally
    /// valid in correct code. Returning a structured error rather than panicking
    /// preserves the caller's transactional rollback guarantees.
    ///
    /// Orthogonality: this variant is distinct from
    /// [`ConflictError::SimplexDataAccessFailed`]. Use `SimplexDataAccessFailed` when
    /// a specific, real simplex key is the subject of the failure; use
    /// `InternalInconsistency` when the failure is structural and has no such key.
    /// Treated as non-retryable by [`InsertionError::is_retryable`] because
    /// perturbing coordinates cannot resolve a logic error.
    ///
    /// The specific violation site is carried in [`InternalInconsistencySite`]
    /// as a typed payload so callers can pattern-match without parsing strings.
    ///
    /// [`InsertionError::is_retryable`]:
    ///     crate::prelude::insertion::InsertionError::is_retryable
    #[error("Internal cavity-boundary inconsistency: {site}")]
    InternalInconsistency {
        /// Structured, typed description of the violated invariant — the index,
        /// counts, and slice lengths that exposed the failure.
        site: InternalInconsistencySite,
    },

    /// Non-manifold facet detected (facet shared by more than 2 conflict simplices).
    #[error(
        "Non-manifold facet detected: facet {facet_hash:#x} shared by {simplex_count} conflict simplices (expected ≤2)"
    )]
    NonManifoldFacet {
        /// Hash of the facet's canonical vertex keys (sorted).
        facet_hash: u64,
        /// Number of conflict simplices incident to this facet.
        simplex_count: usize,
    },

    /// Ridge fan detected (many facets sharing same (D-2)-simplex).
    ///
    /// When a single conflict region contains multiple ridge fans,
    /// [`extract_cavity_boundary`] accumulates the removal candidates from every
    /// fan into `extra_simplices` before returning, so a single cavity-reduction step
    /// can shrink all of them at once. In that case:
    ///
    /// - `facet_count` and `ridge_vertex_count` describe the **first** fan that
    ///   the boundary walk observed (a representative example, not an aggregate).
    /// - `extra_simplices` contains the **union** of extra-simplex candidates across all
    ///   detected fans in the conflict region (deduplicated).
    ///
    /// The error message reports the representative scalars; consult
    /// `extra_simplices.len()` in traces when the conflict region is large enough to
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
        /// Deduplicated simplex keys that contribute the *extra* (3rd, 4th, …)
        /// facets to one or more ridge fans in the conflict region. Removing
        /// these simplices from the conflict region eliminates every currently
        /// detected ridge fan at once, enabling cavity insertion to proceed at
        /// the cost of leaving those simplices temporarily non-Delaunay (the
        /// subsequent flip-repair pass restores the Delaunay property).
        extra_simplices: Vec<SimplexKey>,
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
        /// Simplex keys from the disconnected (unreachable) boundary component.
        /// Removing these simplices from the conflict region makes the cavity boundary
        /// connected, enabling insertion to proceed (the simplices are left temporarily
        /// non-Delaunay and fixed by the subsequent flip-repair pass).
        disconnected_simplices: Vec<SimplexKey>,
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
        /// The conflict-region simplex that contributes the dangling (open) boundary facet.
        /// Removing this simplex from the conflict region closes the open ridge.
        open_simplex: SimplexKey,
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
/// These paths are unreachable in correct code and are guarded to preserve
/// transactional-rollback semantics when an internal invariant is violated.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::{ConflictError, InternalInconsistencySite};
///
/// let site = InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
///     index: 7,
///     boundary_facets_len: 5,
///     extra_facets_len: 3,
/// };
/// let err = ConflictError::InternalInconsistency { site: site.clone() };
/// std::assert_matches!(
///     err,
///     ConflictError::InternalInconsistency {
///         site: InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds { .. }
///     }
/// );
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

fn format_vertex_refs<U, V, const D: usize>(tds: &Tds<U, V, D>, vertex_keys: &[VertexKey]) -> String
where
    U: DataType,
    V: DataType,
{
    let mut refs = String::new();
    for (idx, &vertex_key) in vertex_keys.iter().enumerate() {
        if idx != 0 {
            refs.push_str(", ");
        }
        let uuid = tds.vertex(vertex_key).map_or_else(
            || String::from("missing"),
            |vertex| vertex.uuid().to_string(),
        );
        let _ = write!(&mut refs, "{vertex_key:?}/{uuid}");
    }
    refs
}

fn format_facet_vertices<U, V, const D: usize>(tds: &Tds<U, V, D>, handle: FacetHandle) -> String
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(handle.simplex_key()) else {
        return String::from("<missing-simplex>");
    };

    let facet_index = usize::from(handle.facet_index());
    let vertex_keys: Vec<VertexKey> = simplex
        .vertices()
        .iter()
        .enumerate()
        .filter_map(|(idx, &vertex_key)| (idx != facet_index).then_some(vertex_key))
        .collect();
    format_vertex_refs(tds, &vertex_keys)
}

fn format_simplex_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> String
where
    U: DataType,
    V: DataType,
{
    let Some(simplex) = tds.simplex(simplex_key) else {
        return String::from("<missing-simplex>");
    };
    format_vertex_refs(tds, simplex.vertices())
}

/// Emits a compact one-shot snapshot of the first detected ridge fan in the
/// current process/test binary.
///
/// Enabled via `DELAUNAY_DEBUG_RIDGE_FAN_ONCE`. Output is routed through
/// `tracing::debug!` so it respects the configured tracing subscriber;
/// callers that want these lines during a release-mode run should set
/// `RUST_LOG=debug` (or the matching filter in the large-scale debug harness).
///
/// The snapshot captures the shared ridge vertices, the participating boundary
/// facets, and the extra simplices that cavity reduction would remove.
fn log_first_ridge_fan_dump<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    conflict_simplices: &SimplexKeyBuffer,
    boundary_facets: &CavityBoundaryBuffer,
    info: &RidgeInfo,
    extra_simplices: &[SimplexKey],
) where
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

    let conflict_preview: SimplexKeyBuffer = conflict_simplices.iter().copied().take(16).collect();
    let ridge_vertices = format_vertex_refs(tds, info.ridge_vertices.as_slice());

    let participating_facets: Vec<String> = participating_indices
        .iter()
        .copied()
        .map(|boundary_index| {
            boundary_facets.get(boundary_index).copied().map_or_else(
                || format!("boundary_idx={boundary_index} <missing-boundary-facet>"),
                |handle| {
                    format!(
                        "boundary_idx={} simplex={:?} facet_index={} vertices=[{}]",
                        boundary_index,
                        handle.simplex_key(),
                        handle.facet_index(),
                        format_facet_vertices(tds, handle),
                    )
                },
            )
        })
        .collect();

    let extra_simplex_details: Vec<String> = extra_simplices
        .iter()
        .copied()
        .map(|simplex_key| {
            format!(
                "simplex={simplex_key:?} vertices=[{}]",
                format_simplex_vertices(tds, simplex_key)
            )
        })
        .collect();

    tracing::debug!(
        target: "delaunay::ridge_fan_dump",
        D,
        conflict_simplices = conflict_simplices.len(),
        boundary_facets = boundary_facets.len(),
        facet_count = info.facet_count,
        ridge_vertex_count = info.ridge_vertex_count,
        extra_simplices = ?extra_simplices,
        conflict_preview = ?conflict_preview,
        ridge_vertices = %ridge_vertices,
        participating_boundary_indices = ?participating_indices,
        participating_facets = ?participating_facets,
        extra_simplex_details = ?extra_simplex_details,
        "ridge-fan-dump: first detected ridge fan"
    );
}

fn collect_ridge_fan_extra_simplices(
    boundary_facets: &CavityBoundaryBuffer,
    info: &RidgeInfo,
) -> Result<Vec<SimplexKey>, ConflictError> {
    // Deduplicate: multiple extra facets can come from the same simplex. Downstream code
    // expects unique simplex keys when shrinking the conflict region.
    let mut seen = FastHashSet::<SimplexKey>::default();
    let mut extra_simplices: Vec<SimplexKey> = Vec::new();
    for &fi in &info.extra_facets {
        // Every entry in `info.extra_facets` is a `boundary_facets` index written by the
        // same traversal that populated `boundary_facets`, so any out-of-range value
        // represents an internal invariant violation rather than a data-access failure
        // attributable to a real simplex. Report it as such so the error message is truthful
        // (no fabricated `SimplexKey::default()` placeholder) and stays non-retryable.
        let ck = boundary_facets
            .get(fi)
            .ok_or_else(|| ConflictError::InternalInconsistency {
                site: InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
                    index: fi,
                    boundary_facets_len: boundary_facets.len(),
                    extra_facets_len: info.extra_facets.len(),
                },
            })?
            .simplex_key();
        if seen.insert(ck) {
            extra_simplices.push(ck);
        }
    }
    extra_simplices.sort_unstable();
    Ok(extra_simplices)
}

/// Indicates why facet-walking fell back to a brute-force scan.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::LocateFallbackReason;
///
/// let reason = LocateFallbackReason::StepLimit;
/// assert_eq!(reason, LocateFallbackReason::StepLimit);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocateFallbackReason {
    /// The facet-walking traversal revisited a previously seen simplex.
    CycleDetected,
    /// The facet-walking traversal exceeded the maximum step budget.
    StepLimit,
}

/// Information about a facet-walking fallback.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::{LocateFallback, LocateFallbackReason};
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
/// use delaunay::prelude::algorithms::LocateStats;
/// use delaunay::prelude::tds::SimplexKey;
/// use slotmap::KeyData;
///
/// let stats = LocateStats {
///     start_simplex: SimplexKey::from(KeyData::from_ffi(9)),
///     used_hint: false,
///     walk_steps: 0,
///     fallback: None,
/// };
/// assert!(!stats.fell_back_to_scan());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateStats {
    /// The start simplex used for the facet-walking phase.
    pub start_simplex: SimplexKey,
    /// Whether the caller-provided hint was used as the start simplex.
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

/// Internal locate result that also records the final simplex reached by the walk.
///
/// Exterior insertion uses `terminal_simplex` as a local conflict-region seed so it
/// can avoid a full triangulation scan while still repairing simplices near the hull
/// facet where point location exited the triangulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocateTrace {
    /// Public location classification.
    pub(crate) result: LocateResult,
    /// Public locate diagnostics.
    pub(crate) stats: LocateStats,
    /// Last simplex visited by the facet walk before returning or falling back.
    pub(crate) terminal_simplex: SimplexKey,
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
/// * `hint` - Optional starting simplex (uses arbitrary simplex if None)
///
/// # Returns
///
/// Returns `LocateResult` indicating where the point is located.
///
/// # Errors
///
/// Returns `LocateError` if:
/// - The triangulation is empty
/// - Simplex references are invalid
/// - Geometric predicates fail
///
/// # Examples
///
/// Basic point location in a 4D simplex:
///
/// ```rust
/// use delaunay::prelude::algorithms::*;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Locate(#[from] delaunay::prelude::algorithms::LocateError),
/// #     #[error(transparent)]
/// #     Conflict(#[from] delaunay::prelude::algorithms::ConflictError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 4D simplex (5 vertices)
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let kernel = FastKernel::<f64>::new();
///
/// // Point inside the 4-simplex
/// let inside_point = Point::try_from([0.2, 0.2, 0.2, 0.2])?;
/// let inside = locate(dt.tds(), &kernel, &inside_point, None)?;
/// std::assert_matches!(
///     inside,
///     LocateResult::InsideSimplex(simplex_key) if dt.tds().contains_simplex(simplex_key)
/// );
///
/// // Point outside the convex hull
/// let outside_point = Point::try_from([2.0, 2.0, 2.0, 2.0])?;
/// let outside = locate(dt.tds(), &kernel, &outside_point, None)?;
/// std::assert_matches!(outside, LocateResult::Outside);
/// # Ok(())
/// # }
/// ```
///
/// Using a hint simplex for faster location:
///
/// ```rust
/// use delaunay::prelude::geometry::RobustKernel;
/// use delaunay::prelude::algorithms::*;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Locate(#[from] delaunay::prelude::algorithms::LocateError),
/// #     #[error(transparent)]
/// #     Conflict(#[from] delaunay::prelude::algorithms::ConflictError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 4D simplex
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let kernel = RobustKernel::<f64>::default();
///
/// // Get a simplex to use as hint (spatially close to query point)
/// let Some(hint_simplex) = dt.tds().simplex_keys().next() else {
///     return Ok(());
/// };
/// let query_point = Point::try_from([0.15, 0.15, 0.15, 0.15])?;
///
/// let located = locate(dt.tds(), &kernel, &query_point, Some(hint_simplex))?;
/// std::assert_matches!(located, LocateResult::InsideSimplex(_));
/// # Ok(())
/// # }
/// ```
pub fn locate<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    hint: Option<SimplexKey>,
) -> Result<LocateResult, LocateError>
where
    K: Kernel<D, Scalar = f64>,
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
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Locate(#[from] delaunay::prelude::algorithms::LocateError),
/// #     #[error(transparent)]
/// #     Conflict(#[from] delaunay::prelude::algorithms::ConflictError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 2> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let kernel = FastKernel::<f64>::new();
///
/// let query_point = Point::try_from([0.3, 0.3])?;
/// let (_result, stats) = locate_with_stats(dt.tds(), &kernel, &query_point, None)?;
///
/// // In well-conditioned cases, the facet-walk should converge without falling back.
/// assert!(!stats.fell_back_to_scan());
/// # Ok(())
/// # }
/// ```
pub fn locate_with_stats<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    hint: Option<SimplexKey>,
) -> Result<(LocateResult, LocateStats), LocateError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let trace = locate_with_trace(tds, kernel, point, hint)?;
    Ok((trace.result, trace.stats))
}

/// Locate a point and keep the final walked simplex for local exterior repair.
///
/// This mirrors [`locate_with_stats`] but also exposes the last facet-walk simplex
/// before the algorithm concluded.  For [`LocateResult::Outside`] without a scan
/// fallback, that simplex is adjacent to the hull facet crossed by the query point.
pub(crate) fn locate_with_trace<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    hint: Option<SimplexKey>,
) -> Result<LocateTrace, LocateError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    const MAX_STEPS: usize = 10000;

    if tds.number_of_simplices() == 0 {
        return Err(LocateError::EmptyTriangulation);
    }

    let (start_simplex, used_hint) = match hint {
        Some(key) if tds.contains_simplex(key) => (key, true),
        _ => (
            tds.simplex_keys()
                .next()
                .ok_or(LocateError::EmptyTriangulation)?,
            false,
        ),
    };

    let mut stats = LocateStats {
        start_simplex,
        used_hint,
        walk_steps: 0,
        fallback: None,
    };

    let mut current_simplex = start_simplex;
    let mut visited: FastHashSet<SimplexKey> = FastHashSet::default();

    for step in 0..MAX_STEPS {
        stats.walk_steps = step + 1;

        if !visited.insert(current_simplex) {
            stats.fallback = Some(LocateFallback {
                reason: LocateFallbackReason::CycleDetected,
                steps: stats.walk_steps,
            });
            let result = locate_by_scan(tds, kernel, point)?;
            return Ok(LocateTrace {
                result,
                stats,
                terminal_simplex: current_simplex,
            });
        }

        let simplex = tds
            .simplex(current_simplex)
            .ok_or(LocateError::InvalidSimplex {
                simplex_key: current_simplex,
            })?;

        let facet_count = simplex.number_of_vertices();
        ensure_locate_simplex_arity::<D>(current_simplex, facet_count)?;
        let mut found_outside_facet = false;

        for facet_idx in 0..facet_count {
            if is_point_outside_facet(tds, kernel, current_simplex, facet_idx, point)? {
                if let Some(neighbor_key) = simplex.neighbor_key(facet_idx).flatten() {
                    current_simplex = neighbor_key;
                    found_outside_facet = true;
                    break;
                }
                return Ok(LocateTrace {
                    result: LocateResult::Outside,
                    stats,
                    terminal_simplex: current_simplex,
                });
            }
        }

        if !found_outside_facet {
            return Ok(LocateTrace {
                result: LocateResult::InsideSimplex(current_simplex),
                stats,
                terminal_simplex: current_simplex,
            });
        }
    }

    stats.fallback = Some(LocateFallback {
        reason: LocateFallbackReason::StepLimit,
        steps: stats.walk_steps,
    });
    let result = locate_by_scan(tds, kernel, point)?;
    Ok(LocateTrace {
        result,
        stats,
        terminal_simplex: current_simplex,
    })
}

pub(crate) fn locate_by_scan<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
) -> Result<LocateResult, LocateError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    for (simplex_key, simplex) in tds.simplices() {
        let mut found_outside_facet = false;
        let facet_count = simplex.number_of_vertices();
        ensure_locate_simplex_arity::<D>(simplex_key, facet_count)?;

        for facet_idx in 0..facet_count {
            if is_point_outside_facet(tds, kernel, simplex_key, facet_idx, point)? {
                found_outside_facet = true;
                break;
            }
        }

        if !found_outside_facet {
            return Ok(LocateResult::InsideSimplex(simplex_key));
        }
    }

    Ok(LocateResult::Outside)
}

/// Test if a point is on the outside of a simplex's facet.
///
/// A point is "outside" a facet if walking through that facet moves us closer
/// to the query point. This is determined by comparing orientations with a
/// consistent vertex ordering.
///
/// # Invariant Dependency
///
/// **CRITICAL**: This function relies on the triangulation's topological invariant:
/// - `facet_idx` refers to both the facet AND the vertex opposite to that facet
/// - `simplex.vertices()[facet_idx]` is the vertex opposite the facet
/// - The facet consists of all vertices EXCEPT `vertices[facet_idx]`
/// - This invariant is documented in [`Simplex`](crate::prelude::tds::Simplex) and enforced by
///   [`Tds::assign_neighbors`](crate::prelude::tds::Tds::assign_neighbors).
///
/// It is validated as part of Level 2 structural validation via
/// [`Tds::is_valid`](crate::prelude::tds::Tds::is_valid)
/// (or cumulatively via [`Tds::validate`](crate::prelude::tds::Tds::validate)).
///
/// This correspondence is essential for the canonical ordering used in orientation tests.
/// If this invariant is violated, point location will produce incorrect results.
///
/// Returns `true` if the point is outside the facet and `false` otherwise.
///
/// # Errors
///
/// Returns [`LocateError`] if the simplex or facet cannot provide valid
/// orientation predicate input.
fn is_point_outside_facet<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    simplex_key: SimplexKey,
    facet_idx: usize,
    query_point: &Point<D>,
) -> Result<bool, LocateError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(LocateError::InvalidSimplex { simplex_key })?;

    let simplex_vertex_keys = simplex.vertices();
    let vertex_count = simplex_vertex_keys.len();
    ensure_locate_simplex_arity::<D>(simplex_key, vertex_count)?;

    if facet_idx >= vertex_count {
        return Err(LocateError::InvalidFacetIndex {
            simplex_key,
            facet_index: facet_idx,
            facet_count: vertex_count,
        });
    }

    // The vertex at facet_idx is opposite the facet
    let opposite_key = simplex_vertex_keys[facet_idx];
    let opposite_point = *tds
        .vertex(opposite_key)
        .ok_or(LocateError::MissingSimplexVertex {
            simplex_key,
            vertex_key: opposite_key,
        })?
        .point();

    // Facet keys: all vertex keys except the one at facet_idx
    let facet_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = simplex_vertex_keys
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != facet_idx)
        .map(|(_, &vk)| vk)
        .collect();

    // Build facet simplex + opposite vertex in canonical key order.
    let canonical_simplex = sorted_facet_points_with_extra(tds, &facet_keys, opposite_point)
        .map_err(|error| locate_facet_points_error(simplex_key, error))?;

    let simplex_orientation = kernel.orientation(&canonical_simplex)?;

    // Build query simplex by reusing the canonical facet ordering:
    // replace the last element (opposite → query point).
    let mut query_simplex = canonical_simplex;
    let last = query_simplex.len() - 1;
    query_simplex[last] = *query_point;

    let query_orientation = kernel.orientation(&query_simplex)?;

    // If orientations differ, query point and opposite vertex are on
    // opposite sides of the facet → point is "outside" (should cross)
    // If orientations match, they're on the same side → point is "inside" (should not cross)
    Ok(simplex_orientation * query_orientation < 0)
}

const fn ensure_locate_simplex_arity<const D: usize>(
    simplex_key: SimplexKey,
    vertex_count: usize,
) -> Result<(), LocateError> {
    if vertex_count == D + 1 {
        return Ok(());
    }

    Err(LocateError::InvalidSimplexArity {
        simplex_key,
        expected: D + 1,
        found: vertex_count,
    })
}

const fn locate_facet_points_error(
    simplex_key: SimplexKey,
    error: CanonicalFacetPointError,
) -> LocateError {
    match error {
        CanonicalFacetPointError::InvalidArity { expected, found } => {
            LocateError::InvalidSimplexArity {
                simplex_key,
                expected: expected + 1,
                found: found + 1,
            }
        }
        CanonicalFacetPointError::MissingVertex { vertex_key } => {
            LocateError::MissingSimplexVertex {
                simplex_key,
                vertex_key,
            }
        }
    }
}

const fn conflict_simplex_points_error(
    simplex_key: SimplexKey,
    error: CanonicalSimplexPointError,
) -> ConflictError {
    match error {
        CanonicalSimplexPointError::InvalidArity { expected, found } => {
            ConflictError::InvalidSimplexArity {
                simplex_key,
                expected,
                found,
            }
        }
        CanonicalSimplexPointError::MissingVertex { vertex_key } => {
            ConflictError::MissingSimplexVertex {
                simplex_key,
                vertex_key,
            }
        }
    }
}

/// Find all simplices whose circumspheres contain the query point (conflict region).
///
/// Uses BFS traversal starting from a located simplex to find all simplices in conflict.
/// A simplex is in conflict if the query point lies inside **or on** its circumsphere.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `kernel` - Geometric kernel for `in_sphere` tests
/// * `point` - Query point to test
/// * `start_simplex` - Starting simplex (typically from `locate()`)
///
/// # Returns
///
/// Returns a buffer of all `SimplexKey`s whose circumspheres contain the point.
///
/// # Errors
///
/// Returns `ConflictError` if:
/// - The starting simplex is invalid
/// - Geometric predicates fail
/// - Cannot retrieve simplex vertices
///
/// # Algorithm
///
/// 1. Start BFS from the located simplex
/// 2. For each simplex, test `kernel.in_sphere()`
/// 3. If point is inside or on circumsphere (sign >= 0), add to conflict region
/// 4. Expand search to neighbors of conflicting simplices
/// 5. Track visited simplices with `SimplexSecondaryMap` for O(1) lookups
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::{locate, find_conflict_region, LocateResult};
/// use delaunay::prelude::{DelaunayTriangulation, DelaunayTriangulationBuilder};
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Locate(#[from] delaunay::prelude::algorithms::LocateError),
/// #     #[error(transparent)]
/// #     Conflict(#[from] delaunay::prelude::algorithms::ConflictError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 4D simplex (5 vertices forming a 4-simplex)
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let kernel = FastKernel::<f64>::new();
/// // Point inside the 4-simplex
/// let query_point = Point::try_from([0.2, 0.2, 0.2, 0.2])?;
///
/// // First locate the point
/// let location = locate(dt.tds(), &kernel, &query_point, None)?;
/// if let LocateResult::InsideSimplex(simplex_key) = location {
///     // Find all simplices whose circumspheres contain the point
///     let conflict_simplices = find_conflict_region(dt.tds(), &kernel, &query_point, simplex_key)?;
///     assert_eq!(conflict_simplices.len(), 1); // Single 4-simplex contains the point
/// }
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "function is long due to complex locate logic and should be split when refactoring"
)]
pub fn find_conflict_region<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    start_simplex: SimplexKey,
) -> Result<SimplexKeyBuffer, ConflictError>
where
    K: Kernel<D, Scalar = f64>,
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

    // Validate start simplex exists
    if !tds.contains_simplex(start_simplex) {
        return Err(ConflictError::InvalidStartSimplex {
            simplex_key: start_simplex,
        });
    }

    // Result buffer for conflicting simplices
    let mut conflict_simplices = SimplexKeyBuffer::new();

    // BFS work queue
    let mut queue = SimplexKeyBuffer::new();
    queue.push(start_simplex);

    // Track visited simplices with SparseSecondaryMap (idiomatic for SlotMap)
    let mut visited = SimplexSecondaryMap::new();

    while let Some(simplex_key) = queue.pop() {
        // Skip if already visited
        if visited.contains_key(simplex_key) {
            continue;
        }
        visited.insert(simplex_key, ());

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

        // Get simplex vertices for in_sphere test
        let simplex =
            tds.simplex(simplex_key)
                .ok_or_else(|| ConflictError::SimplexDataAccessFailed {
                    simplex_key,
                    message: "Simplex vanished during BFS traversal".to_string(),
                })?;

        // Collect simplex vertex points in canonical VertexKey order for consistent
        // SoS perturbation priority.
        let simplex_points = sorted_simplex_points(tds, simplex)
            .map_err(|error| conflict_simplex_points_error(simplex_key, error))?;

        #[cfg(debug_assertions)]
        if debug_config.log_conflict {
            tracing::debug!(
                simplex_key = ?simplex_key,
                vertex_keys = ?simplex.vertices(),
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
                        simplex_key = ?simplex_key,
                        vertex_keys = ?simplex.vertices(),
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
                simplex_key = ?simplex_key,
                sign,
                in_conflict = sign >= 0,
                "find_conflict_region: in_sphere classification"
            );
        }

        if sign >= 0 {
            // Point is inside or on circumsphere - simplex is in conflict
            conflict_simplices.push(simplex_key);

            #[cfg(debug_assertions)]
            {
                conflict_count = conflict_count.saturating_add(1);
            }

            // Add neighbors to queue for exploration
            if let Some(neighbors) = simplex.neighbor_keys() {
                for neighbor_opt in neighbors {
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
            // Simplex is NOT in conflict (sign < 0): BFS boundary.
            // Log boundary simplices so investigators can see exactly where
            // and why the BFS stopped expanding.
            #[cfg(debug_assertions)]
            if debug_config.log_conflict {
                let neighbor_keys: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex
                        .neighbor_keys()
                        .map(Iterator::collect)
                        .unwrap_or_default();
                tracing::debug!(
                    simplex_key = ?simplex_key,
                    vertex_keys = ?simplex.vertices(),
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
            conflict_simplices = conflict_simplices.len(),
            neighbor_enqueued,
            elapsed = ?start_time.elapsed(),
            "find_conflict_region: summary"
        );
    }

    Ok(conflict_simplices)
}

/// Verify that a BFS-found conflict region is complete by brute-force scanning all simplices.
///
/// This debug-only function compares the conflict region found by BFS traversal against
/// a full scan of every simplex in the TDS using insphere tests. Any simplex that the BFS missed
/// (i.e., the point is inside its circumsphere but the simplex was not found by BFS) is logged
/// as a "missed" simplex.
///
/// Activated by setting `DELAUNAY_DEBUG_CONFLICT_VERIFY=1`.
///
/// Returns the number of missed simplices (0 means the BFS result is complete).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "diagnostics")]
/// # {
/// use delaunay::prelude::collections::SimplexKeyBuffer;
/// use delaunay::prelude::diagnostics::verify_conflict_region_completeness;
/// use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
/// use delaunay::prelude::tds::Tds;
///
/// let tds: Tds<(), (), 2> = Tds::empty();
/// let kernel = AdaptiveKernel::<f64>::new();
/// let point = Point::try_from([0.25, 0.25])?;
/// let bfs_conflicts = SimplexKeyBuffer::new();
///
/// let missed = verify_conflict_region_completeness(
///     &tds,
///     &kernel,
///     &point,
///     &bfs_conflicts,
/// );
/// assert_eq!(missed, 0);
/// # }
/// ```
#[cfg(any(feature = "diagnostics", all(test, debug_assertions)))]
#[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
pub fn verify_conflict_region_completeness<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    bfs_conflict_simplices: &SimplexKeyBuffer,
) -> usize
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let bfs_set: FastHashSet<SimplexKey> = bfs_conflict_simplices.iter().copied().collect();
    let mut missed_count = 0usize;
    let mut brute_force_count = 0usize;
    let mut malformed_simplices = 0usize;
    let mut predicate_errors = 0usize;

    for (simplex_key, simplex) in tds.simplices() {
        let simplex_points = match sorted_simplex_points(tds, simplex) {
            Ok(points) => points,
            Err(error) => {
                malformed_simplices += 1;
                tracing::debug!(
                    simplex_key = ?simplex_key,
                    vertex_keys = ?simplex.vertices(),
                    error = %error,
                    "verify_conflict_region: skipping malformed simplex"
                );
                continue;
            }
        };
        let Ok(sign) = kernel.in_sphere(&simplex_points, point) else {
            predicate_errors += 1;
            tracing::debug!(
                simplex_key = ?simplex_key,
                vertex_keys = ?simplex.vertices(),
                "verify_conflict_region: in_sphere predicate failed"
            );
            continue;
        };

        if sign >= 0 {
            brute_force_count += 1;
            if !bfs_set.contains(&simplex_key) {
                missed_count += 1;

                // Reachability analysis: determine WHY BFS missed this simplex.
                // Check if any TDS neighbor of the missed simplex is in the BFS
                // conflict set.  This distinguishes two root causes:
                //   - Reachable: a neighbor IS in bfs_set, so BFS reached a
                //     neighbor but an intermediate insphere test rejected it
                //   - Unreachable: NO neighbors are in bfs_set, indicating
                //     broken neighbor pointers or a disconnected pocket
                let (neighbor_in_bfs, neighbor_total, neighbor_none) =
                    simplex.neighbor_keys().map_or((0, 0, 0), |neighbors| {
                        let total = neighbors.len();
                        let mut none_count = 0;
                        let mut in_bfs = 0;
                        for neighbor in neighbors {
                            match neighbor {
                                Some(nk) if bfs_set.contains(&nk) => {
                                    in_bfs += 1;
                                }
                                Some(_) => {}
                                None => {
                                    none_count += 1;
                                }
                            }
                        }
                        (in_bfs, total, none_count)
                    });

                let reachability = if neighbor_in_bfs > 0 {
                    "REACHABLE_BUT_REJECTED"
                } else {
                    "UNREACHABLE"
                };

                tracing::warn!(
                    simplex_key = ?simplex_key,
                    vertex_keys = ?simplex.vertices(),
                    sign,
                    reachability,
                    neighbor_in_bfs,
                    neighbor_total,
                    neighbor_none,
                    bfs_conflict_len = bfs_conflict_simplices.len(),
                    brute_force_conflict_so_far = brute_force_count,
                    "verify_conflict_region: BFS MISSED conflicting simplex"
                );
            }
        }
    }

    if missed_count > 0 || malformed_simplices > 0 || predicate_errors > 0 {
        tracing::warn!(
            bfs_conflict = bfs_conflict_simplices.len(),
            brute_force_conflict = brute_force_count,
            missed = missed_count,
            malformed_simplices,
            predicate_errors,
            query_point = ?point,
            "verify_conflict_region: INCOMPLETE — missed simplices or evaluation failures"
        );
    } else {
        tracing::debug!(
            bfs_conflict = bfs_conflict_simplices.len(),
            brute_force_conflict = brute_force_count,
            "verify_conflict_region: conflict region is COMPLETE"
        );
    }

    missed_count
}

/// Extract boundary facets of a conflict region (cavity).
///
/// Finds all facets where exactly one adjacent simplex is in the conflict region.
/// These boundary facets form the surface that will be connected to the new point.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `conflict_simplices` - Buffer of simplex keys in the conflict region
///
/// # Returns
///
/// Returns a buffer of `FacetHandle`s representing the cavity boundary.
/// Each facet is oriented such that its `simplex_key` is in the conflict region.
///
/// # Errors
///
/// Returns `ConflictError` if:
/// - A conflict simplex cannot be retrieved from the TDS
/// - Simplex neighbor data is inconsistent
///
/// # Algorithm
///
/// 1. Convert `conflict_simplices` to a `FastHashSet` for O(1) lookups
/// 2. For each simplex in the conflict region:
///    - Iterate through all facets (opposite each vertex)
///    - Check if the neighbor across that facet is also in conflict
///    - If neighbor is NOT in conflict (or is None/hull), it's a boundary facet
/// 3. Return all boundary facets as `FacetHandle`s
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::algorithms::extract_cavity_boundary;
/// use delaunay::prelude::collections::SimplexKeyBuffer;
/// use delaunay::prelude::tds::Tds;
///
/// # fn main() -> Result<(), delaunay::prelude::algorithms::ConflictError> {
/// let tds: Tds<(), (), 3> = Tds::empty();
/// let boundary = extract_cavity_boundary(&tds, &SimplexKeyBuffer::new())?;
/// assert!(boundary.is_empty());
/// # Ok(())
/// # }
/// ```
///
///
/// ```rust
/// use delaunay::prelude::algorithms::{locate, find_conflict_region, extract_cavity_boundary, LocateResult};
/// use delaunay::prelude::{DelaunayTriangulation, DelaunayTriangulationBuilder};
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Locate(#[from] delaunay::prelude::algorithms::LocateError),
/// #     #[error(transparent)]
/// #     Conflict(#[from] delaunay::prelude::algorithms::ConflictError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 4D simplex
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let kernel = FastKernel::<f64>::new();
/// let query_point = Point::try_from([0.2, 0.2, 0.2, 0.2])?;
///
/// // Locate and find conflict region
/// let location = locate(dt.tds(), &kernel, &query_point, None)?;
/// if let LocateResult::InsideSimplex(simplex_key) = location {
///     let conflict_simplices = find_conflict_region(dt.tds(), &kernel, &query_point, simplex_key)?;
///     
///     // Extract cavity boundary
///     let boundary_facets = extract_cavity_boundary(dt.tds(), &conflict_simplices)?;
///     
///     // For a single 4-simplex, all 5 facets are on the boundary (convex hull)
///     assert_eq!(boundary_facets.len(), 5);
/// }
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Long function; keep boundary extraction logic in one place for clarity"
)]
pub fn extract_cavity_boundary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    conflict_simplices: &SimplexKeyBuffer,
) -> Result<CavityBoundaryBuffer, ConflictError>
where
    U: DataType,
    V: DataType,
{
    // Empty conflict region => empty boundary
    if conflict_simplices.is_empty() {
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
            conflict_simplices = conflict_simplices.len(),
            "extract_cavity_boundary: start"
        );
    }

    // IMPORTANT:
    // We intentionally do NOT rely on neighbor pointers to classify boundary facets here.
    //
    // Neighbor pointers can be temporarily incomplete during incremental updates (e.g., after
    // simplex removal or before a full neighbor repair). If we rely on `simplex.neighbors()` and a
    // shared facet between two conflict simplices is missing a neighbor pointer, that facet will be
    // misclassified as a boundary facet. This can introduce internal boundary components and
    // break Level-3 Euler/topology validation (observed as χ=2 for Ball(3)).
    //
    // Instead, classify facets purely by facet incidence *within the conflict region*:
    // - A facet is on the cavity boundary iff it is incident to exactly 1 conflict simplex.
    let conflict_set: FastHashSet<SimplexKey> = conflict_simplices.iter().copied().collect();

    // facet_hash -> all facets in the conflict region that contain the facet
    let mut facet_to_conflict: FacetToSimplicesMap = FacetToSimplicesMap::default();

    // facet_hash -> canonical vertex keys (sorted, excluding the opposite vertex)
    // Cached so ridge analysis doesn't have to rebuild facet vertex sets from simplices.
    let mut facet_hash_to_vkeys: FastHashMap<
        u64,
        SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    > = FastHashMap::default();

    for &simplex_key in &conflict_set {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(ConflictError::InvalidStartSimplex { simplex_key })?;

        let facet_count = simplex.number_of_vertices(); // D+1 facets
        for facet_idx in 0..facet_count {
            // Compute canonical facet hash: sorted vertex keys excluding the opposite vertex.
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in simplex.vertices().iter().enumerate() {
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
                u8::try_from(facet_idx).map_err(|_| ConflictError::SimplexDataAccessFailed {
                    simplex_key,
                    message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                })?;

            facet_to_conflict
                .entry(facet_hash)
                .or_default()
                .push(FacetHandle::from_validated(simplex_key, facet_idx_u8));
        }
    }

    let mut boundary_facets = CavityBoundaryBuffer::new();

    // Track ridge incidence for detecting ridge fans and validating boundary connectivity.
    //
    // A ridge is a (D-2)-simplex. For a valid *closed* cavity boundary (a (D-1)-manifold), each
    // ridge must be incident to exactly 2 boundary facets.

    // Map: ridge_hash -> RidgeInfo
    let mut ridge_map: FastHashMap<u64, RidgeInfo> = FastHashMap::default();

    for (facet_hash, simplex_facet_pairs) in &facet_to_conflict {
        match simplex_facet_pairs.as_slice() {
            // Exactly one conflict simplex owns this facet => boundary facet
            [handle] => {
                let simplex_key = handle.simplex_key();
                let facet_idx_u8 = handle.facet_index();

                let boundary_facet_idx = boundary_facets.len();
                boundary_facets.push(FacetHandle::from_validated(simplex_key, facet_idx_u8));
                #[cfg(debug_assertions)]
                {
                    boundary_facet_count = boundary_facet_count.saturating_add(1);
                }

                // Use the cached canonical facet vertex keys for ridge analysis.
                let facet_vkeys = facet_hash_to_vkeys.get(facet_hash).ok_or_else(|| {
                    ConflictError::SimplexDataAccessFailed {
                        simplex_key,
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
                                    // 3rd+ facet: record for fan-simplex identification.
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

            // Two conflict simplices share this facet => internal facet (not on boundary)
            [_, _] => {
                #[cfg(debug_assertions)]
                {
                    internal_facet_count = internal_facet_count.saturating_add(1);
                }
            }

            // >2 conflict simplices share this facet => non-manifold (should be impossible in valid TDS)
            // Treat as a retryable degeneracy.
            many => {
                #[cfg(debug_assertions)]
                if detail_enabled {
                    tracing::debug!(
                        facet_hash = *facet_hash,
                        simplex_count = many.len(),
                        conflict_simplices = conflict_simplices.len(),
                        boundary_facet_count,
                        internal_facet_count,
                        elapsed = ?start_time.elapsed(),
                        "extract_cavity_boundary: non-manifold facet"
                    );
                }
                return Err(ConflictError::NonManifoldFacet {
                    facet_hash: *facet_hash,
                    simplex_count: many.len(),
                });
            }
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            conflict_simplices = conflict_simplices.len(),
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
            conflict_simplices = conflict_simplices.len(),
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
        let mut ridge_fan_extra_simplices: Vec<SimplexKey> = Vec::new();
        let mut ridge_fan_seen_simplices = FastHashSet::<SimplexKey>::default();

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
                // The open facet's simplex is the simplex to remove to close the boundary.
                // `first_facet` is always a valid `boundary_facets` index by construction
                // (it is set during the same boundary-building traversal), so a missing
                // entry is an internal invariant violation rather than a simplex-data-access
                // failure attributable to a real simplex.
                let open_simplex = boundary_facets
                    .get(info.first_facet)
                    .ok_or_else(|| ConflictError::InternalInconsistency {
                        site: InternalInconsistencySite::OpenBoundaryMissingFirstFacet {
                            first_facet: info.first_facet,
                            boundary_facets_len: boundary_facets.len(),
                            facet_count: info.facet_count,
                            ridge_vertex_count: info.ridge_vertex_count,
                        },
                    })
                    .map(FacetHandle::simplex_key)?;
                return Err(ConflictError::OpenBoundary {
                    facet_count: info.facet_count,
                    ridge_vertex_count: info.ridge_vertex_count,
                    open_simplex,
                });
            }
        }

        for info in ridge_map.values() {
            // `second_facet` is populated by the same ridge-map update that increments
            // `facet_count` to 2, so a `None` here is an internal invariant violation.
            // Check this before accumulating ridge fans so error precedence is deterministic.
            if info.facet_count == 2 && info.second_facet.is_none() {
                return Err(ConflictError::InternalInconsistency {
                    site: InternalInconsistencySite::RidgeInfoMissingSecondFacet {
                        first_facet: info.first_facet,
                        boundary_facets_len: boundary_facets.len(),
                        ridge_vertex_count: info.ridge_vertex_count,
                    },
                });
            }
        }

        for info in ridge_map.values() {
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
                // Collect the extra simplices for this fan, but keep scanning so we can shrink
                // all currently-detected ridge fans in one reduction step instead of peeling
                // them one hash-map iteration at a time.
                let extra_simplices = collect_ridge_fan_extra_simplices(&boundary_facets, info)?;
                log_first_ridge_fan_dump(
                    tds,
                    conflict_simplices,
                    &boundary_facets,
                    info,
                    &extra_simplices,
                );
                first_ridge_fan.get_or_insert((info.facet_count, info.ridge_vertex_count));
                for simplex_key in extra_simplices {
                    if ridge_fan_seen_simplices.insert(simplex_key) {
                        ridge_fan_extra_simplices.push(simplex_key);
                    }
                }
                continue;
            }

            // facet_count == 2
            let a = info.first_facet;
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
            ridge_fan_extra_simplices.sort_unstable();
            return Err(ConflictError::RidgeFan {
                facet_count,
                ridge_vertex_count,
                extra_simplices: ridge_fan_extra_simplices,
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
            // Collect de-duplicated simplex keys from the unreachable (disconnected) component
            // so callers can reduce the conflict region to eliminate the disconnection.
            let mut seen = FastHashSet::<SimplexKey>::default();
            let mut disconnected_simplices: Vec<SimplexKey> = boundary_facets
                .iter()
                .enumerate()
                .filter(|(i, _)| !visited[*i])
                .map(|(_, fh)| fh.simplex_key())
                .filter(|ck| seen.insert(*ck))
                .collect();
            disconnected_simplices.sort_unstable();
            return Err(ConflictError::DisconnectedBoundary {
                visited: visited_count,
                total: boundary_len,
                disconnected_simplices,
            });
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            conflict_simplices = conflict_simplices.len(),
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
    use crate::core::collections::NeighborBuffer;
    use crate::core::simplex::Simplex;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::prelude::DelaunayTriangulation;
    use slotmap::KeyData;
    use std::assert_matches;

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
    fn test_format_vertex_and_simplex_references_include_missing_markers() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let tds = dt.tds();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let simplex = tds.simplex(simplex_key).unwrap();

        let formatted_vertices = format_vertex_refs(tds, simplex.vertices());
        assert!(formatted_vertices.contains("VertexKey"));
        assert!(!formatted_vertices.contains("missing"));

        let missing_vertex = VertexKey::from(KeyData::from_ffi(999_999));
        let formatted_missing = format_vertex_refs(tds, &[missing_vertex]);
        assert!(formatted_missing.contains("missing"));

        let facet = FacetHandle::from_validated(simplex_key, 0);
        let formatted_facet = format_facet_vertices(tds, facet);
        assert!(formatted_facet.contains("VertexKey"));

        let formatted_simplex = format_simplex_vertices(tds, simplex_key);
        assert!(formatted_simplex.contains("VertexKey"));

        let missing_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        assert_eq!(
            format_facet_vertices(tds, FacetHandle::from_validated(missing_simplex, 0)),
            "<missing-simplex>"
        );
        assert_eq!(
            format_simplex_vertices(tds, missing_simplex),
            "<missing-simplex>"
        );
    }

    #[test]
    fn test_collect_ridge_fan_extra_simplices_deduplicates_simplices() {
        let simplex_a = SimplexKey::from(KeyData::from_ffi(1));
        let simplex_b = SimplexKey::from(KeyData::from_ffi(2));
        let simplex_c = SimplexKey::from(KeyData::from_ffi(3));
        let simplex_d = SimplexKey::from(KeyData::from_ffi(4));
        let boundary_facets: CavityBoundaryBuffer = [
            FacetHandle::from_validated(simplex_a, 0),
            FacetHandle::from_validated(simplex_b, 1),
            FacetHandle::from_validated(simplex_c, 2),
            FacetHandle::from_validated(simplex_c, 3),
            FacetHandle::from_validated(simplex_d, 0),
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

        let extra_simplices = collect_ridge_fan_extra_simplices(&boundary_facets, &info).unwrap();
        assert_eq!(extra_simplices, vec![simplex_c, simplex_d]);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "regression test spells out the multi-fan cavity fixture"
    )]
    fn test_extract_cavity_boundary_accumulates_multiple_ridge_fans_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let center_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let a2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let a4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let a5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, -1.0]).unwrap(),
            )
            .unwrap();

        let center_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([11.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 1.0]).unwrap(),
            )
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([9.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, -1.0]).unwrap(),
            )
            .unwrap();
        let b4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([11.0, 1.0]).unwrap(),
            )
            .unwrap();
        let b5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([9.0, -1.0]).unwrap(),
            )
            .unwrap();

        let origin_simplices = [
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_a, a0, a1], None).unwrap(),
            )
            .unwrap(),
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_a, a2, a3], None).unwrap(),
            )
            .unwrap(),
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_a, a4, a5], None).unwrap(),
            )
            .unwrap(),
        ];
        let shifted_simplices = [
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_b, b0, b1], None).unwrap(),
            )
            .unwrap(),
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_b, b2, b3], None).unwrap(),
            )
            .unwrap(),
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center_b, b4, b5], None).unwrap(),
            )
            .unwrap(),
        ];

        let all_simplices = [
            origin_simplices[0],
            origin_simplices[1],
            origin_simplices[2],
            shifted_simplices[0],
            shifted_simplices[1],
            shifted_simplices[2],
        ];
        let conflict_simplices: SimplexKeyBuffer = all_simplices.into_iter().collect();

        match extract_cavity_boundary(&tds, &conflict_simplices).unwrap_err() {
            ConflictError::RidgeFan {
                facet_count,
                ridge_vertex_count,
                extra_simplices,
            } => {
                assert_eq!(facet_count, 6);
                assert_eq!(ridge_vertex_count, 1);
                let expected: FastHashSet<SimplexKey> = all_simplices.into_iter().collect();
                let actual: FastHashSet<SimplexKey> = extra_simplices.iter().copied().collect();
                assert_eq!(actual, expected);
                assert_eq!(extra_simplices.len(), expected.len());
            }
            other => panic!("Expected RidgeFan, got {other:?}"),
        }
    }

    #[test]
    fn test_orientation_logic_manual() {
        // Manual test of orientation logic for 2D triangle
        // Triangle: (0,0), (1,0), (0,1)
        // Point inside: (0.3, 0.3)
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get the single simplex
        let simplex_key = dt.tds().simplex_keys().next().unwrap();
        let simplex = dt.tds().simplex(simplex_key).unwrap();

        // Get simplex vertices in order
        let simplex_points: Vec<Point<2>> = simplex
            .vertices()
            .iter()
            .map(|&vkey| *dt.tds().vertex(vkey).unwrap().point())
            .collect();

        println!("Simplex vertices: {simplex_points:?}");

        // Test orientation of full simplex
        let simplex_orientation = kernel.orientation(&simplex_points).unwrap();
        println!("Simplex orientation: {simplex_orientation}");

        // Test query point inside
        let query_inside = Point::from_validated_coords([0.3, 0.3]);

        // For each facet, test if point is outside using the actual function
        for facet_idx in 0..3 {
            let result =
                is_point_outside_facet(dt.tds(), &kernel, simplex_key, facet_idx, &query_inside);
            let is_outside = result.unwrap();

            println!("Facet {facet_idx} (opposite to vertex {facet_idx}): is_outside={is_outside}");

            // Point inside should NOT be outside any facet
            assert!(
                !is_outside,
                "Point inside triangle should not be outside facet {facet_idx}"
            );
        }

        // Test query point outside
        let query_outside = Point::from_validated_coords([2.0, 2.0]);
        let mut found_outside_facet = false;

        for facet_idx in 0..3 {
            let result =
                is_point_outside_facet(dt.tds(), &kernel, simplex_key, facet_idx, &query_outside);
            let is_outside = result.unwrap();

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
        let tds: Tds<(), (), 3> = Tds::empty();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.0, 0.0, 0.0]);

        let result = locate(&tds, &kernel, &point, None);
        assert_matches!(result, Err(LocateError::EmptyTriangulation));
    }

    #[test]
    fn test_locate_point_inside_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the triangle
        let point = Point::from_validated_coords([0.3, 0.3]);
        let result = locate(dt.tds(), &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideSimplex(simplex_key)) => {
                assert!(dt.tds().contains_simplex(simplex_key));
            }
            _ => panic!("Expected point to be inside a simplex, got {result:?}"),
        }
    }

    #[test]
    fn test_locate_point_inside_3d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the tetrahedron
        let point = Point::from_validated_coords([0.25, 0.25, 0.25]);
        let result = locate(dt.tds(), &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideSimplex(simplex_key)) => {
                assert!(dt.tds().contains_simplex(simplex_key));
            }
            _ => panic!("Expected point to be inside a simplex, got {result:?}"),
        }
    }

    #[test]
    fn test_locate_point_outside_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the triangle
        let point = Point::from_validated_coords([10.0, 10.0]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert_matches!(result, Ok(LocateResult::Outside));
    }

    #[test]
    fn test_locate_point_outside_3d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the tetrahedron
        let point = Point::from_validated_coords([2.0, 2.0, 2.0]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert_matches!(result, Ok(LocateResult::Outside));
    }

    #[test]
    fn test_locate_with_hint_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get a valid simplex as hint
        let hint_simplex = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([0.25, 0.25, 0.25]);

        let result = locate(dt.tds(), &kernel, &point, Some(hint_simplex));
        assert_matches!(result, Ok(LocateResult::InsideSimplex(_)));
    }

    #[test]
    fn test_locate_with_robust_kernel() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        let point = Point::from_validated_coords([0.3, 0.3]);
        let result = locate(dt.tds(), &kernel, &point, None);

        assert_matches!(result, Ok(LocateResult::InsideSimplex(_)));
    }

    #[test]
    fn test_locate_with_stats_falls_back_on_cycle() {
        // Construct a valid single-simplex triangulation, then intentionally corrupt the neighbor
        // pointers to create a self-loop. This forces facet walking to revisit a simplex, exercising
        // the cycle-detection fallback path deterministically.
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let simplex_key = dt.tds().simplex_keys().next().unwrap();

        // ⚠️ Dangerous test-only mutation: create a neighbor self-loop on every facet.
        let simplex = dt.tds_mut().simplex_mut(simplex_key).unwrap();
        let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
        neighbors.resize(3, Some(simplex_key));
        simplex.set_neighbors_from_keys(neighbors).unwrap();

        // Point outside the simplex: walking will attempt to cross a facet, hit the self-loop,
        // detect a cycle, and fall back to scan.
        let point = Point::from_validated_coords([10.0, 10.0]);
        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, None).unwrap();

        assert_matches!(result, LocateResult::Outside);
        assert!(stats.fell_back_to_scan());
        assert!(!stats.used_hint);
        assert_eq!(stats.start_simplex, simplex_key);
        assert_eq!(stats.walk_steps, 2);
        assert_matches!(
            stats.fallback,
            Some(LocateFallback {
                reason: LocateFallbackReason::CycleDetected,
                steps: 2,
            })
        );
    }

    #[test]
    fn test_is_point_outside_facet_inside() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let simplex_key = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([0.25, 0.25, 0.25]); // Inside tetrahedron

        // Test all facets - point should not be outside any of them
        for facet_idx in 0..4 {
            let result = is_point_outside_facet(dt.tds(), &kernel, simplex_key, facet_idx, &point);
            assert_matches!(result, Ok(false));
        }
    }

    #[test]
    fn test_is_point_outside_facet_outside() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let simplex_key = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([2.0, 2.0, 2.0]); // Outside tetrahedron

        // At least one facet should show the point as outside
        let mut found_outside = false;
        for facet_idx in 0..4 {
            if matches!(
                is_point_outside_facet(dt.tds(), &kernel, simplex_key, facet_idx, &point),
                Ok(true)
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        // Point very close to an edge but still inside
        let point = Point::from_validated_coords([0.01, 0.01]);
        let result = locate(dt.tds(), &kernel, &point, None);

        // Should either be inside or on the edge, not outside
        match result {
            Ok(LocateResult::InsideSimplex(_) | LocateResult::OnEdge(_)) => { /* OK */ }
            other => panic!("Expected inside or on edge, got {other:?}"),
        }
    }

    // Macro to test locate across dimensions
    macro_rules! test_locate_dimension {
        ($dim:literal, $inside_point:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(crate::core::vertex::Vertex::<(), _>::try_new($coords).unwrap()),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let point = Point::from_validated_coords($inside_point);
            let result = locate(dt.tds(), &kernel, &point, None);

            assert!(
                matches!(result, Ok(LocateResult::InsideSimplex(_))),
                "Expected point to be inside a simplex in {}-simplex",
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
                $(crate::core::vertex::Vertex::<(), _>::try_new($coords).unwrap()),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let start_simplex = dt.tds().simplex_keys().next().unwrap();
            let point = Point::from_validated_coords($inside_point);

            let conflict_simplices = find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();

            assert_eq!(
                conflict_simplices.len(),
                1,
                "Expected 1 simplex in conflict for point inside single {}-simplex",
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
        // Point outside - should find zero simplices in conflict
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_simplex = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([10.0, 10.0, 10.0]); // Far outside

        let conflict_simplices =
            find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();

        // Should find zero simplices in conflict
        assert_eq!(
            conflict_simplices.len(),
            0,
            "Expected 0 simplices in conflict for point far outside"
        );
    }

    #[test]
    fn test_find_conflict_region_invalid_start_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Create invalid simplex key
        let invalid_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let point = Point::from_validated_coords([0.3, 0.3]);

        let result = find_conflict_region(dt.tds(), &kernel, &point, invalid_simplex);

        assert!(
            matches!(result, Err(ConflictError::InvalidStartSimplex { .. })),
            "Expected InvalidStartSimplex error"
        );
    }

    #[test]
    fn test_find_conflict_region_with_robust_kernel() {
        // Test with robust kernel
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        let start_simplex = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([0.3, 0.3]);

        let conflict_simplices =
            find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();

        assert_eq!(
            conflict_simplices.len(),
            1,
            "Robust kernel should also find 1 simplex in conflict"
        );
    }

    // Macro to test cavity boundary extraction across dimensions
    macro_rules! test_cavity_boundary_dimension {
        ($dim:literal, $expected_facets:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(crate::core::vertex::Vertex::<(), _>::try_new($coords).unwrap()),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            let start_simplex = dt.tds().simplex_keys().next().unwrap();
            let mut conflict_simplices = SimplexKeyBuffer::new();
            conflict_simplices.push(start_simplex);

            let boundary = extract_cavity_boundary(dt.tds(), &conflict_simplices).unwrap();

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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let conflict_simplices = SimplexKeyBuffer::new(); // Empty

        let boundary = extract_cavity_boundary(dt.tds(), &conflict_simplices).unwrap();

        assert_eq!(
            boundary.len(),
            0,
            "Expected 0 boundary facets for empty conflict region"
        );
    }

    #[test]
    fn test_locate_with_stats_invalid_hint_uses_arbitrary_start_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.25, 0.25]);

        let invalid_hint = SimplexKey::from(KeyData::from_ffi(999_999));
        let expected_start = dt.tds().simplex_keys().next().unwrap();
        let (result, stats) =
            locate_with_stats(dt.tds(), &kernel, &point, Some(invalid_hint)).unwrap();

        assert_matches!(result, LocateResult::InsideSimplex(_));
        assert_eq!(stats.start_simplex, expected_start);
        assert!(!stats.used_hint);
        assert!(!stats.fell_back_to_scan());
    }

    #[test]
    fn test_locate_by_scan_inside_and_outside() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let expected_simplex = dt.tds().simplex_keys().next().unwrap();

        let inside = Point::from_validated_coords([0.2, 0.2]);
        let outside = Point::from_validated_coords([3.0, 3.0]);

        assert_eq!(
            locate_by_scan(dt.tds(), &kernel, &inside).unwrap(),
            LocateResult::InsideSimplex(expected_simplex)
        );
        assert_eq!(
            locate_by_scan(dt.tds(), &kernel, &outside).unwrap(),
            LocateResult::Outside
        );
    }

    #[test]
    fn test_extract_cavity_boundary_rejects_nonmanifold_facet_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let x_axis = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let y_axis = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let upper_right = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let top_apex = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.5]).unwrap(),
            )
            .unwrap();

        let first_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin, x_axis, y_axis], None).unwrap(),
            )
            .unwrap();
        let second_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin, x_axis, upper_right], None).unwrap(),
            )
            .unwrap();
        let third_simplex = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![origin, x_axis, top_apex], None).unwrap(),
            )
            .unwrap();

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(first_simplex);
        conflict_simplices.push(second_simplex);
        conflict_simplices.push(third_simplex);

        let err = extract_cavity_boundary(&tds, &conflict_simplices).unwrap_err();
        assert_matches!(
            err,
            ConflictError::NonManifoldFacet {
                simplex_count: 3,
                ..
            }
        );
    }

    #[test]
    fn test_extract_cavity_boundary_detects_disconnected_boundary_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let left_origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let left_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let left_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let right_origin = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, 0.0]).unwrap(),
            )
            .unwrap();
        let right_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([4.0, 0.0]).unwrap(),
            )
            .unwrap();
        let right_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, 1.0]).unwrap(),
            )
            .unwrap();

        let left = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![left_origin, left_x, left_y], None).unwrap(),
            )
            .unwrap();
        let right = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![right_origin, right_x, right_y], None).unwrap(),
            )
            .unwrap();

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(left);
        conflict_simplices.push(right);

        let err = extract_cavity_boundary(&tds, &conflict_simplices).unwrap_err();
        match err {
            ConflictError::DisconnectedBoundary {
                visited,
                total,
                disconnected_simplices,
            } => {
                assert!(visited < total);
                assert_eq!(total, 6);
                assert!(
                    !disconnected_simplices.is_empty(),
                    "disconnected_simplices should be non-empty"
                );
                for ck in &disconnected_simplices {
                    assert!(
                        tds.contains_simplex(*ck),
                        "disconnected simplex key {ck:?} should be present in the TDS"
                    );
                }
            }
            other => panic!("Expected DisconnectedBoundary, got {other:?}"),
        }
    }

    #[test]
    fn test_locate_with_stats_valid_hint_marks_used_hint() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.2, 0.2]);
        let hint = dt.tds().simplex_keys().next().unwrap();

        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, Some(hint)).unwrap();

        assert_matches!(result, LocateResult::InsideSimplex(_));
        assert_eq!(stats.start_simplex, hint);
        assert!(stats.used_hint);
        assert!(!stats.fell_back_to_scan());
    }

    #[test]
    fn test_locate_by_scan_empty_returns_outside() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.0, 0.0]);

        assert_eq!(
            locate_by_scan(&tds, &kernel, &point).unwrap(),
            LocateResult::Outside
        );
    }

    #[test]
    fn test_extract_cavity_boundary_invalid_conflict_simplex_key() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let invalid = SimplexKey::from(KeyData::from_ffi(424_242));
        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(invalid);

        let err = extract_cavity_boundary(dt.tds(), &conflict_simplices).unwrap_err();
        assert_matches!(
            err,
            ConflictError::InvalidStartSimplex { simplex_key } if simplex_key == invalid
        );
    }

    /// A simplex with fewer than D+1 vertex keys is rejected before orientation.
    #[test]
    fn test_is_point_outside_facet_underdimensioned_simplex_returns_invalid_arity() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Shrink simplex to only 2 vertices (D+1 = 3 required for D=2).
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3, 0.3]);
        let result = is_point_outside_facet(&tds, &kernel, simplex_key, 0, &point);
        assert!(
            matches!(
                result,
                Err(LocateError::InvalidSimplexArity {
                    simplex_key: key,
                    expected: 3,
                    found: 2,
                }) if key == simplex_key
            ),
            "underdimensioned simplex should return InvalidSimplexArity, got {result:?}"
        );
    }

    #[test]
    fn test_is_point_outside_facet_invalid_facet_index_returns_invalid_facet_index() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3, 0.3]);
        let result = is_point_outside_facet(&tds, &kernel, simplex_key, 3, &point);

        assert!(
            matches!(
                result,
                Err(LocateError::InvalidFacetIndex {
                    simplex_key: key,
                    facet_index: 3,
                    facet_count: 3,
                }) if key == simplex_key
            ),
            "invalid facet index should return InvalidFacetIndex, got {result:?}"
        );
    }

    /// When the vertex at `facet_idx` (the opposite vertex) is unresolvable,
    /// `is_point_outside_facet` returns a typed error before reaching the
    /// canonical ordering helpers.
    #[test]
    fn test_is_point_outside_facet_unresolvable_opposite_vertex_returns_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Replace vertex at index 0 (the opposite vertex for facet_idx=0)
        // with a missing key, keeping the simplex at D+1 vertex keys.
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(missing); // index 0 = opposite vertex for facet_idx=0
            simplex.push_vertex_key(v1);
            simplex.push_vertex_key(v2);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3, 0.3]);
        // facet_idx=0 → opposite_key = missing → unresolvable.
        let result = is_point_outside_facet(&tds, &kernel, simplex_key, 0, &point);
        assert!(
            matches!(
                result,
                Err(LocateError::MissingSimplexVertex {
                    simplex_key: key,
                    vertex_key,
                }) if key == simplex_key && vertex_key == missing
            ),
            "unresolvable opposite vertex should return MissingSimplexVertex, got {result:?}"
        );
    }

    /// `is_point_outside_facet` resolves vertex points via the canonical ordering
    /// helper `sorted_facet_points_with_extra`.  A simplex whose vertex-key list
    /// contains a key absent from the TDS causes the helper to return an error,
    /// which the function preserves as a typed locate error.
    #[test]
    fn test_is_point_outside_facet_degenerate_simplex_missing_vertex_returns_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Build a valid simplex first, then mutate its vertex list to include a missing key.
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let existing_vertices = tds.simplex(simplex_key).unwrap().vertices().to_vec();
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(existing_vertices[0]);
            simplex.push_vertex_key(existing_vertices[1]);
            simplex.push_vertex_key(missing);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3_f64, 0.3_f64]);

        // One facet vertex is missing from the TDS, so canonical point collection fails.
        let result = is_point_outside_facet(&tds, &kernel, simplex_key, 0, &point);
        assert!(
            matches!(
                result,
                Err(LocateError::MissingSimplexVertex {
                    simplex_key: key,
                    vertex_key,
                }) if key == simplex_key && vertex_key == missing
            ),
            "degenerate simplex with missing vertex should return MissingSimplexVertex, got {result:?}"
        );
    }

    /// When a conflict simplex's neighbor list references a non-existent simplex key,
    /// the BFS in `find_conflict_region` pops that key and fails to retrieve the
    /// simplex, returning `SimplexDataAccessFailed` with a "vanished" message.
    #[test]
    fn test_find_conflict_region_vanished_neighbor_returns_simplex_data_access_failed() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Wire a neighbor that doesn't exist in the TDS.
        let ghost = SimplexKey::from(KeyData::from_ffi(777_777));
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys([Some(ghost), None, None])
                .unwrap();
        }

        let kernel = FastKernel::<f64>::new();
        // Point inside the circumcircle so the start simplex is in conflict
        // and BFS tries to visit the ghost neighbor.
        let point = Point::from_validated_coords([0.2, 0.2]);
        let result = find_conflict_region(&tds, &kernel, &point, simplex_key);
        assert!(
            matches!(
                result,
                Err(ConflictError::SimplexDataAccessFailed { simplex_key: ck, .. }) if ck == ghost
            ),
            "expected SimplexDataAccessFailed for vanished neighbor, got {result:?}"
        );
    }

    /// When a simplex in the BFS has fewer than D+1 vertex keys, canonical point
    /// collection rejects it before predicate evaluation.
    #[test]
    fn test_find_conflict_region_underdimensioned_simplex_returns_invalid_simplex_arity() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Shrink simplex to only 2 vertices (both valid).
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3, 0.3]);
        let result = find_conflict_region(&tds, &kernel, &point, simplex_key);
        assert!(
            matches!(
                result,
                Err(ConflictError::InvalidSimplexArity {
                    simplex_key: ck,
                    expected: 3,
                    found: 2,
                }) if ck == simplex_key
            ),
            "expected InvalidSimplexArity for underdimensioned simplex, got {result:?}"
        );
    }

    /// `find_conflict_region` uses `sorted_simplex_points` in the BFS loop;
    /// a conflict simplex whose vertex-key list contains a key absent from the TDS
    /// causes the helper to return an error, yielding `Err(MissingSimplexVertex)`.
    #[test]
    fn test_find_conflict_region_degenerate_simplex_returns_missing_simplex_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Build valid simplex then mutate one vertex to a missing key.
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let existing_vertices = tds.simplex(simplex_key).unwrap().vertices().to_vec();
        let missing = VertexKey::from(KeyData::from_ffi(999_999));
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(existing_vertices[0]);
            simplex.push_vertex_key(existing_vertices[1]);
            simplex.push_vertex_key(missing);
        }

        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.3_f64, 0.3_f64]);

        // BFS visits the simplex; canonical point collection rejects the missing vertex.
        let result = find_conflict_region(&tds, &kernel, &point, simplex_key);
        assert!(
            matches!(
                result,
                Err(ConflictError::MissingSimplexVertex {
                    simplex_key: ck,
                    vertex_key,
                }) if ck == simplex_key && vertex_key == missing
            ),
            "expected MissingSimplexVertex for degenerate simplex, got {result:?}"
        );
    }

    /// Calling `locate_with_stats` with `hint = None` exercises the `_ =>` fallback arm
    /// of the hint-match, which picks an arbitrary start simplex and records `used_hint = false`.
    #[test]
    fn test_locate_with_stats_none_hint_picks_arbitrary_start_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.25_f64, 0.25_f64]);

        let (result, stats) = locate_with_stats(dt.tds(), &kernel, &point, None).unwrap();

        assert_matches!(result, LocateResult::InsideSimplex(_));
        assert!(!stats.used_hint, "None hint should set used_hint = false");
        assert!(!stats.fell_back_to_scan());
    }

    // =============================================================================
    // VERIFY CONFLICT REGION COMPLETENESS TESTS
    // =============================================================================
    // The production diagnostic is feature-gated; keep the unit coverage in
    // debug builds to avoid adding cost to release/bench test runs.

    #[cfg(debug_assertions)]
    /// Macro to test `verify_conflict_region_completeness` across dimensions.
    /// Builds a single-simplex Delaunay triangulation, finds the conflict region
    /// for an interior point via BFS, then verifies the brute-force check agrees.
    macro_rules! test_verify_conflict_region_complete_dimension {
        ($dim:literal, $inside_point:expr, $($coords:expr),+ $(,)?) => {{
            let vertices: Vec<_> = vec![
                $(crate::core::vertex::Vertex::<(), _>::try_new($coords).unwrap()),+
            ];
            let dt = DelaunayTriangulation::new(&vertices).unwrap();
            let kernel = FastKernel::<f64>::new();

            let start_simplex = dt.tds().simplex_keys().next().unwrap();
            let point = Point::from_validated_coords($inside_point);

            let conflict_simplices = find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();
            assert!(
                !conflict_simplices.is_empty(),
                "BFS should find at least 1 conflict simplex for interior point in {}D",
                $dim
            );

            let missed = verify_conflict_region_completeness(
                dt.tds(),
                &kernel,
                &point,
                &conflict_simplices,
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

    /// An empty BFS result should detect all conflict simplices as missed.
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_empty_bfs_detects_missed() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the circumcircle — the single simplex should be in conflict.
        let point = Point::from_validated_coords([0.3, 0.3]);
        let empty_bfs = SimplexKeyBuffer::new();

        let missed = verify_conflict_region_completeness(dt.tds(), &kernel, &point, &empty_bfs);
        assert!(
            missed > 0,
            "Empty BFS result should detect missed conflict simplices"
        );
    }

    /// Point far outside produces no conflict simplices; verify returns 0 missed.
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_outside_point_zero_missed() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_simplex = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([10.0, 10.0, 10.0]);

        let conflict_simplices =
            find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();
        assert!(conflict_simplices.is_empty());

        let missed =
            verify_conflict_region_completeness(dt.tds(), &kernel, &point, &conflict_simplices);
        assert_eq!(
            missed, 0,
            "Outside point should produce zero missed simplices"
        );
    }

    /// Truncated multi-simplex BFS result detects missed conflict simplices.
    ///
    /// Builds a 2D triangulation with 4 vertices (2 triangles sharing an edge),
    /// finds a multi-simplex conflict region, then drops one simplex from the BFS
    /// result and verifies that `verify_conflict_region_completeness` catches
    /// the omission.  Because the two triangles are adjacent, the dropped simplex
    /// has a neighbor still in the truncated BFS set, so the internal
    /// classification logs `REACHABLE_BUT_REJECTED` (observable via tracing).
    #[cfg(debug_assertions)]
    #[test]
    fn test_verify_conflict_region_completeness_truncated_multi_simplex_detects_missed() {
        // Four corners of a rectangle — DT produces 2 triangles sharing a diagonal.
        // All 4 points are co-circular, so a center-ish query point is strictly
        // inside both circumcircles → conflict region has 2 simplices.
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([4.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([4.0, 3.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 3.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let start_simplex = dt.tds().simplex_keys().next().unwrap();
        let point = Point::from_validated_coords([2.0, 1.5]);

        let full_conflict = find_conflict_region(dt.tds(), &kernel, &point, start_simplex).unwrap();
        assert!(
            full_conflict.len() >= 2,
            "Expected ≥2 conflict simplices for center query in 2-triangle mesh, got {}",
            full_conflict.len()
        );

        // Truncate: keep only the first simplex, drop the rest.
        let mut truncated = SimplexKeyBuffer::new();
        truncated.push(full_conflict[0]);

        let missed = verify_conflict_region_completeness(dt.tds(), &kernel, &point, &truncated);
        assert!(
            missed >= 1,
            "Truncated BFS should detect at least 1 missed conflict simplex, got {missed}"
        );
    }

    #[test]
    fn test_extract_cavity_boundary_rejects_ridge_fan_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let center = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let axis_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let axis_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let axis_neg_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let axis_neg_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let far_x = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();
        let far_y = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0]).unwrap(),
            )
            .unwrap();

        let first_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, axis_x, axis_y], None).unwrap(),
            )
            .unwrap();
        let second_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, axis_neg_x, axis_neg_y], None).unwrap(),
            )
            .unwrap();
        let third_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![center, far_x, far_y], None).unwrap(),
            )
            .unwrap();

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(first_simplex);
        conflict_simplices.push(second_simplex);
        conflict_simplices.push(third_simplex);

        match extract_cavity_boundary(&tds, &conflict_simplices).unwrap_err() {
            ConflictError::RidgeFan {
                facet_count,
                ridge_vertex_count,
                extra_simplices,
            } => {
                assert!(facet_count >= 3);
                assert_eq!(ridge_vertex_count, 1);
                // After deduplication, extra_simplices contains unique simplex keys contributing
                // the 3rd, 4th, … facets. Its length is ≤ facet_count - 2 and ≥ 1 here.
                assert!(
                    !extra_simplices.is_empty() && extra_simplices.len() <= facet_count - 2,
                    "deduped extra_simplices should be non-empty and not exceed facet_count - 2; got {} vs {}",
                    extra_simplices.len(),
                    facet_count - 2
                );
                // All entries must be valid keys from the TDS and unique.
                let mut seen = FastHashSet::default();
                for ck in &extra_simplices {
                    assert!(
                        tds.contains_simplex(*ck),
                        "extra simplex key {ck:?} should be present in the TDS"
                    );
                    assert!(seen.insert(*ck), "duplicate key {ck:?} in extra_simplices");
                }
            }
            other => panic!("Expected RidgeFan, got {other:?}"),
        }
    }
}
