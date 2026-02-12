//! Incremental Delaunay insertion using cavity-based algorithm.
//!
//! This module implements efficient incremental insertion following CGAL's approach:
//! 1. Locate the cell containing the new point (facet walking)
//! 2. Find conflict region (BFS with in_sphere tests)
//! 3. Extract cavity boundary facets
//! 4. Remove conflict cells
//! 5. Fill cavity (create new cells connecting boundary to new vertex)
//! 6. Wire neighbors locally (no global assign_neighbors call)
//!
//! ## Hull Extension and Visibility
//!
//! When inserting a vertex outside the current convex hull, the algorithm finds
//! *visible* boundary facets using orientation tests:
//! - A facet is **strictly visible** if the new point and the opposite vertex
//!   have opposite orientations relative to the facet's supporting hyperplane.
//! - **Coplanar cases** for the *query point* (orientation == 0) are treated as
//!   **weakly visible** to avoid missing horizon facets when the point lies on the
//!   hull plane. In **2D**, collinear cases are handled explicitly:
//!   - on-segment points trigger a boundary-edge split
//!   - off-segment collinearity is **not** treated as visible (avoids degenerate triangles)
//!   This still treats degeneracies of the hull facet itself (orientation_with_opposite == 0)
//!   as non-visible.
//! - For numerically robust weak visibility beyond coplanar cases, a threshold-based
//!   approach would be needed (not currently implemented).

#![forbid(unsafe_code)]

use crate::core::algorithms::locate::{ConflictError, LocateError, extract_cavity_boundary};
use crate::core::cell::Cell;
use crate::core::collections::{
    CellKeyBuffer, FastHashMap, FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
    VertexKeyBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::TriangulationConstructionError;
use crate::core::triangulation::TriangulationValidationError;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateScalar;
use std::hash::{Hash, Hasher};

pub use crate::core::operations::{InsertionOutcome, InsertionResult, InsertionStatistics};

/// Reason for hull extension failure.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::incremental_insertion::HullExtensionReason;
///
/// let reason = HullExtensionReason::NoVisibleFacets;
/// assert!(matches!(reason, HullExtensionReason::NoVisibleFacets));
/// ```
#[derive(Debug, Clone)]
pub enum HullExtensionReason {
    /// No visible boundary facets (coplanar with hull surface).
    NoVisibleFacets,
    /// Visible facets form an invalid patch.
    InvalidPatch {
        /// Details about why the patch was invalid.
        details: String,
    },
    /// Other failure.
    Other {
        /// Underlying error message.
        message: String,
    },
}

impl std::fmt::Display for HullExtensionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoVisibleFacets => f.write_str(
                "No visible boundary facets found for exterior vertex (may be coplanar with hull surface)",
            ),
            Self::InvalidPatch { details } => write!(
                f,
                "Visible boundary facets are not a valid patch: {details}"
            ),
            Self::Other { message } => f.write_str(message),
        }
    }
}

/// Error during incremental insertion.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::incremental_insertion::InsertionError;
///
/// let err = InsertionError::DuplicateCoordinates {
///     coordinates: "[0.0, 0.0, 0.0]".to_string(),
/// };
/// assert!(matches!(err, InsertionError::DuplicateCoordinates { .. }));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum InsertionError {
    /// Conflict region finding failed
    #[error("Conflict region error: {0}")]
    ConflictRegion(#[from] ConflictError),

    /// Point location failed
    #[error("Location error: {0}")]
    Location(#[from] LocateError),

    /// Triangulation construction failed
    #[error("Construction error: {0}")]
    Construction(#[from] TriangulationConstructionError),

    /// Cavity filling failed
    #[error("Cavity filling failed: {message}")]
    CavityFilling {
        /// Error message
        message: String,
    },

    /// Neighbor wiring failed
    #[error("Neighbor wiring failed: {message}")]
    NeighborWiring {
        /// Error message
        message: String,
    },

    /// Non-manifold topology detected during neighbor wiring.
    ///
    /// This occurs when a facet is shared by more than 2 cells, violating
    /// the manifold property. This is typically caused by geometric degeneracy
    /// and can often be resolved via coordinate perturbation.
    #[error(
        "Non-manifold topology: facet {facet_hash:#x} shared by {cell_count} cells (expected ≤2)"
    )]
    NonManifoldTopology {
        /// Hash of the facet vertices
        facet_hash: u64,
        /// Number of cells sharing this facet
        cell_count: usize,
    },

    /// Hull extension failed (finding visible boundary facets).
    #[error("Hull extension failed: {reason}")]
    HullExtension {
        /// Structured reason for failure.
        reason: HullExtensionReason,
    },

    /// Global Delaunay validation failed after insertion.
    ///
    /// This indicates the triangulation is structurally valid but violates the
    /// empty-circumsphere property (Level 4).
    #[error("Delaunay validation failed: {message}")]
    DelaunayValidationFailed {
        /// Error message
        message: String,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// String representation of the duplicate coordinates.
        coordinates: String,
    },

    /// Attempted to insert an entity with a UUID that already exists.
    #[error("Duplicate UUID: {entity:?} with UUID {uuid} already exists")]
    DuplicateUuid {
        /// The type of entity.
        entity: crate::core::triangulation_data_structure::EntityKind,
        /// The UUID that was duplicated.
        uuid: uuid::Uuid,
    },

    /// Topology validation or repair failed.
    #[error("Topology validation error: {0}")]
    TopologyValidation(#[from] crate::core::triangulation_data_structure::TdsValidationError),

    /// Level 3 topology validation failed (Triangulation layer).
    ///
    /// This preserves the structured [`TriangulationValidationError`] without wrapping it into a
    /// [`TdsValidationError`](crate::core::triangulation_data_structure::TdsValidationError),
    /// avoiding lower-layer (`Tds`) errors depending on higher-layer (`Triangulation`) errors.
    #[error("{message}: {source}")]
    TopologyValidationFailed {
        /// High-level context for when the topology validation failed.
        message: String,

        /// The underlying Level 3 validation error.
        #[source]
        source: TriangulationValidationError,
    },
}

impl InsertionError {
    /// Returns true if this error is retryable via coordinate perturbation.
    ///
    /// Retryable errors are geometric degeneracies that may be resolved by
    /// slightly perturbing the vertex coordinates:
    /// - Non-manifold topology (facets shared by >2 cells, ridge fans)
    /// - Topology validation failures during repair
    /// - Conflict-region boundary degeneracies (disconnected/open boundaries)
    ///
    /// Non-retryable errors are structural failures that won't be fixed by perturbation:
    /// - Duplicate UUIDs
    /// - Duplicate coordinates
    /// - Generic construction or wiring failures
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::algorithms::incremental_insertion::{HullExtensionReason, InsertionError};
    ///
    /// let retryable = InsertionError::NonManifoldTopology {
    ///     facet_hash: 1,
    ///     cell_count: 3,
    /// };
    /// assert!(retryable.is_retryable());
    ///
    /// let not_retryable = InsertionError::DuplicateCoordinates {
    ///     coordinates: "[0.0, 0.0, 0.0]".to_string(),
    /// };
    /// assert!(!not_retryable.is_retryable());
    ///
    /// let hull = InsertionError::HullExtension {
    ///     reason: HullExtensionReason::NoVisibleFacets,
    /// };
    /// assert!(hull.is_retryable());
    /// ```
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            // Non-manifold topology and topology validation errors are retryable via perturbation
            Self::NonManifoldTopology { .. }
            | Self::TopologyValidation(_)
            | Self::TopologyValidationFailed { .. } => true,
            // Legacy neighbor wiring errors: check message for non-manifold (backwards compatibility)
            Self::NeighborWiring { message } => message.contains("Non-manifold"),
            // Conflict region errors: non-manifold facets, ridge fans, or disconnected/open cavity
            // boundaries indicate degeneracy.
            Self::ConflictRegion(ce) => {
                matches!(
                    ce,
                    ConflictError::NonManifoldFacet { .. }
                        | ConflictError::RidgeFan { .. }
                        | ConflictError::DisconnectedBoundary { .. }
                        | ConflictError::OpenBoundary { .. }
                )
            }
            // All other errors are not retryable, except for the specific hull-extension
            // degeneracy handled below.
            //
            // Location errors are treated as non-retryable: `locate()` falls back to a scan when
            // facet-walking fails to make progress (cycle / step limit). Remaining location errors
            // are structural (invalid cell references) or predicate failures.
            Self::HullExtension { reason } => {
                // Hull extension can fail when the query point is nearly coplanar with the hull
                // surface (no *strictly* visible facets). This is a geometric degeneracy that may
                // be resolved by a perturbation retry.
                matches!(
                    reason,
                    HullExtensionReason::NoVisibleFacets | HullExtensionReason::InvalidPatch { .. }
                )
            }
            Self::Location(_)
            | Self::Construction(_)
            | Self::CavityFilling { .. }
            | Self::DelaunayValidationFailed { .. }
            | Self::DuplicateCoordinates { .. }
            | Self::DuplicateUuid { .. } => false,
        }
    }
}

/// Fill cavity by creating new cells connecting boundary facets to new vertex.
///
/// Each boundary facet becomes the base of a new (D+1)-cell with the new vertex as apex.
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_vertex_key` - Key of the newly inserted vertex
/// - `boundary_facets` - Facets forming the cavity boundary
///
/// # Returns
/// Buffer of newly created cell keys
///
/// # Errors
/// Returns error if cell creation or insertion fails.
///
/// # Partial Mutation on Error
///
/// **IMPORTANT**: If this function returns an error, the TDS may be left in a
/// partially updated state with some new cells already inserted. This function
/// does NOT rollback cells created before the error occurred.
///
/// This behavior is intentional for performance reasons (avoiding double validation)
/// and is safe because:
/// - This function is only called during vertex insertion operations
/// - If insertion fails, the entire triangulation is typically discarded
/// - Callers should not attempt to use or recover a triangulation after
///   receiving an error from this function
///
/// If you need transactional semantics (all-or-nothing insertion), you must
/// implement rollback logic at a higher level by cloning the TDS before calling
/// this function.
///
/// **Note (Debug Builds)**: In debug builds, this function checks for duplicate
/// boundary facets and logs warnings if found. Duplicate facets will create
/// overlapping cells, which will be detected and repaired by subsequent topology
/// validation passes (see `detect_local_facet_issues` / `repair_local_facet_issues`).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::incremental_insertion::fill_cavity;
/// use delaunay::core::facet::FacetHandle;
/// use delaunay::prelude::query::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let mut tds = dt.tds().clone();
/// let vkey = tds.vertex_keys().next().unwrap();
/// let boundary_facets: Vec<FacetHandle> = Vec::new();
///
/// let new_cells = fill_cavity(&mut tds, vkey, &boundary_facets).unwrap();
/// assert!(new_cells.is_empty());
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Cavity filling includes detailed debug instrumentation and error handling"
)]
pub fn fill_cavity<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    new_vertex_key: VertexKey,
    boundary_facets: &[FacetHandle],
) -> Result<CellKeyBuffer, InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    #[cfg(debug_assertions)]
    {
        let log_enabled = std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();
        // Check for duplicate boundary facets
        let mut seen_facets: FastHashMap<u64, Vec<FacetHandle>> = FastHashMap::default();
        for facet_handle in boundary_facets {
            if let Some(boundary_cell) = tds.get_cell(facet_handle.cell_key()) {
                let facet_idx = usize::from(facet_handle.facet_index());
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vertex_key) in boundary_cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vertex_key);
                    }
                }
                facet_vkeys.sort_unstable();
                let facet_hash = compute_facet_hash(&facet_vkeys);
                seen_facets
                    .entry(facet_hash)
                    .or_default()
                    .push(*facet_handle);
            }
        }
        let duplicates: Vec<_> = seen_facets
            .iter()
            .filter(|(_, handles)| handles.len() > 1)
            .collect();
        if !duplicates.is_empty() {
            tracing::warn!(
                duplicate_facets = duplicates.len(),
                "fill_cavity: duplicate boundary facets will create overlapping cells"
            );
            for (hash, handles) in &duplicates {
                tracing::warn!(
                    facet_hash = *hash,
                    instances = handles.len(),
                    handles = ?handles,
                    "fill_cavity: duplicate boundary facet"
                );
            }
        } else if log_enabled {
            tracing::debug!(
                boundary_facets = boundary_facets.len(),
                "fill_cavity: no duplicate boundary facet hashes"
            );
        }

        if log_enabled {
            let mut ridge_counts: FastHashMap<u64, usize> = FastHashMap::default();
            let mut ridge_vertices_map: FastHashMap<u64, VertexKeyBuffer> = FastHashMap::default();

            for facet_handle in boundary_facets {
                let Some(boundary_cell) = tds.get_cell(facet_handle.cell_key()) else {
                    tracing::warn!(
                        cell_key = ?facet_handle.cell_key(),
                        "fill_cavity: missing boundary cell while building ridge incidence"
                    );
                    continue;
                };
                let facet_idx = usize::from(facet_handle.facet_index());
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vertex_key) in boundary_cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vertex_key);
                    }
                }

                if facet_vkeys.len() < 2 {
                    continue;
                }

                for omit in 0..facet_vkeys.len() {
                    let mut ridge_vertices =
                        SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                    for (j, &vkey) in facet_vkeys.iter().enumerate() {
                        if j != omit {
                            ridge_vertices.push(vkey);
                        }
                    }
                    ridge_vertices.sort_unstable();
                    let ridge_hash = compute_facet_hash(&ridge_vertices);
                    *ridge_counts.entry(ridge_hash).or_insert(0) += 1;
                    ridge_vertices_map
                        .entry(ridge_hash)
                        .or_insert_with(|| ridge_vertices.iter().copied().collect());
                }
            }

            let mut ridge_boundary = 0usize;
            let mut ridge_internal = 0usize;
            let mut ridge_over_shared = 0usize;
            for count in ridge_counts.values() {
                match *count {
                    1 => ridge_boundary += 1,
                    2 => ridge_internal += 1,
                    _ => ridge_over_shared += 1,
                }
            }

            tracing::debug!(
                boundary_facets = boundary_facets.len(),
                ridge_boundary,
                ridge_internal,
                ridge_over_shared,
                "fill_cavity: boundary ridge incidence summary"
            );

            if ridge_over_shared > 0 {
                let mut logged = 0usize;
                for (&ridge_hash, &count) in &ridge_counts {
                    if count <= 2 {
                        continue;
                    }
                    if logged >= 10 {
                        break;
                    }
                    tracing::debug!(
                        ridge_hash,
                        ridge_count = count,
                        ridge_vertices = ?ridge_vertices_map.get(&ridge_hash),
                        "fill_cavity: ridge shared by >2 boundary facets"
                    );
                    logged += 1;
                }
            }
        }
    }

    let mut new_cells = CellKeyBuffer::new();

    for facet_handle in boundary_facets {
        let boundary_cell =
            tds.get_cell(facet_handle.cell_key())
                .ok_or_else(|| InsertionError::CavityFilling {
                    message: format!(
                        "Boundary facet cell {:?} not found",
                        facet_handle.cell_key()
                    ),
                })?;

        // Validate boundary cell has correct dimensionality (D+1 vertices)
        if boundary_cell.number_of_vertices() != D + 1 {
            return Err(InsertionError::CavityFilling {
                message: format!(
                    "Boundary cell {:?} has {} vertices, expected {} (D+1)",
                    facet_handle.cell_key(),
                    boundary_cell.number_of_vertices(),
                    D + 1
                ),
            });
        }

        let facet_idx = usize::from(facet_handle.facet_index());
        let mut new_cell_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        // Get vertices of the facet (all except the opposite vertex)
        for (i, &vertex_key) in boundary_cell.vertices().iter().enumerate() {
            if i != facet_idx {
                new_cell_vertices.push(vertex_key);
            }
        }

        // Add the new vertex as the apex
        new_cell_vertices.push(new_vertex_key);

        // Create and insert the new cell
        let new_cell =
            Cell::new(new_cell_vertices, None).map_err(|e| InsertionError::CavityFilling {
                message: format!("Failed to create cell: {e}"),
            })?;
        let cell_key =
            tds.insert_cell_with_mapping(new_cell)
                .map_err(|e| InsertionError::CavityFilling {
                    message: format!("Failed to insert cell: {e}"),
                })?;

        new_cells.push(cell_key);
    }

    // Defensive check: 1:1 correspondence is guaranteed by construction
    // (one iteration per boundary facet, one cell push per iteration)
    debug_assert_eq!(
        boundary_facets.len(),
        new_cells.len(),
        "Created {} cells for {} boundary facets (should be 1:1)",
        new_cells.len(),
        boundary_facets.len()
    );

    Ok(new_cells)
}

/// Wire neighbor relationships for newly created cavity cells.
///
/// This function wires:
/// - **Internal facets** between newly created cells (new↔new)
/// - **Boundary facets** between a new cell and an existing cell, using caller-supplied
///   external facet handles (new↔existing)
///
/// The design goal is to keep wiring **local**: callers provide the small set of
/// existing facets that bound the cavity/horizon, avoiding an O(#cells) global scan.
///
/// The algorithm:
/// 1. Index all facets of `new_cells` by a canonical facet hash (sorted vertex keys)
/// 2. For each `external_facet`, add it to the facet hash entry *only* if the entry
///    currently has exactly 1 incident cell (i.e., a new-cell boundary facet)
/// 3. Wire mutual neighbor relationships for facet-hash entries with exactly 2 incidents
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_cells` - Keys of newly created cells
/// - `external_facets` - Facets on existing cells that should be glued to the new cells
/// - `conflict_cells` - Optional set of cells being removed (for debug classification only)
///
/// # Returns
/// Ok(()) if wiring succeeds
///
/// # Errors
/// Returns error if neighbor wiring fails or cells cannot be found.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::incremental_insertion::wire_cavity_neighbors;
/// use delaunay::core::collections::CellKeyBuffer;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// let mut tds: Tds<f64, (), (), 3> = Tds::empty();
/// let new_cells = CellKeyBuffer::new();
///
/// wire_cavity_neighbors(&mut tds, &new_cells, [], None).unwrap();
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Neighbor wiring keeps cohesive logic and debug accounting together"
)]
pub fn wire_cavity_neighbors<T, U, V, const D: usize, I>(
    tds: &mut Tds<T, U, V, D>,
    new_cells: &CellKeyBuffer,
    external_facets: I,
    conflict_cells: Option<&CellKeyBuffer>,
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    I: IntoIterator<Item = FacetHandle>,
{
    type FacetIncidents = SmallBuffer<(CellKey, u8), 2>;
    type FacetMap = FastHashMap<u64, FacetIncidents>;
    let mut facet_map: FacetMap = FastHashMap::default();

    // `conflict_cells` is used only for debug instrumentation, but CI also compiles in
    // release mode with `-D warnings`.
    #[cfg(not(debug_assertions))]
    let _conflict_cells = conflict_cells;

    #[cfg(debug_assertions)]
    let log_enabled = std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();
    #[cfg(debug_assertions)]
    let ridge_link_debug = std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some();
    #[cfg(debug_assertions)]
    let mut skipped_external_matches: Vec<(u64, CellKey, usize)> = Vec::new();
    #[cfg(debug_assertions)]
    let mut skipped_external_count = 0usize;
    #[cfg(debug_assertions)]
    let mut unmatched_external: Vec<(CellKey, u8)> = Vec::new();
    #[cfg(debug_assertions)]
    let mut unmatched_external_count = 0usize;

    // Index all facets of new cells.
    for &cell_key in new_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("New cell {cell_key:?} not found"),
            })?;

        for facet_idx in 0..cell.number_of_vertices() {
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }

            facet_vkeys.sort_unstable();
            let facet_key = compute_facet_hash(&facet_vkeys);

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| InsertionError::NeighborWiring {
                    message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                })?;

            facet_map
                .entry(facet_key)
                .or_default()
                .push((cell_key, facet_idx_u8));
        }
    }

    // Index caller-supplied external facets (existing cells) that should glue to
    // new-cell boundary facets.
    for external in external_facets {
        let cell_key = external.cell_key();
        let facet_idx = usize::from(external.facet_index());

        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("External facet cell {cell_key:?} not found"),
            })?;

        if facet_idx >= cell.number_of_vertices() {
            return Err(InsertionError::NeighborWiring {
                message: format!(
                    "External facet index {facet_idx} out of range for cell {cell_key:?}"
                ),
            });
        }

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for (i, &vkey) in cell.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vkeys.push(vkey);
            }
        }
        facet_vkeys.sort_unstable();
        let facet_key = compute_facet_hash(&facet_vkeys);

        let Some(incidents) = facet_map.get_mut(&facet_key) else {
            #[cfg(debug_assertions)]
            {
                unmatched_external_count = unmatched_external_count.saturating_add(1);
                if unmatched_external.len() < 10 {
                    unmatched_external.push((cell_key, external.facet_index()));
                }
            }
            continue;
        };

        // Only glue to boundary facets (len == 1). If len >= 2, the facet is already
        // shared by multiple new cells (internal). Adding an external cell would create
        // a non-manifold facet shared by 3+ cells.
        if incidents.len() == 1 {
            incidents.push((cell_key, external.facet_index()));
        } else {
            #[cfg(debug_assertions)]
            if ridge_link_debug {
                skipped_external_count = skipped_external_count.saturating_add(1);
                if skipped_external_matches.len() < 10 {
                    skipped_external_matches.push((facet_key, cell_key, incidents.len()));
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    let conflict_set: FastHashSet<CellKey> = conflict_cells
        .map(|cells| cells.iter().copied().collect())
        .unwrap_or_default();
    #[cfg(debug_assertions)]
    let new_cells_set: FastHashSet<CellKey> = new_cells.iter().copied().collect();

    // Wire all matching facets (both internal and external).
    // Two cells share a facet if they have the same facet key.
    for (facet_key, cells) in &facet_map {
        if cells.len() == 2 {
            let (c1, idx1) = cells[0];
            let (c2, idx2) = cells[1];

            set_neighbor(tds, c1, idx1, Some(c2))?;
            set_neighbor(tds, c2, idx2, Some(c1))?;
        } else if cells.len() > 2 {
            #[cfg(debug_assertions)]
            {
                let cell_types: Vec<String> = cells
                    .iter()
                    .map(|(ck, _)| {
                        if new_cells_set.contains(ck) {
                            format!("NEW:{ck:?}")
                        } else if conflict_set.contains(ck) {
                            format!("CONFLICT:{ck:?}")
                        } else {
                            format!("EXISTING:{ck:?}")
                        }
                    })
                    .collect();
                tracing::warn!(
                    facet_hash = *facet_key,
                    cell_count = cells.len(),
                    cell_types = ?cell_types,
                    "wire_cavity_neighbors: non-manifold facet shared by >2 cells"
                );
            }
            return Err(InsertionError::NonManifoldTopology {
                facet_hash: *facet_key,
                cell_count: cells.len(),
            });
        }
        // cells.len() == 1 means it's a boundary facet (no neighbor)
    }

    #[cfg(debug_assertions)]
    if log_enabled {
        let mut boundary_facets = 0usize;
        let mut internal_facets = 0usize;
        let mut over_shared_facets = 0usize;
        for cells in facet_map.values() {
            match cells.len() {
                1 => boundary_facets += 1,
                2 => internal_facets += 1,
                _ => over_shared_facets += 1,
            }
        }
        tracing::debug!(
            new_cells = new_cells.len(),
            conflict_cells = conflict_cells.map_or(0, CellKeyBuffer::len),
            internal_facets,
            boundary_facets,
            over_shared_facets,
            "wire_cavity_neighbors: facet summary"
        );

        if over_shared_facets > 0 {
            let mut logged = 0usize;
            for (facet_key, cells) in &facet_map {
                if cells.len() <= 2 {
                    continue;
                }
                if logged >= 10 {
                    break;
                }
                tracing::debug!(
                    facet_hash = *facet_key,
                    cell_count = cells.len(),
                    cells = ?cells,
                    "wire_cavity_neighbors: facet shared by >2 cells in map"
                );
                logged += 1;
            }
        }
    }

    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_NEIGHBORS").is_some() {
        let mut mismatches = 0usize;
        for &cell_key in new_cells {
            let Some(cell) = tds.get_cell(cell_key) else {
                continue;
            };
            let Some(neighbors) = cell.neighbors() else {
                continue;
            };
            for (facet_idx, neighbor_opt) in neighbors.iter().enumerate() {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };
                let Some(neighbor_cell) = tds.get_cell(*neighbor_key) else {
                    continue;
                };
                let Some(mirror_idx) = cell.mirror_facet_index(facet_idx, neighbor_cell) else {
                    mismatches += 1;
                    tracing::warn!(
                        cell = ?cell_key,
                        facet_idx,
                        neighbor = ?neighbor_key,
                        "wire_cavity_neighbors: missing mirror facet index"
                    );
                    continue;
                };
                let neighbor_back = neighbor_cell
                    .neighbors()
                    .and_then(|ns| ns.get(mirror_idx).copied().flatten());
                if neighbor_back != Some(cell_key) {
                    mismatches += 1;
                    tracing::warn!(
                        cell = ?cell_key,
                        facet_idx,
                        neighbor = ?neighbor_key,
                        mirror_idx,
                        neighbor_back = ?neighbor_back,
                        "wire_cavity_neighbors: asymmetric neighbor pointer"
                    );
                }
            }
        }

        tracing::debug!(
            mismatches,
            "wire_cavity_neighbors: neighbor symmetry check complete"
        );
    }

    #[cfg(debug_assertions)]
    if ridge_link_debug {
        if skipped_external_count > 0 {
            tracing::debug!(
                skipped_external_count,
                skipped_external_matches = ?skipped_external_matches,
                "wire_cavity_neighbors: skipped external-facet matches (facet already shared by >1 new cell)"
            );
        }

        if unmatched_external_count > 0 {
            tracing::debug!(
                unmatched_external_count,
                unmatched_external = ?unmatched_external,
                "wire_cavity_neighbors: external facets did not match any new-cell facet hash"
            );
        }

        let mut total_slots = 0usize;
        let mut neighbor_new = 0usize;
        let mut neighbor_existing = 0usize;
        let mut neighbor_conflict = 0usize;
        let mut neighbor_missing = 0usize;
        let mut neighbor_none = 0usize;
        let mut anomaly_samples: Vec<(CellKey, usize, Option<CellKey>, String)> = Vec::new();

        for &cell_key in new_cells {
            let Some(cell) = tds.get_cell(cell_key) else {
                continue;
            };

            let vertex_count = cell.number_of_vertices();
            total_slots = total_slots.saturating_add(vertex_count);

            let Some(neighbors) = cell.neighbors() else {
                neighbor_none = neighbor_none.saturating_add(vertex_count);
                continue;
            };

            for (facet_idx, neighbor_opt) in neighbors.iter().enumerate() {
                match neighbor_opt {
                    None => {
                        neighbor_none = neighbor_none.saturating_add(1);
                    }
                    Some(neighbor_key) => {
                        if new_cells_set.contains(neighbor_key) {
                            neighbor_new = neighbor_new.saturating_add(1);
                        } else if conflict_set.contains(neighbor_key) {
                            neighbor_conflict = neighbor_conflict.saturating_add(1);
                            if anomaly_samples.len() < 10 {
                                anomaly_samples.push((
                                    cell_key,
                                    facet_idx,
                                    Some(*neighbor_key),
                                    "CONFLICT".to_string(),
                                ));
                            }
                        } else if tds.contains_cell(*neighbor_key) {
                            neighbor_existing = neighbor_existing.saturating_add(1);
                        } else {
                            neighbor_missing = neighbor_missing.saturating_add(1);
                            if anomaly_samples.len() < 10 {
                                anomaly_samples.push((
                                    cell_key,
                                    facet_idx,
                                    Some(*neighbor_key),
                                    "MISSING".to_string(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!(
            new_cells = new_cells.len(),
            total_slots,
            neighbor_new,
            neighbor_existing,
            neighbor_conflict,
            neighbor_missing,
            neighbor_none,
            "wire_cavity_neighbors: new-cell neighbor classification summary"
        );

        if neighbor_conflict > 0 || neighbor_missing > 0 {
            tracing::warn!(
                neighbor_conflict,
                neighbor_missing,
                anomaly_samples = ?anomaly_samples,
                "wire_cavity_neighbors: unexpected neighbor classifications for new cells"
            );
        }
    }

    Ok(())
}

/// Collect facets on existing (non-internal) cells that share a facet with the internal boundary.
///
/// Given:
/// - `internal_cells`: a set of cells that will be removed/replaced
/// - `boundary_facets`: facet handles on *internal* cells that lie on the boundary of that set
///
/// This returns facet handles on *external* cells (neighbors of `internal_cells`) whose facet
/// vertex sets match one of the boundary facets.
///
/// This is used to wire new cells to the pre-existing triangulation without performing a global
/// scan over all cells.
pub(crate) fn external_facets_for_boundary<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    internal_cells: &CellKeyBuffer,
    boundary_facets: &[FacetHandle],
) -> Result<SmallBuffer<FacetHandle, 64>, InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    if internal_cells.is_empty() || boundary_facets.is_empty() {
        return Ok(SmallBuffer::new());
    }

    let internal_set: FastHashSet<CellKey> = internal_cells.iter().copied().collect();

    // Hashes of boundary facets on internal cells.
    let mut boundary_hashes: FastHashSet<u64> = FastHashSet::default();
    for &facet in boundary_facets {
        let cell_key = facet.cell_key();
        let facet_idx = usize::from(facet.facet_index());

        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("Boundary facet cell {cell_key:?} not found"),
            })?;

        if facet_idx >= cell.number_of_vertices() {
            return Err(InsertionError::NeighborWiring {
                message: format!(
                    "Boundary facet index {facet_idx} out of range for cell {cell_key:?}"
                ),
            });
        }

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for (i, &vkey) in cell.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vkeys.push(vkey);
            }
        }
        facet_vkeys.sort_unstable();
        boundary_hashes.insert(compute_facet_hash(&facet_vkeys));
    }

    // Candidate external cells are those reachable via neighbor pointers from the internal set.
    let mut candidate_cells: FastHashSet<CellKey> = FastHashSet::default();
    for &cell_key in internal_cells {
        let Some(cell) = tds.get_cell(cell_key) else {
            continue;
        };
        let Some(neighbors) = cell.neighbors() else {
            continue;
        };

        for &neighbor_opt in neighbors {
            let Some(neighbor_key) = neighbor_opt else {
                continue;
            };
            if !internal_set.contains(&neighbor_key) {
                candidate_cells.insert(neighbor_key);
            }
        }
    }

    let mut external_facets: SmallBuffer<FacetHandle, 64> = SmallBuffer::new();

    for &cell_key in &candidate_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("External cell {cell_key:?} not found"),
            })?;

        for facet_idx in 0..cell.number_of_vertices() {
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();
            let facet_hash = compute_facet_hash(&facet_vkeys);

            if !boundary_hashes.contains(&facet_hash) {
                continue;
            }

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| InsertionError::NeighborWiring {
                    message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                })?;
            external_facets.push(FacetHandle::new(cell_key, facet_idx_u8));
        }
    }

    Ok(external_facets)
}

/// Helper: Set a single neighbor relationship
fn set_neighbor<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    cell_key: CellKey,
    facet_idx: u8,
    neighbor: Option<CellKey>,
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let cell = tds
        .get_cell_by_key_mut(cell_key)
        .ok_or_else(|| InsertionError::NeighborWiring {
            message: format!("Cell {cell_key:?} not found"),
        })?;

    let mut neighbors = cell.neighbors().map_or_else(
        || SmallBuffer::from_elem(None, D + 1),
        |n| n.iter().copied().collect(),
    );

    neighbors[usize::from(facet_idx)] = neighbor;
    cell.neighbors = Some(neighbors);

    Ok(())
}

/// Compute a hash for a facet from sorted vertex keys.
///
/// Uses [`FastHasher`] for deterministic hashing consistent with other
/// internal collections ([`FastHashMap`], [`FastHashSet`]).
fn compute_facet_hash(sorted_vkeys: &[VertexKey]) -> u64 {
    let mut hasher = FastHasher::default();
    for &vkey in sorted_vkeys {
        vkey.hash(&mut hasher);
    }
    hasher.finish()
}

/// Repair neighbor pointers using a global facet-incidence rebuild.
///
/// This performs a **global** reconstruction of the cell-neighbor graph:
/// - Build a facet → incident-cells map from vertex keys (purely combinatorial)
/// - Wire mutual neighbors for facets shared by exactly 2 cells
/// - Clear all other neighbor slots (boundary facets)
///
/// Unlike the original "scan for missing neighbors" approach, this avoids an
/// O(n²) facet-matching loop and runs in O(n·D) time.
///
/// # Returns
/// `Ok(n)` where `n` is the number of neighbor-pointer slots whose values changed.
///
/// # Errors
/// Returns [`InsertionError::NonManifoldTopology`] if any facet is shared by more than
/// 2 cells, since neighbor pointers are not well-defined in that case.
#[expect(
    clippy::too_many_lines,
    reason = "Neighbor rebuild keeps facet indexing + wiring + application cohesive; prefer correctness and debuggability"
)]
pub fn repair_neighbor_pointers<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
) -> Result<usize, InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    type FacetIncidents = SmallBuffer<(CellKey, u8), 2>;

    let cell_keys: Vec<CellKey> = tds.cells().map(|(key, _)| key).collect();

    #[cfg(debug_assertions)]
    tracing::trace!(
        cells = cell_keys.len(),
        "repair_neighbor_pointers: rebuilding neighbor pointers"
    );

    if cell_keys.is_empty() {
        return Ok(0);
    }

    // facet_hash -> [(cell_key, facet_index_opposite_to_facet)]
    let mut facet_map: FastHashMap<u64, FacetIncidents> = FastHashMap::default();

    // cell_key -> neighbor buffer (len = #vertices in the cell)
    let mut neighbors_by_cell: FastHashMap<
        CellKey,
        SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE>,
    > = FastHashMap::default();

    for &cell_key in &cell_keys {
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("Cell {cell_key:?} not found"),
            })?;

        let vertex_count = cell.number_of_vertices();
        let mut neighbors = SmallBuffer::with_capacity(vertex_count);
        neighbors.resize(vertex_count, None);
        neighbors_by_cell.insert(cell_key, neighbors);

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        for facet_idx in 0..vertex_count {
            facet_vkeys.clear();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();
            let facet_hash = compute_facet_hash(&facet_vkeys);

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| InsertionError::NeighborWiring {
                    message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                })?;

            let entry = facet_map.entry(facet_hash).or_default();
            entry.push((cell_key, facet_idx_u8));

            if entry.len() > 2 {
                return Err(InsertionError::NonManifoldTopology {
                    facet_hash,
                    cell_count: entry.len(),
                });
            }
        }
    }

    // Wire mutual neighbors for facets shared by exactly 2 cells.
    for (facet_hash, incidents) in facet_map {
        match incidents.as_slice() {
            [(c1, i1), (c2, i2)] => {
                {
                    let n1 = neighbors_by_cell.get_mut(c1).ok_or_else(|| {
                        InsertionError::NeighborWiring {
                            message: format!("Cell {c1:?} not found during neighbor rebuild"),
                        }
                    })?;
                    n1[usize::from(*i1)] = Some(*c2);
                }
                {
                    let n2 = neighbors_by_cell.get_mut(c2).ok_or_else(|| {
                        InsertionError::NeighborWiring {
                            message: format!("Cell {c2:?} not found during neighbor rebuild"),
                        }
                    })?;
                    n2[usize::from(*i2)] = Some(*c1);
                }
            }
            [_] | [] => {
                // Boundary facet => leave as None.
            }
            many => {
                return Err(InsertionError::NonManifoldTopology {
                    facet_hash,
                    cell_count: many.len(),
                });
            }
        }
    }

    // Apply rebuilt neighbors and count changed slots.
    let mut total_neighbor_slots_fixed: usize = 0;
    for (cell_key, rebuilt) in neighbors_by_cell {
        let old_neighbors: SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE> = {
            let cell = tds
                .get_cell(cell_key)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("Cell {cell_key:?} not found"),
                })?;

            cell.neighbors().map_or_else(
                || SmallBuffer::from_elem(None, rebuilt.len()),
                |ns| ns.iter().copied().collect(),
            )
        };

        total_neighbor_slots_fixed = total_neighbor_slots_fixed.saturating_add(
            old_neighbors
                .iter()
                .zip(rebuilt.iter())
                .filter(|(a, b)| a != b)
                .count(),
        );

        let cell =
            tds.get_cell_by_key_mut(cell_key)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("Cell {cell_key:?} not found"),
                })?;

        if rebuilt.iter().all(Option::is_none) {
            cell.neighbors = None;
        } else {
            cell.neighbors = Some(rebuilt);
        }
    }

    #[cfg(debug_assertions)]
    tracing::trace!(
        neighbor_pointers_updated = total_neighbor_slots_fixed,
        "repair_neighbor_pointers: neighbor rebuild complete"
    );

    // Validate no cycles were introduced (debug mode only)
    #[cfg(debug_assertions)]
    validate_no_neighbor_cycles(tds)?;

    Ok(total_neighbor_slots_fixed)
}

/// Debug-only sanity check for neighbor pointers.
///
/// This does **not** attempt to prove the neighbor graph is acyclic (triangulations
/// naturally contain cycles). Instead, it ensures that walking neighbor pointers from a
/// few sample cells:
/// - terminates (by visiting each discovered cell at most once), and
/// - does not encounter pointers to missing cell keys.
///
/// **Performance**: O(n·D) in the worst case for each sampled start cell.
///
/// # Errors
/// Returns `NeighborWiring` if a neighbor pointer references a missing cell key.
#[cfg(debug_assertions)]
fn validate_no_neighbor_cycles<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Sample a few cells and try walking through their neighbor graph.
    let sample_cells: Vec<CellKey> = tds.cells().map(|(key, _)| key).take(10).collect();
    let max_cells = tds.number_of_cells();

    for &start_cell in &sample_cells {
        let mut visited: FastHashSet<CellKey> = FastHashSet::default();
        let mut to_visit = vec![start_cell];
        visited.insert(start_cell);

        while let Some(current) = to_visit.pop() {
            let cell = tds
                .get_cell(current)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("Neighbor walk encountered missing cell {current:?}"),
                })?;

            let Some(neighbors) = cell.neighbors() else {
                continue;
            };

            for &neighbor_opt in neighbors {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };

                if neighbor_key == current {
                    return Err(InsertionError::NeighborWiring {
                        message: format!("Cell {current:?} has a self-neighbor pointer"),
                    });
                }

                if !tds.contains_cell(neighbor_key) {
                    return Err(InsertionError::NeighborWiring {
                        message: format!(
                            "Cell {current:?} has neighbor pointer to missing cell {neighbor_key:?}"
                        ),
                    });
                }

                if visited.insert(neighbor_key) {
                    to_visit.push(neighbor_key);
                    if visited.len() > max_cells {
                        return Err(InsertionError::NeighborWiring {
                            message: format!(
                                "Neighbor walk visited {} unique cells but triangulation contains {max_cells} cells",
                                visited.len()
                            ),
                        });
                    }
                }
            }
        }
    }

    tracing::trace!("validate_no_neighbor_cycles: neighbor walk terminated");
    Ok(())
}

/// Extend the convex hull by connecting an exterior vertex to visible boundary facets.
///
/// This function is used when a vertex is outside the current convex hull.
/// It finds all visible boundary facets and creates new cells connecting them to the new vertex.
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `kernel` - Geometric kernel for orientation tests
/// - `new_vertex_key` - Key of the newly inserted vertex
/// - `point` - Coordinates of the new vertex
///
/// # Returns
/// Buffer of newly created cell keys
///
/// # Errors
/// Returns error if:
/// - Finding visible facets fails
/// - Cavity filling or neighbor wiring fails
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::incremental_insertion::extend_hull;
/// use delaunay::core::triangulation_data_structure::Tds;
/// use delaunay::core::triangulation_data_structure::VertexKey;
/// use delaunay::geometry::kernel::FastKernel;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use slotmap::Key;
///
/// let mut tds: Tds<f64, (), (), 3> = Tds::empty();
/// let vkey = VertexKey::null();
/// let kernel = FastKernel::<f64>::new();
/// let point = Point::new([2.0, 2.0, 2.0]);
///
/// let result = extend_hull(&mut tds, &kernel, vkey, &point);
/// assert!(result.is_err());
/// ```
pub fn extend_hull<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    kernel: &K,
    new_vertex_key: VertexKey,
    point: &Point<K::Scalar, D>,
) -> Result<CellKeyBuffer, InsertionError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    // 2D special-case: if the point is collinear with a boundary edge and lies on
    // that edge segment, split the edge instead of building new hull triangles.
    if D == 2
        && let Some(edge_facet) = find_boundary_edge_split_facet(tds, kernel, point)?
    {
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
            tracing::debug!(
                point = ?point,
                cell_key = ?edge_facet.cell_key(),
                facet_index = usize::from(edge_facet.facet_index()),
                "extend_hull: 2D boundary-edge split"
            );
        }

        let mut conflict_cells = CellKeyBuffer::new();
        conflict_cells.push(edge_facet.cell_key());

        let mut boundary_facets = extract_cavity_boundary(tds, &conflict_cells)
            .map_err(InsertionError::ConflictRegion)?;
        boundary_facets.retain(|facet| {
            facet.cell_key() != edge_facet.cell_key()
                || facet.facet_index() != edge_facet.facet_index()
        });

        if boundary_facets.len() != 2 {
            return Err(InsertionError::HullExtension {
                reason: HullExtensionReason::Other {
                    message: format!(
                        "2D boundary edge split expected 2 facets, got {}",
                        boundary_facets.len()
                    ),
                },
            });
        }

        let external_facets = external_facets_for_boundary(tds, &conflict_cells, &boundary_facets)?;

        let new_cells = fill_cavity(tds, new_vertex_key, &boundary_facets)?;
        wire_cavity_neighbors(
            tds,
            &new_cells,
            external_facets.iter().copied(),
            Some(&conflict_cells),
        )?;
        let _ = tds.remove_cells_by_keys(&conflict_cells);

        return Ok(new_cells);
    }

    // Find visible boundary facets
    let visible_facets = match find_visible_boundary_facets(tds, kernel, point) {
        Ok(facets) => facets,
        Err(err) => {
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                tracing::debug!(error = ?err, "extend_hull: visibility computation failed");
            }
            return Err(err);
        }
    };

    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
        let total_boundary = tds.boundary_facets().map_or(0, Iterator::count);
        tracing::debug!(
            point = ?point,
            visible_facets = visible_facets.len(),
            total_boundary,
            "extend_hull: visibility summary"
        );
    }

    if visible_facets.is_empty() {
        return Err(InsertionError::HullExtension {
            reason: HullExtensionReason::NoVisibleFacets,
        });
    }

    // Fill cavity with new cells
    let new_cells = fill_cavity(tds, new_vertex_key, &visible_facets)?;

    // Wire neighbors using comprehensive facet matching
    // For hull extension, no conflict cells (nothing is removed)
    wire_cavity_neighbors(tds, &new_cells, visible_facets.iter().copied(), None)?;

    Ok(new_cells)
}

#[expect(
    clippy::too_many_lines,
    reason = "Visibility and edge-split checks are kept together for clarity"
)]
fn find_boundary_edge_split_facet<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
) -> Result<Option<FacetHandle>, InsertionError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    if D != 2 {
        return Ok(None);
    }

    let mut match_facet: Option<FacetHandle> = None;
    let tol = K::Scalar::default_tolerance();

    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::HullExtension {
            reason: HullExtensionReason::Other {
                message: format!("Failed to get boundary facets: {e}"),
            },
        })?;

    for facet_view in boundary_facets {
        let cell_key = facet_view.cell_key();
        let facet_index = facet_view.facet_index();
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::HullExtension {
                reason: HullExtensionReason::Other {
                    message: format!("Boundary facet cell {cell_key:?} not found"),
                },
            })?;

        let mut edge_points =
            SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut opposite_point: Option<Point<K::Scalar, D>> = None;

        for (i, &vkey) in cell.vertices().iter().enumerate() {
            let vertex =
                tds.get_vertex_by_key(vkey)
                    .ok_or_else(|| InsertionError::HullExtension {
                        reason: HullExtensionReason::Other {
                            message: format!("Vertex {vkey:?} not found in TDS"),
                        },
                    })?;
            if i == usize::from(facet_index) {
                opposite_point = Some(*vertex.point());
            } else {
                edge_points.push(*vertex.point());
            }
        }

        if edge_points.len() != 2 {
            continue;
        }

        let opposite_point = opposite_point.ok_or_else(|| InsertionError::HullExtension {
            reason: HullExtensionReason::Other {
                message: format!(
                    "Opposite vertex missing for facet {facet_index} in cell {cell_key:?}"
                ),
            },
        })?;

        let mut simplex_points =
            SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        simplex_points.extend(edge_points.iter().copied());
        simplex_points.push(opposite_point);
        let orientation_with_opposite =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::Other {
                        message: format!("Orientation test failed: {e}"),
                    },
                })?;

        if orientation_with_opposite == 0 {
            continue;
        }

        let mut edge_line = SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        edge_line.extend(edge_points.iter().copied());
        edge_line.push(*point);
        let orientation_with_point =
            kernel
                .orientation(&edge_line)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::Other {
                        message: format!("Orientation test failed: {e}"),
                    },
                })?;

        if orientation_with_point != 0 {
            continue;
        }

        let p0 = edge_points[0].coords();
        let p1 = edge_points[1].coords();
        let p = point.coords();
        let (min_x, max_x) = if p0[0] <= p1[0] {
            (p0[0], p1[0])
        } else {
            (p1[0], p0[0])
        };
        let (min_y, max_y) = if p0[1] <= p1[1] {
            (p0[1], p1[1])
        } else {
            (p1[1], p0[1])
        };
        let on_segment = p[0] >= min_x - tol
            && p[0] <= max_x + tol
            && p[1] >= min_y - tol
            && p[1] <= max_y + tol;

        if on_segment {
            let handle = FacetHandle::new(cell_key, facet_index);
            if match_facet.is_some() {
                return Err(InsertionError::HullExtension {
                    reason: HullExtensionReason::Other {
                        message: "2D boundary edge split matched multiple facets".to_string(),
                    },
                });
            }
            match_facet = Some(handle);
        }
    }

    Ok(match_facet)
}

/// Find all boundary facets visible from a point.
///
/// A boundary facet is visible from a point if the point is on the positive side
/// of the facet's supporting hyperplane (determined by orientation test).
///
/// **Visibility criterion:**
/// - **Strictly visible**: Opposite orientations (orientation signs differ)
/// - **Coplanar** (query orientation == 0): Treated as **weakly visible** to avoid
///   missing horizon facets when the point lies on the hull plane.
/// - **Facet degeneracy** (opposite orientation == 0): Treated as non-visible.
/// - For numerically robust weak visibility beyond coplanar cases, the orientation
///   test logic would need an epsilon-based threshold (not currently implemented).
///
/// # Arguments
/// - `tds` - The triangulation data structure
/// - `kernel` - Geometric kernel for orientation tests
/// - `point` - The query point
///
/// # Returns
/// Vector of facet handles representing visible boundary facets
///
/// # Errors
/// Returns `HullExtension` error if:
/// - Boundary facets cannot be retrieved
/// - Orientation tests fail
/// - Cell/vertex lookups fail
#[expect(
    clippy::too_many_lines,
    reason = "Visibility checks and diagnostic summaries are kept in a single routine"
)]
fn find_visible_boundary_facets<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
) -> Result<Vec<FacetHandle>, InsertionError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    let mut visible_facets = Vec::new();

    #[cfg(debug_assertions)]
    let log_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL").is_some();
    #[cfg(debug_assertions)]
    let detail_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL_DETAIL").is_some();
    #[cfg(debug_assertions)]
    let track_orientations = detail_enabled || log_enabled;
    #[cfg(debug_assertions)]
    let mut boundary_facets_count = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_positive = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_negative = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_zero = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_positive = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_negative = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_zero = 0usize;
    #[cfg(debug_assertions)]
    let mut visible_facets_strict = 0usize;
    #[cfg(debug_assertions)]
    let mut visible_facets_weak = 0usize;
    #[cfg(debug_assertions)]
    let mut degenerate_facets: Vec<FacetHandle> = Vec::new();

    // Get all boundary facets
    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::HullExtension {
            reason: HullExtensionReason::Other {
                message: format!("Failed to get boundary facets: {e}"),
            },
        })?;

    // Test each boundary facet for visibility
    for facet_view in boundary_facets {
        #[cfg(debug_assertions)]
        if track_orientations {
            boundary_facets_count += 1;
        }
        let cell_key = facet_view.cell_key();
        let facet_index = facet_view.facet_index();

        // Get the cell and its vertices
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::HullExtension {
                reason: HullExtensionReason::Other {
                    message: format!("Boundary facet cell {cell_key:?} not found"),
                },
            })?;

        // Collect points for the simplex in canonical order: facet vertices + opposite vertex.
        let mut simplex_points =
            SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut opposite_point: Option<Point<K::Scalar, D>> = None;

        for (i, &vkey) in cell.vertices().iter().enumerate() {
            let vertex =
                tds.get_vertex_by_key(vkey)
                    .ok_or_else(|| InsertionError::HullExtension {
                        reason: HullExtensionReason::Other {
                            message: format!("Vertex {vkey:?} not found in TDS"),
                        },
                    })?;
            if i == usize::from(facet_index) {
                opposite_point = Some(*vertex.point());
            } else {
                simplex_points.push(*vertex.point());
            }
        }

        let opposite_point = opposite_point.ok_or_else(|| InsertionError::HullExtension {
            reason: HullExtensionReason::Other {
                message: format!(
                    "Opposite vertex missing for facet {facet_index} in cell {cell_key:?}"
                ),
            },
        })?;

        // Append opposite vertex in canonical order.
        simplex_points.push(opposite_point);

        // Test orientation: if point is on same side as inside of hull, facet is visible.
        // For a boundary facet, we want to know if the new point is on the "outside" side
        // relative to the opposite vertex.
        let orientation_with_opposite =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::Other {
                        message: format!("Orientation test failed: {e}"),
                    },
                })?;

        // Replace opposite vertex with query point (last entry in canonical order).
        let last_index = simplex_points.len() - 1;
        simplex_points[last_index] = *point;
        let orientation_with_point =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::Other {
                        message: format!("Orientation test failed: {e}"),
                    },
                })?;

        #[cfg(debug_assertions)]
        if log_enabled && D == 2 && orientation_with_point == 0 {
            let p0 = simplex_points[0].coords();
            let p1 = simplex_points[1].coords();
            let p = point.coords();
            let tol = K::Scalar::default_tolerance();
            let (min_x, max_x) = if p0[0] <= p1[0] {
                (p0[0], p1[0])
            } else {
                (p1[0], p0[0])
            };
            let (min_y, max_y) = if p0[1] <= p1[1] {
                (p0[1], p1[1])
            } else {
                (p1[1], p0[1])
            };
            let min_x = min_x - tol;
            let max_x = max_x + tol;
            let min_y = min_y - tol;
            let max_y = max_y + tol;
            let on_segment = p[0] >= min_x && p[0] <= max_x && p[1] >= min_y && p[1] <= max_y;
            tracing::debug!(
                cell_key = ?cell_key,
                facet_index = usize::from(facet_index),
                point = ?point,
                edge_start = ?p0,
                edge_end = ?p1,
                on_segment,
                "find_visible_boundary_facets: query point coplanar with boundary edge"
            );
        }

        // Facet is visible if the point lies on the opposite side of the facet
        // relative to the opposite vertex. This avoids assuming consistent facet
        // orientations across the boundary.
        //
        // Weak visibility: treat query-point coplanarity as visible (orientation == 0).
        // Hull-facet degeneracy (orientation_with_opposite == 0) is still treated as
        // non-visible to avoid propagating degenerate facets.
        //
        // In 2D, collinear cases are handled via explicit boundary-edge splitting instead;
        // do not treat coplanar edges as visible here to avoid zero-area triangles.
        let is_strict_visible = orientation_with_opposite != 0
            && orientation_with_point != 0
            && orientation_with_opposite.signum() != orientation_with_point.signum();
        let is_weak_visible = if D == 2 {
            false
        } else {
            orientation_with_opposite != 0 && orientation_with_point == 0
        };
        let is_visible = is_strict_visible || is_weak_visible;

        #[cfg(debug_assertions)]
        if track_orientations {
            match orientation_with_opposite.cmp(&0) {
                std::cmp::Ordering::Greater => orientation_opposite_positive += 1,
                std::cmp::Ordering::Less => orientation_opposite_negative += 1,
                std::cmp::Ordering::Equal => orientation_opposite_zero += 1,
            }
            match orientation_with_point.cmp(&0) {
                std::cmp::Ordering::Greater => orientation_point_positive += 1,
                std::cmp::Ordering::Less => orientation_point_negative += 1,
                std::cmp::Ordering::Equal => orientation_point_zero += 1,
            }
            if is_strict_visible {
                visible_facets_strict += 1;
            }
            if is_weak_visible {
                visible_facets_weak += 1;
            }
        }
        #[cfg(debug_assertions)]
        if log_enabled && orientation_with_opposite == 0 && degenerate_facets.len() < 10 {
            degenerate_facets.push(FacetHandle::new(cell_key, facet_index));
        }
        #[cfg(debug_assertions)]
        if detail_enabled {
            tracing::trace!(
                cell_key = ?cell_key,
                facet_index = usize::from(facet_index),
                orientation_with_opposite,
                orientation_with_point,
                is_strict_visible,
                is_weak_visible,
                "find_visible_boundary_facets: facet orientation"
            );
        }

        if is_visible {
            visible_facets.push(FacetHandle::new(cell_key, facet_index));
        }
    }

    #[cfg(debug_assertions)]
    if log_enabled && visible_facets.is_empty() {
        tracing::debug!(
            point = ?point,
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            degenerate_facets = ?degenerate_facets,
            "find_visible_boundary_facets: no visible facets"
        );
    }

    if D >= 2 && !visible_facets.is_empty() {
        let mut ridge_to_facets: FastHashMap<u64, Vec<usize>> = FastHashMap::default();
        let mut ridge_counts: FastHashMap<u64, usize> = FastHashMap::default();
        let mut ridge_vertices_map: FastHashMap<u64, VertexKeyBuffer> = FastHashMap::default();

        for (facet_idx, facet_handle) in visible_facets.iter().enumerate() {
            let Some(cell) = tds.get_cell(facet_handle.cell_key()) else {
                #[cfg(debug_assertions)]
                tracing::warn!(
                    cell_key = ?facet_handle.cell_key(),
                    "find_visible_boundary_facets: missing cell while summarizing ridges"
                );
                continue;
            };
            let facet_index = usize::from(facet_handle.facet_index());
            let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_index {
                    facet_vertices.push(vkey);
                }
            }

            if facet_vertices.len() < 2 {
                continue;
            }

            for omit in 0..facet_vertices.len() {
                let mut ridge_vertices =
                    SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (j, &vkey) in facet_vertices.iter().enumerate() {
                    if j != omit {
                        ridge_vertices.push(vkey);
                    }
                }
                ridge_vertices.sort_unstable();
                let ridge_hash = compute_facet_hash(&ridge_vertices);
                ridge_to_facets
                    .entry(ridge_hash)
                    .or_default()
                    .push(facet_idx);
                *ridge_counts.entry(ridge_hash).or_insert(0) += 1;
                ridge_vertices_map
                    .entry(ridge_hash)
                    .or_insert_with(|| ridge_vertices.clone());
            }
        }

        let mut boundary_ridges = 0usize;
        #[cfg(debug_assertions)]
        let mut internal_ridges = 0usize;
        let mut over_shared_ridges = 0usize;
        for count in ridge_counts.values() {
            match *count {
                1 => boundary_ridges += 1,
                2 => {
                    #[cfg(debug_assertions)]
                    {
                        internal_ridges += 1;
                    }
                }
                _ => over_shared_ridges += 1,
            }
        }

        let mut boundary_components = 0usize;
        let mut boundary_subface_nonmanifold = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_component_sizes: Vec<usize> = Vec::new();
        #[cfg(debug_assertions)]
        let mut boundary_degree_min: Option<usize> = None;
        #[cfg(debug_assertions)]
        let mut boundary_degree_max: Option<usize> = None;
        #[cfg(debug_assertions)]
        let mut boundary_degree_zero = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_one = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_two = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_over = 0usize;
        #[cfg(debug_assertions)]
        let want_subface_samples = detail_enabled || log_enabled;
        #[cfg(debug_assertions)]
        let mut subface_samples: Vec<(u64, usize, Option<VertexKeyBuffer>)> = Vec::new();
        if D >= 3 && boundary_ridges > 0 {
            let mut boundary_ridge_keys: Vec<u64> = Vec::with_capacity(boundary_ridges);
            for (ridge_hash, count) in &ridge_counts {
                if *count == 1 {
                    boundary_ridge_keys.push(*ridge_hash);
                }
            }

            let mut face_to_ridges: FastHashMap<u64, Vec<usize>> = FastHashMap::default();
            let mut subface_vertices_map: FastHashMap<u64, VertexKeyBuffer> =
                FastHashMap::default();
            let mut subface_vertices: VertexKeyBuffer = VertexKeyBuffer::new();

            for (idx, ridge_hash) in boundary_ridge_keys.iter().enumerate() {
                let Some(ridge_vertices) = ridge_vertices_map.get(ridge_hash) else {
                    continue;
                };
                if ridge_vertices.len() < 2 {
                    continue;
                }
                for omit in 0..ridge_vertices.len() {
                    subface_vertices.clear();
                    for (j, &vk) in ridge_vertices.iter().enumerate() {
                        if j != omit {
                            subface_vertices.push(vk);
                        }
                    }
                    subface_vertices.sort_unstable();
                    let subface_hash = compute_facet_hash(&subface_vertices);
                    face_to_ridges.entry(subface_hash).or_default().push(idx);
                    subface_vertices_map
                        .entry(subface_hash)
                        .or_insert_with(|| subface_vertices.iter().copied().collect());
                }
            }

            let mut ridge_adjacency: Vec<Vec<usize>> = vec![Vec::new(); boundary_ridge_keys.len()];
            for (subface_hash, ridges) in &face_to_ridges {
                if ridges.len() != 2 {
                    boundary_subface_nonmanifold += 1;
                    #[cfg(debug_assertions)]
                    if want_subface_samples && subface_samples.len() < 10 {
                        subface_samples.push((
                            *subface_hash,
                            ridges.len(),
                            subface_vertices_map.get(subface_hash).cloned(),
                        ));
                    }
                }
                #[cfg(not(debug_assertions))]
                let _subface_hash = subface_hash;
                if ridges.len() < 2 {
                    continue;
                }
                for i in 0..ridges.len() {
                    for j in (i + 1)..ridges.len() {
                        let a = ridges[i];
                        let b = ridges[j];
                        ridge_adjacency[a].push(b);
                        ridge_adjacency[b].push(a);
                    }
                }
            }

            #[cfg(debug_assertions)]
            for adj in &ridge_adjacency {
                let degree = adj.len();
                match degree {
                    0 => boundary_degree_zero += 1,
                    1 => boundary_degree_one += 1,
                    2 => boundary_degree_two += 1,
                    _ => boundary_degree_over += 1,
                }
                boundary_degree_min =
                    Some(boundary_degree_min.map_or(degree, |min| min.min(degree)));
                boundary_degree_max =
                    Some(boundary_degree_max.map_or(degree, |max| max.max(degree)));
            }

            let mut visited = vec![false; boundary_ridge_keys.len()];
            for start in 0..boundary_ridge_keys.len() {
                if visited[start] {
                    continue;
                }
                boundary_components += 1;
                let mut stack = vec![start];
                visited[start] = true;
                #[cfg(debug_assertions)]
                let mut component_size = 0usize;
                while let Some(r) = stack.pop() {
                    #[cfg(debug_assertions)]
                    {
                        component_size += 1;
                    }
                    for &n in &ridge_adjacency[r] {
                        if !visited[n] {
                            visited[n] = true;
                            stack.push(n);
                        }
                    }
                }
                #[cfg(debug_assertions)]
                boundary_component_sizes.push(component_size);
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); visible_facets.len()];
        for facets in ridge_to_facets.values() {
            if facets.len() < 2 {
                continue;
            }
            for i in 0..facets.len() {
                for j in (i + 1)..facets.len() {
                    let a = facets[i];
                    let b = facets[j];
                    adjacency[a].push(b);
                    adjacency[b].push(a);
                }
            }
        }

        let mut visited = vec![false; visible_facets.len()];
        let mut components = 0usize;
        #[cfg(debug_assertions)]
        let mut component_sizes: Vec<usize> = Vec::new();
        for start in 0..visible_facets.len() {
            if visited[start] {
                continue;
            }
            components += 1;
            let mut stack = vec![start];
            visited[start] = true;
            #[cfg(debug_assertions)]
            let mut component_size = 0usize;
            while let Some(f) = stack.pop() {
                #[cfg(debug_assertions)]
                {
                    component_size += 1;
                }
                for &n in &adjacency[f] {
                    if !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
            #[cfg(debug_assertions)]
            component_sizes.push(component_size);
        }

        if over_shared_ridges > 0
            || components > 1
            || boundary_ridges == 0
            || (D >= 3 && boundary_components > 1)
            || (D >= 3 && boundary_subface_nonmanifold > 0)
        {
            #[cfg(debug_assertions)]
            if detail_enabled || log_enabled {
                let visible_sample = &visible_facets[..visible_facets.len().min(10)];
                tracing::debug!(
                    point = ?point,
                    visible_facets = visible_facets.len(),
                    visible_facets_sample = ?visible_sample,
                    ridge_boundary = boundary_ridges,
                    ridge_internal = internal_ridges,
                    ridge_over_shared = over_shared_ridges,
                    components,
                    component_sizes = ?component_sizes,
                    boundary_components,
                    boundary_component_sizes = ?boundary_component_sizes,
                    boundary_subface_nonmanifold,
                    boundary_degree_min,
                    boundary_degree_max,
                    boundary_degree_zero,
                    boundary_degree_one,
                    boundary_degree_two,
                    boundary_degree_over,
                    orientation_opposite_zero,
                    orientation_point_zero,
                    degenerate_facets = ?degenerate_facets,
                    subface_samples = ?subface_samples,
                    "find_visible_boundary_facets: invalid patch"
                );
            }
            return Err(InsertionError::HullExtension {
                reason: HullExtensionReason::InvalidPatch {
                    details: format!(
                        "boundary_ridges={boundary_ridges}, ridge_fans={over_shared_ridges}, components={components}, boundary_components={boundary_components}, boundary_subface_nonmanifold={boundary_subface_nonmanifold}",
                    ),
                },
            });
        }

        #[cfg(debug_assertions)]
        if detail_enabled {
            tracing::debug!(
                visible_facets = visible_facets.len(),
                ridge_boundary = boundary_ridges,
                ridge_internal = internal_ridges,
                ridge_over_shared = over_shared_ridges,
                components,
                component_sizes = ?component_sizes,
                boundary_components,
                boundary_component_sizes = ?boundary_component_sizes,
                boundary_subface_nonmanifold,
                boundary_degree_min,
                boundary_degree_max,
                boundary_degree_zero,
                boundary_degree_one,
                boundary_degree_two,
                boundary_degree_over,
                orientation_opposite_zero,
                orientation_point_zero,
                "find_visible_boundary_facets: ridge connectivity summary"
            );
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        let mixed_orientation =
            orientation_opposite_positive > 0 && orientation_opposite_negative > 0;
        tracing::debug!(
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            mixed_orientation,
            "find_visible_boundary_facets: boundary facet orientation consistency"
        );
        tracing::debug!(
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            "find_visible_boundary_facets: orientation summary"
        );
    }

    Ok(visible_facets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::CellKeyBuffer;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::core::triangulation_data_structure::TdsValidationError;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use slotmap::KeyData;

    /// Macro to generate cavity filling tests for different dimensions
    macro_rules! test_fill_cavity {
        ($dim:literal, $initial_vertices:expr, $new_vertex:expr, $expected_facets:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_fill_cavity_ $dim d>]() {
                    // Create initial simplex
                    let vertices = $initial_vertices;
                    let mut dt = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).unwrap();
                    let tds = dt.tds_mut();

                    // Insert new vertex
                    let new_vertex = $new_vertex;
                    let new_vkey = tds.insert_vertex_with_mapping(new_vertex).unwrap();

                    // Find the single cell and create boundary facets (one per face)
                    let cell_key = tds.cell_keys().next().unwrap();
                    let boundary_facets: Vec<FacetHandle> = (0..=$dim)
                        .map(|i| FacetHandle::new(cell_key, i))
                        .collect();

                    // Verify expected number of facets
                    assert_eq!(boundary_facets.len(), $expected_facets);

                    // Fill cavity
                    let new_cells = fill_cavity(tds, new_vkey, &boundary_facets).unwrap();

                    // Should create one cell per boundary facet
                    assert_eq!(new_cells.len(), $expected_facets);

                    // Wire neighbors (glue new cells to the original cell's facets)
                    wire_cavity_neighbors(
                        tds,
                        &new_cells,
                        boundary_facets.iter().copied(),
                        None,
                    )
                    .unwrap();

                    // Validate neighbor consistency
                    assert!(
                        tds.is_valid().is_ok(),
                        "TDS should be valid after cavity filling and neighbor wiring: {:?}",
                        tds.is_valid().err()
                    );

                    // Verify all new cells have correct vertex count
                    for &cell_key in &new_cells {
                        let cell = tds.get_cell(cell_key).unwrap();
                        assert_eq!(
                            cell.number_of_vertices(),
                            $dim + 1,
                            "New cell should have D+1 vertices"
                        );
                    }
                }
            }
        };
    }

    // Generate tests for dimensions 2-5
    test_fill_cavity!(
        2,
        vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ],
        vertex!([0.5, 0.5]),
        3 // D+1 facets for a 2-simplex
    );

    test_fill_cavity!(
        3,
        vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ],
        vertex!([0.25, 0.25, 0.25]),
        4 // D+1 facets for a 3-simplex
    );

    test_fill_cavity!(
        4,
        vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ],
        vertex!([0.2, 0.2, 0.2, 0.2]),
        5 // D+1 facets for a 4-simplex
    );

    test_fill_cavity!(
        5,
        vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
        vertex!([0.15, 0.15, 0.15, 0.15, 0.15]),
        6 // D+1 facets for a 5-simplex
    );

    // Error case tests

    #[test]
    fn test_fill_cavity_with_invalid_vertex_key() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        let cell_key = tds.cell_keys().next().unwrap();
        let boundary_facets: Vec<FacetHandle> =
            (0..=2).map(|i| FacetHandle::new(cell_key, i)).collect();

        let result = fill_cavity(tds, invalid_vkey, &boundary_facets);
        assert!(result.is_err());

        if let Err(InsertionError::CavityFilling { message }) = result {
            assert!(
                message.contains("not found")
                    || message.contains("invalid")
                    || message.contains("does not exist")
            );
        } else {
            panic!("Expected CavityFilling error");
        }
    }

    #[test]
    fn test_fill_cavity_with_invalid_facet_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let new_vkey = tds.insert_vertex_with_mapping(vertex!([0.5, 0.5])).unwrap();
        let invalid_cell_key = CellKey::from(KeyData::from_ffi(u64::MAX));
        let invalid_boundary_facets: Vec<FacetHandle> = (0..=2)
            .map(|i| FacetHandle::new(invalid_cell_key, i))
            .collect();

        let result = fill_cavity(tds, new_vkey, &invalid_boundary_facets);
        assert!(result.is_err());
        assert!(matches!(result, Err(InsertionError::CavityFilling { .. })));
    }

    #[test]
    fn test_wire_cavity_neighbors_with_invalid_cells() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let mut invalid_cells = CellKeyBuffer::new();
        invalid_cells.push(CellKey::from(KeyData::from_ffi(u64::MAX)));
        invalid_cells.push(CellKey::from(KeyData::from_ffi(u64::MAX - 1)));

        let result = wire_cavity_neighbors(tds, &invalid_cells, [], None);
        assert!(result.is_err());
        assert!(matches!(result, Err(InsertionError::NeighborWiring { .. })));
    }

    #[test]
    fn test_external_facets_for_boundary_finds_shared_edge_only() {
        // Two triangles share one edge; only that edge should be returned as an external facet.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v0, v3], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut internal_cells = CellKeyBuffer::new();
        internal_cells.push(c1);

        // Internal set has a single cell, so all its facets are boundary facets.
        let boundary_facets: Vec<FacetHandle> = (0..=2).map(|i| FacetHandle::new(c1, i)).collect();

        let external_facets =
            external_facets_for_boundary(&tds, &internal_cells, &boundary_facets).unwrap();
        assert_eq!(external_facets.len(), 1);

        let external = external_facets[0];
        assert_eq!(external.cell_key(), c2);

        let cell = tds.get_cell(external.cell_key()).unwrap();
        let facet_idx = usize::from(external.facet_index());

        let mut edge: SmallBuffer<VertexKey, 2> = SmallBuffer::new();
        for (i, &vkey) in cell.vertices().iter().enumerate() {
            if i != facet_idx {
                edge.push(vkey);
            }
        }
        assert_eq!(edge.len(), 2);
        assert!(edge.contains(&v0));
        assert!(edge.contains(&v1));
    }

    #[test]
    fn test_fill_cavity_with_empty_boundary_facets() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let new_vkey = tds.insert_vertex_with_mapping(vertex!([0.5, 0.5])).unwrap();
        let empty_facets: Vec<FacetHandle> = vec![];
        let result = fill_cavity(tds, new_vkey, &empty_facets);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_fill_cavity_errors_on_boundary_cell_wrong_vertex_count() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        // Insert a new vertex (apex)
        let new_vkey = tds.insert_vertex_with_mapping(vertex!([0.5, 0.5])).unwrap();

        // Corrupt the single boundary cell by adding one extra vertex key.
        let cell_key = tds.cell_keys().next().unwrap();
        let extra_vkey = tds.get_cell(cell_key).unwrap().vertices()[0];
        tds.get_cell_by_key_mut(cell_key)
            .unwrap()
            .push_vertex_key(extra_vkey);

        let boundary_facets = vec![FacetHandle::new(cell_key, 0)];
        let err = fill_cavity(tds, new_vkey, &boundary_facets).unwrap_err();

        assert!(matches!(err, InsertionError::CavityFilling { .. }));
    }

    #[test]
    fn test_wire_cavity_neighbors_reports_non_manifold_topology() {
        // Three triangles sharing the same edge (v_a,v_b) is non-manifold in 2D.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([2.0, 0.0])).unwrap();

        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();
        let c3 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_e], None).unwrap())
            .unwrap();

        let mut new_cells = CellKeyBuffer::new();
        new_cells.push(c1);
        new_cells.push(c2);
        new_cells.push(c3);

        let err = wire_cavity_neighbors(&mut tds, &new_cells, [], None).unwrap_err();
        assert!(matches!(
            err,
            InsertionError::NonManifoldTopology { cell_count: 3, .. }
        ));
    }

    #[test]
    fn test_wire_cavity_neighbors_errors_on_facet_index_overflow() {
        // Force a cell to have > 255 vertices so facet_idx -> u8 conversion fails.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let cell_key = tds.cell_keys().next().unwrap();
        let vkey0 = tds.get_cell(cell_key).unwrap().vertices()[0];

        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            while cell.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                cell.push_vertex_key(vkey0);
            }
        }

        let mut new_cells = CellKeyBuffer::new();
        new_cells.push(cell_key);

        let err = wire_cavity_neighbors(tds, &new_cells, [], None).unwrap_err();
        assert!(matches!(
            err,
            InsertionError::NeighborWiring { message } if message.contains("Facet index")
        ));
    }

    // InsertionError::is_retryable() tests

    #[test]
    fn test_insertion_error_retryable() {
        // Retryable errors
        assert!(
            InsertionError::NonManifoldTopology {
                facet_hash: 0x12345,
                cell_count: 3
            }
            .is_retryable()
        );

        assert!(
            InsertionError::TopologyValidation(TdsValidationError::InconsistentDataStructure {
                message: "test".to_string()
            })
            .is_retryable()
        );

        let level3_err =
            TriangulationValidationError::from(TdsValidationError::InconsistentDataStructure {
                message: "test".to_string(),
            });
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: level3_err,
            }
            .is_retryable()
        );

        assert!(
            InsertionError::NeighborWiring {
                message: "Non-manifold topology detected".to_string()
            }
            .is_retryable()
        );

        // Conflict-region errors
        assert!(
            InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
                facet_hash: 0x12345_u64,
                cell_count: 3,
            })
            .is_retryable()
        );
        assert!(
            InsertionError::ConflictRegion(ConflictError::RidgeFan {
                facet_count: 3,
                ridge_vertex_count: 2,
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InvalidStartCell {
                cell_key: CellKey::from(KeyData::from_ffi(u64::MAX)),
            })
            .is_retryable()
        );

        // Non-retryable errors
        assert!(
            !InsertionError::DuplicateUuid {
                entity: crate::core::triangulation_data_structure::EntityKind::Vertex,
                uuid: uuid::Uuid::new_v4(),
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::DuplicateCoordinates {
                coordinates: "0,0,0".to_string()
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::CavityFilling {
                message: "test".to_string()
            }
            .is_retryable()
        );

        assert!(
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::HullExtension {
                reason: HullExtensionReason::Other {
                    message: "Failed to get boundary facets: test".to_string()
                }
            }
            .is_retryable()
        );
    }

    // repair_neighbor_pointers tests

    /// Macro to generate `repair_neighbor_pointers` tests for different dimensions
    macro_rules! test_repair_neighbors {
        ($dim:literal, $initial_vertices:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_repair_neighbor_pointers_ $dim d>]() {
                    let vertices = $initial_vertices;
                    let mut dt = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).unwrap();
                    let tds = dt.tds_mut();

                    // Verify all neighbor pointers are initially valid
                    for (_, cell) in tds.cells() {
                        if let Some(neighbors) = cell.neighbors() {
                            for &neighbor_opt in neighbors {
                                if let Some(neighbor_key) = neighbor_opt {
                                    assert!(tds.contains_cell(neighbor_key), "Neighbor should exist");
                                }
                            }
                        }
                    }

                    // Repair should succeed (no-op since pointers are valid)
                    let fixed = repair_neighbor_pointers(tds).unwrap();
                    assert_eq!(fixed, 0);

                    // Verify all pointers still valid after repair
                    for (_, cell) in tds.cells() {
                        if let Some(neighbors) = cell.neighbors() {
                            for &neighbor_opt in neighbors {
                                if let Some(neighbor_key) = neighbor_opt {
                                    assert!(tds.contains_cell(neighbor_key), "Neighbor should still exist after repair");
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    test_repair_neighbors!(
        2,
        vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ]
    );

    test_repair_neighbors!(
        3,
        vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.25, 0.25, 0.25]),
        ]
    );

    test_repair_neighbors!(
        4,
        vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
            vertex!([0.2, 0.2, 0.2, 0.2]),
        ]
    );

    test_repair_neighbors!(
        5,
        vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
            vertex!([0.15, 0.15, 0.15, 0.15, 0.15]),
        ]
    );

    #[test]
    fn test_repair_neighbor_pointers_reconstructs_missing_neighbors() {
        // Create a simple 2D triangulation with two triangles.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.1]), // break cocircular symmetry
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        // Remove all neighbor pointers.
        tds.clear_all_neighbors();
        assert!(tds.cells().all(|(_, c)| c.neighbors().is_none()));

        // Repair should rebuild internal adjacencies.
        let fixed = repair_neighbor_pointers(tds).unwrap();
        assert!(
            fixed > 0,
            "Expected at least one neighbor pointer to be repaired"
        );

        let any_internal_neighbor = tds
            .cells()
            .any(|(_, c)| c.neighbors().is_some_and(|n| n.iter().any(Option::is_some)));
        assert!(
            any_internal_neighbor,
            "Expected at least one internal neighbor after repair"
        );

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_extend_hull_adds_cells_for_exterior_vertex() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let kernel = FastKernel::<f64>::new();
        let p = Point::new([2.0, 2.0]);
        let new_vkey = tds.insert_vertex_with_mapping(vertex!([2.0, 2.0])).unwrap();

        let new_cells = extend_hull(tds, &kernel, new_vkey, &p).unwrap();
        assert!(!new_cells.is_empty());
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_extend_hull_errors_when_no_visible_facets() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let kernel = FastKernel::<f64>::new();
        let p = Point::new([0.25, 0.25]); // inside
        let new_vkey = tds
            .insert_vertex_with_mapping(vertex!([0.25, 0.25]))
            .unwrap();

        let err = extend_hull(tds, &kernel, new_vkey, &p).unwrap_err();
        assert!(matches!(
            err,
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets
            }
        ));
    }
}
