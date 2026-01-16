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
//! - **Coplanar cases** (orientation == 0) are conservatively treated as non-visible
//!   to avoid numerical instability. This may cause "no visible facets" errors for
//!   points nearly on the hull surface.
//! - For "weakly visible" behavior, a threshold-based approach would be needed
//!   (not currently implemented).

use crate::core::algorithms::locate::{ConflictError, LocateError};
use crate::core::cell::Cell;
use crate::core::collections::{
    CellKeyBuffer, FastHashMap, FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
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

/// Error during incremental insertion.
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

    /// Hull extension failed (finding visible boundary facets)
    #[error("Hull extension failed: {message}")]
    HullExtension {
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
    /// - Locate cycles (numerical degeneracy during point location)
    /// - Non-manifold topology (facets shared by >2 cells, ridge fans)
    /// - Topology validation failures during repair
    ///
    /// Non-retryable errors are structural failures that won't be fixed by perturbation:
    /// - Duplicate UUIDs
    /// - Duplicate coordinates
    /// - Generic construction or wiring failures
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            // Locate errors: cycles indicate numerical degeneracy
            Self::Location(le) => {
                matches!(le, LocateError::CycleDetected { .. })
            }
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
            // All other errors are not retryable
            Self::Construction(_)
            | Self::CavityFilling { .. }
            | Self::HullExtension { .. }
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
            eprintln!(
                "WARNING: {} duplicate boundary facets will create overlapping cells!",
                duplicates.len()
            );
            for (hash, handles) in &duplicates {
                eprintln!(
                    "  Facet hash {}: {} instances: {:?}",
                    hash,
                    handles.len(),
                    handles
                );
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
/// This function uses comprehensive facet matching to wire neighbors correctly
/// for both interior cavity filling and hull extension:
/// - Interior: New cells replace removed cells, share facets with boundary
/// - Hull extension: New cells extend hull, share facets with original boundary cells
///
/// The algorithm:
/// 1. Index all facets of new cells
/// 2. Index relevant boundary facets (from both conflict cells and neighbors)
/// 3. Match facets by their vertex sets (using hash)
/// 4. Wire mutual neighbor relationships for matched facets
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_cells` - Keys of newly created cells
/// - `conflict_cells` - Optional set of cells being removed (for interior insertion)
///
/// # Returns
/// Ok(()) if wiring succeeds
///
/// # Errors
/// Returns error if neighbor wiring fails or cells cannot be found.
pub fn wire_cavity_neighbors<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    new_cells: &CellKeyBuffer,
    conflict_cells: Option<&CellKeyBuffer>,
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Build a comprehensive facet map for ALL facets of new cells and boundary cells
    // This approach works for both interior cavity filling and hull extension
    type FacetMap = FastHashMap<u64, Vec<(CellKey, u8)>>;
    let mut facet_map: FacetMap = FastHashMap::default();

    // Note: We don't assert new_cells.len() == boundary_facets.len() here because:
    // 1. boundary_facets is not available in this function (only passed to fill_cavity)
    // 2. For hull extension, the relationship is different than interior cavity filling
    // The assertion is done in fill_cavity where boundary_facets is available.

    // Index all facets of new cells
    for &cell_key in new_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("New cell {cell_key:?} not found"),
            })?;

        for facet_idx in 0..cell.number_of_vertices() {
            // Compute facet key (hash of vertex keys excluding facet_idx)
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }

            // Sort vertices to get canonical facet key
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

    // Build conflict set for O(1) lookup
    let conflict_set: FastHashSet<CellKey> = conflict_cells
        .map(|cells| cells.iter().copied().collect())
        .unwrap_or_default();

    // Build new cells set for O(1) lookup
    let new_cells_set: FastHashSet<CellKey> = new_cells.iter().copied().collect();

    // Collect existing cell keys first to avoid borrow issues
    let existing_cell_keys: Vec<CellKey> = tds
        .cells()
        .map(|(key, _)| key)
        .filter(|&key| !new_cells_set.contains(&key) && !conflict_set.contains(&key))
        .collect();

    // Index facets from existing non-conflict cells to find matches with new cells
    // This catches cases where existing cells should neighbor new cells but weren't
    // in the boundary_facets list
    for existing_cell_key in existing_cell_keys {
        let existing_cell =
            tds.get_cell(existing_cell_key)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("Existing cell {existing_cell_key:?} not found"),
                })?;

        for facet_idx in 0..existing_cell.number_of_vertices() {
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in existing_cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }

            facet_vkeys.sort_unstable();
            let facet_key = compute_facet_hash(&facet_vkeys);

            // Only add if this facet matches one from new cells
            if let Some(existing_facet_cells) = facet_map.get_mut(&facet_key) {
                // CRITICAL: Only wire to boundary facets (len == 1)
                //
                // Why: If len >= 2, the facet is already shared by multiple new cells (internal).
                // Adding an existing cell would create a non-manifold configuration where the facet
                // is shared by 3+ cells (>2 new cells + existing cell).
                //
                // By only wiring to boundary facets (len == 1), we ensure:
                // - Boundary facets get their external neighbor (existing cell)
                // - Internal facets remain paired between new cells only
                // - Manifold property is preserved (each facet shared by ≤2 cells)
                if existing_facet_cells.len() == 1 {
                    let facet_idx_u8 =
                        u8::try_from(facet_idx).map_err(|_| InsertionError::NeighborWiring {
                            message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                        })?;

                    existing_facet_cells.push((existing_cell_key, facet_idx_u8));
                }
            }
        }
    }

    // Wire all matching facets (both internal and external)
    // Two cells share a facet if they have the same facet key
    for (facet_key, cells) in &facet_map {
        if cells.len() == 2 {
            let (c1, idx1) = cells[0];
            let (c2, idx2) = cells[1];

            // Set mutual neighbors
            set_neighbor(tds, c1, idx1, Some(c2))?;
            set_neighbor(tds, c2, idx2, Some(c1))?;
        } else if cells.len() > 2 {
            // Non-manifold topology detected
            // Return structured error for proper retry handling via coordinate perturbation
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
                eprintln!(
                    "Non-manifold topology during wiring: facet {facet_key:#x} shared by {} cells (expected ≤2). Cell types: {cell_types:?}",
                    cells.len()
                );
            }
            return Err(InsertionError::NonManifoldTopology {
                facet_hash: *facet_key,
                cell_count: cells.len(),
            });
        }
        // cells.len() == 1 means it's a boundary facet (no neighbor)
    }

    Ok(())
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

/// Repair neighbor pointers by fixing only broken/None pointers.
///
/// This function scans all cells and for each None or invalid neighbor pointer,
/// finds the correct neighbor by matching facets. Unlike `assign_neighbors`,
/// this preserves existing correct neighbor relationships.
///
/// **Performance**: O(k·n·D) where k = cells with broken neighbors, n = total cells.
///
/// **Use case**: After removing cells during topology repair, when some neighbor
/// pointers are stale (point to removed cells) but most are still correct.
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
///
/// # Returns
/// `Ok(n)` where `n` is the number of neighbor pointer slots that were updated.
///
/// # Errors
/// Returns error if facet indexing, neighbor setting, or cell retrieval fails.
///
/// # Algorithm
/// 1. For each cell, check each neighbor pointer
/// 2. If neighbor is None or points to non-existent cell, mark for repair
/// 3. Build facet hash for that specific facet
/// 4. Scan all other cells to find the one sharing that facet
/// 5. Wire mutual neighbors only for the broken facets
///
/// # Debug Validation
/// In debug builds, this function performs additional cycle detection via BFS
/// (see `validate_no_neighbor_cycles`). This adds overhead but helps catch
/// neighbor graph corruption early during development.
#[expect(
    clippy::too_many_lines,
    reason = "Long function; keep the repair algorithm in one place for clarity"
)]
pub fn repair_neighbor_pointers<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
) -> Result<usize, InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Collect all cell keys first
    let all_cell_keys: Vec<CellKey> = tds.cells().map(|(key, _)| key).collect();

    #[cfg(debug_assertions)]
    eprintln!(
        "repair_neighbor_pointers: scanning {} cells",
        all_cell_keys.len()
    );

    // Track wired facet pairs to avoid double-wiring
    // Key: (min_cell, max_cell, facet_hash) to ensure unique facet pairs
    let mut wired_pairs: FastHashSet<(CellKey, CellKey, u64)> = FastHashSet::default();

    let mut total_neighbor_slots_fixed: usize = 0;

    #[cfg(debug_assertions)]
    let mut total_pairs_repaired: usize = 0;

    // For each cell, find facets that need repair (None or invalid neighbors)
    for &cell_key in &all_cell_keys {
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::NeighborWiring {
                message: format!("Cell {cell_key:?} not found"),
            })?;

        let num_vertices = cell.number_of_vertices();
        let mut facets_to_repair: Vec<usize> = Vec::new();

        // Check which facets need repair
        if let Some(neighbors) = cell.neighbors() {
            for facet_idx in 0..num_vertices {
                let neighbor = neighbors.get(facet_idx).copied().flatten();
                // Repair if None or neighbor doesn't exist
                if neighbor.is_none_or(|neighbor_key| !tds.contains_cell(neighbor_key)) {
                    facets_to_repair.push(facet_idx);
                }
            }
        } else {
            // No neighbors set at all - repair all facets
            facets_to_repair.extend(0..num_vertices);
        }

        if facets_to_repair.is_empty() {
            continue; // All neighbors are valid
        }

        // For each facet that needs repair, find its neighbor by facet matching
        for facet_idx in facets_to_repair {
            // Build facet vertex keys (all except facet_idx)
            let cell_vertices =
                tds.get_cell_vertices(cell_key)
                    .map_err(|e| InsertionError::NeighborWiring {
                        message: format!("Cell {cell_key:?} not found during facet matching: {e}"),
                    })?;
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in cell_vertices.iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();
            let facet_hash = compute_facet_hash(&facet_vkeys);

            // Scan all other cells to find one sharing this facet
            let mut matching_cell: Option<(CellKey, usize)> = None;

            for &other_key in &all_cell_keys {
                if other_key == cell_key {
                    continue;
                }

                let other_cell =
                    tds.get_cell(other_key)
                        .ok_or_else(|| InsertionError::NeighborWiring {
                            message: format!("Cell {other_key:?} not found during facet matching"),
                        })?;

                // Check each facet of other_cell
                for other_facet_idx in 0..other_cell.number_of_vertices() {
                    let mut other_facet_vkeys =
                        SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                    for (i, &vkey) in other_cell.vertices().iter().enumerate() {
                        if i != other_facet_idx {
                            other_facet_vkeys.push(vkey);
                        }
                    }
                    other_facet_vkeys.sort_unstable();
                    let other_facet_hash = compute_facet_hash(&other_facet_vkeys);

                    if facet_hash == other_facet_hash {
                        // Found matching facet
                        matching_cell = Some((other_key, other_facet_idx));
                        break;
                    }
                }

                if matching_cell.is_some() {
                    break;
                }
            }

            // Wire mutual neighbors if we found a match
            if let Some((other_key, _other_facet_idx)) = matching_cell {
                // Check if we've already wired this facet pair
                let pair_key = if cell_key < other_key {
                    (cell_key, other_key, facet_hash)
                } else {
                    (other_key, cell_key, facet_hash)
                };

                if wired_pairs.contains(&pair_key) {
                    // Already wired - skip to avoid double-wiring
                    #[cfg(debug_assertions)]
                    eprintln!("  Skipping already-wired pair: {cell_key:?} <-> {other_key:?}");
                    continue;
                }

                // Use Cell::mirror_facet_index to find the correct opposite vertex in neighbor
                let cell =
                    tds.get_cell(cell_key)
                        .ok_or_else(|| InsertionError::NeighborWiring {
                            message: format!("Cell {cell_key:?} not found during neighbor wiring"),
                        })?;
                let other_cell =
                    tds.get_cell(other_key)
                        .ok_or_else(|| InsertionError::NeighborWiring {
                            message: format!("Cell {other_key:?} not found during neighbor wiring"),
                        })?;

                // Find the mirror facet index: which vertex in other_cell is opposite the shared facet?
                let mirror_idx = cell.mirror_facet_index(facet_idx, other_cell).ok_or_else(
                    || InsertionError::NeighborWiring {
                        message: format!(
                            "Could not find mirror facet: cell {cell_key:?} facet {facet_idx} -> other cell {other_key:?}"
                        ),
                    },
                )?;

                #[cfg(debug_assertions)]
                eprintln!(
                    "  Wiring: cell {cell_key:?}[{facet_idx}] <-> other {other_key:?}[{mirror_idx}]"
                );

                let facet_idx_u8 =
                    u8::try_from(facet_idx).map_err(|_| InsertionError::NeighborWiring {
                        message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                    })?;
                let mirror_idx_u8 =
                    u8::try_from(mirror_idx).map_err(|_| InsertionError::NeighborWiring {
                        message: format!("Mirror facet index {mirror_idx} exceeds u8::MAX"),
                    })?;

                let desired_for_cell = Some(other_key);
                let desired_for_other = Some(cell_key);

                let before_cell = tds.get_cell(cell_key).and_then(|c| {
                    c.neighbors()
                        .and_then(|ns| ns.get(facet_idx).copied().flatten())
                });
                let before_other = tds.get_cell(other_key).and_then(|c| {
                    c.neighbors()
                        .and_then(|ns| ns.get(mirror_idx).copied().flatten())
                });

                if before_cell != desired_for_cell {
                    set_neighbor(tds, cell_key, facet_idx_u8, desired_for_cell)?;
                    total_neighbor_slots_fixed += 1;
                }

                if before_other != desired_for_other {
                    set_neighbor(tds, other_key, mirror_idx_u8, desired_for_other)?;
                    total_neighbor_slots_fixed += 1;
                }

                wired_pairs.insert(pair_key);

                #[cfg(debug_assertions)]
                if before_cell != desired_for_cell || before_other != desired_for_other {
                    total_pairs_repaired += 1;
                }
            }
            // If no match found, leave as None (boundary facet)
        }
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "repair_neighbor_pointers: repaired {total_pairs_repaired} facet pairs ({total_neighbor_slots_fixed} neighbor pointers updated)"
    );

    // Validate no cycles were introduced (debug mode only)
    #[cfg(debug_assertions)]
    validate_no_neighbor_cycles(tds)?;

    Ok(total_neighbor_slots_fixed)
}

/// Validate that the neighbor graph has no cycles using BFS.
///
/// This catches cycles early during development/testing. Cycles in the neighbor
/// graph would cause infinite loops during point location.
///
/// **Performance**: O(n·D) - visits each cell once, checks D neighbors per cell.
///
/// # Errors
/// Returns `NeighborWiring` error if a cycle is detected.
#[cfg(debug_assertions)]
fn validate_no_neighbor_cycles<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    const MAX_WALK_STEPS: usize = 10000;

    // Sample a few cells and try walking through their neighbor graph
    let sample_cells: Vec<CellKey> = tds.cells().map(|(key, _)| key).take(10).collect();

    for &start_cell in &sample_cells {
        let mut visited = FastHashSet::default();
        let mut to_visit = vec![start_cell];
        let mut steps = 0;

        while let Some(current) = to_visit.pop() {
            steps += 1;
            if steps > MAX_WALK_STEPS {
                return Err(InsertionError::NeighborWiring {
                    message: format!(
                        "Possible cycle detected: BFS exceeded {MAX_WALK_STEPS} steps from cell {start_cell:?}"
                    ),
                });
            }

            if !visited.insert(current) {
                continue; // Already visited
            }

            // Add all neighbors to visit queue
            if let Some(cell) = tds.get_cell(current)
                && let Some(neighbors) = cell.neighbors()
            {
                for &neighbor_opt in neighbors {
                    if let Some(neighbor_key) = neighbor_opt
                        && tds.contains_cell(neighbor_key)
                        && !visited.contains(&neighbor_key)
                    {
                        to_visit.push(neighbor_key);
                    }
                }
            }
        }
    }

    eprintln!("validate_no_neighbor_cycles: passed (no cycles detected)");
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
    // Find visible boundary facets
    let visible_facets = find_visible_boundary_facets(tds, kernel, point)?;

    if visible_facets.is_empty() {
        return Err(InsertionError::HullExtension {
            message: "No visible boundary facets found for exterior vertex (may be coplanar with hull surface)".to_string(),
        });
    }

    // Fill cavity with new cells
    let new_cells = fill_cavity(tds, new_vertex_key, &visible_facets)?;

    // Wire neighbors using comprehensive facet matching
    // For hull extension, no conflict cells (nothing is removed)
    wire_cavity_neighbors(tds, &new_cells, None)?;

    Ok(new_cells)
}

/// Find all boundary facets visible from a point.
///
/// A boundary facet is visible from a point if the point is on the positive side
/// of the facet's supporting hyperplane (determined by orientation test).
///
/// **Visibility criterion:**
/// - **Strictly visible**: Opposite orientations (orientation signs differ)
/// - **Coplanar** (orientation == 0): Currently treated as non-visible to avoid
///   numerical instability. For "weakly visible" behavior that includes nearly
///   coplanar facets, the orientation test logic would need an epsilon-based
///   threshold (not currently implemented).
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

    // Get all boundary facets
    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::HullExtension {
            message: format!("Failed to get boundary facets: {e}"),
        })?;

    // Test each boundary facet for visibility
    for facet_view in boundary_facets {
        let cell_key = facet_view.cell_key();
        let facet_index = facet_view.facet_index();

        // Get the cell and its vertices
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::HullExtension {
                message: format!("Boundary facet cell {cell_key:?} not found"),
            })?;

        // Collect points for the simplex: facet vertices + opposite vertex
        let mut simplex_points =
            SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        for &vkey in cell.vertices() {
            let vertex =
                tds.get_vertex_by_key(vkey)
                    .ok_or_else(|| InsertionError::HullExtension {
                        message: format!("Vertex {vkey:?} not found in TDS"),
                    })?;
            simplex_points.push(*vertex.point());
        }

        // Test orientation: if point is on same side as inside of hull, facet is visible
        // For a boundary facet, we want to know if the new point is on the "outside" side
        // The facet vertices are ordered such that the opposite vertex (at facet_index) is "inside"
        // So we test if replacing the opposite vertex with our point gives opposite orientation
        let orientation_with_opposite =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    message: format!("Orientation test failed: {e}"),
                })?;

        // Replace opposite vertex with query point
        simplex_points[usize::from(facet_index)] = *point;
        let orientation_with_point =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    message: format!("Orientation test failed: {e}"),
                })?;

        // Facet is visible if orientations have opposite sign (point is on opposite side)
        // orientation() returns i32: positive, negative, or zero
        //
        // Note: Coplanar cases (either orientation == 0) are treated as non-visible.
        // This conservative approach avoids numerical instability but may cause
        // "no visible facets" errors for points nearly on the hull surface.
        // For "weakly visible" behavior, use: orientation_with_opposite * orientation_with_point < 0
        // (but this requires careful epsilon-based handling to avoid degenerate cases)
        //
        // TODO: Investigate threshold-based approaches for weakly visible behavior.
        // This would allow treating nearly coplanar facets as visible, reducing failures
        // for points close to the hull surface. Implementation would need:
        // - Configurable epsilon threshold based on coordinate type and scale
        // - Careful handling of edge cases to avoid creating degenerate cells
        // - Testing with various numerical precision scenarios (f32 vs f64)
        let is_visible = (orientation_with_opposite > 0 && orientation_with_point < 0)
            || (orientation_with_opposite < 0 && orientation_with_point > 0);

        if is_visible {
            visible_facets.push(FacetHandle::new(cell_key, facet_index));
        }
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

                    // Wire neighbors
                    wire_cavity_neighbors(tds, &new_cells, None).unwrap();

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

        let result = wire_cavity_neighbors(tds, &invalid_cells, None);
        assert!(result.is_err());
        assert!(matches!(result, Err(InsertionError::NeighborWiring { .. })));
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

        let err = wire_cavity_neighbors(&mut tds, &new_cells, None).unwrap_err();
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

        let err = wire_cavity_neighbors(tds, &new_cells, None).unwrap_err();
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
            InsertionError::Location(LocateError::CycleDetected { steps: 1000 }).is_retryable()
        );

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
            !InsertionError::HullExtension {
                message: "test".to_string()
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
            InsertionError::HullExtension { message } if message.contains("No visible boundary facets")
        ));
    }
}
