//! Incremental Delaunay insertion using cavity-based algorithm.
//!
//! This module implements efficient incremental insertion following CGAL's approach:
//! 1. Locate the cell containing the new point (facet walking)
//! 2. Find conflict region (BFS with in_sphere tests)
//! 3. Extract cavity boundary facets
//! 4. Remove conflict cells
//! 5. Fill cavity (create new cells connecting boundary to new vertex)
//! 6. Wire neighbors locally (no global assign_neighbors call)

use crate::core::algorithms::locate::{ConflictError, LocateError};
use crate::core::cell::Cell;
use crate::core::collections::{
    CellKeyBuffer, FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, VertexKey,
};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Error during incremental insertion.
#[derive(Debug, thiserror::Error)]
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

    // Validate we created one cell per boundary facet (1:1 correspondence)
    if boundary_facets.len() != new_cells.len() {
        return Err(InsertionError::CavityFilling {
            message: format!(
                "Created {} cells for {} boundary facets (should be 1:1)",
                new_cells.len(),
                boundary_facets.len()
            ),
        });
    }

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
                // Only add if we don't already have 2 cells for this facet
                // (prevents non-manifold topology)
                if existing_facet_cells.len() < 2 {
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
            // This should never happen - indicates non-manifold topology
            return Err(InsertionError::NeighborWiring {
                message: format!(
                    "Non-manifold topology detected: facet {} shared by {} cells (expected ≤2)",
                    facet_key,
                    cells.len()
                ),
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

/// Compute a hash for a facet from sorted vertex keys
fn compute_facet_hash(sorted_vkeys: &[VertexKey]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &vkey in sorted_vkeys {
        vkey.hash(&mut hasher);
    }
    hasher.finish()
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
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::vertex;

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
}
