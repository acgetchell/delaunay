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
    CellKeyBuffer, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
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
use std::iter::Sum;
use std::ops::{AddAssign, SubAssign};

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

    Ok(new_cells)
}

/// Wire neighbor relationships for newly created cavity cells.
///
/// Updates both internal neighbors (between new cells) and external neighbors
/// (between new cells and existing boundary cells).
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_cells` - Keys of newly created cells
/// - `boundary_facets` - Facets that formed the cavity boundary
///
/// # Returns
/// Ok(()) if wiring succeeds
///
/// # Errors
/// Returns error if neighbor wiring fails or cells cannot be found.
pub fn wire_cavity_neighbors<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    new_cells: &CellKeyBuffer,
    boundary_facets: &[FacetHandle],
) -> Result<(), InsertionError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Build facet map: facet_vertices -> (cell_key, facet_idx)
    type FacetMap = FastHashMap<u64, Vec<(CellKey, u8)>>;
    let mut facet_map: FacetMap = FastHashMap::default();

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

    // Wire internal neighbors (facets shared by 2 new cells)
    for cells in facet_map.values() {
        if cells.len() == 2 {
            let (c1, idx1) = cells[0];
            let (c2, idx2) = cells[1];

            // Set mutual neighbors
            set_neighbor(tds, c1, idx1, Some(c2))?;
            set_neighbor(tds, c2, idx2, Some(c1))?;
        }
    }

    // Wire external neighbors (boundary facets to existing cells)
    for (i, facet_handle) in boundary_facets.iter().enumerate() {
        if i >= new_cells.len() {
            break;
        }

        let new_cell_key = new_cells[i];

        // Get the external neighbor across this boundary facet
        let boundary_cell = tds.get_cell(facet_handle.cell_key()).ok_or_else(|| {
            InsertionError::NeighborWiring {
                message: format!("Boundary cell {:?} not found", facet_handle.cell_key()),
            }
        })?;

        let external_neighbor = boundary_cell
            .neighbors()
            .and_then(|neighbors| neighbors.get(usize::from(facet_handle.facet_index())))
            .and_then(|&opt| opt);

        // The new cell's facet opposite the last vertex (new vertex) connects to external
        let new_cell =
            tds.get_cell(new_cell_key)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("New cell {new_cell_key:?} not found"),
                })?;

        let last_facet_idx = new_cell.number_of_vertices() - 1;
        let last_facet_idx_u8 =
            u8::try_from(last_facet_idx).map_err(|_| InsertionError::NeighborWiring {
                message: format!("Last facet index {last_facet_idx} exceeds u8::MAX"),
            })?;

        set_neighbor(tds, new_cell_key, last_facet_idx_u8, external_neighbor)?;

        // Update external cell to point back to new cell
        if let Some(ext_key) = external_neighbor {
            // Find which facet of external cell points to the old boundary cell
            let ext_cell = tds
                .get_cell(ext_key)
                .ok_or_else(|| InsertionError::NeighborWiring {
                    message: format!("External cell {ext_key:?} not found"),
                })?;

            if let Some(ext_neighbors) = ext_cell.neighbors() {
                for (ext_idx, &ext_neighbor_opt) in ext_neighbors.iter().enumerate() {
                    if ext_neighbor_opt == Some(facet_handle.cell_key()) {
                        let ext_idx_u8 =
                            u8::try_from(ext_idx).map_err(|_| InsertionError::NeighborWiring {
                                message: format!("External index {ext_idx} exceeds u8::MAX"),
                            })?;
                        set_neighbor(tds, ext_key, ext_idx_u8, Some(new_cell_key))?;
                        break;
                    }
                }
            }
        }
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
    K::Scalar: AddAssign + SubAssign + Sum,
    U: DataType,
    V: DataType,
{
    // Find visible boundary facets
    let visible_facets = find_visible_boundary_facets(tds, kernel, point)?;

    if visible_facets.is_empty() {
        return Err(InsertionError::CavityFilling {
            message: "No visible boundary facets found for exterior vertex".to_string(),
        });
    }

    // Fill cavity with new cells
    let new_cells = fill_cavity(tds, new_vertex_key, &visible_facets)?;

    // Wire neighbors
    wire_cavity_neighbors(tds, &new_cells, &visible_facets)?;

    Ok(new_cells)
}

/// Find all boundary facets visible from a point.
///
/// A boundary facet is visible from a point if the point is on the positive side
/// of the facet's supporting hyperplane (determined by orientation test).
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
/// Returns error if boundary facets cannot be retrieved or orientation tests fail
fn find_visible_boundary_facets<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
) -> Result<Vec<FacetHandle>, InsertionError>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum,
    U: DataType,
    V: DataType,
{
    let mut visible_facets = Vec::new();

    // Get all boundary facets
    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::CavityFilling {
            message: format!("Failed to get boundary facets: {e}"),
        })?;

    // Test each boundary facet for visibility
    for facet_view in boundary_facets {
        let cell_key = facet_view.cell_key();
        let facet_index = facet_view.facet_index();

        // Get the cell and its vertices
        let cell = tds
            .get_cell(cell_key)
            .ok_or_else(|| InsertionError::CavityFilling {
                message: format!("Boundary facet cell {cell_key:?} not found"),
            })?;

        // Collect points for the simplex: facet vertices + opposite vertex
        let mut simplex_points =
            SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        for &vkey in cell.vertices() {
            let vertex =
                tds.get_vertex_by_key(vkey)
                    .ok_or_else(|| InsertionError::CavityFilling {
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
                .map_err(|e| InsertionError::CavityFilling {
                    message: format!("Orientation test failed: {e}"),
                })?;

        // Replace opposite vertex with query point
        simplex_points[usize::from(facet_index)] = *point;
        let orientation_with_point =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::CavityFilling {
                    message: format!("Orientation test failed: {e}"),
                })?;

        // Facet is visible if orientations have opposite sign (point is on opposite side)
        // orientation() returns i32: positive, negative, or zero
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
    use crate::vertex;

    #[test]
    fn test_fill_cavity_2d() {
        // Create simple 2D triangle
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        tds.assign_neighbors().unwrap();

        // Insert new vertex
        let new_vertex = vertex!([0.5, 0.5]);
        let new_vkey = tds.insert_vertex_with_mapping(new_vertex).unwrap();

        // Find the single cell and create boundary facets
        let cell_key = tds.cell_keys().next().unwrap();
        let boundary_facets = vec![
            FacetHandle::new(cell_key, 0),
            FacetHandle::new(cell_key, 1),
            FacetHandle::new(cell_key, 2),
        ];

        // Fill cavity
        let new_cells = fill_cavity(&mut tds, new_vkey, &boundary_facets).unwrap();

        // Should create 3 new cells (one for each boundary edge)
        assert_eq!(new_cells.len(), 3);
    }
}
