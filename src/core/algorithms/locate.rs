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
    CellKeyBuffer, CellSecondaryMap, FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE,
    SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};

/// Result of point location query.
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
#[derive(Debug, Clone, thiserror::Error)]
pub enum LocateError {
    /// Triangulation has no cells
    #[error("Cannot locate in empty triangulation")]
    EmptyTriangulation,

    /// Cell reference is invalid
    #[error("Invalid cell reference: {cell_key:?}")]
    InvalidCell {
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

    /// Cycle detected during walking (numerical issues)
    #[error("Cycle detected after {steps} steps - possible numerical degeneracy")]
    CycleDetected {
        /// Number of steps before cycle detection
        steps: usize,
    },
}

/// Error during conflict region finding.
#[derive(Debug, Clone, thiserror::Error)]
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

    /// Failed to retrieve cell vertices
    #[error("Failed to get vertices for cell {cell_key:?}: {message}")]
    VertexRetrievalFailed {
        /// The cell key that failed
        cell_key: CellKey,
        /// Error message
        message: String,
    },

    /// Duplicate boundary facets detected (geometric degeneracy)
    #[error(
        "Duplicate boundary facets detected: {count} duplicates found (indicates degenerate geometry requiring perturbation)"
    )]
    DuplicateBoundaryFacets {
        /// Number of duplicate facets found
        count: usize,
    },

    /// Ridge fan detected (many facets sharing same (D-2)-simplex)
    #[error(
        "Ridge fan detected: {facet_count} facets share ridge with {ridge_vertex_count} vertices (indicates degenerate geometry requiring perturbation)"
    )]
    RidgeFan {
        /// Number of facets in the fan
        facet_count: usize,
        /// Number of vertices in the shared ridge
        ridge_vertex_count: usize,
    },
}

/// Locate a point in the triangulation using facet walking.
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
/// - Geometric predicates fail
/// - A cycle is detected (numerical degeneracy)
///
/// # Examples
///
/// Basic point location in a 4D simplex:
///
/// ```rust
/// use delaunay::prelude::*;
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
/// use delaunay::prelude::*;
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
    K::Scalar: std::iter::Sum,
    U: DataType,
    V: DataType,
{
    const MAX_STEPS: usize = 10000; // Safety limit for cycle detection

    // Check if triangulation has any cells
    if tds.number_of_cells() == 0 {
        return Err(LocateError::EmptyTriangulation);
    }

    // Start from hint or arbitrary first cell
    let mut current_cell = match hint {
        Some(key) if tds.contains_cell(key) => key,
        _ => tds
            .cell_keys()
            .next()
            .ok_or(LocateError::EmptyTriangulation)?,
    };

    // Track visited cells to detect cycles
    let mut visited = FastHashSet::default();

    // Walk toward the query point
    for step in 0..MAX_STEPS {
        // Detect cycles
        if !visited.insert(current_cell) {
            return Err(LocateError::CycleDetected { steps: step });
        }

        // Get current cell
        let cell = tds.get_cell(current_cell).ok_or(LocateError::InvalidCell {
            cell_key: current_cell,
        })?;

        // Test orientation relative to each facet
        let facet_count = cell.number_of_vertices();
        let mut found_outside_facet = false;

        for facet_idx in 0..facet_count {
            // Check if we should cross this facet
            if is_point_outside_facet(tds, kernel, current_cell, facet_idx, point)? == Some(true) {
                // Try to cross to neighbor
                if let Some(neighbor_key) = cell
                    .neighbors()
                    .and_then(|neighbors| neighbors.get(facet_idx))
                    .and_then(|&opt_key| opt_key)
                {
                    current_cell = neighbor_key;
                    found_outside_facet = true;
                    break; // Move to next cell
                }
                // No neighbor = boundary facet = outside hull
                return Ok(LocateResult::Outside);
            }
        }

        // If we didn't cross any facet, point is in this cell
        if !found_outside_facet {
            return Ok(LocateResult::InsideCell(current_cell));
        }
    }

    // Reached step limit - treat as cycle
    Err(LocateError::CycleDetected { steps: MAX_STEPS })
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
///   [`Tds::assign_neighbors`](crate::core::triangulation_data_structure::Tds::assign_neighbors).
///
/// It is validated as part of Level 2 structural validation via
/// [`Tds::is_valid`](crate::core::triangulation_data_structure::Tds::is_valid)
/// (or cumulatively via [`Tds::validate`](crate::core::triangulation_data_structure::Tds::validate)).
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
    K::Scalar: std::iter::Sum,
    U: DataType,
    V: DataType,
{
    let cell = tds
        .get_cell(cell_key)
        .ok_or(LocateError::InvalidCell { cell_key })?;

    // Get all cell vertices
    let cell_vertices: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> = cell
        .vertices()
        .iter()
        .filter_map(|&vkey| tds.get_vertex_by_key(vkey).map(|v| *v.point()))
        .collect();

    if cell_vertices.len() != D + 1 {
        return Ok(None); // Degenerate cell
    }

    // Get the opposite vertex (the one NOT on the facet)
    let opposite_vertex = cell_vertices[facet_idx];

    // Build facet simplex + opposite vertex in canonical order:
    // All facet vertices first, then opposite vertex
    let mut canonical_cell =
        SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
    for (i, point) in cell_vertices.iter().enumerate() {
        if i != facet_idx {
            canonical_cell.push(*point);
        }
    }
    canonical_cell.push(opposite_vertex);

    let cell_orientation = kernel.orientation(&canonical_cell)?;

    // Build facet simplex + query point in same canonical order
    let mut query_simplex = SmallBuffer::<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
    for (i, point) in cell_vertices.iter().enumerate() {
        if i != facet_idx {
            query_simplex.push(*point);
        }
    }
    query_simplex.push(*query_point);

    let query_orientation = kernel.orientation(&query_simplex)?;

    // If orientations differ, query point and opposite vertex are on
    // opposite sides of the facet → point is "outside" (should cross)
    // If orientations match, they're on the same side → point is "inside" (should not cross)
    Ok(Some(cell_orientation * query_orientation < 0))
}

/// Find all cells whose circumspheres contain the query point (conflict region).
///
/// Uses BFS traversal starting from a located cell to find all cells in conflict.
/// A cell is in conflict if the query point lies strictly inside its circumsphere.
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
/// 3. If point is inside circumsphere (sign > 0), add to conflict region
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
pub fn find_conflict_region<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    start_cell: CellKey,
) -> Result<CellKeyBuffer, ConflictError>
where
    K: Kernel<D>,
    K::Scalar: std::iter::Sum,
    U: DataType,
    V: DataType,
{
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

        // Get cell vertices for in_sphere test
        let cell = tds
            .get_cell(cell_key)
            .ok_or(ConflictError::InvalidStartCell { cell_key })?;

        // Collect cell vertex points
        let simplex_points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> = cell
            .vertices()
            .iter()
            .filter_map(|&vkey| tds.get_vertex_by_key(vkey).map(|v| *v.point()))
            .collect();

        if simplex_points.len() != D + 1 {
            return Err(ConflictError::VertexRetrievalFailed {
                cell_key,
                message: format!("Expected {} vertices, got {}", D + 1, simplex_points.len()),
            });
        }

        // Test if point is inside circumsphere
        let sign = kernel.in_sphere(&simplex_points, point)?;

        if sign > 0 {
            // Point is inside circumsphere - cell is in conflict
            conflict_cells.push(cell_key);

            // Add neighbors to queue for exploration
            if let Some(neighbors) = cell.neighbors() {
                for &neighbor_opt in neighbors {
                    if let Some(neighbor_key) = neighbor_opt
                        && !visited.contains_key(neighbor_key)
                    {
                        queue.push(neighbor_key);
                    }
                }
            }
        }
        // If sign <= 0, cell is not in conflict, don't explore further in this direction
    }

    Ok(conflict_cells)
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
pub fn extract_cavity_boundary<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    conflict_cells: &CellKeyBuffer,
) -> Result<SmallBuffer<FacetHandle, 64>, ConflictError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    use crate::core::collections::FastHasher;
    use std::hash::{Hash, Hasher};

    // Convert conflict cells to set for O(1) lookup
    let conflict_set: FastHashSet<CellKey> = conflict_cells.iter().copied().collect();

    // Use a set to deduplicate boundary facets by their canonical vertex keys
    // Deduplication prevents creating multiple identical cells in fill_cavity
    let mut facet_set: FastHashSet<u64> = FastHashSet::default();
    let mut boundary_facets = SmallBuffer::new();
    let mut duplicate_count = 0;

    // Track ridge incidence for detecting ridge fans
    // Map: ridge_hash -> (ridge_vertex_count, number_of_facets_sharing_this_ridge)
    let mut ridge_map: FastHashMap<u64, (usize, usize)> = FastHashMap::default();

    // Examine each cell in the conflict region
    for &cell_key in conflict_cells {
        let cell = tds
            .get_cell(cell_key)
            .ok_or(ConflictError::InvalidStartCell { cell_key })?;

        // Check each facet of the cell
        let facet_count = cell.number_of_vertices(); // D+1 facets
        for facet_idx in 0..facet_count {
            // Get neighbor across this facet
            let neighbor_opt = cell
                .neighbors()
                .and_then(|neighbors| neighbors.get(facet_idx))
                .and_then(|&opt_key| opt_key);

            // Boundary facet if:
            // 1. No neighbor (hull facet), OR
            // 2. Neighbor exists but is NOT in conflict
            let is_boundary =
                neighbor_opt.is_none_or(|neighbor_key| !conflict_set.contains(&neighbor_key));

            if is_boundary {
                // Get facet vertices (all except opposite vertex at facet_idx)
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vkey) in cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vkey);
                    }
                }

                // Sort to get canonical representation
                facet_vkeys.sort_unstable();

                // Compute facet hash
                let mut hasher = FastHasher::default();
                for &vkey in &facet_vkeys {
                    vkey.hash(&mut hasher);
                }
                let facet_hash = hasher.finish();

                // Check for duplicates
                if facet_set.contains(&facet_hash) {
                    duplicate_count += 1;
                    continue; // Skip duplicate
                }

                // Insert into set and buffer
                facet_set.insert(facet_hash);
                let facet_idx_u8 =
                    u8::try_from(facet_idx).map_err(|_| ConflictError::VertexRetrievalFailed {
                        cell_key,
                        message: format!("Facet index {facet_idx} exceeds u8::MAX"),
                    })?;
                boundary_facets.push(FacetHandle::new(cell_key, facet_idx_u8));

                // Track ridge incidence for fan detection
                // A ridge is a (D-2)-simplex, which is the facet with one more vertex removed
                // For each facet, check all possible ridges (D different ridges per facet)
                if facet_vkeys.len() >= 2 {
                    for ridge_idx in 0..facet_vkeys.len() {
                        let mut ridge_vkeys =
                            SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                        for (i, &vkey) in facet_vkeys.iter().enumerate() {
                            if i != ridge_idx {
                                ridge_vkeys.push(vkey);
                            }
                        }
                        ridge_vkeys.sort_unstable();

                        let mut ridge_hasher = FastHasher::default();
                        for &vkey in &ridge_vkeys {
                            vkey.hash(&mut ridge_hasher);
                        }
                        let ridge_hash = ridge_hasher.finish();

                        ridge_map
                            .entry(ridge_hash)
                            .and_modify(|(_, count)| *count += 1)
                            .or_insert((ridge_vkeys.len(), 1));
                    }
                }
            }
        }
    }

    // Check for duplicate facets (geometric degeneracy)
    if duplicate_count > 0 {
        return Err(ConflictError::DuplicateBoundaryFacets {
            count: duplicate_count,
        });
    }

    // Check for ridge fans (many facets sharing same ridge)
    // In a manifold boundary, a ridge should be shared by at most 2 facets
    // More than 2 indicates a degenerate fan configuration
    for (ridge_vertex_count, facet_count) in ridge_map.values() {
        const RIDGE_FAN_THRESHOLD: usize = 3;
        if *facet_count >= RIDGE_FAN_THRESHOLD {
            return Err(ConflictError::RidgeFan {
                facet_count: *facet_count,
                ridge_vertex_count: *ridge_vertex_count,
            });
        }
    }

    Ok(boundary_facets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::prelude::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

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
}
