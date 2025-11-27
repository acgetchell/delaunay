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

use crate::core::collections::{FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateConversionError;

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
#[derive(Debug, thiserror::Error)]
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
/// Basic point location in a 3D tetrahedron:
///
/// ```rust
/// use delaunay::core::algorithms::locate::{locate, LocateResult};
/// use delaunay::core::triangulation_data_structure::Tds;
/// use delaunay::geometry::kernel::FastKernel;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::vertex;
///
/// // Create a 3D tetrahedron
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
/// let kernel = FastKernel::<f64>::new();
///
/// // Point inside the tetrahedron
/// let inside_point = Point::new([0.25, 0.25, 0.25]);
/// match locate(&tds, &kernel, &inside_point, None) {
///     Ok(LocateResult::InsideCell(cell_key)) => {
///         assert!(tds.contains_cell(cell_key));
///     }
///     _ => panic!("Expected point to be inside a cell"),
/// }
///
/// // Point outside the convex hull
/// let outside_point = Point::new([2.0, 2.0, 2.0]);
/// match locate(&tds, &kernel, &outside_point, None) {
///     Ok(LocateResult::Outside) => { /* Expected */ }
///     _ => panic!("Expected point to be outside convex hull"),
/// }
/// ```
///
/// Using a hint cell for faster location:
///
/// ```rust
/// use delaunay::core::algorithms::locate::{locate, LocateResult};
/// use delaunay::core::triangulation_data_structure::Tds;
/// use delaunay::geometry::kernel::RobustKernel;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::vertex;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
/// let kernel = RobustKernel::<f64>::default();
///
/// // Get a cell to use as hint
/// let hint_cell = tds.cell_keys().next().unwrap();
/// let query_point = Point::new([0.3, 0.3]);
///
/// match locate(&tds, &kernel, &query_point, Some(hint_cell)) {
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
///   [`Tds::assign_neighbors`](crate::core::triangulation_data_structure::Tds::assign_neighbors)
///   and validated by [`Tds::validate_neighbors`](crate::core::triangulation_data_structure::Tds::validate_neighbors)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;

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
        let mut tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        tds.assign_neighbors().unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get the single cell
        let cell_key = tds.cell_keys().next().unwrap();
        let cell = tds.get_cell(cell_key).unwrap();

        // Get cell vertices in order
        let cell_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|&vkey| *tds.get_vertex_by_key(vkey).unwrap().point())
            .collect();

        println!("Cell vertices: {cell_points:?}");

        // Test orientation of full cell
        let cell_orientation = kernel.orientation(&cell_points).unwrap();
        println!("Cell orientation: {cell_orientation}");

        // Test query point inside
        let query_inside = Point::new([0.3, 0.3]);

        // For each facet, test if point is outside using the actual function
        for facet_idx in 0..3 {
            let result = is_point_outside_facet(&tds, &kernel, cell_key, facet_idx, &query_inside);
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
            let result = is_point_outside_facet(&tds, &kernel, cell_key, facet_idx, &query_outside);
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
        let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the triangle
        let point = Point::new([0.3, 0.3]);
        let result = locate(&tds, &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideCell(cell_key)) => {
                assert!(tds.contains_cell(cell_key));
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
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the tetrahedron
        let point = Point::new([0.25, 0.25, 0.25]);
        let result = locate(&tds, &kernel, &point, None);

        match result {
            Ok(LocateResult::InsideCell(cell_key)) => {
                assert!(tds.contains_cell(cell_key));
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
        let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the triangle
        let point = Point::new([10.0, 10.0]);
        let result = locate(&tds, &kernel, &point, None);

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
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point far outside the tetrahedron
        let point = Point::new([2.0, 2.0, 2.0]);
        let result = locate(&tds, &kernel, &point, None);

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
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Get a valid cell as hint
        let hint_cell = tds.cell_keys().next().unwrap();
        let point = Point::new([0.25, 0.25, 0.25]);

        let result = locate(&tds, &kernel, &point, Some(hint_cell));
        assert!(matches!(result, Ok(LocateResult::InsideCell(_))));
    }

    #[test]
    fn test_locate_with_robust_kernel() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        let point = Point::new([0.3, 0.3]);
        let result = locate(&tds, &kernel, &point, None);

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
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let cell_key = tds.cell_keys().next().unwrap();
        let point = Point::new([0.25, 0.25, 0.25]); // Inside tetrahedron

        // Test all facets - point should not be outside any of them
        for facet_idx in 0..4 {
            let result = is_point_outside_facet(&tds, &kernel, cell_key, facet_idx, &point);
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
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        let cell_key = tds.cell_keys().next().unwrap();
        let point = Point::new([2.0, 2.0, 2.0]); // Outside tetrahedron

        // At least one facet should show the point as outside
        let mut found_outside = false;
        for facet_idx in 0..4 {
            if matches!(
                is_point_outside_facet(&tds, &kernel, cell_key, facet_idx, &point),
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
        let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();
        let kernel = RobustKernel::<f64>::default();

        // Point very close to an edge but still inside
        let point = Point::new([0.01, 0.01]);
        let result = locate(&tds, &kernel, &point, None);

        // Should either be inside or on the edge, not outside
        match result {
            Ok(LocateResult::InsideCell(_) | LocateResult::OnEdge(_)) => { /* OK */ }
            other => panic!("Expected inside or on edge, got {other:?}"),
        }
    }

    #[test]
    fn test_locate_4d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();

        // Point inside the 4-simplex
        let point = Point::new([0.2, 0.2, 0.2, 0.2]);
        let result = locate(&tds, &kernel, &point, None);

        assert!(matches!(result, Ok(LocateResult::InsideCell(_))));
    }
}
