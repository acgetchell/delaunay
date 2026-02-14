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

#![forbid(unsafe_code)]

use crate::core::collections::{
    CavityBoundaryBuffer, CellKeyBuffer, CellSecondaryMap, FacetToCellsMap, FastHashMap,
    FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::FacetHandle;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateScalar, ScalarSummable,
};
use std::hash::{Hash, Hasher};

/// Result of point location query.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::algorithms::locate::LocateResult;
/// use delaunay::core::triangulation_data_structure::VertexKey;
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
/// use delaunay::core::triangulation_data_structure::CellKey;
/// use slotmap::KeyData;
///
/// let cell_key = CellKey::from(KeyData::from_ffi(5));
/// let err = ConflictError::InvalidStartCell { cell_key };
/// assert!(matches!(err, ConflictError::InvalidStartCell { .. }));
/// ```
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

    /// Failed to access required cell data (e.g., vertices) or build facet identifiers.
    #[error("Failed to access required data for cell {cell_key:?}: {message}")]
    CellDataAccessFailed {
        /// The cell key for which required data could not be accessed.
        cell_key: CellKey,
        /// Human-readable details about what data could not be accessed.
        message: String,
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
    },
}

/// Ridge incidence information used for cavity-boundary validation.
#[derive(Debug, Clone, Copy)]
struct RidgeInfo {
    ridge_vertex_count: usize,
    facet_count: usize,
    first_facet: usize,
    second_facet: Option<usize>,
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
/// use delaunay::core::triangulation_data_structure::CellKey;
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
    K::Scalar: ScalarSummable,
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
    K::Scalar: ScalarSummable,
    U: DataType,
    V: DataType,
{
    const MAX_STEPS: usize = 10000; // Safety limit; cycles/step limits fall back to a scan

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
    K::Scalar: ScalarSummable,
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
    K::Scalar: ScalarSummable,
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
pub fn find_conflict_region<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    start_cell: CellKey,
) -> Result<CellKeyBuffer, ConflictError>
where
    K: Kernel<D>,
    K::Scalar: ScalarSummable,
    U: DataType,
    V: DataType,
{
    #[cfg(debug_assertions)]
    let log_conflict = std::env::var_os("DELAUNAY_DEBUG_CONFLICT").is_some();

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
            return Err(ConflictError::CellDataAccessFailed {
                cell_key,
                message: format!("Expected {} vertices, got {}", D + 1, simplex_points.len()),
            });
        }

        #[cfg(debug_assertions)]
        if log_conflict {
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
                if log_conflict {
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
        if log_conflict {
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
        // If sign < 0, cell is not in conflict, don't explore further in this direction
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
/// use delaunay::core::algorithms::locate::extract_cavity_boundary;
/// use delaunay::core::collections::CellKeyBuffer;
/// use delaunay::core::triangulation_data_structure::Tds;
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
    let detail_enabled = std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();

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
                        let mut ridge_hasher = FastHasher::default();
                        for (i, &vkey) in facet_vkeys.iter().enumerate() {
                            if i != ridge_idx {
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
                                }
                            })
                            .or_insert(RidgeInfo {
                                ridge_vertex_count,
                                facet_count: 1,
                                first_facet: boundary_facet_idx,
                                second_facet: None,
                            });
                    }
                }
            }

            // Two conflict cells share this facet => internal facet (not on boundary)
            [_, _] => {}

            // >2 conflict cells share this facet => non-manifold (should be impossible in valid TDS)
            // Treat as a retryable degeneracy.
            many => {
                return Err(ConflictError::NonManifoldFacet {
                    facet_hash: *facet_hash,
                    cell_count: many.len(),
                });
            }
        }
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

        for info in ridge_map.values() {
            // Closed manifold boundary requires exactly 2 incident facets per ridge.
            if info.facet_count == 1 {
                return Err(ConflictError::OpenBoundary {
                    facet_count: info.facet_count,
                    ridge_vertex_count: info.ridge_vertex_count,
                });
            }

            if info.facet_count >= 3 {
                return Err(ConflictError::RidgeFan {
                    facet_count: info.facet_count,
                    ridge_vertex_count: info.ridge_vertex_count,
                });
            }

            // facet_count == 2
            let a = info.first_facet;
            let b = info.second_facet.ok_or_else(|| {
                // This should be impossible by construction; treat as an internal consistency error.
                let fallback_cell_key = boundary_facets.first().map_or_else(
                    || {
                        // boundary_facets is non-empty by the enclosing `if`, but keep this
                        // branch to avoid panics and satisfy strict clippy.
                        CellKey::default()
                    },
                    FacetHandle::cell_key,
                );
                let cell_key = boundary_facets
                    .get(a)
                    .map_or(fallback_cell_key, FacetHandle::cell_key);

                ConflictError::CellDataAccessFailed {
                    cell_key,
                    message: "RidgeInfo missing second_facet when facet_count == 2".to_string(),
                }
            })?;
            adjacency[a].push(b);
            adjacency[b].push(a);
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
            return Err(ConflictError::DisconnectedBoundary {
                visited: visited_count,
                total: boundary_len,
            });
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        tracing::debug!(
            boundary_facets = boundary_facets.len(),
            "extract_cavity_boundary: boundary connectivity validated"
        );
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
}
