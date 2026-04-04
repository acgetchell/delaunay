//! Bounded deterministic PL-manifold topology repair.
//!
//! This module implements a `pub(crate)` repair algorithm that attempts to bring
//! a triangulation closer to satisfying the
//! [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold)
//! invariant by removing cells that cause codimension-1 facet over-sharing
//! (facets incident to more than 2 cells).
//!
//! # Algorithm
//!
//! 1. **Structural precheck**: validate Levels 1–2 via `Tds::validate()`. If
//!    the facet-degree invariant already holds, return early with zero work.
//! 2. **Iterative facet over-sharing repair**: build the facet-to-cells map,
//!    identify facets with degree > 2, deterministically select the worst-quality
//!    cell per over-shared facet for removal, remove the batch, and rebuild
//!    neighbors/incidence.
//! 3. **Termination**: the loop terminates on success (all facets have degree
//!    ≤ 2), budget exhaustion (`max_iterations` or `max_cells_removed`), or
//!    no-progress (zero cells removed in a pass).
//!
//! # Determinism
//!
//! Cell removal order is deterministic: candidates are sorted by
//! (ascending quality, canonicalized vertex tuple, cell UUID) so that
//! repeated runs on identical input produce identical output.

#![forbid(unsafe_code)]

use crate::core::cell::Cell;
use crate::core::collections::{CellKeySet, SmallBuffer, fast_hash_set_with_capacity};
use crate::core::facet::FacetHandle;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, TdsError, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::geometry::util::norms::hypot;
use crate::topology::manifold::validate_facet_degree;
use num_traits::NumCast;
use slotmap::Key;
use std::cmp::Ordering;
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the bounded PL-manifold repair pass.
///
/// # Defaults
///
/// - `max_iterations`: 64
/// - `max_cells_removed`: 10,000
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlManifoldRepairConfig {
    /// Maximum number of repair iterations (each iteration removes a batch of
    /// cells).
    pub max_iterations: usize,
    /// Maximum total number of cells that may be removed across all iterations.
    pub max_cells_removed: usize,
}

impl Default for PlManifoldRepairConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            max_cells_removed: 10_000,
        }
    }
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Statistics and artifacts collected during PL-manifold repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunayize::PlManifoldRepairStats;
///
/// let stats = PlManifoldRepairStats::<f64, (), (), 3>::default();
/// assert_eq!(stats.iterations, 0);
/// assert_eq!(stats.cells_removed, 0);
/// assert!(stats.removed_cells.is_empty());
/// assert!(stats.removed_vertices.is_empty());
/// assert!(!stats.succeeded);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlManifoldRepairStats<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Number of repair iterations executed.
    pub iterations: usize,
    /// Total number of cells removed.
    pub cells_removed: usize,
    /// Cells that were removed, preserving user data for callers that need to
    /// migrate or inspect it. Identifiable by [`Cell::uuid()`].
    pub removed_cells: Vec<Cell<T, U, V, D>>,
    /// Vertices that became isolated after cell removal and were removed from
    /// the TDS. Identifiable by [`Vertex::uuid()`].
    pub removed_vertices: Vec<Vertex<T, U, D>>,
    /// Whether the facet-degree invariant was satisfied at termination.
    pub succeeded: bool,
}

impl<T, U, V, const D: usize> Default for PlManifoldRepairStats<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn default() -> Self {
        Self {
            iterations: 0,
            cells_removed: 0,
            removed_cells: Vec::new(),
            removed_vertices: Vec::new(),
            succeeded: false,
        }
    }
}

// =============================================================================
// ERRORS
// =============================================================================

/// Errors that can occur during PL-manifold repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunayize::PlManifoldRepairError;
///
/// let err = PlManifoldRepairError::BudgetExhausted {
///     iterations: 64,
///     cells_removed: 100,
///     max_iterations: 64,
///     max_cells_removed: 10_000,
/// };
/// assert!(err.to_string().contains("budget exhausted"));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlManifoldRepairError {
    /// The underlying TDS is structurally inconsistent.
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// Iteration budget exhausted before the facet-degree invariant was satisfied.
    #[error(
        "PL-manifold repair budget exhausted: {iterations} iterations, {cells_removed} cells removed (max_iterations={max_iterations}, max_cells_removed={max_cells_removed})"
    )]
    BudgetExhausted {
        /// Iterations executed.
        iterations: usize,
        /// Cells removed so far.
        cells_removed: usize,
        /// Configured iteration limit.
        max_iterations: usize,
        /// Configured cell-removal limit.
        max_cells_removed: usize,
    },

    /// No progress: a repair pass found over-shared facets but could not remove
    /// any cells (all candidates were already removed or the TDS is in a state
    /// where removal is not possible).
    #[error(
        "PL-manifold repair made no progress: {over_shared_facets} over-shared facets remain after {iterations} iterations ({cells_removed} cells removed before stalling)"
    )]
    NoProgress {
        /// Number of over-shared facets remaining.
        over_shared_facets: usize,
        /// Iterations executed so far.
        iterations: usize,
        /// Total cells removed before progress stalled.
        cells_removed: usize,
    },
}

// =============================================================================
// REPAIR ALGORITHM
// =============================================================================

/// Attempts to repair facet over-sharing (degree > 2) by removing the
/// worst-quality cell per over-shared facet in deterministic order.
///
/// This targets only the codimension-1 facet-degree invariant and does **not**
/// guarantee full PL-manifoldness (ridge-link / vertex-link conditions are not
/// addressed).
///
/// # Errors
///
/// Returns [`PlManifoldRepairError`] if:
/// - The TDS fails structural validation (Levels 1–2).
/// - The iteration or cell-removal budget is exhausted.
/// - A pass finds violations but cannot remove any cells.
pub fn repair_facet_oversharing<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    config: &PlManifoldRepairConfig,
) -> Result<PlManifoldRepairStats<T, U, V, D>, PlManifoldRepairError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut stats = PlManifoldRepairStats::default();

    // Structural precheck: validate everything EXCEPT facet sharing, since
    // facet over-sharing is exactly what we are trying to repair. A full
    // `tds.validate()` would reject over-shared facets and return before the
    // repair loop could run.
    tds.validate_vertex_mappings()?;
    tds.validate_cell_mappings()?;
    tds.validate_cell_vertex_keys()?;

    // Fast path: if the facet-degree invariant already holds, nothing to do.
    let facet_map = tds.build_facet_to_cells_map()?;
    if validate_facet_degree(&facet_map).is_ok() {
        stats.succeeded = true;
        return Ok(stats);
    }

    for iteration in 0..config.max_iterations {
        stats.iterations = iteration + 1;

        // Rebuild the facet map after mutations.
        let facet_map = tds.build_facet_to_cells_map()?;

        // Collect cells to remove: for each over-shared facet (degree > 2),
        // deterministically pick the worst candidate.
        let mut removal_candidates: CellKeySet = fast_hash_set_with_capacity(16);

        for handles in facet_map.values() {
            if handles.len() <= 2 {
                continue;
            }

            // Among the cells sharing this facet, pick the one to remove.
            // Strategy: compute a deterministic quality score for each cell
            // and remove the worst (highest score = worst quality). Ties are
            // broken by canonicalized vertex-key tuple, then by cell UUID.
            let worst = pick_worst_cell(tds, handles);
            if let Some(cell_key) = worst {
                removal_candidates.insert(cell_key);
            }
        }

        if removal_candidates.is_empty() {
            // We had violations but couldn't pick any cell — no progress.
            let remaining = facet_map.values().filter(|h| h.len() > 2).count();
            return Err(PlManifoldRepairError::NoProgress {
                over_shared_facets: remaining,
                iterations: stats.iterations,
                cells_removed: stats.cells_removed,
            });
        }

        // Check cell-removal budget.
        let batch_size = removal_candidates.len();
        if stats.cells_removed + batch_size > config.max_cells_removed {
            return Err(PlManifoldRepairError::BudgetExhausted {
                iterations: stats.iterations,
                cells_removed: stats.cells_removed,
                max_iterations: config.max_iterations,
                max_cells_removed: config.max_cells_removed,
            });
        }

        // Snapshot cells before removal (sorted by UUID for determinism).
        let mut keys: Vec<CellKey> = removal_candidates.into_iter().collect();
        keys.sort_by(|a, b| {
            let uuid_a = tds.get_cell(*a).map(Cell::uuid);
            let uuid_b = tds.get_cell(*b).map(Cell::uuid);
            uuid_a.cmp(&uuid_b)
        });
        for &ck in &keys {
            if let Some(cell) = tds.get_cell(ck) {
                stats.removed_cells.push(cell.clone());
            }
        }

        // Remove the batch. `remove_cells_by_keys` handles local neighbor
        // back-reference clearing and incident-cell repair. Full neighbor
        // rebuild is deferred to after the loop to avoid O(cells²) work.
        let removed = tds.remove_cells_by_keys(&keys);
        stats.cells_removed += removed;

        // Remove orphaned vertices (required for PL-manifold validity).
        remove_orphaned_vertices(tds, &mut stats);

        // Check if the invariant now holds.
        let facet_map = tds.build_facet_to_cells_map()?;
        if validate_facet_degree(&facet_map).is_ok() {
            stats.succeeded = true;
            // Rebuild full neighbor/incidence pointers before returning.
            tds.assign_neighbors().map_err(PlManifoldRepairError::Tds)?;
            tds.assign_incident_cells()
                .map_err(|e| PlManifoldRepairError::Tds(e.into()))?;
            return Ok(stats);
        }
    }

    Err(PlManifoldRepairError::BudgetExhausted {
        iterations: stats.iterations,
        cells_removed: stats.cells_removed,
        max_iterations: config.max_iterations,
        max_cells_removed: config.max_cells_removed,
    })
}

/// Deterministic quality score for a cell: lower = better quality.
///
/// Uses an edge-length aspect-ratio metric (max/min edge) that operates
/// directly on the `Tds` without requiring a full `Triangulation` or
/// circumsphere computation.
fn cell_quality_score<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>, cell_key: CellKey) -> f64
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Ok(vertices) = tds.get_cell_vertices(cell_key) else {
        return f64::MAX;
    };

    let mut edge_lengths: SmallBuffer<f64, 16> = SmallBuffer::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let Some(vi) = tds.get_vertex_by_key(vertices[i]) else {
                return f64::MAX;
            };
            let Some(vj) = tds.get_vertex_by_key(vertices[j]) else {
                return f64::MAX;
            };

            let mut diff = [T::zero(); D];
            for (idx, d) in diff.iter_mut().enumerate() {
                *d = vi.point().coords()[idx] - vj.point().coords()[idx];
            }
            let len: f64 = NumCast::from(hypot(&diff)).unwrap_or(f64::MAX);
            edge_lengths.push(len);
        }
    }

    if edge_lengths.is_empty() {
        return f64::MAX;
    }

    let Some(n): Option<f64> = NumCast::from(edge_lengths.len()) else {
        return f64::MAX;
    };
    let mean = edge_lengths.iter().sum::<f64>() / n;
    let variance = edge_lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n;
    let min_edge = edge_lengths.iter().copied().fold(f64::INFINITY, f64::min);
    let max_edge = edge_lengths
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if min_edge <= 0.0 {
        return f64::MAX;
    }
    // Primary: aspect ratio; secondary: edge-length variance as tiebreaker.
    (max_edge / min_edge) + variance * 1e-12
}

/// Among the cells incident to an over-shared facet, deterministically select
/// the worst-quality cell for removal.
///
/// Selection order: highest quality score first (worst cell), then canonicalized
/// vertex keys (ascending), then cell UUID (ascending).
fn pick_worst_cell<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    handles: &[FacetHandle],
) -> Option<CellKey>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    struct Candidate {
        cell_key: CellKey,
        score: f64,
        vertex_keys: Vec<u64>,
        uuid: Uuid,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(handles.len());

    for handle in handles {
        let cell_key = handle.cell_key();
        let Some(cell) = tds.get_cell(cell_key) else {
            continue;
        };

        let score = cell_quality_score(tds, cell_key);

        let mut vertex_keys: Vec<u64> = cell
            .vertices()
            .iter()
            .map(|vk| vk.data().as_ffi())
            .collect();
        vertex_keys.sort_unstable();

        candidates.push(Candidate {
            cell_key,
            score,
            vertex_keys,
            uuid: cell.uuid(),
        });
    }

    // Worst quality first, then vertex keys, then UUID.
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.vertex_keys.cmp(&b.vertex_keys))
            .then_with(|| a.uuid.cmp(&b.uuid))
    });

    candidates.first().map(|c| c.cell_key)
}

/// Remove vertices with no incident cell from the TDS, collecting them into
/// `stats.removed_vertices` so callers can recover their data.
///
/// Vertices are sorted by UUID before removal for deterministic ordering.
fn remove_orphaned_vertices<T, U, V, const D: usize>(
    tds: &mut Tds<T, U, V, D>,
    stats: &mut PlManifoldRepairStats<T, U, V, D>,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut orphaned: Vec<(VertexKey, Uuid)> = tds
        .vertices()
        .filter(|(_, v)| v.incident_cell.is_none())
        .map(|(k, v)| (k, v.uuid()))
        .collect();
    orphaned.sort_by_key(|(_, uuid)| *uuid);

    for (vk, _) in orphaned {
        if let Some(vertex) = tds.get_vertex_by_key(vk) {
            stats.removed_vertices.push(*vertex);
        }
        tds.remove_isolated_vertex(vk);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::vertex;

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    fn init_tracing() {
        let _ = tracing_subscriber::fmt::try_init();
    }

    // =============================================================================
    // CONFIG DEFAULT TESTS
    // =============================================================================

    #[test]
    fn test_config_defaults() {
        init_tracing();
        let config = PlManifoldRepairConfig::default();
        assert_eq!(config.max_iterations, 64);
        assert_eq!(config.max_cells_removed, 10_000);
    }

    // =============================================================================
    // ALREADY PL-MANIFOLD TESTS
    // =============================================================================

    #[test]
    fn test_already_pl_manifold_is_noop() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let config = PlManifoldRepairConfig::default();
        let stats = repair_facet_oversharing(&mut tds, &config).unwrap();

        assert!(stats.succeeded);
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.cells_removed, 0);
    }

    // =============================================================================
    // BUDGET EXHAUSTION TESTS
    // =============================================================================

    #[test]
    fn test_budget_exhaustion_zero_iterations() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        // Zero iterations — should succeed because already PL-manifold.
        let config = PlManifoldRepairConfig {
            max_iterations: 0,
            max_cells_removed: 10_000,
        };
        let stats = repair_facet_oversharing(&mut tds, &config).unwrap();
        assert!(stats.succeeded);
    }

    // =============================================================================
    // 2D PL-MANIFOLD TESTS
    // =============================================================================

    #[test]
    fn test_2d_already_pl_manifold() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let config = PlManifoldRepairConfig::default();
        let stats = repair_facet_oversharing(&mut tds, &config).unwrap();

        assert!(stats.succeeded);
        assert_eq!(stats.cells_removed, 0);
    }

    // =============================================================================
    // STATS POPULATION TESTS
    // =============================================================================

    #[test]
    fn test_stats_populated_on_success() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let stats = repair_facet_oversharing(&mut tds, &PlManifoldRepairConfig::default()).unwrap();

        assert!(stats.succeeded);
        // iterations == 0 for already-valid
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.cells_removed, 0);
    }

    // =============================================================================
    // DETERMINISM TESTS
    // =============================================================================

    // =============================================================================
    // FACET OVER-SHARING REPAIR TESTS
    // =============================================================================

    /// Helper: create a TDS with over-shared facets by duplicating a cell in a
    /// multi-cell triangulation. Interior facets go from degree 2 to degree 3.
    fn make_overshared_tds() -> Tds<f64, (), (), 3> {
        // 5 points
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        assert!(
            tds.number_of_cells() > 1,
            "Need multiple cells for interior facets"
        );

        // Duplicate the first cell → its facets go from degree 2 to degree 3.
        let cell_key = tds.cell_keys().next().unwrap();
        let vkeys = tds.get_cell_vertices(cell_key).unwrap();
        let dup_cell = Cell::new(vkeys.to_vec(), None).unwrap();
        tds.insert_cell_with_mapping(dup_cell).unwrap();

        // Sanity: at least one facet should now be over-shared.
        let facet_map = tds.build_facet_to_cells_map().unwrap();
        assert!(
            validate_facet_degree(&facet_map).is_err(),
            "Expected over-shared facets after duplicating a cell"
        );

        tds
    }

    /// Create a TDS with over-shared facets and verify that repair removes
    /// cells until the facet-degree invariant is satisfied.
    #[test]
    fn test_repair_removes_overshared_cells() {
        init_tracing();
        let mut tds = make_overshared_tds();
        let cells_before = tds.number_of_cells();

        let stats = repair_facet_oversharing(&mut tds, &PlManifoldRepairConfig::default()).unwrap();

        assert!(stats.succeeded);
        assert!(stats.cells_removed > 0);
        assert!(stats.iterations > 0);
        assert!(tds.number_of_cells() < cells_before);

        // removed_cells should contain exactly cells_removed entries.
        assert_eq!(stats.removed_cells.len(), stats.cells_removed);

        let facet_map = tds.build_facet_to_cells_map().unwrap();
        assert!(validate_facet_degree(&facet_map).is_ok());
    }

    /// Verify that a tight cell-removal budget triggers `BudgetExhausted`.
    #[test]
    fn test_repair_budget_exhausted_on_overshared_tds() {
        init_tracing();
        let mut tds = make_overshared_tds();

        let config = PlManifoldRepairConfig {
            max_iterations: 64,
            max_cells_removed: 0,
        };
        let result = repair_facet_oversharing(&mut tds, &config);
        assert!(
            matches!(result, Err(PlManifoldRepairError::BudgetExhausted { .. })),
            "Expected BudgetExhausted, got: {result:?}"
        );
    }

    /// Verify that `max_iterations: 0` on an over-shared TDS triggers
    /// `BudgetExhausted` (iteration budget, not cell budget).
    #[test]
    fn test_repair_iteration_budget_exhausted_on_overshared_tds() {
        init_tracing();
        let mut tds = make_overshared_tds();

        let config = PlManifoldRepairConfig {
            max_iterations: 0,
            max_cells_removed: 10_000,
        };
        let result = repair_facet_oversharing(&mut tds, &config);
        assert!(
            matches!(result, Err(PlManifoldRepairError::BudgetExhausted { .. })),
            "Expected BudgetExhausted from iteration limit, got: {result:?}"
        );
    }

    // =============================================================================
    // QUALITY SCORE TESTS
    // =============================================================================

    /// Verify that `cell_quality_score` returns a finite, positive value for a
    /// valid cell and is deterministic across calls.
    #[test]
    fn test_cell_quality_score_finite_and_deterministic() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tds = dt.tds();
        let cell_key = tds.cell_keys().next().unwrap();

        let score1 = cell_quality_score(tds, cell_key);
        let score2 = cell_quality_score(tds, cell_key);

        assert!(score1.is_finite(), "Score should be finite, got {score1}");
        assert!(score1 > 0.0, "Score should be positive, got {score1}");
        approx::assert_relative_eq!(score1, score2, epsilon = 0.0);
    }

    // =============================================================================
    // DETERMINISM TESTS
    // =============================================================================

    /// Verify that repair of an over-shared TDS produces identical stats
    /// across repeated runs on identical input.
    #[test]
    fn test_deterministic_repair_on_overshared_tds() {
        init_tracing();

        let mut tds1 = make_overshared_tds();
        let mut tds2 = make_overshared_tds();

        let config = PlManifoldRepairConfig::default();
        let stats1 = repair_facet_oversharing(&mut tds1, &config).unwrap();
        let stats2 = repair_facet_oversharing(&mut tds2, &config).unwrap();

        assert_eq!(stats1, stats2, "Repair should be deterministic");
    }

    #[test]
    fn test_deterministic_repeated_runs() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let config = PlManifoldRepairConfig::default();

        let mut tds1 = dt.tds().clone();
        let stats1 = repair_facet_oversharing(&mut tds1, &config).unwrap();

        let mut tds2 = dt.tds().clone();
        let stats2 = repair_facet_oversharing(&mut tds2, &config).unwrap();

        assert_eq!(stats1, stats2);
    }
}
