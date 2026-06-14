//! Bounded deterministic PL-manifold topology repair.
//!
//! This module implements a `pub(crate)` repair algorithm that attempts to bring
//! a triangulation closer to satisfying the
//! [`TopologyGuarantee::PLManifold`](crate::prelude::validation::TopologyGuarantee::PLManifold)
//! invariant by removing simplices that cause codimension-1 facet over-sharing
//! (facets incident to more than 2 simplices).
//!
//! # Algorithm
//!
//! 1. **Structural precheck**: validate Levels 1–2 via `Tds::validate()`. If
//!    the facet-degree invariant already holds, return early with zero work.
//! 2. **Iterative facet over-sharing repair**: build the facet-to-simplices map,
//!    identify facets with degree > 2, deterministically select the worst-quality
//!    simplex per over-shared facet for removal, remove the batch, and rebuild
//!    neighbors/incidence.
//! 3. **Termination**: the loop terminates on success (all facets have degree
//!    ≤ 2), budget exhaustion (`max_iterations` or `max_simplices_removed`), or
//!    no-progress (zero simplices removed in a pass).
//!
//! # Determinism
//!
//! Simplex removal order is deterministic: candidates are sorted by
//! (descending quality, canonicalized vertex tuple, simplex UUID) so that
//! repeated runs on identical input produce identical output.

#![forbid(unsafe_code)]

use crate::core::collections::{SimplexKeySet, SmallBuffer, fast_hash_set_with_capacity};
use crate::core::facet::FacetHandle;
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, Tds, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::util::norms::hypot;
use crate::topology::manifold::validate_facet_degree;
use num_traits::NumCast;
use slotmap::Key;
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
/// - `max_simplices_removed`: 10,000
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlManifoldRepairConfig {
    /// Maximum number of repair iterations (each iteration removes a batch of
    /// simplices).
    pub max_iterations: usize,
    /// Maximum total number of simplices that may be removed across all iterations.
    pub max_simplices_removed: usize,
}

impl Default for PlManifoldRepairConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            max_simplices_removed: 10_000,
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
/// use delaunay::prelude::delaunayize::PlManifoldRepairStats;
///
/// let stats = PlManifoldRepairStats::<(), (), 3>::default();
/// assert_eq!(stats.iterations, 0);
/// assert_eq!(stats.simplices_removed, 0);
/// assert!(stats.removed_simplices.is_empty());
/// assert!(stats.removed_vertices.is_empty());
/// assert!(!stats.succeeded);
/// ```
#[derive(Debug, Clone)]
pub struct PlManifoldRepairStats<U, V, const D: usize> {
    /// Number of repair iterations executed.
    pub iterations: usize,
    /// Total number of simplices removed.
    pub simplices_removed: usize,
    /// Simplices that were removed, preserving user data for callers that need to
    /// migrate or inspect it. Identifiable by [`Simplex::uuid()`].
    pub removed_simplices: Vec<Simplex<V, D>>,
    /// Vertices that became isolated after simplex removal and were removed from
    /// the TDS. Identifiable by [`Vertex::uuid()`].
    pub removed_vertices: Vec<Vertex<U, D>>,
    /// Whether the facet-degree invariant was satisfied at termination.
    pub succeeded: bool,
}

impl<U, V, const D: usize> PartialEq for PlManifoldRepairStats<U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        self.iterations == other.iterations
            && self.simplices_removed == other.simplices_removed
            && self.removed_simplices == other.removed_simplices
            && self.removed_vertices == other.removed_vertices
            && self.succeeded == other.succeeded
    }
}

impl<U, V, const D: usize> Eq for PlManifoldRepairStats<U, V, D> {}

impl<U, V, const D: usize> Default for PlManifoldRepairStats<U, V, D> {
    fn default() -> Self {
        Self {
            iterations: 0,
            simplices_removed: 0,
            removed_simplices: Vec::new(),
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
/// use delaunay::prelude::delaunayize::PlManifoldRepairError;
///
/// let err = PlManifoldRepairError::BudgetExhausted {
///     iterations: 64,
///     simplices_removed: 100,
///     max_iterations: 64,
///     max_simplices_removed: 10_000,
/// };
/// assert!(err.to_string().contains("budget exhausted"));
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum PlManifoldRepairError {
    /// The underlying TDS is structurally inconsistent.
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// Iteration budget exhausted before the facet-degree invariant was satisfied.
    #[error(
        "PL-manifold repair budget exhausted: {iterations} iterations, {simplices_removed} simplices removed (max_iterations={max_iterations}, max_simplices_removed={max_simplices_removed})"
    )]
    BudgetExhausted {
        /// Iterations executed.
        iterations: usize,
        /// Simplices removed so far.
        simplices_removed: usize,
        /// Configured iteration limit.
        max_iterations: usize,
        /// Configured simplex-removal limit.
        max_simplices_removed: usize,
    },

    /// No progress: a repair pass found over-shared facets but could not remove
    /// any simplices (all candidates were already removed or the TDS is in a state
    /// where removal is not possible).
    #[error(
        "PL-manifold repair made no progress: {over_shared_facets} over-shared facets remain after {iterations} iterations ({simplices_removed} simplices removed before stalling)"
    )]
    NoProgress {
        /// Number of over-shared facets remaining.
        over_shared_facets: usize,
        /// Iterations executed so far.
        iterations: usize,
        /// Total simplices removed before progress stalled.
        simplices_removed: usize,
    },
}

// =============================================================================
// REPAIR ALGORITHM
// =============================================================================

/// Attempts to repair facet over-sharing (degree > 2) by removing the
/// worst-quality simplex per over-shared facet in deterministic order.
///
/// This targets only the codimension-1 facet-degree invariant and does **not**
/// guarantee full PL-manifoldness (ridge-link / vertex-link conditions are not
/// addressed).
///
/// # Errors
///
/// Returns [`PlManifoldRepairError`] if:
/// - The TDS fails structural validation (Levels 1–2).
/// - The iteration or simplex-removal budget is exhausted.
/// - A pass finds violations but cannot remove any simplices.
pub fn repair_facet_oversharing<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    config: &PlManifoldRepairConfig,
) -> Result<PlManifoldRepairStats<U, V, D>, PlManifoldRepairError>
where
    U: DataType,
    V: DataType,
{
    let mut stats = PlManifoldRepairStats::default();

    // Structural precheck: validate everything EXCEPT facet sharing, since
    // facet over-sharing is exactly what we are trying to repair. A full
    // `tds.validate()` would reject over-shared facets and return before the
    // repair loop could run.
    tds.validate_vertex_mappings()?;
    tds.validate_simplex_mappings()?;
    tds.validate_simplex_vertex_keys()?;

    // Fast path: if the facet-degree invariant already holds, nothing to do.
    let facet_map = tds.build_facet_to_simplices_map()?;
    if validate_facet_degree(&facet_map).is_ok() {
        stats.succeeded = true;
        return Ok(stats);
    }

    for iteration in 0..config.max_iterations {
        stats.iterations = iteration + 1;

        // Rebuild the facet map after mutations.
        let facet_map = tds.build_facet_to_simplices_map()?;

        // Collect simplices to remove: for each over-shared facet (degree > 2),
        // deterministically pick the worst candidate.
        let mut removal_candidates: SimplexKeySet = fast_hash_set_with_capacity(16);

        for handles in facet_map.values() {
            if handles.len() <= 2 {
                continue;
            }

            // Among the simplices sharing this facet, pick the one to remove.
            // Strategy: compute a deterministic quality score for each simplex
            // and remove the worst (highest score = worst quality). Ties are
            // broken by canonicalized vertex-key tuple, then by simplex UUID.
            let worst = pick_worst_simplex(tds, handles);
            if let Some(simplex_key) = worst {
                removal_candidates.insert(simplex_key);
            }
        }

        if removal_candidates.is_empty() {
            // We had violations but couldn't pick any simplex — no progress.
            let remaining = facet_map.values().filter(|h| h.len() > 2).count();
            return Err(PlManifoldRepairError::NoProgress {
                over_shared_facets: remaining,
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
            });
        }

        // Check simplex-removal budget.
        let batch_size = removal_candidates.len();
        if stats.simplices_removed + batch_size > config.max_simplices_removed {
            return Err(PlManifoldRepairError::BudgetExhausted {
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
                max_iterations: config.max_iterations,
                max_simplices_removed: config.max_simplices_removed,
            });
        }

        // Snapshot simplices before removal (sorted by UUID for determinism).
        let mut keys: Vec<SimplexKey> = removal_candidates.into_iter().collect();
        keys.sort_by(|a, b| {
            let uuid_a = tds.simplex(*a).map(Simplex::uuid);
            let uuid_b = tds.simplex(*b).map(Simplex::uuid);
            uuid_a.cmp(&uuid_b)
        });
        for &ck in &keys {
            if let Some(simplex) = tds.simplex(ck) {
                stats.removed_simplices.push(simplex.clone());
            }
        }

        // Remove the batch. `remove_simplices_by_keys` handles local neighbor
        // back-reference clearing and incident-simplex repair. Full neighbor
        // rebuild is deferred to after the loop to avoid O(simplices²) work.
        let removed = tds.remove_simplices_by_keys(&keys);
        stats.simplices_removed += removed;

        // Remove orphaned vertices (required for PL-manifold validity).
        remove_orphaned_vertices(tds, &mut stats);

        // Check if the invariant now holds.
        let facet_map = tds.build_facet_to_simplices_map()?;
        if validate_facet_degree(&facet_map).is_ok() {
            stats.succeeded = true;
            // Rebuild full neighbor/incidence pointers before returning.
            tds.assign_neighbors().map_err(PlManifoldRepairError::Tds)?;
            tds.assign_incident_simplices()
                .map_err(|e| PlManifoldRepairError::Tds(e.into()))?;
            return Ok(stats);
        }
    }

    Err(PlManifoldRepairError::BudgetExhausted {
        iterations: stats.iterations,
        simplices_removed: stats.simplices_removed,
        max_iterations: config.max_iterations,
        max_simplices_removed: config.max_simplices_removed,
    })
}

/// Deterministic quality score for a simplex: lower = better quality.
///
/// Uses an edge-length aspect-ratio metric (max/min edge) that operates
/// directly on the `Tds` without requiring a full `Triangulation` or
/// circumsphere computation.
fn simplex_quality_score<U, V, const D: usize>(tds: &Tds<U, V, D>, simplex_key: SimplexKey) -> f64
where
    U: DataType,
    V: DataType,
{
    let Ok(vertices) = tds.simplex_vertices(simplex_key) else {
        return f64::MAX;
    };

    let mut edge_lengths: SmallBuffer<f64, 16> = SmallBuffer::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let Some(vi) = tds.vertex(vertices[i]) else {
                return f64::MAX;
            };
            let Some(vj) = tds.vertex(vertices[j]) else {
                return f64::MAX;
            };

            let mut diff = [0.0; D];
            for (idx, d) in diff.iter_mut().enumerate() {
                *d = vi.point().coords()[idx] - vj.point().coords()[idx];
            }
            let len = hypot(&diff);
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

/// Among the simplices incident to an over-shared facet, deterministically select
/// the worst-quality simplex for removal.
///
/// Selection order: highest quality score first (worst simplex), then canonicalized
/// vertex keys (ascending), then simplex UUID (ascending).
fn pick_worst_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handles: &[FacetHandle],
) -> Option<SimplexKey>
where
    U: DataType,
    V: DataType,
{
    struct Candidate {
        simplex_key: SimplexKey,
        score: f64,
        vertex_keys: Vec<u64>,
        uuid: Uuid,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(handles.len());

    for handle in handles {
        let simplex_key = handle.simplex_key();
        let Some(simplex) = tds.simplex(simplex_key) else {
            continue;
        };

        let score = simplex_quality_score(tds, simplex_key);

        let mut vertex_keys: Vec<u64> = simplex
            .vertices()
            .iter()
            .map(|vk| vk.data().as_ffi())
            .collect();
        vertex_keys.sort_unstable();

        candidates.push(Candidate {
            simplex_key,
            score,
            vertex_keys,
            uuid: simplex.uuid(),
        });
    }

    // Worst quality first, then vertex keys, then UUID.
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.vertex_keys.cmp(&b.vertex_keys))
            .then_with(|| a.uuid.cmp(&b.uuid))
    });

    candidates.first().map(|c| c.simplex_key)
}

/// Remove vertices with no incident simplex from the TDS, collecting them into
/// `stats.removed_vertices` so callers can recover their data.
///
/// Vertices are sorted by UUID before removal for deterministic ordering.
fn remove_orphaned_vertices<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    stats: &mut PlManifoldRepairStats<U, V, D>,
) where
    U: DataType,
    V: DataType,
{
    let mut orphaned: Vec<(VertexKey, Uuid)> = tds
        .vertices()
        .filter(|(_, v)| v.incident_simplex().is_none())
        .map(|(k, v)| (k, v.uuid()))
        .collect();
    orphaned.sort_by_key(|(_, uuid)| *uuid);

    for (vk, _) in orphaned {
        if let Some(vertex) = tds.vertex(vk) {
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
    use crate::triangulation::DelaunayTriangulation;

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
        assert_eq!(config.max_simplices_removed, 10_000);
    }

    #[test]
    fn stats_default_does_not_require_data_type_metadata() {
        struct NonDataType(String);

        let stats: PlManifoldRepairStats<String, NonDataType, 3> = PlManifoldRepairStats::default();

        assert_eq!(stats.iterations, 0);
        assert!(stats.removed_simplices.is_empty());
        assert!(stats.removed_vertices.is_empty());

        let metadata = NonDataType("owned metadata".to_string());
        assert_eq!(metadata.0, "owned metadata");
    }

    // =============================================================================
    // ALREADY PL-MANIFOLD TESTS
    // =============================================================================

    #[test]
    fn test_already_pl_manifold_is_noop() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let config = PlManifoldRepairConfig::default();
        let stats = repair_facet_oversharing(&mut tds, &config).unwrap();

        assert!(stats.succeeded);
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.simplices_removed, 0);
    }

    // =============================================================================
    // BUDGET EXHAUSTION TESTS
    // =============================================================================

    #[test]
    fn test_budget_exhaustion_zero_iterations() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        // Zero iterations — should succeed because already PL-manifold.
        let config = PlManifoldRepairConfig {
            max_iterations: 0,
            max_simplices_removed: 10_000,
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let config = PlManifoldRepairConfig::default();
        let stats = repair_facet_oversharing(&mut tds, &config).unwrap();

        assert!(stats.succeeded);
        assert_eq!(stats.simplices_removed, 0);
    }

    // =============================================================================
    // STATS POPULATION TESTS
    // =============================================================================

    #[test]
    fn test_stats_populated_on_success() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();

        let stats = repair_facet_oversharing(&mut tds, &PlManifoldRepairConfig::default()).unwrap();

        assert!(stats.succeeded);
        // iterations == 0 for already-valid
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.simplices_removed, 0);
    }

    // =============================================================================
    // DETERMINISM TESTS
    // =============================================================================

    // =============================================================================
    // FACET OVER-SHARING REPAIR TESTS
    // =============================================================================

    /// Helper: create a TDS with over-shared facets by duplicating a simplex in a
    /// multi-simplex triangulation. Interior facets go from degree 2 to degree 3.
    fn make_overshared_tds() -> Tds<(), (), 3> {
        // 5 points
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        assert!(
            tds.number_of_simplices() > 1,
            "Need multiple simplices for interior facets"
        );

        // Duplicate the first simplex → its facets go from degree 2 to degree 3.
        let simplex_key = tds.simplex_keys().next().unwrap();
        let vkeys = tds.simplex_vertices(simplex_key).unwrap();
        let dup_simplex = Simplex::try_new_with_data(vkeys.to_vec(), None).unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(dup_simplex)
            .unwrap();

        // Sanity: at least one facet should now be over-shared.
        let facet_map = tds.build_facet_to_simplices_map().unwrap();
        assert!(
            validate_facet_degree(&facet_map).is_err(),
            "Expected over-shared facets after duplicating a simplex"
        );

        tds
    }

    /// Create a TDS with over-shared facets and verify that repair removes
    /// simplices until the facet-degree invariant is satisfied.
    #[test]
    fn test_repair_removes_overshared_simplices() {
        init_tracing();
        let mut tds = make_overshared_tds();
        let simplices_before = tds.number_of_simplices();

        let stats = repair_facet_oversharing(&mut tds, &PlManifoldRepairConfig::default()).unwrap();

        assert!(stats.succeeded);
        assert!(stats.simplices_removed > 0);
        assert!(stats.iterations > 0);
        assert!(tds.number_of_simplices() < simplices_before);

        // removed_simplices should contain exactly simplices_removed entries.
        assert_eq!(stats.removed_simplices.len(), stats.simplices_removed);

        let facet_map = tds.build_facet_to_simplices_map().unwrap();
        assert!(validate_facet_degree(&facet_map).is_ok());
    }

    /// Verify that a tight simplex-removal budget triggers `BudgetExhausted`.
    #[test]
    fn test_repair_budget_exhausted_on_overshared_tds() {
        init_tracing();
        let mut tds = make_overshared_tds();

        let config = PlManifoldRepairConfig {
            max_iterations: 64,
            max_simplices_removed: 0,
        };
        let result = repair_facet_oversharing(&mut tds, &config);
        assert!(
            matches!(result, Err(PlManifoldRepairError::BudgetExhausted { .. })),
            "Expected BudgetExhausted, got: {result:?}"
        );
    }

    /// Verify that `max_iterations: 0` on an over-shared TDS triggers
    /// `BudgetExhausted` (iteration budget, not simplex budget).
    #[test]
    fn test_repair_iteration_budget_exhausted_on_overshared_tds() {
        init_tracing();
        let mut tds = make_overshared_tds();

        let config = PlManifoldRepairConfig {
            max_iterations: 0,
            max_simplices_removed: 10_000,
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

    /// Verify that `simplex_quality_score` returns a finite, positive value for a
    /// valid simplex and is deterministic across calls.
    #[test]
    fn test_simplex_quality_score_finite_and_deterministic() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tds = dt.tds();
        let simplex_key = tds.simplex_keys().next().unwrap();

        let score1 = simplex_quality_score(tds, simplex_key);
        let score2 = simplex_quality_score(tds, simplex_key);

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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
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
