//! Bounded deterministic PL-manifold topology repair.
//!
//! This module implements repair algorithms that attempt to bring a
//! triangulation closer to satisfying the
//! [`TopologyGuarantee::PLManifold`](crate::prelude::validation::TopologyGuarantee::PLManifold)
//! invariant by removing simplices that cause facet over-sharing, non-closed
//! boundary ridges, non-manifold ridge links, or non-manifold vertex links.
//!
//! # Algorithm
//!
//! 1. **Structural precheck**: validate Levels 1–2 via `Tds::validate()`. If
//!    the facet-degree invariant already holds, return early with zero work.
//! 2. **Iterative facet over-sharing repair**: build the facet-to-simplices map,
//!    identify facets with degree > 2, deterministically select the worst-quality
//!    simplex per over-shared facet for removal, remove the batch, and rebuild
//!    neighbors/incidence.
//! 3. **Targeted topology repair**: validate boundary-ridge multiplicity,
//!    ridge-link manifoldness, and vertex-link manifoldness, then remove the
//!    worst-quality simplex in the local violating star until each targeted
//!    condition passes or the repair budget is exhausted.
//! 4. **Termination**: the loop terminates on success (all targeted PL-manifold
//!    checks pass), budget exhaustion (`max_iterations` or
//!    `max_simplices_removed`), or no-progress (a violation has no removable
//!    local candidate).
//!
//! The narrower `repair_facet_oversharing` entrypoint remains available for
//! callers that only want codimension-1 facet-degree repair:
//! the loop terminates on success (all facets have degree
//!    ≤ 2), budget exhaustion (`max_iterations` or `max_simplices_removed`), or
//!    no-progress (zero simplices removed in a pass).
//!
//! # Determinism
//!
//! Simplex removal order is deterministic: candidates are sorted by
//! (descending quality, canonicalized vertex tuple, simplex UUID) so that
//! repeated runs on identical input produce identical output.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};

use crate::core::collections::{
    SimplexKeyBuffer, SimplexKeySet, VertexKeyBuffer, fast_hash_set_with_capacity,
};
use crate::core::facet::{FacetHandle, facet_key_from_vertices};
use crate::core::simplex::Simplex;
use crate::core::tds::{SimplexKey, Tds, TdsError, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::util::norms::hypot;
use crate::topology::manifold::{
    BoundaryFacetClassification, ManifoldError, ValidatedFacetDegreeMap, classify_boundary_facet,
    validate_closed_boundary_from_validated_facet_map, validate_ridge_links,
    validate_vertex_links_from_validated_facet_map,
};
use crate::topology::ridge::{build_ridge_star_map, simplex_star_simplices};
use crate::topology::traits::topological_space::GlobalTopology;
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

/// Targeted PL-manifold repair stage that produced a topology-repair diagnostic.
///
/// The stages match the validation layers repaired after codimension-1 facet
/// degree repair: boundary-ridge multiplicity, ridge-link manifoldness, and
/// vertex-link manifoldness.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::delaunayize::PlManifoldRepairStage;
///
/// let stage = PlManifoldRepairStage::RidgeLink;
/// assert_eq!(format!("{stage:?}"), "RidgeLink");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlManifoldRepairStage {
    /// Repair of boundary ridges with invalid boundary-facet multiplicity.
    BoundaryRidgeMultiplicity,
    /// Repair of non-manifold codimension-2 ridge links.
    RidgeLink,
    /// Repair of non-manifold vertex links.
    VertexLink,
}

impl Display for PlManifoldRepairStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::BoundaryRidgeMultiplicity => "boundary-ridge multiplicity",
            Self::RidgeLink => "ridge-link",
            Self::VertexLink => "vertex-link",
        };
        f.write_str(label)
    }
}

/// Returns whether a manifold validation error belongs to a targeted repair stage.
///
/// Benchmark fixtures use this as the shared source of truth for the
/// stage-to-error contract exercised by targeted PL-manifold repair.
#[cfg(any(test, feature = "bench"))]
pub(crate) const fn manifold_error_matches_repair_stage(
    source: &ManifoldError,
    stage: PlManifoldRepairStage,
) -> bool {
    matches!(
        (stage, source),
        (
            PlManifoldRepairStage::BoundaryRidgeMultiplicity,
            ManifoldError::BoundaryRidgeMultiplicity { .. }
        ) | (
            PlManifoldRepairStage::RidgeLink,
            ManifoldError::RidgeLinkNotManifold { .. }
        ) | (
            PlManifoldRepairStage::VertexLink,
            ManifoldError::VertexLinkNotManifold { .. }
        )
    )
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
#[must_use]
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
    /// Whether the requested PL-manifold repair target was satisfied at termination.
    ///
    /// For facet-over-sharing repair, this means the codimension-1 facet-degree
    /// invariant holds. For the full
    /// [`delaunayize_by_flips`](crate::delaunayize::delaunayize_by_flips)
    /// topology stage, this means all targeted boundary-ridge, ridge-link, and
    /// vertex-link checks passed.
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

    /// A targeted topology stage exhausted the shared repair budget before its
    /// manifold invariant was satisfied.
    #[error(
        "PL-manifold {stage} repair budget exhausted: {iterations} iterations, {simplices_removed} simplices removed (max_iterations={max_iterations}, max_simplices_removed={max_simplices_removed}); remaining violation: {source}"
    )]
    TargetedBudgetExhausted {
        /// Targeted repair stage that could not finish within budget.
        stage: PlManifoldRepairStage,
        /// The validation error that was still present.
        #[source]
        source: Box<ManifoldError>,
        /// Iterations executed across all topology repair stages.
        iterations: usize,
        /// Simplices removed across all topology repair stages.
        simplices_removed: usize,
        /// Configured iteration limit.
        max_iterations: usize,
        /// Configured simplex-removal limit.
        max_simplices_removed: usize,
    },

    /// A targeted topology stage found a violation but could not identify or
    /// remove a local candidate simplex.
    #[error(
        "PL-manifold {stage} repair made no progress after {iterations} iterations ({simplices_removed} simplices removed); remaining violation: {source}"
    )]
    TargetedNoProgress {
        /// Targeted repair stage that stalled.
        stage: PlManifoldRepairStage,
        /// The validation error that could not be repaired.
        #[source]
        source: Box<ManifoldError>,
        /// Iterations executed across all topology repair stages.
        iterations: usize,
        /// Simplices removed across all topology repair stages.
        simplices_removed: usize,
    },

    /// A stage validator reported an error outside the violation shape owned
    /// by the current targeted repair stage.
    #[error("PL-manifold validation failed during {stage} repair: {source}")]
    TargetedValidation {
        /// Targeted repair stage being evaluated.
        stage: PlManifoldRepairStage,
        /// The typed manifold validation error.
        #[source]
        source: Box<ManifoldError>,
    },

    /// Final targeted topology validation reported a manifold error outside
    /// the set handled by targeted repair stages.
    #[error("PL-manifold targeted repair postcondition validation failed: {source}")]
    TargetedPostconditionValidation {
        /// Typed manifold validation error reported by the final roll-up.
        #[source]
        source: Box<ManifoldError>,
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
    U: Clone,
    V: Clone,
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
    if ValidatedFacetDegreeMap::try_from_facet_map(&facet_map).is_ok() {
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
            prepare_error_return_topology(tds);
            return Err(PlManifoldRepairError::NoProgress {
                over_shared_facets: remaining,
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
            });
        }

        // Check simplex-removal budget.
        let batch_size = removal_candidates.len();
        if stats.simplices_removed.saturating_add(batch_size) > config.max_simplices_removed {
            prepare_error_return_topology(tds);
            return Err(PlManifoldRepairError::BudgetExhausted {
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
                max_iterations: config.max_iterations,
                max_simplices_removed: config.max_simplices_removed,
            });
        }

        // Snapshot simplices before removal (sorted by UUID for determinism).
        // Precompute UUID keys once so sorting does not repeatedly probe TDS storage.
        let mut removals: Vec<(Option<Uuid>, SimplexKey)> = removal_candidates
            .into_iter()
            .map(|simplex_key| (tds.simplex(simplex_key).map(Simplex::uuid), simplex_key))
            .collect();
        removals.sort_unstable_by_key(|(uuid, _)| *uuid);

        let mut keys = Vec::with_capacity(removals.len());
        for (_, ck) in removals {
            if let Some(simplex) = tds.simplex(ck) {
                stats.removed_simplices.push(simplex.clone());
            }
            keys.push(ck);
        }

        // Remove the batch. `remove_simplices_by_keys` handles local neighbor
        // back-reference clearing and incident-simplex repair. Full neighbor
        // rebuild is deferred to after the loop to avoid O(simplices²) work.
        let removed = tds
            .remove_simplices_by_keys(&keys)
            .map_err(|e| PlManifoldRepairError::Tds(e.into_inner()))?;
        stats.simplices_removed += removed;

        // Remove orphaned vertices (required for PL-manifold validity).
        remove_orphaned_vertices(tds, &mut stats)?;

        // Check if the invariant now holds.
        let facet_map = tds.build_facet_to_simplices_map()?;
        if ValidatedFacetDegreeMap::try_from_facet_map(&facet_map).is_ok() {
            stats.succeeded = true;
            // Rebuild full neighbor/incidence pointers before returning.
            rebuild_success_topology(tds)?;
            return Ok(stats);
        }
    }

    prepare_error_return_topology(tds);
    Err(PlManifoldRepairError::BudgetExhausted {
        iterations: stats.iterations,
        simplices_removed: stats.simplices_removed,
        max_iterations: config.max_iterations,
        max_simplices_removed: config.max_simplices_removed,
    })
}

/// Attempts full targeted PL-manifold repair after codimension-1 facet repair.
///
/// This orchestrator first delegates to [`repair_facet_oversharing`], then
/// runs dedicated bounded stages for boundary-ridge multiplicity, ridge-link
/// manifoldness, and vertex-link manifoldness. Each targeted stage removes the
/// worst-quality simplex from the local star responsible for the first
/// reported violation.
///
/// # Errors
///
/// Returns [`PlManifoldRepairError`] if the codimension-1 repair fails, a
/// targeted repair stage exhausts the shared budget, a targeted violation has no
/// removable local candidate, a stage validator reports a manifold error outside
/// the stage currently being repaired, or final targeted validation reports an
/// unrepaired manifold error outside the targeted stage set.
pub(crate) fn repair_pl_manifold_topology<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    config: &PlManifoldRepairConfig,
) -> Result<PlManifoldRepairStats<U, V, D>, PlManifoldRepairError>
where
    U: Clone,
    V: Clone,
{
    let mut stats = repair_facet_oversharing(tds, config)?;
    stats.succeeded = false;

    loop {
        let iterations_before_cycle = stats.iterations;
        let removed_before_cycle = stats.simplices_removed;

        for stage in TARGETED_REPAIR_STAGES {
            repair_targeted_topology_stage(tds, global_topology, config, &mut stats, stage)?;
        }

        match validate_targeted_topology(tds, global_topology) {
            Ok(()) => {
                stats.succeeded = true;
                rebuild_success_topology(tds)?;
                return Ok(stats);
            }
            Err(source) => {
                let violation = match TargetedTopologyViolation::try_from(source) {
                    Ok(violation) => violation,
                    Err(source) => {
                        prepare_error_return_topology(tds);
                        return Err(PlManifoldRepairError::TargetedPostconditionValidation {
                            source: Box::new(source),
                        });
                    }
                };
                let stage = violation.stage();

                let no_progress = stats.iterations == iterations_before_cycle
                    && stats.simplices_removed == removed_before_cycle;
                if no_progress {
                    prepare_error_return_topology(tds);
                    return Err(PlManifoldRepairError::TargetedNoProgress {
                        stage,
                        source: Box::new(violation.source().clone()),
                        iterations: stats.iterations,
                        simplices_removed: stats.simplices_removed,
                    });
                }

                if stats.iterations >= config.max_iterations {
                    prepare_error_return_topology(tds);
                    return Err(PlManifoldRepairError::TargetedBudgetExhausted {
                        stage,
                        source: Box::new(violation.source().clone()),
                        iterations: stats.iterations,
                        simplices_removed: stats.simplices_removed,
                        max_iterations: config.max_iterations,
                        max_simplices_removed: config.max_simplices_removed,
                    });
                }
            }
        }
    }
}

const TARGETED_REPAIR_STAGES: [PlManifoldRepairStage; 3] = [
    PlManifoldRepairStage::BoundaryRidgeMultiplicity,
    PlManifoldRepairStage::RidgeLink,
    PlManifoldRepairStage::VertexLink,
];

/// Repairable targeted topology violation with its stage and candidate key parsed.
#[derive(Clone, Debug, PartialEq)]
enum TargetedTopologyViolation {
    /// Boundary ridge whose incident boundary-facet multiplicity is invalid.
    BoundaryRidgeMultiplicity {
        /// Offending ridge key.
        ridge_key: u64,
        /// Original typed validation source.
        source: ManifoldError,
    },
    /// Ridge whose link is not manifold.
    RidgeLink {
        /// Offending ridge key.
        ridge_key: u64,
        /// Original typed validation source.
        source: ManifoldError,
    },
    /// Vertex whose link is not manifold.
    VertexLink {
        /// Offending vertex key.
        vertex_key: VertexKey,
        /// Original typed validation source.
        source: ManifoldError,
    },
}

impl TargetedTopologyViolation {
    /// Returns the repair stage that owns this parsed violation.
    const fn stage(&self) -> PlManifoldRepairStage {
        match self {
            Self::BoundaryRidgeMultiplicity { .. } => {
                PlManifoldRepairStage::BoundaryRidgeMultiplicity
            }
            Self::RidgeLink { .. } => PlManifoldRepairStage::RidgeLink,
            Self::VertexLink { .. } => PlManifoldRepairStage::VertexLink,
        }
    }

    /// Returns the original typed validation source.
    const fn source(&self) -> &ManifoldError {
        match self {
            Self::BoundaryRidgeMultiplicity { source, .. }
            | Self::RidgeLink { source, .. }
            | Self::VertexLink { source, .. } => source,
        }
    }

    /// Returns the original typed validation source.
    fn into_source(self) -> ManifoldError {
        match self {
            Self::BoundaryRidgeMultiplicity { source, .. }
            | Self::RidgeLink { source, .. }
            | Self::VertexLink { source, .. } => source,
        }
    }
}

impl TryFrom<ManifoldError> for TargetedTopologyViolation {
    type Error = ManifoldError;

    fn try_from(source: ManifoldError) -> Result<Self, Self::Error> {
        match source {
            source @ ManifoldError::BoundaryRidgeMultiplicity { ridge_key, .. } => {
                Ok(Self::BoundaryRidgeMultiplicity { ridge_key, source })
            }
            source @ ManifoldError::RidgeLinkNotManifold { ridge_key, .. } => {
                Ok(Self::RidgeLink { ridge_key, source })
            }
            source @ ManifoldError::VertexLinkNotManifold { vertex_key, .. } => {
                Ok(Self::VertexLink { vertex_key, source })
            }
            source => Err(source),
        }
    }
}

/// Runs one targeted topology repair stage until that stage's validator passes.
fn repair_targeted_topology_stage<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    config: &PlManifoldRepairConfig,
    stats: &mut PlManifoldRepairStats<U, V, D>,
    stage: PlManifoldRepairStage,
) -> Result<(), PlManifoldRepairError>
where
    U: Clone,
    V: Clone,
{
    loop {
        let Some(violation) = targeted_stage_violation(tds, global_topology, stage)? else {
            return Ok(());
        };

        if stats.iterations >= config.max_iterations
            || stats.simplices_removed.saturating_add(1) > config.max_simplices_removed
        {
            prepare_error_return_topology(tds);
            return Err(PlManifoldRepairError::TargetedBudgetExhausted {
                stage,
                source: Box::new(violation.source().clone()),
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
                max_iterations: config.max_iterations,
                max_simplices_removed: config.max_simplices_removed,
            });
        }

        let candidates = candidate_simplices_for_violation(tds, global_topology, &violation)?;
        let Some(simplex_key) = pick_worst_simplex_key(tds, candidates.as_slice()) else {
            prepare_error_return_topology(tds);
            return Err(PlManifoldRepairError::TargetedNoProgress {
                stage,
                source: Box::new(violation.source().clone()),
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
            });
        };

        let removed = remove_targeted_simplex(tds, stats, simplex_key)?;
        if removed == 0 {
            prepare_error_return_topology(tds);
            return Err(PlManifoldRepairError::TargetedNoProgress {
                stage,
                source: Box::new(violation.into_source()),
                iterations: stats.iterations,
                simplices_removed: stats.simplices_removed,
            });
        }
        stats.iterations = stats.iterations.saturating_add(1);
    }
}

/// Returns the first violation owned by a targeted repair stage.
fn targeted_stage_violation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    stage: PlManifoldRepairStage,
) -> Result<Option<TargetedTopologyViolation>, PlManifoldRepairError> {
    let result = match stage {
        PlManifoldRepairStage::BoundaryRidgeMultiplicity => {
            validate_boundary_ridge_multiplicity(tds, global_topology)
        }
        PlManifoldRepairStage::RidgeLink => validate_ridge_links(tds),
        PlManifoldRepairStage::VertexLink => {
            validate_vertex_link_manifoldness(tds, global_topology)
        }
    };

    match result {
        Ok(()) => Ok(None),
        Err(source) => match TargetedTopologyViolation::try_from(source) {
            Ok(violation) if violation.stage() == stage => Ok(Some(violation)),
            Ok(violation) => Err(PlManifoldRepairError::TargetedValidation {
                stage,
                source: Box::new(violation.into_source()),
            }),
            Err(source) => Err(PlManifoldRepairError::TargetedValidation {
                stage,
                source: Box::new(source),
            }),
        },
    }
}

/// Validates all targeted PL-manifold conditions in pipeline order.
fn validate_targeted_topology<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    validate_boundary_ridge_multiplicity(tds, global_topology)?;
    validate_ridge_links(tds)?;
    validate_vertex_link_manifoldness(tds, global_topology)
}

/// Validates the boundary-ridge multiplicity stage against a fresh facet map.
fn validate_boundary_ridge_multiplicity<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    let facet_to_simplices = tds.build_facet_to_simplices_map()?;
    let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
    validate_closed_boundary_from_validated_facet_map(tds, facet_to_simplices, global_topology)
}

/// Validates vertex-link manifoldness against a fresh facet map.
fn validate_vertex_link_manifoldness<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    let facet_to_simplices = tds.build_facet_to_simplices_map()?;
    let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
    validate_vertex_links_from_validated_facet_map(tds, facet_to_simplices, global_topology)
}

/// Returns the local simplex candidates responsible for a targeted violation.
fn candidate_simplices_for_violation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    violation: &TargetedTopologyViolation,
) -> Result<SimplexKeyBuffer, PlManifoldRepairError> {
    let stage = violation.stage();
    let candidates = match violation {
        TargetedTopologyViolation::BoundaryRidgeMultiplicity { ridge_key, .. } => {
            boundary_ridge_candidate_simplices(tds, global_topology, *ridge_key)
        }
        TargetedTopologyViolation::RidgeLink { ridge_key, .. } => {
            ridge_link_candidate_simplices(tds, *ridge_key)
        }
        TargetedTopologyViolation::VertexLink { vertex_key, .. } => {
            vertex_link_candidate_simplices(tds, *vertex_key)
        }
    }
    .map_err(|source| PlManifoldRepairError::TargetedValidation {
        stage,
        source: Box::new(source),
    })?;
    Ok(candidates)
}

/// Finds simplices incident to boundary facets containing the offending boundary ridge.
fn boundary_ridge_candidate_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    ridge_key: u64,
) -> Result<SimplexKeyBuffer, ManifoldError> {
    let facet_index = tds.build_facet_to_simplices_index()?;
    let mut candidates = SimplexKeyBuffer::new();
    let mut seen: SimplexKeySet = fast_hash_set_with_capacity(4);
    let mut facet_vertices = VertexKeyBuffer::with_capacity(D);
    let mut ridge_vertices = VertexKeyBuffer::with_capacity(D.saturating_sub(1));

    for incidence in facet_index.iter() {
        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet(incidence, global_topology)?
        else {
            continue;
        };
        simplex_facet_vertices(tds, handle, &mut facet_vertices)?;
        if !facet_contains_ridge_key(&facet_vertices, ridge_key, &mut ridge_vertices) {
            continue;
        }
        push_unique_candidate(&mut candidates, &mut seen, handle.simplex_key());
    }

    sort_simplex_candidates_by_uuid(tds, &mut candidates);
    Ok(candidates)
}

/// Finds simplices in the offending ridge's star.
fn ridge_link_candidate_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_key: u64,
) -> Result<SimplexKeyBuffer, ManifoldError> {
    let ridge_to_star = build_ridge_star_map(tds)?;
    let mut candidates = SimplexKeyBuffer::new();
    if let Some(star) = ridge_to_star.get(&ridge_key) {
        candidates.extend(star.star_simplices.iter().copied());
    }
    sort_simplex_candidates_by_uuid(tds, &mut candidates);
    Ok(candidates)
}

/// Finds simplices in the offending vertex's star.
fn vertex_link_candidate_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<SimplexKeyBuffer, ManifoldError> {
    let star_simplices = simplex_star_simplices(tds, &[vertex_key])?;
    let mut candidates = SimplexKeyBuffer::with_capacity(star_simplices.len());
    candidates.extend(star_simplices.iter().copied());
    sort_simplex_candidates_by_uuid(tds, &mut candidates);
    Ok(candidates)
}

/// Extracts one facet's vertices from its owning simplex.
fn simplex_facet_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
    facet_vertices: &mut VertexKeyBuffer,
) -> Result<(), ManifoldError> {
    let simplex_key = handle.simplex_key();
    let facet_index: usize = handle.facet_index().into();
    let simplex_vertices = tds.simplex_vertices(simplex_key)?;
    if facet_index >= simplex_vertices.len() {
        return Err(TdsError::IndexOutOfBounds {
            index: facet_index,
            bound: simplex_vertices.len(),
            context: format!("targeted repair facet index for simplex {simplex_key:?}"),
        }
        .into());
    }

    facet_vertices.clear();
    for (index, &vertex_key) in simplex_vertices.iter().enumerate() {
        if index != facet_index {
            facet_vertices.push(vertex_key);
        }
    }

    Ok(())
}

/// Checks whether a boundary facet contains the reported boundary ridge key.
fn facet_contains_ridge_key(
    facet_vertices: &[VertexKey],
    target_ridge_key: u64,
    ridge_vertices: &mut VertexKeyBuffer,
) -> bool {
    for omit in 0..facet_vertices.len() {
        ridge_vertices.clear();
        for (index, &vertex_key) in facet_vertices.iter().enumerate() {
            if index != omit {
                ridge_vertices.push(vertex_key);
            }
        }
        if facet_key_from_vertices(ridge_vertices.as_slice()) == target_ridge_key {
            return true;
        }
    }
    false
}

/// Appends a simplex candidate once while preserving first-seen order.
fn push_unique_candidate(
    candidates: &mut SimplexKeyBuffer,
    seen: &mut SimplexKeySet,
    simplex_key: SimplexKey,
) {
    if seen.insert(simplex_key) {
        candidates.push(simplex_key);
    }
}

/// Sorts local simplex candidates by stable simplex UUID for deterministic ties.
fn sort_simplex_candidates_by_uuid<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    candidates: &mut SimplexKeyBuffer,
) {
    candidates.sort_unstable_by_key(|simplex_key| tds.simplex(*simplex_key).map(Simplex::uuid));
}

/// Removes one targeted repair candidate and updates aggregate repair stats.
fn remove_targeted_simplex<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    stats: &mut PlManifoldRepairStats<U, V, D>,
    simplex_key: SimplexKey,
) -> Result<usize, PlManifoldRepairError>
where
    U: Clone,
    V: Clone,
{
    if let Some(simplex) = tds.simplex(simplex_key) {
        stats.removed_simplices.push(simplex.clone());
    }
    let removed = tds
        .remove_simplices_by_keys(&[simplex_key])
        .map_err(|e| PlManifoldRepairError::Tds(e.into_inner()))?;
    if removed == 0 {
        return Ok(0);
    }
    stats.simplices_removed += removed;
    remove_orphaned_vertices(tds, stats)?;
    tds.assign_incident_simplices()
        .map_err(|e| PlManifoldRepairError::Tds(e.into()))?;
    Ok(removed)
}

/// Rebuilds topology metadata before a successful PL-manifold repair result is observed.
///
/// Successful repair has restored the facet-degree invariant, so neighbor and
/// incident-simplex metadata can be rebuilt strictly and any failure should be
/// surfaced to the caller.
///
/// # Errors
///
/// Returns [`PlManifoldRepairError::Tds`] if neighbor assignment or
/// incident-simplex assignment fails while restoring topology metadata.
fn rebuild_success_topology<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
) -> Result<(), PlManifoldRepairError> {
    tds.assign_neighbors().map_err(PlManifoldRepairError::Tds)?;
    tds.assign_incident_simplices()
        .map_err(|e| PlManifoldRepairError::Tds(e.into()))
}

/// Best-effort topology metadata repair before returning a non-convergence error.
///
/// Error exits may still contain over-shared facets, so strict neighbor assignment
/// can fail even though the intended public result is `NoProgress` or
/// `BudgetExhausted`. This helper preserves that primary repair outcome while
/// still repairing incident simplices and rebuilding neighbors when the remaining
/// topology permits it.
fn prepare_error_return_topology<U, V, const D: usize>(tds: &mut Tds<U, V, D>) {
    if let Err(error) = tds.assign_incident_simplices() {
        tracing::debug!(
            ?error,
            "PL-manifold repair could not rebuild incident simplices before error return"
        );
    }

    if let Err(error) = tds.assign_neighbors() {
        tracing::debug!(
            ?error,
            "PL-manifold repair could not rebuild neighbors before error return"
        );
    }
}

/// Candidate metadata used to order simplex removals deterministically.
struct SimplexRemovalCandidate {
    simplex_key: SimplexKey,
    score: f64,
    vertex_keys: Vec<u64>,
    uuid: Uuid,
}

/// Deterministic deletion quality score for a simplex: lower = better quality.
///
/// Uses the edge-length aspect-ratio conditioning heuristic described in
/// `REFERENCES.md`'s mesh quality and simplex-shape references, especially
/// Shewchuk's "What Is a Good Linear Element?". The score operates directly on
/// the `Tds` without requiring a full `Triangulation` or circumsphere
/// computation. Invalid handles, missing vertices, empty edge sets,
/// zero-length edges, and non-finite geometry are ranked as [`f64::MAX`] so
/// they sort as the worst deletion candidates.
fn simplex_quality_score<U, V, const D: usize>(tds: &Tds<U, V, D>, simplex_key: SimplexKey) -> f64 {
    let Ok(vertices) = tds.simplex_vertices(simplex_key) else {
        return f64::MAX;
    };

    let mut edge_count = 0_usize;
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    let mut min_edge = f64::INFINITY;
    let mut max_edge = f64::NEG_INFINITY;
    let mut edge_count_scalar = 0.0;

    for (i, &vi_key) in vertices.iter().enumerate() {
        let Some(vi) = tds.vertex(vi_key) else {
            return f64::MAX;
        };
        for &vj_key in vertices.iter().skip(i + 1) {
            let Some(vj) = tds.vertex(vj_key) else {
                return f64::MAX;
            };

            let mut diff = [0.0; D];
            for (idx, d) in diff.iter_mut().enumerate() {
                *d = vi.point().coords()[idx] - vj.point().coords()[idx];
            }
            let len = hypot(&diff);
            if !len.is_finite() {
                return f64::MAX;
            }
            edge_count += 1;
            let Some(current_edge_count): Option<f64> = NumCast::from(edge_count) else {
                return f64::MAX;
            };
            let delta = len - mean;
            mean += delta / current_edge_count;
            sum_squared_deviations += delta * (len - mean);
            edge_count_scalar = current_edge_count;
            min_edge = min_edge.min(len);
            max_edge = max_edge.max(len);
        }
    }

    if edge_count == 0 {
        return f64::MAX;
    }

    let variance = sum_squared_deviations / edge_count_scalar;
    if !variance.is_finite() {
        return f64::MAX;
    }
    let variance = variance.max(0.0);
    if !min_edge.is_finite() || !max_edge.is_finite() || min_edge <= 0.0 {
        return f64::MAX;
    }
    // Primary: aspect ratio; secondary: edge-length variance as tiebreaker.
    let score = (max_edge / min_edge) + variance * 1e-12;
    if score.is_finite() { score } else { f64::MAX }
}

/// Among the simplices incident to an over-shared facet, deterministically select
/// the worst-quality simplex for removal.
///
/// Selection order: highest quality score first (worst simplex), then canonicalized
/// vertex keys (ascending), then simplex UUID (ascending).
fn pick_worst_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handles: &[FacetHandle],
) -> Option<SimplexKey> {
    pick_worst_candidate(tds, handles.iter().map(FacetHandle::simplex_key))
}

/// Selects the worst-quality simplex from a direct simplex-key candidate set.
fn pick_worst_simplex_key<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_keys: &[SimplexKey],
) -> Option<SimplexKey> {
    pick_worst_candidate(tds, simplex_keys.iter().copied())
}

/// Builds deterministic removal-order metadata for one live simplex.
fn simplex_removal_candidate<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> Option<SimplexRemovalCandidate> {
    let simplex = tds.simplex(simplex_key)?;
    let score = simplex_quality_score(tds, simplex_key);
    let mut vertex_keys: Vec<u64> = simplex
        .vertices()
        .iter()
        .map(|vk| vk.data().as_ffi())
        .collect();
    vertex_keys.sort_unstable();

    Some(SimplexRemovalCandidate {
        simplex_key,
        score,
        vertex_keys,
        uuid: simplex.uuid(),
    })
}

/// Applies the shared removal ordering and returns the highest-priority candidate.
fn pick_worst_candidate<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_keys: impl IntoIterator<Item = SimplexKey>,
) -> Option<SimplexKey> {
    simplex_keys
        .into_iter()
        .filter_map(|simplex_key| simplex_removal_candidate(tds, simplex_key))
        .max_by(compare_removal_candidates)
        .map(|candidate| candidate.simplex_key)
}

/// Orders candidates by removal priority: worse quality, then stable identity.
fn compare_removal_candidates(
    a: &SimplexRemovalCandidate,
    b: &SimplexRemovalCandidate,
) -> Ordering {
    a.score
        .total_cmp(&b.score)
        .then_with(|| b.vertex_keys.cmp(&a.vertex_keys))
        .then_with(|| b.uuid.cmp(&a.uuid))
}

/// Remove vertices with no incident simplex from the TDS, collecting them into
/// `stats.removed_vertices` so callers can recover their data.
///
/// Vertices are sorted by UUID before removal for deterministic ordering.
fn remove_orphaned_vertices<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    stats: &mut PlManifoldRepairStats<U, V, D>,
) -> Result<(), PlManifoldRepairError>
where
    U: Clone,
{
    let mut orphaned: Vec<(VertexKey, Uuid)> = tds
        .vertices()
        .filter(|(vertex_key, _)| {
            tds.simplex_keys_containing_vertex(*vertex_key)
                .next()
                .is_none()
        })
        .map(|(k, v)| (k, v.uuid()))
        .collect();
    orphaned.sort_by_key(|(_, uuid)| *uuid);

    for (vk, _) in orphaned {
        if let Some(vertex) = tds.vertex(vk) {
            stats.removed_vertices.push(vertex.clone());
        }
        tds.remove_isolated_vertex(vk)
            .map_err(|e| PlManifoldRepairError::Tds(e.into_inner()))?;
    }

    Ok(())
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();
        assert!(
            tds.number_of_simplices() > 1,
            "Need multiple simplices for interior facets"
        );

        // Duplicate the first simplex → its facets go from degree 2 to degree 3.
        let simplex_key = tds.simplex_keys().next().unwrap();
        let vkeys = tds.simplex_vertices(simplex_key).unwrap().to_vec();
        let dup_simplex = Simplex::try_new_with_data(vkeys, None).unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(dup_simplex)
            .unwrap();

        // Sanity: at least one facet should now be over-shared.
        let facet_map = tds.build_facet_to_simplices_map().unwrap();
        assert!(
            ValidatedFacetDegreeMap::try_from_facet_map(&facet_map).is_err(),
            "Expected over-shared facets after duplicating a simplex"
        );

        tds
    }

    /// Create a TDS that cannot be fully repaired in a single iteration.
    fn make_multi_duplicate_overshared_tds() -> Tds<(), (), 3> {
        let mut tds = make_overshared_tds();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let vkeys = tds.simplex_vertices(simplex_key).unwrap().to_vec();

        for _ in 0..5 {
            let dup_simplex = Simplex::try_new_with_data(vkeys.clone(), None).unwrap();
            tds.insert_simplex_bypassing_topology_checks_for_test(dup_simplex)
                .unwrap();
        }

        tds
    }

    /// Build two tetrahedra sharing an edge but no facet, so that edge has four
    /// incident boundary facets.
    fn make_boundary_ridge_multiplicity_tds() -> Tds<(), (), 3> {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let shared_v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let shared_v1 = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let tet1_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.1, 1.0, 0.2]).unwrap())
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.3, 1.3]).unwrap())
            .unwrap();
        let tet2_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.4, -1.1, 0.7]).unwrap())
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.6, 0.2, -1.4]).unwrap())
            .unwrap();

        for tet in [
            [shared_v0, shared_v1, tet1_v2, tet1_v3],
            [shared_v0, shared_v1, tet2_v2, tet2_v3],
        ] {
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![tet[0], tet[1], tet[2], tet[3]], None).unwrap(),
            )
            .unwrap();
        }

        tds
    }

    /// Build two closed 2D sphere complexes sharing one vertex, yielding a
    /// disconnected ridge link at that shared vertex.
    fn make_disconnected_ridge_link_tds() -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]).unwrap())
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]).unwrap())
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]).unwrap())
            .unwrap();

        for tri in [
            [v0, v1, v2],
            [v0, v1, v3],
            [v0, v2, v3],
            [v1, v2, v3],
            [v0, v4, v5],
            [v0, v4, v6],
            [v0, v5, v6],
            [v4, v5, v6],
        ] {
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2]], None).unwrap(),
            )
            .unwrap();
        }

        tds
    }

    /// Build a cone over a small triangulated torus, whose apex has a torus
    /// vertex link rather than a sphere or ball.
    fn make_cone_on_torus_tds() -> Tds<(), (), 3> {
        const N: usize = 3;
        const M: usize = 3;

        let mut tds: Tds<(), (), 3> = Tds::empty();
        let mut grid: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in grid.iter_mut().enumerate() {
            for (j, vertex_key) in row.iter_mut().enumerate() {
                let i_f: f64 = u32::try_from(i).unwrap().into();
                let j_f: f64 = u32::try_from(j).unwrap().into();
                *vertex_key = tds
                    .insert_vertex_with_mapping(vertex!([i_f, j_f, 0.0]).unwrap())
                    .unwrap();
            }
        }
        let apex = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 1.0]).unwrap())
            .unwrap();

        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;
                let v00 = grid[i][j];
                let v10 = grid[i1][j];
                let v01 = grid[i][j1];
                let v11 = grid[i1][j1];
                for tri in [[v00, v10, v01], [v10, v11, v01]] {
                    tds.insert_simplex_with_mapping(
                        Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2], apex], None)
                            .unwrap(),
                    )
                    .unwrap();
                }
            }
        }

        tds
    }

    /// Set every vertex incident pointer to a dangling key so repair must rebuild them.
    fn poison_incident_simplices(tds: &mut Tds<(), (), 3>) {
        let dangling = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        let vertex_keys: Vec<_> = tds.vertex_keys().collect();
        for vertex_key in vertex_keys {
            tds.vertex_mut(vertex_key)
                .unwrap()
                .set_incident_simplex(Some(dangling));
        }
    }

    /// Assert that every present incident pointer references a containing simplex.
    fn assert_incident_simplices_are_coherent(tds: &Tds<(), (), 3>) {
        for (vertex_key, vertex) in tds.vertices() {
            let Some(simplex_key) = vertex.incident_simplex() else {
                continue;
            };
            let simplex = tds
                .simplex(simplex_key)
                .expect("incident simplex should exist");
            assert!(
                simplex.contains_vertex(vertex_key),
                "incident simplex {simplex_key:?} should contain vertex {vertex_key:?}"
            );
        }
    }

    #[test]
    fn targeted_topology_violation_parses_stage_and_preserves_source() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(0xCAFE));
        let cases = [
            (
                ManifoldError::BoundaryRidgeMultiplicity {
                    ridge_key: 0x100,
                    boundary_facet_count: 1,
                },
                PlManifoldRepairStage::BoundaryRidgeMultiplicity,
            ),
            (
                ManifoldError::RidgeLinkNotManifold {
                    ridge_key: 0x200,
                    link_vertex_count: 3,
                    link_edge_count: 1,
                    max_degree: 1,
                    degree_one_vertices: 2,
                    connected: false,
                },
                PlManifoldRepairStage::RidgeLink,
            ),
            (
                ManifoldError::VertexLinkNotManifold {
                    vertex_key,
                    link_vertex_count: 4,
                    link_simplex_count: 2,
                    boundary_facet_count: 3,
                    max_degree: 1,
                    connected: false,
                    interior_vertex: true,
                },
                PlManifoldRepairStage::VertexLink,
            ),
        ];

        for (source, expected_stage) in cases {
            let violation = TargetedTopologyViolation::try_from(source.clone()).unwrap();

            assert_eq!(violation.stage(), expected_stage);
            assert!(manifold_error_matches_repair_stage(&source, expected_stage));
            for mismatched_stage in [
                PlManifoldRepairStage::BoundaryRidgeMultiplicity,
                PlManifoldRepairStage::RidgeLink,
                PlManifoldRepairStage::VertexLink,
            ] {
                if mismatched_stage != expected_stage {
                    assert!(!manifold_error_matches_repair_stage(
                        &source,
                        mismatched_stage
                    ));
                }
            }
            assert_eq!(violation.source(), &source);
            assert_eq!(violation.into_source(), source);
        }

        let non_targeted = ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 0x300,
            simplex_count: 3,
        };
        assert_eq!(
            TargetedTopologyViolation::try_from(non_targeted.clone()).unwrap_err(),
            non_targeted
        );
        assert!(!manifold_error_matches_repair_stage(
            &non_targeted,
            PlManifoldRepairStage::BoundaryRidgeMultiplicity
        ));
    }

    #[test]
    fn targeted_stage_violation_preserves_non_targeted_validation_error() {
        let tds = make_overshared_tds();

        let result = targeted_stage_violation(
            &tds,
            GlobalTopology::Euclidean,
            PlManifoldRepairStage::BoundaryRidgeMultiplicity,
        );

        assert_matches!(
            result,
            Err(PlManifoldRepairError::TargetedValidation {
                stage: PlManifoldRepairStage::BoundaryRidgeMultiplicity,
                source,
            }) if matches!(*source, ManifoldError::ManifoldFacetMultiplicity { .. })
        );
    }

    #[test]
    fn boundary_ridge_candidates_include_incident_simplices_across_interior_facet() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]).unwrap())
            .unwrap();
        let first = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let second = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();
        let ridge_key = facet_key_from_vertices(&[v0, v1]);

        let candidates =
            boundary_ridge_candidate_simplices(&tds, GlobalTopology::Euclidean, ridge_key).unwrap();

        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&first));
        assert!(candidates.contains(&second));
    }

    #[test]
    fn simplex_facet_vertices_rejects_out_of_bounds_index() {
        let tds = make_boundary_ridge_multiplicity_tds();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let handle = FacetHandle::from_validated(simplex_key, u8::MAX);
        let mut facet_vertices = VertexKeyBuffer::new();

        let result = simplex_facet_vertices(&tds, handle, &mut facet_vertices);

        assert_matches!(
            result,
            Err(ManifoldError::Tds(TdsError::IndexOutOfBounds {
                index,
                bound: 4,
                ..
            })) if index == <usize as From<u8>>::from(u8::MAX)
        );
        assert!(facet_vertices.is_empty());
    }

    #[test]
    fn remove_targeted_simplex_reports_zero_for_missing_key_without_mutating_stats() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let mut stats = PlManifoldRepairStats::default();
        let missing_simplex_key = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let removed = remove_targeted_simplex(&mut tds, &mut stats, missing_simplex_key).unwrap();

        assert_eq!(removed, 0);
        assert_eq!(stats.simplices_removed, 0);
        assert!(stats.removed_simplices.is_empty());
        assert!(stats.removed_vertices.is_empty());
    }

    /// Assert that a targeted violation fails before removal when simplex budget is zero.
    fn assert_targeted_simplex_budget_exhausted<const D: usize>(
        mut tds: Tds<(), (), D>,
        expected_stage: PlManifoldRepairStage,
    ) {
        let config = PlManifoldRepairConfig {
            max_iterations: 64,
            max_simplices_removed: 0,
        };

        let result = repair_pl_manifold_topology(&mut tds, GlobalTopology::Euclidean, &config);

        let Err(PlManifoldRepairError::TargetedBudgetExhausted {
            stage,
            source,
            iterations,
            simplices_removed,
            max_iterations,
            max_simplices_removed,
        }) = result
        else {
            panic!("expected TargetedBudgetExhausted for {expected_stage:?}, got {result:?}");
        };
        assert_eq!(stage, expected_stage);
        assert_eq!(iterations, 0);
        assert_eq!(simplices_removed, 0);
        assert_eq!(max_iterations, 64);
        assert_eq!(max_simplices_removed, 0);
        match stage {
            PlManifoldRepairStage::BoundaryRidgeMultiplicity => {
                assert_matches!(*source, ManifoldError::BoundaryRidgeMultiplicity { .. });
            }
            PlManifoldRepairStage::RidgeLink => {
                assert_matches!(*source, ManifoldError::RidgeLinkNotManifold { .. });
            }
            PlManifoldRepairStage::VertexLink => {
                assert_matches!(*source, ManifoldError::VertexLinkNotManifold { .. });
            }
        }
    }

    #[test]
    fn targeted_postcondition_validation_preserves_typed_source() {
        let source = ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 0x1234,
            simplex_count: 3,
        };
        let err = PlManifoldRepairError::TargetedPostconditionValidation {
            source: Box::new(source.clone()),
        };
        let message = err.to_string();

        let PlManifoldRepairError::TargetedPostconditionValidation { source: err_source } = err
        else {
            panic!("expected TargetedPostconditionValidation");
        };
        assert_eq!(*err_source, source);
        assert!(message.contains("postcondition validation failed"));
    }

    #[test]
    fn targeted_stage_display_uses_user_facing_labels() {
        assert_eq!(
            PlManifoldRepairStage::BoundaryRidgeMultiplicity.to_string(),
            "boundary-ridge multiplicity"
        );
        assert_eq!(PlManifoldRepairStage::RidgeLink.to_string(), "ridge-link");
        assert_eq!(PlManifoldRepairStage::VertexLink.to_string(), "vertex-link");

        let err = PlManifoldRepairError::TargetedNoProgress {
            stage: PlManifoldRepairStage::BoundaryRidgeMultiplicity,
            source: Box::new(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key: 0x1234,
                boundary_facet_count: 1,
            }),
            iterations: 2,
            simplices_removed: 1,
        };
        assert!(
            err.to_string()
                .contains("boundary-ridge multiplicity repair")
        );
    }

    #[test]
    fn test_repair_pl_manifold_topology_repairs_boundary_ridge_multiplicity() {
        init_tracing();
        let mut tds = make_boundary_ridge_multiplicity_tds();
        assert_matches!(
            validate_boundary_ridge_multiplicity(&tds, GlobalTopology::Euclidean),
            Err(ManifoldError::BoundaryRidgeMultiplicity { .. })
        );

        let stats = repair_pl_manifold_topology(
            &mut tds,
            GlobalTopology::Euclidean,
            &PlManifoldRepairConfig::default(),
        )
        .unwrap();

        assert!(stats.succeeded);
        assert!(stats.simplices_removed > 0);
        validate_targeted_topology(&tds, GlobalTopology::Euclidean).unwrap();
    }

    #[test]
    fn test_repair_pl_manifold_topology_repairs_ridge_link() {
        init_tracing();
        let mut tds = make_disconnected_ridge_link_tds();
        assert_matches!(
            validate_ridge_links(&tds),
            Err(ManifoldError::RidgeLinkNotManifold { .. })
        );

        let stats = repair_pl_manifold_topology(
            &mut tds,
            GlobalTopology::Euclidean,
            &PlManifoldRepairConfig::default(),
        )
        .unwrap();

        assert!(stats.succeeded);
        assert!(stats.simplices_removed > 0);
        validate_targeted_topology(&tds, GlobalTopology::Euclidean).unwrap();
    }

    #[test]
    fn test_repair_pl_manifold_topology_repairs_vertex_link() {
        init_tracing();
        let mut tds = make_cone_on_torus_tds();
        assert_matches!(
            validate_vertex_link_manifoldness(&tds, GlobalTopology::Euclidean),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );

        let stats = repair_pl_manifold_topology(
            &mut tds,
            GlobalTopology::Euclidean,
            &PlManifoldRepairConfig::default(),
        )
        .unwrap();

        assert!(stats.succeeded);
        assert!(stats.simplices_removed > 0);
        validate_targeted_topology(&tds, GlobalTopology::Euclidean).unwrap();
    }

    #[test]
    fn test_repair_pl_manifold_topology_targeted_simplex_budget_exhausted_by_stage() {
        init_tracing();
        assert_targeted_simplex_budget_exhausted(
            make_boundary_ridge_multiplicity_tds(),
            PlManifoldRepairStage::BoundaryRidgeMultiplicity,
        );
        assert_targeted_simplex_budget_exhausted(
            make_disconnected_ridge_link_tds(),
            PlManifoldRepairStage::RidgeLink,
        );
        assert_targeted_simplex_budget_exhausted(
            make_cone_on_torus_tds(),
            PlManifoldRepairStage::VertexLink,
        );
    }

    #[test]
    fn test_repair_pl_manifold_topology_targeted_iteration_budget_exhausted() {
        init_tracing();
        let mut tds = make_boundary_ridge_multiplicity_tds();
        let config = PlManifoldRepairConfig {
            max_iterations: 0,
            max_simplices_removed: 10_000,
        };

        let result = repair_pl_manifold_topology(&mut tds, GlobalTopology::Euclidean, &config);

        let Err(PlManifoldRepairError::TargetedBudgetExhausted {
            stage,
            iterations,
            simplices_removed,
            max_iterations,
            max_simplices_removed,
            ..
        }) = result
        else {
            panic!("expected TargetedBudgetExhausted from iteration limit, got {result:?}");
        };
        assert_eq!(stage, PlManifoldRepairStage::BoundaryRidgeMultiplicity);
        assert_eq!(iterations, 0);
        assert_eq!(simplices_removed, 0);
        assert_eq!(max_iterations, 0);
        assert_eq!(max_simplices_removed, 10_000);
    }

    #[test]
    fn test_repair_pl_manifold_topology_targeted_error_rebuilds_incident_simplices() {
        init_tracing();
        let mut tds = make_cone_on_torus_tds();
        poison_incident_simplices(&mut tds);
        let config = PlManifoldRepairConfig {
            max_iterations: 1,
            max_simplices_removed: 10_000,
        };

        let result = repair_pl_manifold_topology(&mut tds, GlobalTopology::Euclidean, &config);

        let Err(PlManifoldRepairError::TargetedBudgetExhausted {
            stage,
            simplices_removed,
            ..
        }) = result
        else {
            panic!(
                "expected TargetedBudgetExhausted after partial targeted repair, got {result:?}"
            );
        };
        assert_eq!(stage, PlManifoldRepairStage::VertexLink);
        assert!(
            simplices_removed > 0,
            "test should exercise an error return after targeted simplex removal"
        );
        assert_incident_simplices_are_coherent(&tds);
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
        assert!(ValidatedFacetDegreeMap::try_from_facet_map(&facet_map).is_ok());
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

    #[test]
    fn test_repair_budget_exhausted_after_mutation_rebuilds_incident_simplices() {
        init_tracing();
        let mut tds = make_multi_duplicate_overshared_tds();
        poison_incident_simplices(&mut tds);

        let config = PlManifoldRepairConfig {
            max_iterations: 1,
            max_simplices_removed: 10_000,
        };
        let result = repair_facet_oversharing(&mut tds, &config);

        let Err(PlManifoldRepairError::BudgetExhausted {
            simplices_removed, ..
        }) = result
        else {
            panic!("Expected BudgetExhausted after partial repair, got: {result:?}");
        };
        assert!(
            simplices_removed > 0,
            "test should exercise an error return after simplex removal"
        );
        assert_incident_simplices_are_coherent(&tds);
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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tds = dt.tds();
        let simplex_key = tds.simplex_keys().next().unwrap();

        let score1 = simplex_quality_score(tds, simplex_key);
        let score2 = simplex_quality_score(tds, simplex_key);

        assert!(score1.is_finite(), "Score should be finite, got {score1}");
        assert!(score1 > 0.0, "Score should be positive, got {score1}");
        approx::assert_relative_eq!(score1, score2, epsilon = 0.0);
        let invalid_score =
            simplex_quality_score(tds, SimplexKey::from(KeyData::from_ffi(u64::MAX)));
        assert_eq!(invalid_score.to_bits(), f64::MAX.to_bits());
    }

    #[test]
    fn simplex_quality_score_ranks_degenerate_geometry_as_worst() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vertex_keys: Vec<_> = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
        ]
        .iter()
        .map(|vertex| tds.insert_vertex_with_mapping(*vertex).unwrap())
        .collect();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new(vertex_keys).unwrap())
            .unwrap();

        let score = simplex_quality_score(&tds, simplex_key);

        assert_eq!(score.to_bits(), f64::MAX.to_bits());
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
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        let config = PlManifoldRepairConfig::default();

        let mut tds1 = dt.tds().clone();
        let stats1 = repair_facet_oversharing(&mut tds1, &config).unwrap();

        let mut tds2 = dt.tds().clone();
        let stats2 = repair_facet_oversharing(&mut tds2, &config).unwrap();

        assert_eq!(stats1, stats2);
    }
}
