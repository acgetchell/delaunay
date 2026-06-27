//! Local topology repair for generic triangulations.
//!
//! This module owns local facet issue detection/repair, stale incident-simplex
//! repair, and vertex-removal cavity retriangulation for [`Triangulation`](crate::prelude::triangulation::Triangulation).

#![forbid(unsafe_code)]

use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, CavityRepairStage, InsertionError, InsertionTopologyValidationContext,
    external_facets_for_boundary, fill_cavity_replacing_simplices, repair_neighbor_pointers,
    repair_neighbor_pointers_local, wire_cavity_neighbors,
};
use crate::core::algorithms::locate::extract_cavity_boundary;
use crate::core::collections::{
    CLEANUP_OPERATION_BUFFER_SIZE, FacetIssuesMap, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE,
    SimplexKeyBuffer, SimplexKeySet, SmallBuffer, VertexKeySet, fast_hash_set_with_capacity,
};
use crate::core::facet::FacetHandle;
use crate::core::rollback::TriangulationRollbackTransaction;
use crate::core::tds::{InvariantError, SimplexKey, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::validation::{TriangulationValidationError, insertion_error_to_invariant_error};
use crate::geometry::kernel::Kernel;
use crate::geometry::quality::{QualityError, QualitySimplexVerticesError, radius_ratio};
use std::env;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use uuid::Uuid;

static FORCE_GLOBAL_NEIGHBOR_REBUILD_ENABLED: OnceLock<bool> = OnceLock::new();

const ORIENTATION_NORMALIZATION_BUDGET: usize = 3;

type IncidentSimplexRepairUpdates =
    SmallBuffer<IncidentSimplexRepairUpdate, CLEANUP_OPERATION_BUFFER_SIZE>;

/// Stack-first vertex frontier for incident-simplex hint repair.
///
/// Insertion and local repair usually touch only a handful of vertices. Keeping
/// this scope as a small buffer preserves the explicit affected-vertex contract
/// without adding hash-table allocation to every insertion.
pub(crate) type IncidentRepairVertexBuffer = SmallBuffer<VertexKey, CLEANUP_OPERATION_BUFFER_SIZE>;

/// Deferred vertex-hint update computed from the canonical incidence index.
#[derive(Clone, Debug)]
struct IncidentSimplexRepairUpdate {
    vertex_key: VertexKey,
    vertex_uuid: Uuid,
    incident_simplex: Option<SimplexKey>,
}

/// Adds a vertex key to an incident-repair frontier without heap-backed hashing.
fn push_incident_repair_vertex_once(
    affected_vertices: &mut IncidentRepairVertexBuffer,
    vertex_key: VertexKey,
) {
    if !affected_vertices.contains(&vertex_key) {
        affected_vertices.push(vertex_key);
    }
}

/// Returns whether local neighbor repair should be bypassed for regression isolation.
fn force_global_neighbor_rebuild_enabled() -> bool {
    *FORCE_GLOBAL_NEIGHBOR_REBUILD_ENABLED
        .get_or_init(|| env::var_os("DELAUNAY_FORCE_GLOBAL_NEIGHBOR_REBUILD").is_some())
}

/// Preserve typed TDS lookup/count failures from quality evaluation where possible.
fn quality_error_to_tds_error(simplex_key: SimplexKey, error: QualityError) -> TdsError {
    match error {
        QualityError::SimplexVertices { source, .. } => match source {
            QualitySimplexVerticesError::SimplexNotFound {
                simplex_key,
                context,
            } => TdsError::SimplexNotFound {
                simplex_key,
                context,
            },
            QualitySimplexVerticesError::ReferencedVertexNotFound {
                vertex_key,
                context,
            } => TdsError::VertexNotFound {
                vertex_key,
                context,
            },
            QualitySimplexVerticesError::UnexpectedTdsFailure { source } => *source,
        },
        QualityError::VertexNotFound { vertex_key } => TdsError::VertexNotFound {
            vertex_key,
            context: format!("quality evaluation for simplex {simplex_key:?}"),
        },
        QualityError::InvalidSimplexArity {
            actual,
            expected,
            dimension,
        } => TdsError::DimensionMismatch {
            expected,
            actual,
            context: format!("quality evaluation for {dimension}D simplex {simplex_key:?}"),
        },
        other => TdsError::InconsistentDataStructure {
            message: format!("Quality evaluation failed for simplex {simplex_key:?}: {other}"),
        },
    }
}

/// Internal result from over-shared-facet repair, including the surviving frontier
/// that should seed local neighbor-pointer repair.
#[derive(Debug)]
pub(crate) struct LocalFacetRepairOutcome {
    /// Number of simplices actually removed from the TDS.
    pub(crate) removed_count: usize,
    /// Simplices selected for removal before they were deleted.
    #[cfg_attr(
        not(debug_assertions),
        expect(
            dead_code,
            reason = "Removed-simplex keys are retained for debug logging and future local repair diagnostics"
        )
    )]
    pub(crate) removed_simplices: SimplexKeyBuffer,
    /// Surviving one-hop neighbors whose back-references may have been cleared.
    pub(crate) frontier_simplices: SimplexKeyBuffer,
    /// Vertices touched by simplices removed during local repair.
    pub(crate) affected_vertices: IncidentRepairVertexBuffer,
}

/// Internal result from vertex deletion, including local simplices that should
/// seed downstream Delaunay repair.
#[derive(Debug)]
pub(crate) struct VertexRemovalOutcome {
    /// Number of simplices actually removed from the TDS.
    pub(crate) simplices_removed: usize,
    /// Live local simplices whose facets, ridges, or neighbors changed.
    pub(crate) repair_seed_simplices: SimplexKeyBuffer,
}

impl VertexRemovalOutcome {
    fn without_repair_seeds(simplices_removed: usize) -> Self {
        Self {
            simplices_removed,
            repair_seed_simplices: SimplexKeyBuffer::new(),
        }
    }

    const fn with_repair_seed_simplices(
        simplices_removed: usize,
        repair_seed_simplices: SimplexKeyBuffer,
    ) -> Self {
        Self {
            simplices_removed,
            repair_seed_simplices,
        }
    }
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
    /// Adds one vertex key to an incident repair scope.
    pub(crate) fn push_incident_repair_vertex(
        affected_vertices: &mut IncidentRepairVertexBuffer,
        vertex_key: VertexKey,
    ) {
        push_incident_repair_vertex_once(affected_vertices, vertex_key);
    }

    /// Extends a vertex repair scope with vertices from currently live simplices.
    pub(crate) fn extend_incident_repair_vertices_from_simplices(
        &self,
        simplex_keys: &[SimplexKey],
        affected_vertices: &mut IncidentRepairVertexBuffer,
    ) {
        for &simplex_key in simplex_keys {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue;
            };
            for &vertex_key in simplex.vertices() {
                push_incident_repair_vertex_once(affected_vertices, vertex_key);
            }
        }
    }

    /// Repair stale incident-simplex pointers and detect truly isolated affected vertices.
    ///
    /// After cavity filling and simplex removal, pre-existing boundary vertices may
    /// still reference deleted conflict-region simplices via a stale `incident_simplex`.
    /// For affected vertices with stale or missing hints, this uses the
    /// maintained vertex-to-simplices incidence index rather than scanning
    /// simplex storage. Returns an error only if an affected vertex is in zero
    /// simplices or the canonical incidence index is stale.
    ///
    /// # Errors
    ///
    /// Returns [`InsertionError::TopologyValidationFailed`] when an affected
    /// vertex has no live incident simplex. Returns
    /// [`InsertionError::TopologyValidation`] when the maintained incidence
    /// index names a missing simplex or a simplex that does not contain the
    /// affected vertex.
    pub(crate) fn repair_stale_incident_simplices(
        &mut self,
        affected_vertices: &IncidentRepairVertexBuffer,
    ) -> Result<(), InsertionError> {
        if affected_vertices.is_empty() {
            return Ok(());
        }

        let mut updates = IncidentSimplexRepairUpdates::new();
        for &vertex_key in affected_vertices {
            let Some(vertex) = self.tds.vertex(vertex_key) else {
                continue;
            };

            let needs_repair = !vertex.incident_simplex().is_some_and(|simplex_key| {
                self.tds
                    .simplex(simplex_key)
                    .is_some_and(|simplex| simplex.contains_vertex(vertex_key))
            });
            if !needs_repair {
                continue;
            }

            updates.push(IncidentSimplexRepairUpdate {
                vertex_key,
                vertex_uuid: vertex.uuid(),
                incident_simplex: self.first_indexed_incident_simplex(vertex_key)?,
            });
        }

        if updates.is_empty() {
            return Ok(());
        }

        #[cfg(debug_assertions)]
        if env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
            tracing::debug!(
                stale_count = updates.len(),
                "repairing stale incident-simplex pointers"
            );
        }

        for update in updates {
            if let Some(simplex_key) = update.incident_simplex {
                if let Some(vertex) = self.tds.vertex_mut(update.vertex_key) {
                    vertex.set_incident_simplex(Some(simplex_key));
                }
            } else {
                return Err(InsertionError::TopologyValidationFailed {
                    context: InsertionTopologyValidationContext::StaleIncidentSimplexRepair,
                    source: TriangulationValidationError::IsolatedVertex {
                        vertex_key: update.vertex_key,
                        vertex_uuid: update.vertex_uuid,
                    },
                });
            }
        }
        Ok(())
    }

    /// Reads and verifies the canonical incidence hint before storing it on a vertex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::SimplexNotFound`] when the incidence index names a
    /// removed simplex. Returns [`TdsError::InconsistentDataStructure`] when
    /// the indexed simplex does not actually contain `vertex_key`.
    fn first_indexed_incident_simplex(
        &self,
        vertex_key: VertexKey,
    ) -> Result<Option<SimplexKey>, TdsError> {
        let Some(simplex_key) = self.tds.first_simplex_containing_vertex(vertex_key) else {
            return Ok(None);
        };

        let simplex = self
            .tds
            .simplex(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: format!(
                    "repairing stale incident_simplex pointer for vertex {vertex_key:?}"
                ),
            })?;

        if !simplex.contains_vertex(vertex_key) {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "Vertex-to-simplices index lists simplex {simplex_key:?} for vertex {vertex_key:?}, but the simplex does not contain that vertex"
                ),
            });
        }

        Ok(Some(simplex_key))
    }
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Repair neighbor pointers after local simplex removal without scanning the full TDS.
    pub(crate) fn repair_neighbors_after_local_simplex_removal(
        &mut self,
        new_simplices: &SimplexKeyBuffer,
        frontier_simplices: &[SimplexKey],
    ) -> Result<usize, InsertionError> {
        #[cfg(debug_assertions)]
        tracing::debug!(
            simplices = self.tds.number_of_simplices(),
            surviving_new_simplex_seeds = new_simplices
                .iter()
                .filter(|&&simplex_key| self.tds.contains_simplex(simplex_key))
                .count(),
            frontier_simplex_seeds = frontier_simplices
                .iter()
                .filter(|&&simplex_key| self.tds.contains_simplex(simplex_key))
                .count(),
            "Before local neighbor-pointer repair"
        );

        if force_global_neighbor_rebuild_enabled() {
            #[cfg(debug_assertions)]
            tracing::debug!(
                "DELAUNAY_FORCE_GLOBAL_NEIGHBOR_REBUILD set; using global neighbor rebuild"
            );
            return repair_neighbor_pointers(&mut self.tds).map_err(|source| {
                CavityFillingError::NeighborRebuild {
                    reason: source.into(),
                }
                .into()
            });
        }

        #[cfg(debug_assertions)]
        {
            match repair_neighbor_pointers_local(
                &mut self.tds,
                new_simplices,
                Some(frontier_simplices),
            ) {
                Ok(repaired) => Ok(repaired),
                Err(local_error) => {
                    tracing::warn!(
                        error = %local_error,
                        "Local neighbor-pointer repair failed; falling back to global rebuild in debug mode"
                    );
                    repair_neighbor_pointers(&mut self.tds).map_err(|source| {
                        CavityFillingError::NeighborRebuild {
                            reason: source.into(),
                        }
                        .into()
                    })
                }
            }
        }

        #[cfg(not(debug_assertions))]
        {
            repair_neighbor_pointers_local(&mut self.tds, new_simplices, Some(frontier_simplices))
                .map_err(|source| {
                    CavityFillingError::NeighborRebuild {
                        reason: source.into(),
                    }
                    .into()
                })
        }
    }
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation maintains topological consistency by:
    /// 1. Finding all simplices containing the vertex
    /// 2. Removing those simplices (creating a cavity)
    /// 3. Extracting the cavity boundary facets
    /// 4. Filling the cavity with a fan triangulation (pick apex, connect to all boundary facets)
    /// 5. Wiring neighbors to maintain consistency
    /// 6. Removing the vertex itself
    ///
    /// **Fan Triangulation**: The cavity is filled by picking one boundary vertex as an apex
    /// and connecting it to all boundary facets. This follows the local cavity-retriangulation
    /// lineage used by Bowyer-Watson insertion and the computational-geometry treatment in
    /// Edelsbrunner and Preparata-Shamos. See `REFERENCES.md` sections "Triangulation
    /// Construction Algorithms", "Lifted Paraboloid Method", and "Robust Geometric
    /// Predicates" for vertex-deletion source context and Shewchuk's robust-predicate
    /// background.
    ///
    /// The vertex-deletion fan step is numerically and topologically fragile when the cavity is
    /// degenerate or nearly coplanar, when epsilon thresholds are too small for the active scalar
    /// range, or when candidate simplices are inverted. Mitigate those cases with robust
    /// predicates, explicit epsilon thresholds, bounded repair budgets, and transactional
    /// validation fallbacks.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - Key of the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of simplices removed along with live local simplices that
    /// should seed downstream Delaunay repair.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] if the removal cannot be completed while maintaining
    /// triangulation invariants. The error preserves structured information from whichever
    /// layer (TDS or Topology) detected the failure.
    pub(crate) fn remove_vertex_with_repair_seeds(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<VertexRemovalOutcome, InvariantError> {
        if self.tds.vertex(vertex_key).is_none() {
            // Vertex not found, nothing to remove.
            return Ok(VertexRemovalOutcome::without_repair_seeds(0));
        }

        let simplices_to_remove: SimplexKeyBuffer = self
            .tds
            .simplex_keys_containing_vertex(vertex_key)
            .collect();

        if simplices_to_remove.is_empty() {
            // Vertex exists but has no incident simplices; remove it only if the
            // resulting triangulation satisfies the same invariant checks.
            return self
                .remove_vertex_with_invariant_checks(vertex_key)
                .map(VertexRemovalOutcome::without_repair_seeds);
        }

        let boundary_facets =
            extract_cavity_boundary(&self.tds, &simplices_to_remove).map_err(|e| {
                TdsError::InconsistentDataStructure {
                    message: format!("Failed to extract cavity boundary: {e}"),
                }
            })?;

        if boundary_facets.is_empty() {
            // Use TDS removal for the empty-boundary case, then validate so
            // lower-dimensional remnants are rejected and rolled back.
            return self
                .remove_vertex_with_invariant_checks(vertex_key)
                .map(VertexRemovalOutcome::without_repair_seeds);
        }

        let affected_vertices =
            self.affected_vertices_for_vertex_removal(&simplices_to_remove, vertex_key)?;

        let apex_vertex_key = self.pick_fan_apex(&boundary_facets)?;

        self.remove_vertex_with_fan_retriangulation(
            vertex_key,
            &simplices_to_remove,
            &boundary_facets,
            &affected_vertices,
            apex_vertex_key,
        )
    }

    /// Removes a vertex through fan cavity filling and rolls back on postcondition failure.
    fn remove_vertex_with_fan_retriangulation(
        &mut self,
        vertex_key: VertexKey,
        simplices_to_remove: &SimplexKeyBuffer,
        boundary_facets: &[FacetHandle],
        affected_vertices: &VertexKeySet,
        apex_vertex_key: VertexKey,
    ) -> Result<VertexRemovalOutcome, InvariantError> {
        let mut transaction = TriangulationRollbackTransaction::begin(self);
        let retriangulation_result = {
            let tri = transaction.triangulation_mut();
            (|| -> Result<VertexRemovalOutcome, InvariantError> {
                // Fill cavity with fan triangulation BEFORE removing old simplices
                // Use fan triangulation that skips boundary facets which already include the apex
                let new_simplices = tri
                    .fan_fill_cavity(apex_vertex_key, boundary_facets)
                    .map_err(|e| {
                        insertion_error_to_invariant_error(e, "Fan triangulation failed")
                    })?;
                tri.canonicalize_positive_orientation_for_simplices(&new_simplices)
                    .map_err(|e| {
                        insertion_error_to_invariant_error(
                            e,
                            "Orientation canonicalization failed after fan filling",
                        )
                    })?;

                // Wire neighbors for the new simplices (while both old and new simplices exist)
                let external_facets =
                    external_facets_for_boundary(&tri.tds, simplices_to_remove, boundary_facets)
                        .map_err(|e| {
                            insertion_error_to_invariant_error(
                                e,
                                "External-facet collection failed",
                            )
                        })?;
                wire_cavity_neighbors(
                    &mut tri.tds,
                    &new_simplices,
                    external_facets.iter().copied(),
                    Some(simplices_to_remove),
                )
                .map_err(|e| insertion_error_to_invariant_error(e, "Neighbor wiring failed"))?;

                // Remove the simplices containing the vertex (now that new simplices are wired up)
                // Note: remove_simplices_by_keys() automatically clears neighbor pointers in surviving
                // simplices that reference removed simplices (sets them to None/boundary)
                let mut simplices_removed = tri
                    .tds
                    .remove_simplices_by_keys(simplices_to_remove)
                    .map_err(|e| InvariantError::Tds(e.into_inner()))?;
                let max_repair_simplices_removed = simplices_to_remove.len();
                let mut post_repair_frontier = SimplexKeyBuffer::new();

                tri.repair_vertex_removal_facet_issues(
                    &new_simplices,
                    &mut simplices_removed,
                    &mut post_repair_frontier,
                    max_repair_simplices_removed,
                )?;

                tri.tds
                    .remove_vertex(vertex_key)
                    .map_err(|e| InvariantError::Tds(e.into_inner()))?;

                let surviving_new_simplices = tri.live_simplices_from(&new_simplices);
                let validation_scope = tri.vertex_removal_validation_scope(
                    &new_simplices,
                    &external_facets,
                    &post_repair_frontier,
                );
                tri.normalize_vertex_removal_orientation(&validation_scope)?;
                tri.repair_affected_vertex_incidence_from_scope(
                    affected_vertices,
                    &validation_scope,
                )?;
                tri.validate_vertex_removal_postconditions(
                    vertex_key,
                    affected_vertices,
                    &surviving_new_simplices,
                    &validation_scope,
                )?;

                Ok(VertexRemovalOutcome::with_repair_seed_simplices(
                    simplices_removed,
                    validation_scope,
                ))
            })()
        };

        match retriangulation_result {
            Ok(outcome) => {
                transaction.commit();
                Ok(outcome)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }

    /// Repairs over-shared facets introduced by vertex-removal fan filling.
    fn repair_vertex_removal_facet_issues(
        &mut self,
        new_simplices: &SimplexKeyBuffer,
        simplices_removed: &mut usize,
        post_repair_frontier: &mut SimplexKeyBuffer,
        max_simplices_removed: usize,
    ) -> Result<(), InvariantError> {
        let mut remaining_budget = max_simplices_removed;

        loop {
            if self.tds.validate_facet_sharing().is_ok() {
                return Ok(());
            }

            let Some(issues) = self.detect_local_facet_issues(new_simplices)? else {
                return Ok(());
            };

            #[cfg(debug_assertions)]
            tracing::warn!(
                "Warning: {} over-shared facets detected after vertex removal, repairing...",
                issues.len()
            );

            let repair_outcome = self
                .repair_local_facet_issues_with_frontier(&issues, remaining_budget)
                .map_err(|e| {
                    insertion_error_to_invariant_error(
                        e,
                        "Local facet repair after vertex removal failed",
                    )
                })?;
            let removed = repair_outcome.removed_count;
            if removed == 0 {
                return Err(insertion_error_to_invariant_error(
                    CavityFillingError::InvalidFacetSharingAfterRepair {
                        stage: CavityRepairStage::FanTriangulation,
                    }
                    .into(),
                    "Local facet repair after vertex removal stalled",
                ));
            }

            remaining_budget = remaining_budget.saturating_sub(removed);
            post_repair_frontier.extend(repair_outcome.frontier_simplices.iter().copied());
            *simplices_removed += removed;

            #[cfg(debug_assertions)]
            tracing::debug!(
                "Repaired by removing {removed} additional simplices ({remaining_budget} budget remaining)"
            );

            self.repair_neighbors_after_local_simplex_removal(new_simplices, post_repair_frontier)
                .map_err(|e| {
                    insertion_error_to_invariant_error(
                        e,
                        "Neighbor repair after facet issue repair failed",
                    )
                })?;
        }
    }

    /// Normalizes coherence globally but promotes only the vertex-removal scope geometrically.
    fn normalize_vertex_removal_orientation(
        &mut self,
        validation_scope: &SimplexKeyBuffer,
    ) -> Result<(), InvariantError> {
        for _ in 0..ORIENTATION_NORMALIZATION_BUDGET {
            self.canonicalize_positive_orientation_for_simplices(validation_scope)
                .map_err(|e| {
                    insertion_error_to_invariant_error(
                        e,
                        "Local orientation promotion failed after vertex removal",
                    )
                })?;
            self.tds
                .normalize_coherent_orientation()
                .map_err(InvariantError::Tds)?;
            if self
                .validate_geometric_simplex_orientation_for_simplices(validation_scope)
                .is_ok()
            {
                return Ok(());
            }
        }

        if self
            .validate_geometric_simplex_orientation_for_simplices(validation_scope)
            .is_ok()
        {
            return Ok(());
        }

        self.normalize_and_promote_positive_orientation()
            .map_err(|e| {
                insertion_error_to_invariant_error(
                    e,
                    "Global orientation fallback failed after local vertex-removal normalization",
                )
            })
    }

    /// Repairs incident-simplex pointers for vertices touched by vertex removal.
    fn repair_affected_vertex_incidence_from_scope(
        &mut self,
        affected_vertices: &VertexKeySet,
        validation_scope: &SimplexKeyBuffer,
    ) -> Result<(), InvariantError> {
        for &vertex_key in affected_vertices {
            let needs_repair = self.tds.vertex(vertex_key).is_some_and(|vertex| {
                !vertex.incident_simplex().is_some_and(|simplex_key| {
                    self.tds
                        .simplex(simplex_key)
                        .is_some_and(|simplex| simplex.contains_vertex(vertex_key))
                })
            });
            if !needs_repair {
                continue;
            }

            let incident_simplex = validation_scope.iter().copied().find(|&simplex_key| {
                self.tds
                    .simplex(simplex_key)
                    .is_some_and(|simplex| simplex.contains_vertex(vertex_key))
            });

            let Some(simplex_key) = incident_simplex else {
                let Some(vertex) = self.tds.vertex(vertex_key) else {
                    continue;
                };
                return Err(InvariantError::Triangulation(
                    TriangulationValidationError::IsolatedVertex {
                        vertex_key,
                        vertex_uuid: vertex.uuid(),
                    },
                ));
            };

            if let Some(vertex) = self.tds.vertex_mut(vertex_key) {
                vertex.set_incident_simplex(Some(simplex_key));
            }
        }

        Ok(())
    }

    /// Removes a vertex via direct TDS mutation and rolls back unless all triangulation
    /// invariants still hold.
    ///
    /// This handles fallback paths that do not retriangulate a cavity, such as isolated vertices
    /// or empty-boundary removals. Those paths can otherwise leave lower-dimensional remnants that
    /// are structurally valid at the TDS layer but invalid as a triangulation.
    fn remove_vertex_with_invariant_checks(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<usize, InvariantError> {
        let mut transaction = TriangulationRollbackTransaction::begin(self);
        let result = {
            let tri = transaction.triangulation_mut();
            (|| -> Result<usize, InvariantError> {
                let simplices_removed = tri
                    .tds
                    .remove_vertex(vertex_key)
                    .map_err(|e| InvariantError::Tds(e.into_inner()))?;
                tri.tds.is_valid().map_err(InvariantError::Tds)?;
                tri.is_valid_topology()?;
                Ok(simplices_removed)
            })()
        };

        match result {
            Ok(simplices_removed) => {
                transaction.commit();
                Ok(simplices_removed)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }

    /// Collects boundary-star vertices whose incident simplices may change during removal.
    fn affected_vertices_for_vertex_removal(
        &self,
        simplices_to_remove: &[SimplexKey],
        removed_vertex: VertexKey,
    ) -> Result<VertexKeySet, InvariantError> {
        let mut affected_vertices =
            fast_hash_set_with_capacity(simplices_to_remove.len().saturating_mul(D));
        for &simplex_key in simplices_to_remove {
            let simplex = self.tds.simplex(simplex_key).ok_or_else(|| {
                InvariantError::Tds(TdsError::SimplexNotFound {
                    simplex_key,
                    context: "collecting affected vertices for vertex removal".to_string(),
                })
            })?;
            for &vertex_key in simplex.vertices() {
                if vertex_key != removed_vertex {
                    affected_vertices.insert(vertex_key);
                }
            }
        }
        Ok(affected_vertices)
    }

    /// Returns live simplex keys from a mutation-produced buffer without duplicates.
    fn live_simplices_from(&self, simplices: &SimplexKeyBuffer) -> SimplexKeyBuffer {
        let mut live_simplices = SimplexKeyBuffer::new();
        let mut seen = SimplexKeySet::default();
        seen.reserve(simplices.len());
        for &simplex_key in simplices {
            self.push_live_simplex_once(&mut live_simplices, &mut seen, simplex_key);
        }
        live_simplices
    }

    /// Builds the local simplex scope whose facets, ridges, neighbors, and orientation changed.
    fn vertex_removal_validation_scope(
        &self,
        new_simplices: &SimplexKeyBuffer,
        external_facets: &[FacetHandle],
        post_repair_frontier: &SimplexKeyBuffer,
    ) -> SimplexKeyBuffer {
        let mut scope = SimplexKeyBuffer::new();
        let mut seen = SimplexKeySet::default();
        seen.reserve(new_simplices.len() + external_facets.len() + post_repair_frontier.len());

        for &simplex_key in new_simplices {
            self.push_live_simplex_once(&mut scope, &mut seen, simplex_key);
        }
        for facet in external_facets {
            self.push_live_simplex_once(&mut scope, &mut seen, facet.simplex_key());
        }
        for &simplex_key in post_repair_frontier {
            self.push_live_simplex_once(&mut scope, &mut seen, simplex_key);
        }

        scope
    }

    /// Pushes a simplex key into a local validation scope only when it still exists.
    fn push_live_simplex_once(
        &self,
        scope: &mut SimplexKeyBuffer,
        seen: &mut SimplexKeySet,
        simplex_key: SimplexKey,
    ) {
        if self.tds.contains_simplex(simplex_key) && seen.insert(simplex_key) {
            scope.push(simplex_key);
        }
    }

    /// Validates the local postconditions that make successful vertex removal topology-preserving.
    fn validate_vertex_removal_postconditions(
        &self,
        removed_vertex: VertexKey,
        affected_vertices: &VertexKeySet,
        surviving_new_simplices: &SimplexKeyBuffer,
        validation_scope: &SimplexKeyBuffer,
    ) -> Result<(), InvariantError> {
        if self.tds.contains_vertex_key(removed_vertex) {
            return Err(InvariantError::Tds(TdsError::InconsistentDataStructure {
                message: format!(
                    "Removed vertex {removed_vertex:?} still exists after vertex-removal finalization"
                ),
            }));
        }

        if self.tds.number_of_simplices() == 0
            || surviving_new_simplices.is_empty()
            || validation_scope.is_empty()
        {
            self.tds.is_valid().map_err(InvariantError::Tds)?;
            self.is_valid_topology()?;
            return Ok(());
        }

        self.validate_connectedness(surviving_new_simplices)
            .map_err(|e| {
                insertion_error_to_invariant_error(
                    e,
                    "Vertex-removal connectedness validation failed",
                )
            })?;
        self.validate_required_topology_links_for_simplices(validation_scope)?;
        self.validate_affected_vertices_non_isolated(affected_vertices)?;

        #[cfg(debug_assertions)]
        {
            self.tds.is_valid().map_err(InvariantError::Tds)?;
            self.is_valid_topology()?;
        }

        Ok(())
    }

    /// Ensures every vertex in the removed star still has a live incident simplex.
    fn validate_affected_vertices_non_isolated(
        &self,
        affected_vertices: &VertexKeySet,
    ) -> Result<(), InvariantError> {
        for &vertex_key in affected_vertices {
            let Some(vertex) = self.tds.vertex(vertex_key) else {
                continue;
            };
            if vertex.incident_simplex().is_some_and(|simplex_key| {
                self.tds
                    .simplex(simplex_key)
                    .is_some_and(|simplex| simplex.contains_vertex(vertex_key))
            }) {
                continue;
            }

            return Err(InvariantError::Triangulation(
                TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    vertex_uuid: vertex.uuid(),
                },
            ));
        }
        Ok(())
    }

    /// Pick an apex vertex for fan triangulation.
    ///
    /// Selects the first vertex from the first boundary facet as the apex.
    /// The fan will connect this apex to all boundary facets.
    ///
    /// # Arguments
    ///
    /// * `boundary_facets` - The cavity boundary facets
    ///
    /// # Returns
    ///
    /// The vertex key to use as apex.
    ///
    /// # Errors
    ///
    /// Returns a typed [`TdsError`] if the boundary is empty, references a missing
    /// simplex, or carries an out-of-range facet index.
    fn pick_fan_apex(&self, boundary_facets: &[FacetHandle]) -> Result<VertexKey, TdsError> {
        // Get first boundary facet
        let first_facet = boundary_facets
            .first()
            .ok_or_else(|| TdsError::DimensionMismatch {
                expected: 1,
                actual: 0,
                context: "fan apex selection requires at least one cavity-boundary facet"
                    .to_string(),
            })?;
        let simplex = self.tds.simplex(first_facet.simplex_key()).ok_or_else(|| {
            TdsError::SimplexNotFound {
                simplex_key: first_facet.simplex_key(),
                context: "fan apex selection".to_string(),
            }
        })?;

        // Get the first vertex from this facet (any vertex that's not the opposite one)
        let facet_idx = <usize as From<_>>::from(first_facet.facet_index());
        if facet_idx >= simplex.number_of_vertices() {
            return Err(TdsError::IndexOutOfBounds {
                index: facet_idx,
                bound: simplex.number_of_vertices(),
                context: format!(
                    "fan apex selection for boundary simplex {:?}",
                    first_facet.simplex_key()
                ),
            });
        }
        simplex
            .vertices()
            .iter()
            .enumerate()
            .find(|(i, _)| *i != facet_idx)
            .map(|(_, &vkey)| vkey)
            .ok_or_else(|| TdsError::DimensionMismatch {
                expected: 2,
                actual: simplex.number_of_vertices(),
                context: format!(
                    "fan apex selection for boundary simplex {:?}",
                    first_facet.simplex_key()
                ),
            })
    }

    /// Fan-specific cavity fill: connect an existing apex vertex to boundary facets
    /// that do not already include the apex. This avoids creating degenerate simplices
    /// with duplicate vertices when the apex lies on a boundary facet.
    fn fan_fill_cavity(
        &mut self,
        apex_vertex_key: VertexKey,
        boundary_facets: &[FacetHandle],
    ) -> Result<SimplexKeyBuffer, InsertionError> {
        let fan_boundary_facets =
            self.fan_boundary_facets_excluding_apex(apex_vertex_key, boundary_facets)?;
        fill_cavity_replacing_simplices(&mut self.tds, apex_vertex_key, &fan_boundary_facets)
    }

    /// Filters out boundary facets that already contain the fan apex.
    fn fan_boundary_facets_excluding_apex(
        &self,
        apex_vertex_key: VertexKey,
        boundary_facets: &[FacetHandle],
    ) -> Result<SmallBuffer<FacetHandle, 64>, InsertionError> {
        let mut fan_boundary_facets = SmallBuffer::new();

        for facet_handle in boundary_facets {
            let boundary_simplex =
                self.tds
                    .simplex(facet_handle.simplex_key())
                    .ok_or_else(|| CavityFillingError::MissingBoundarySimplex {
                        simplex_key: facet_handle.simplex_key(),
                    })?;

            let facet_idx = <usize as From<_>>::from(facet_handle.facet_index());
            if facet_idx >= boundary_simplex.number_of_vertices() {
                return Err(CavityFillingError::InvalidFacetIndex {
                    simplex_key: facet_handle.simplex_key(),
                    facet_index: facet_idx,
                    vertex_count: boundary_simplex.number_of_vertices(),
                }
                .into());
            }

            let facet_contains_apex = boundary_simplex
                .vertices()
                .iter()
                .enumerate()
                .any(|(i, &vkey)| i != facet_idx && vkey == apex_vertex_key);
            if !facet_contains_apex {
                fan_boundary_facets.push(*facet_handle);
            }
        }

        if fan_boundary_facets.is_empty() {
            return Err(CavityFillingError::EmptyFanTriangulation.into());
        }

        Ok(fan_boundary_facets)
    }

    /// Detects over-shared facets
    ///
    /// This is an **O(k * D)** operation where k = number of simplices to check,
    /// unlike global validation which is O(N * D) for the entire triangulation.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(k * D) where k = `simplices.len()`, D = dimension
    /// - **Use case**: Detect issues in newly created simplices after insertion/removal
    /// - **Comparison**: Global detection is O(N * D) where N = total simplices
    ///
    /// # Arguments
    ///
    /// * `simplices` - Keys of simplices to check (typically newly created simplices)
    ///
    /// # Returns
    ///
    /// `Ok(None)` if all facets are valid (≤2 simplices per facet).
    /// `Ok(Some(issues))` if over-shared facets are detected, where issues is a map
    /// from facet hash to the simplices sharing that facet.
    ///
    /// # Errors
    ///
    /// Returns error if simplices cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // A single simplex has no over-shared facets.
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let simplex_keys: Vec<_> = dt.simplices().map(|(ck, _)| ck).collect();
    /// let issues = dt
    ///     .as_triangulation()
    ///     .detect_local_facet_issues(&simplex_keys)
    ///     ?;
    /// assert!(issues.is_none());
    ///
    /// // Note: This method is most useful for checking newly created simplices
    /// // after insertion/removal operations.
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect_local_facet_issues(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<Option<FacetIssuesMap>, TdsError> {
        // Build facet map for ONLY the specified simplices
        // This is O(k * D) instead of O(N * D)
        let mut facet_to_simplices = FacetIssuesMap::default();

        // Index facets from the specified simplices
        for &simplex_key in simplices {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue; // Simplex was removed, skip
            };

            // For each facet of this simplex
            for facet_idx in 0..simplex.number_of_vertices() {
                // Compute facet hash from sorted vertex keys
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vkey) in simplex.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vkey);
                    }
                }
                facet_vkeys.sort_unstable();

                // Hash the facet
                let mut hasher = FastHasher::default();
                for &vkey in &facet_vkeys {
                    vkey.hash(&mut hasher);
                }
                let facet_hash = hasher.finish();

                // Track this simplex/facet pair
                let facet_idx_u8 =
                    u8::try_from(facet_idx).map_err(|_| TdsError::IndexOutOfBounds {
                        index: facet_idx,
                        bound: u8::MAX as usize + 1,
                        context: "facet index exceeds u8 range (dimension too high)".to_string(),
                    })?;
                facet_to_simplices
                    .entry(facet_hash)
                    .or_insert_with(SmallBuffer::new)
                    .push((simplex_key, facet_idx_u8));
            }
        }

        // Filter to only over-shared facets (> 2 simplices) in a single pass
        facet_to_simplices.retain(|_, simplex_facet_pairs| simplex_facet_pairs.len() > 2);

        if facet_to_simplices.is_empty() {
            Ok(None)
        } else {
            Ok(Some(facet_to_simplices))
        }
    }

    /// Select simplices to remove for over-shared-facet repair without mutating the TDS.
    fn simplices_for_local_facet_issue_repair(
        &self,
        issues: &FacetIssuesMap,
    ) -> Result<SimplexKeyBuffer, TdsError> {
        let mut simplices_to_remove = SimplexKeySet::default();

        // For each over-shared facet, select simplices to remove
        for simplex_facet_pairs in issues.values() {
            // Compute quality for each simplex - propagate errors from quality evaluation
            let mut simplex_qualities: Vec<(SimplexKey, f64, Uuid)> = Vec::new();
            for &(simplex_key, _) in simplex_facet_pairs {
                let simplex =
                    self.tds
                        .simplex(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "facet repair quality evaluation".to_string(),
                        })?;
                let uuid = simplex.uuid();

                // Propagate quality evaluation errors
                let ratio = radius_ratio(self, simplex_key)
                    .map_err(|error| quality_error_to_tds_error(simplex_key, error))?;
                if ratio.is_finite() {
                    simplex_qualities.push((simplex_key, ratio, uuid));
                } else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Non-finite quality ratio {ratio} for simplex {simplex_key:?}"
                        ),
                    });
                }
            }

            // Quality-based selection: keep 2 best, remove rest
            // Note: simplex_qualities always has all involved_simplices at this point since
            // any quality computation failure results in an early error return above
            simplex_qualities
                .sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

            // Mark simplices beyond the top 2 for removal
            for (simplex_key, _, _) in simplex_qualities.iter().skip(2) {
                if self.tds.contains_simplex(*simplex_key) {
                    simplices_to_remove.insert(*simplex_key);
                }
            }
        }

        Ok(simplices_to_remove.into_iter().collect())
    }

    /// Collect surviving simplices that need local neighbor repair after simplex removal.
    fn collect_local_repair_frontier(
        &self,
        issues: &FacetIssuesMap,
        simplices_to_remove: &[SimplexKey],
    ) -> SimplexKeyBuffer {
        if simplices_to_remove.is_empty() {
            return SimplexKeyBuffer::new();
        }

        let removal_set: SimplexKeySet = simplices_to_remove.iter().copied().collect();
        let mut frontier = SimplexKeyBuffer::new();
        let mut issue_simplex_count = 0;
        for pairs in issues.values() {
            issue_simplex_count += pairs.len();
        }
        let mut seen =
            fast_hash_set_with_capacity(simplices_to_remove.len() * (D + 1) + issue_simplex_count);

        for &simplex_key in simplices_to_remove {
            let Some(simplex) = self.tds.simplex(simplex_key) else {
                continue;
            };
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };
            for neighbor_key in neighbors.flatten() {
                if removal_set.contains(&neighbor_key) || !self.tds.contains_simplex(neighbor_key) {
                    continue;
                }
                if seen.insert(neighbor_key) {
                    frontier.push(neighbor_key);
                }
            }
        }

        for simplex_facet_pairs in issues.values() {
            for &(simplex_key, _) in simplex_facet_pairs {
                if removal_set.contains(&simplex_key) || !self.tds.contains_simplex(simplex_key) {
                    continue;
                }
                if seen.insert(simplex_key) {
                    frontier.push(simplex_key);
                }
            }
        }

        frontier
    }

    /// Repair over-shared facets and return the local frontier for neighbor repair.
    ///
    /// An empty issue map returns an empty outcome without touching the TDS, which lets the public
    /// transactional wrapper avoid snapshotting for no-op repairs.
    pub(crate) fn repair_local_facet_issues_with_frontier(
        &mut self,
        issues: &FacetIssuesMap,
        max_simplices_removed: usize,
    ) -> Result<LocalFacetRepairOutcome, InsertionError> {
        if issues.is_empty() {
            return Ok(LocalFacetRepairOutcome {
                removed_count: 0,
                removed_simplices: SimplexKeyBuffer::new(),
                frontier_simplices: SimplexKeyBuffer::new(),
                affected_vertices: IncidentRepairVertexBuffer::new(),
            });
        }

        let to_remove = self
            .simplices_for_local_facet_issue_repair(issues)
            .map_err(InsertionError::TopologyValidation)?;
        let attempted = to_remove.len();
        if attempted > max_simplices_removed {
            return Err(InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            });
        }
        let mut affected_vertices = IncidentRepairVertexBuffer::new();
        self.extend_incident_repair_vertices_from_simplices(&to_remove, &mut affected_vertices);
        let frontier_simplices = self.collect_local_repair_frontier(issues, &to_remove);
        let removed_count = self
            .tds
            .remove_simplices_by_keys(&to_remove)
            .map_err(|e| InsertionError::TopologyValidation(e.into_inner()))?;

        Ok(LocalFacetRepairOutcome {
            removed_count,
            removed_simplices: to_remove,
            frontier_simplices,
            affected_vertices,
        })
    }

    /// Repairs over-shared facets by removing lower-quality simplices.
    ///
    /// Uses geometric quality metrics (`radius_ratio`) to select which simplices to keep
    /// when a facet is shared by more than 2 simplices. UUID ordering is used as a tie-breaker
    /// when simplices have equal quality. Errors if quality computation or conversion fails.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(m * q) where m = number of problematic facets, q = quality computation cost
    /// - **Localized**: Only processes simplices involved in detected issues
    ///
    /// # Arguments
    ///
    /// * `issues` - Detected facet issues map from `detect_local_facet_issues()`
    /// * `max_simplices_removed` - Maximum simplices this repair may remove
    ///
    /// # Returns
    ///
    /// Number of simplices removed during repair.
    ///
    /// This public wrapper is transactional: if removal, neighbor repair,
    /// incident-simplex assignment, or final validation fails, the TDS is
    /// restored to its pre-call state.
    ///
    /// # Errors
    ///
    /// Returns an [`InsertionError`] if quality evaluation, facet bookkeeping,
    /// neighbor repair, incident-simplex assignment, or final topology
    /// validation fails. Returns
    /// [`InsertionError::MaxSimplicesRemovedExceeded`] when the selected repair
    /// would remove more simplices than `max_simplices_removed` allows; in that
    /// case the original TDS is restored before returning the error.
    ///
    /// `repair_local_facet_issues` uses a localized radius-ratio heuristic to choose
    /// problematic simplices for removal and repair. The heuristic is inspired by the same
    /// local cavity and simplex-quality ideas cited for vertex deletion; see `REFERENCES.md`
    /// entries \[1\]-\[5\]. It may fail or choose an overly aggressive repair near degenerate or
    /// nearly-coplanar cavities, inverted simplices, or scalar ranges where small numeric epsilons
    /// hide facet distinctions. Use robust predicates, explicit epsilon thresholds, bounded
    /// budgets, and transactional fallbacks when calling it from public repair paths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::insertion::InsertionError;
    /// use delaunay::prelude::triangulation::{FacetIssuesMap};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Insertion(#[from] InsertionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Start with a valid 2D simplex.
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Empty issues map => nothing to remove.
    /// let mut tri = dt.as_triangulation().clone();
    /// let removed = tri.repair_local_facet_issues(&FacetIssuesMap::default(), 0)?;
    /// assert_eq!(removed, 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// In practice, this method is typically called with issues detected by
    /// [`detect_local_facet_issues`](Self::detect_local_facet_issues) after insertion/removal
    /// operations.
    pub fn repair_local_facet_issues(
        &mut self,
        issues: &FacetIssuesMap,
        max_simplices_removed: usize,
    ) -> Result<usize, InsertionError> {
        if issues.is_empty() {
            return Ok(0);
        }

        let mut transaction = TriangulationRollbackTransaction::begin(self);
        let repair_result = {
            let tri = transaction.triangulation_mut();
            (|| -> Result<usize, InsertionError> {
                let outcome =
                    tri.repair_local_facet_issues_with_frontier(issues, max_simplices_removed)?;
                if outcome.removed_count == 0 {
                    return Ok(0);
                }

                let new_simplices = SimplexKeyBuffer::new();
                tri.repair_neighbors_after_local_simplex_removal(
                    &new_simplices,
                    &outcome.frontier_simplices,
                )?;
                tri.tds
                    .assign_incident_simplices()
                    .map_err(|error| InsertionError::TopologyValidation(error.into_inner()))?;
                tri.validate()
                    .map_err(Self::invariant_error_to_insertion_error)?;

                Ok(outcome.removed_count)
            })()
        };

        match repair_result {
            Ok(removed_count) => {
                transaction.commit();
                Ok(removed_count)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::collections::{CavityBoundaryBuffer, NeighborBuffer};
    use crate::core::simplex::{NeighborSlot, Simplex};
    use crate::core::tds::Tds;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    use std::assert_matches;

    use slotmap::KeyData;

    /// Helper: build a minimal 3D triangulation with one tetrahedron and valid
    /// incident-simplex pointers for all four vertices.
    fn build_single_tet() -> (
        Triangulation<FastKernel<f64>, (), (), 3>,
        [VertexKey; 4],
        SimplexKey,
    ) {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let v0 = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.0, 0.0, 0.0].unwrap())
            .unwrap();
        let v1 = tri
            .tds
            .insert_vertex_with_mapping(vertex![1.0, 0.0, 0.0].unwrap())
            .unwrap();
        let v2 = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.0, 1.0, 0.0].unwrap())
            .unwrap();
        let v3 = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.0, 0.0, 1.0].unwrap())
            .unwrap();

        let ck = tri
            .tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        for vk in [v0, v1, v2, v3] {
            tri.tds
                .vertex_mut(vk)
                .unwrap()
                .set_incident_simplex(Some(ck));
        }

        (tri, [v0, v1, v2, v3], ck)
    }

    /// Builds the explicit vertex frontier consumed by stale incident-simplex repair.
    fn incident_repair_scope(
        vertices: impl IntoIterator<Item = VertexKey>,
    ) -> IncidentRepairVertexBuffer {
        let mut scope = IncidentRepairVertexBuffer::new();
        for vertex_key in vertices {
            push_incident_repair_vertex_once(&mut scope, vertex_key);
        }
        scope
    }

    /// Build a deliberately invalid 2D fixture with three triangles sharing
    /// one edge. The fixture is useful for local facet-repair tests because
    /// removing one triangle leaves a small frontier whose survivor neighbor
    /// slots need rewiring.
    fn build_overshared_edge_fixture() -> (
        Triangulation<FastKernel<f64>, (), (), 2>,
        [SimplexKey; 3],
        VertexKey,
        VertexKey,
    ) {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(vertex![0.0, 0.0].unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex![1.0, 0.0].unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex![0.0, 1.0].unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex![0.0, -1.0].unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex![1.0, 1.0].unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();
        let c3 = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v_a, v_b, v_e], None).unwrap(),
            )
            .unwrap();

        for (simplex_key, neighbor_key) in [(c1, c2), (c2, c3), (c3, c1)] {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.resize(3, None);
            neighbors[2] = Some(neighbor_key);
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }
        tds.assign_incident_simplices().unwrap();

        (
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds),
            [c1, c2, c3],
            v_a,
            v_b,
        )
    }

    /// Consolidated macro for facet validation tests across dimensions.
    ///
    /// Verifies the manifold topology invariant: each facet shared by at most 2 simplices.
    /// Consolidates detection and repair tests into comprehensive suites.
    macro_rules! test_facet_validation {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_detect_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<(), $dim>> = vec![
                        $(vertex!($simplex_coords).unwrap()),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Valid simplex: should have no issues
                    let simplex_keys: Vec<_> = tri.tds.simplex_keys().collect();
                    assert_eq!(simplex_keys.len(), 1);
                    let issues = tri.detect_local_facet_issues(&simplex_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Valid simplex should have no facet issues", $dim);

                    // Empty list: should return None
                    let issues = tri.detect_local_facet_issues(&[]).unwrap();
                    assert!(issues.is_none(), "{}D: Empty list should have no issues", $dim);

                    // Nonexistent simplices: should be skipped gracefully
                    let fake_keys = vec![SimplexKey::default()];
                    let issues = tri.detect_local_facet_issues(&fake_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Nonexistent simplices should be skipped", $dim);

                    // Verify neighbors (all should be explicit boundary slots for a single simplex)
                    let (_, simplex) = tri.tds.simplices().next().unwrap();
                    let neighbors = simplex
                        .neighbor_slots()
                        .expect("single simplex should assign boundary neighbor slots");
                    assert!(
                        neighbors.iter().all(|slot| *slot == NeighborSlot::Boundary),
                        "{}D: Single simplex should have boundary slots",
                        $dim
                    );
                }

                #[test]
                fn [<test_repair_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<(), $dim>> = vec![
                        $(vertex!($simplex_coords).unwrap()),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let mut tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Empty issues map: should remove nothing
                    let empty_issues = FacetIssuesMap::default();
                    let removed = tri.repair_local_facet_issues(&empty_issues, 0).unwrap();
                    assert_eq!(removed, 0, "{}D: Empty issues should remove 0 simplices", $dim);
                    assert_eq!(tri.tds.number_of_simplices(), 1, "{}D: Should still have 1 simplex", $dim);
                }
            }
        };
    }

    /// Dimension-parametric `delete_vertex` tests.
    ///
    /// Verifies that vertex deletion maintains neighbor pointer integrity and
    /// triangulation validity across dimensions.
    macro_rules! test_delete_vertex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_delete_vertex_neighbor_pointers_ $dim d>]() {
                    // Build triangulation with D+1 simplex vertices + 1 interior point
                    let vertices: Vec<Vertex<(), $dim>> = {
                        let mut v = vec![$(vertex!($simplex_coords).unwrap()),+];
                        v.push(vertex!($interior_point).unwrap());
                        v
                    };

                    let mut dt = DelaunayTriangulation::try_new(&vertices)
                        .expect("Failed to create triangulation");

                    // Find and remove the interior vertex
                    let interior_vertex_key = dt
                        .vertices()
                        .find(|(_, v)| {
                            let coords = v.point().coords();
                            coords.iter()
                                .zip($interior_point.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-10)
                        })
                        .map(|(k, _)| k)
                        .expect("Interior vertex not found");

                    let initial_simplex_count = dt.tds().number_of_simplices();
                    dt.delete_vertex(interior_vertex_key)
                        .expect("Failed to remove vertex");

                    // After removal, should have fewer simplices (or same if just 1 simplex left)
                    assert!(dt.tds().number_of_simplices() <= initial_simplex_count,
                        "{}D: Simplex count should not increase after removal", $dim);

                    // Verify neighbor pointer consistency:
                    // 1. No dangling pointers (all neighbor keys exist)
                    // 2. Neighbor relationships are symmetric
                    for (simplex_key, simplex) in dt.tds().simplices() {
                        if let Some(neighbors) = simplex.neighbors() {
                            for (facet_idx, neighbor_opt) in neighbors.enumerate() {
                                if let Some(neighbor_key) = neighbor_opt {
                                    // Verify neighbor exists
                                    assert!(
                                        dt.tds().contains_simplex(neighbor_key),
                                        "{}D: Simplex {simplex_key:?} has neighbor pointer to non-existent simplex {neighbor_key:?}",
                                        $dim
                                    );

                                    // Verify symmetry: neighbor should point back to us
                                    let neighbor_simplex = dt
                                        .tds()
                                        .simplex(neighbor_key)
                                        .expect("Neighbor simplex should exist");
                                    if let Some(mut neighbor_neighbors) = neighbor_simplex.neighbors() {
                                        let points_back = neighbor_neighbors
                                            .any(|neighbor| neighbor == Some(simplex_key));
                                        assert!(
                                            points_back,
                                            "{}D: Simplex {simplex_key:?} has neighbor {neighbor_key:?} at facet {facet_idx}, but neighbor doesn't point back",
                                            $dim
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Verify triangulation is still valid (Levels 1–3; removal does not guarantee Delaunay)
                    let validation = dt.as_triangulation().validate();
                    assert!(
                        validation.is_ok(),
                        "{}D: Triangulation should be structurally valid after vertex removal: {:?}",
                        $dim,
                        validation.err()
                    );
                }
            }
        };
    }

    // Facet validation tests (2D - 5D)
    test_facet_validation!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_facet_validation!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    // Delete vertex tests (2D - 5D)
    test_delete_vertex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.3, 0.3]);
    test_delete_vertex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.25, 0.25, 0.25]
    );
    test_delete_vertex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2]
    );
    test_delete_vertex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.16, 0.16, 0.16, 0.16, 0.16]
    );

    // ---- repair_stale_incident_simplices tests ----

    #[test]
    fn test_repair_stale_incident_simplices_noop_when_all_valid() {
        let (mut tri, [v0, v1, v2, v3], ck) = build_single_tet();
        let scope = incident_repair_scope([v0, v1, v2, v3]);

        assert!(tri.repair_stale_incident_simplices(&scope).is_ok());

        // Pointers unchanged.
        for vk in [v0, v1, v2, v3] {
            assert_eq!(tri.tds.vertex_mut(vk).unwrap().incident_simplex(), Some(ck));
        }
    }

    #[test]
    fn test_repair_stale_incident_simplices_repairs_none_pointer() {
        let (mut tri, [_, _, _, v3], ck) = build_single_tet();

        // Corrupt v3 to have no incident simplex.
        tri.tds.vertex_mut(v3).unwrap().set_incident_simplex(None);
        let scope = incident_repair_scope([v3]);

        assert!(tri.repair_stale_incident_simplices(&scope).is_ok());
        assert_eq!(
            tri.tds.vertex_mut(v3).unwrap().incident_simplex(),
            Some(ck),
            "v3 should be repaired to point to the tetrahedron"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_repairs_stale_pointer() {
        let (mut tri, [_, _, _, v3], ck) = build_single_tet();

        // Point v3 to a non-existent simplex key (simulates a deleted conflict simplex).
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));
        tri.tds
            .vertex_mut(v3)
            .unwrap()
            .set_incident_simplex(Some(stale));
        let scope = incident_repair_scope([v3]);

        assert!(tri.repair_stale_incident_simplices(&scope).is_ok());
        assert_eq!(
            tri.tds.vertex_mut(v3).unwrap().incident_simplex(),
            Some(ck),
            "stale pointer should be repaired to the valid simplex"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_repairs_multiple_stale_pointers() {
        let (mut tri, [v0, v1, v2, v3], ck) = build_single_tet();

        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));
        tri.tds.vertex_mut(v0).unwrap().set_incident_simplex(None);
        tri.tds
            .vertex_mut(v2)
            .unwrap()
            .set_incident_simplex(Some(stale));
        let scope = incident_repair_scope([v0, v2]);

        assert!(tri.repair_stale_incident_simplices(&scope).is_ok());
        for vk in [v0, v1, v2, v3] {
            assert_eq!(
                tri.tds.vertex(vk).unwrap().incident_simplex(),
                Some(ck),
                "all vertices should point to the live tetrahedron after one repair pass"
            );
        }
    }

    #[test]
    fn test_repair_stale_incident_simplices_only_repairs_supplied_scope() {
        let (mut tri, [v0, _, v2, _], ck) = build_single_tet();

        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD_BEEF));
        tri.tds
            .vertex_mut(v0)
            .unwrap()
            .set_incident_simplex(Some(stale));
        tri.tds.vertex_mut(v2).unwrap().set_incident_simplex(None);
        let scope = incident_repair_scope([v2]);

        assert!(tri.repair_stale_incident_simplices(&scope).is_ok());
        assert_eq!(
            tri.tds.vertex(v2).unwrap().incident_simplex(),
            Some(ck),
            "scoped vertex should be repaired from canonical incidence"
        );
        assert_eq!(
            tri.tds.vertex(v0).unwrap().incident_simplex(),
            Some(stale),
            "unscoped stale vertex should remain untouched"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_errors_on_truly_isolated_vertex() {
        let (mut tri, _, _) = build_single_tet();

        // Insert a vertex that is NOT referenced by any simplex.
        let iso = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.5, 0.5, 0.5].unwrap())
            .unwrap();

        let scope = incident_repair_scope([iso]);
        let result = tri.repair_stale_incident_simplices(&scope);
        assert!(
            matches!(
                &result,
                Err(InsertionError::TopologyValidationFailed {
                    source, ..
                }) if matches!(
                    source,
                    TriangulationValidationError::IsolatedVertex { vertex_key, .. }
                        if *vertex_key == iso
                )
            ),
            "Truly isolated vertex should produce IsolatedVertex error: {result:?}"
        );
    }

    #[test]
    fn test_repair_stale_incident_simplices_errors_on_stale_incidence_index() {
        let (mut tri, [v0, _, _, _], ck) = build_single_tet();
        tri.tds.remove_simplex_storage_only_for_test(ck);
        tri.tds.vertex_mut(v0).unwrap().set_incident_simplex(None);
        let scope = incident_repair_scope([v0]);

        let result = tri.repair_stale_incident_simplices(&scope);

        assert_matches!(
            result,
            Err(InsertionError::TopologyValidation(TdsError::SimplexNotFound {
                simplex_key,
                ..
            })) if simplex_key == ck
        );
    }

    // =========================================================================
    // DETECT / REPAIR LOCAL FACET ISSUES
    // =========================================================================

    #[test]
    fn test_detect_local_facet_issues_none_for_valid_triangulation() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![1.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
        let issues = tri.detect_local_facet_issues(&simplex_keys).unwrap();
        assert!(issues.is_none());
    }

    #[test]
    fn test_pick_fan_apex_errors_for_empty_facets() {
        let (tri, _, _) = build_single_tet();
        assert_matches!(
            tri.pick_fan_apex(&[]),
            Err(TdsError::DimensionMismatch {
                expected: 1,
                actual: 0,
                ..
            })
        );
    }

    #[test]
    fn test_pick_fan_apex_preserves_missing_boundary_simplex() {
        let (tri, _, _) = build_single_tet();
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(0xBAD));
        let facets = [FacetHandle::from_validated(missing_simplex, 0)];

        assert_matches!(
            tri.pick_fan_apex(&facets),
            Err(TdsError::SimplexNotFound {
                simplex_key,
                ..
            }) if simplex_key == missing_simplex
        );
    }

    #[test]
    fn test_vertex_removal_affected_vertices_excludes_removed_vertex() {
        let (tri, [v0, v1, v2, v3], simplex_key) = build_single_tet();

        let affected = tri
            .affected_vertices_for_vertex_removal(&[simplex_key], v0)
            .unwrap();

        assert_eq!(affected.len(), 3);
        assert!(!affected.contains(&v0));
        assert!(affected.contains(&v1));
        assert!(affected.contains(&v2));
        assert!(affected.contains(&v3));
    }

    #[test]
    fn test_vertex_removal_affected_vertices_errors_on_missing_simplex() {
        let (tri, [v0, ..], _) = build_single_tet();
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(0xBAD));

        let result = tri.affected_vertices_for_vertex_removal(&[missing_simplex], v0);

        assert_matches!(
            result,
            Err(InvariantError::Tds(TdsError::SimplexNotFound {
                simplex_key,
                ..
            })) if simplex_key == missing_simplex
        );
    }

    #[test]
    fn test_vertex_removal_scope_helpers_keep_live_simplices_once() {
        let (tri, _, simplex_key) = build_single_tet();
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(0xBAD));
        let mut candidate_simplices = SimplexKeyBuffer::new();
        candidate_simplices.push(simplex_key);
        candidate_simplices.push(missing_simplex);
        candidate_simplices.push(simplex_key);
        let external_facets = [FacetHandle::from_validated(simplex_key, 0)];

        let live_simplices = tri.live_simplices_from(&candidate_simplices);
        let validation_scope = tri.vertex_removal_validation_scope(
            &candidate_simplices,
            &external_facets,
            &live_simplices,
        );

        assert_eq!(live_simplices.as_slice(), &[simplex_key]);
        assert_eq!(validation_scope.as_slice(), &[simplex_key]);
    }

    #[test]
    fn test_vertex_removal_outcome_returns_local_repair_seeds_for_fan_path() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![0.5, 0.5].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let vertex_key = dt
            .vertices()
            .find(|(_, vertex)| {
                let coords = vertex.point().coords();
                (coords[0] - 0.5).abs() < 1e-10 && (coords[1] - 0.5).abs() < 1e-10
            })
            .map(|(key, _)| key)
            .unwrap();
        let mut tri = dt.as_triangulation().clone();

        let outcome = tri.remove_vertex_with_repair_seeds(vertex_key).unwrap();

        assert!(outcome.simplices_removed > 0);
        assert!(!outcome.repair_seed_simplices.is_empty());
        assert!(tri.tds.vertex(vertex_key).is_none());
        assert!(
            outcome
                .repair_seed_simplices
                .iter()
                .all(|&simplex_key| tri.tds.simplex(simplex_key).is_some())
        );
        tri.validate().unwrap();
    }

    #[test]
    fn test_vertex_removal_outcome_noops_for_missing_vertex_key() {
        let vertices = [
            vertex![0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tri = dt.as_triangulation().clone();
        let missing_vertex = VertexKey::from(KeyData::from_ffi(0xBAD));
        let vertex_count = tri.tds.number_of_vertices();
        let simplex_count = tri.tds.number_of_simplices();

        let outcome = tri
            .remove_vertex_with_repair_seeds(missing_vertex)
            .expect("missing vertex removal should be a no-op at the core repair layer");

        assert_eq!(outcome.simplices_removed, 0);
        assert!(outcome.repair_seed_simplices.is_empty());
        assert_eq!(tri.tds.number_of_vertices(), vertex_count);
        assert_eq!(tri.tds.number_of_simplices(), simplex_count);
        tri.validate().unwrap();
    }

    #[test]
    fn test_vertex_removal_outcome_removes_isolated_vertex_without_repair_seeds() {
        let vertices = [
            vertex![0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tri = dt.as_triangulation().clone();
        let isolated_vertex = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.5, 0.5, 0.5].unwrap())
            .unwrap();
        let simplex_count = tri.tds.number_of_simplices();

        let outcome = tri
            .remove_vertex_with_repair_seeds(isolated_vertex)
            .expect("isolated vertex removal should restore the prior valid triangulation");

        assert_eq!(outcome.simplices_removed, 0);
        assert!(outcome.repair_seed_simplices.is_empty());
        assert!(!tri.tds.contains_vertex_key(isolated_vertex));
        assert_eq!(tri.tds.number_of_simplices(), simplex_count);
        tri.validate().unwrap();
    }

    #[test]
    fn test_vertex_removal_outcome_rolls_back_failed_isolated_vertex_removal() {
        let (mut tri, _, simplex_key) = build_single_tet();
        let isolated_to_remove = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.5, 0.5, 0.5].unwrap())
            .unwrap();
        let isolated_survivor = tri
            .tds
            .insert_vertex_with_mapping(vertex![0.25, 0.25, 0.25].unwrap())
            .unwrap();
        let vertex_count = tri.tds.number_of_vertices();
        let simplex_count = tri.tds.number_of_simplices();

        let result = tri.remove_vertex_with_repair_seeds(isolated_to_remove);

        assert_matches!(
            result,
            Err(InvariantError::Triangulation(
                TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    ..
                },
            )) if vertex_key == isolated_survivor
        );
        assert_eq!(tri.tds.number_of_vertices(), vertex_count);
        assert_eq!(tri.tds.number_of_simplices(), simplex_count);
        assert!(tri.tds.contains_vertex_key(isolated_to_remove));
        assert!(tri.tds.contains_vertex_key(isolated_survivor));
        assert!(tri.tds.contains_simplex(simplex_key));
    }

    #[test]
    fn test_repair_vertex_removal_facet_issues_noops_without_local_issues() {
        let (mut tri, _, simplex_key) = build_single_tet();
        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(simplex_key);
        let mut simplices_removed = 0;
        let mut post_repair_frontier = SimplexKeyBuffer::new();

        tri.repair_vertex_removal_facet_issues(
            &new_simplices,
            &mut simplices_removed,
            &mut post_repair_frontier,
            usize::MAX,
        )
        .unwrap();

        assert_eq!(simplices_removed, 0);
        assert!(post_repair_frontier.is_empty());
    }

    #[test]
    fn test_repair_affected_vertex_incidence_from_scope_repairs_missing_pointer() {
        let (mut tri, [_, _, _, v3], simplex_key) = build_single_tet();
        tri.tds.vertex_mut(v3).unwrap().set_incident_simplex(None);
        let mut affected_vertices = VertexKeySet::default();
        affected_vertices.insert(v3);
        let mut validation_scope = SimplexKeyBuffer::new();
        validation_scope.push(simplex_key);

        tri.repair_affected_vertex_incidence_from_scope(&affected_vertices, &validation_scope)
            .unwrap();

        assert_eq!(
            tri.tds.vertex(v3).unwrap().incident_simplex(),
            Some(simplex_key)
        );
    }

    #[test]
    fn test_repair_affected_vertex_incidence_from_scope_reports_isolated_vertex() {
        let (mut tri, [_, _, _, v3], _) = build_single_tet();
        tri.tds.vertex_mut(v3).unwrap().set_incident_simplex(None);
        let mut affected_vertices = VertexKeySet::default();
        affected_vertices.insert(v3);
        let validation_scope = SimplexKeyBuffer::new();

        let result =
            tri.repair_affected_vertex_incidence_from_scope(&affected_vertices, &validation_scope);

        assert_matches!(
            result,
            Err(InvariantError::Triangulation(
                TriangulationValidationError::IsolatedVertex { vertex_key, .. }
            )) if vertex_key == v3
        );
    }

    #[test]
    fn test_vertex_removal_postconditions_reject_still_present_vertex() {
        let (tri, [v0, ..], simplex_key) = build_single_tet();
        let affected_vertices = VertexKeySet::default();
        let mut surviving_new_simplices = SimplexKeyBuffer::new();
        surviving_new_simplices.push(simplex_key);
        let mut validation_scope = SimplexKeyBuffer::new();
        validation_scope.push(simplex_key);

        let result = tri.validate_vertex_removal_postconditions(
            v0,
            &affected_vertices,
            &surviving_new_simplices,
            &validation_scope,
        );

        assert_matches!(
            result,
            Err(InvariantError::Tds(
                TdsError::InconsistentDataStructure { .. }
            ))
        );
    }

    #[test]
    fn test_fan_boundary_facets_excluding_apex_keeps_only_facets_without_apex() {
        let (tri, vkeys, simplex_key) = build_single_tet();
        let boundary_facets: CavityBoundaryBuffer = (0..=3)
            .map(|i| FacetHandle::from_validated(simplex_key, i))
            .collect();

        let fan_facets = tri
            .fan_boundary_facets_excluding_apex(vkeys[0], &boundary_facets)
            .unwrap();

        assert_eq!(
            fan_facets.as_slice(),
            &[FacetHandle::from_validated(simplex_key, 0)]
        );
    }

    #[test]
    fn test_quality_error_to_tds_error_preserves_lookup_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(2));

        let simplex_error = QualityError::SimplexVertices {
            simplex_key,
            source: QualitySimplexVerticesError::SimplexNotFound {
                simplex_key,
                context: "quality lookup".to_string(),
            },
        };
        assert_matches!(
            quality_error_to_tds_error(simplex_key, simplex_error),
            TdsError::SimplexNotFound { simplex_key: key, .. } if key == simplex_key
        );

        let vertex_error = QualityError::VertexNotFound { vertex_key };
        assert_matches!(
            quality_error_to_tds_error(simplex_key, vertex_error),
            TdsError::VertexNotFound { vertex_key: key, .. } if key == vertex_key
        );
    }

    #[test]
    fn test_quality_error_to_tds_error_preserves_arity_mismatch() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let error = QualityError::InvalidSimplexArity {
            actual: 3,
            expected: 4,
            dimension: 3,
        };

        assert_matches!(
            quality_error_to_tds_error(simplex_key, error),
            TdsError::DimensionMismatch {
                expected: 4,
                actual: 3,
                ..
            }
        );
    }

    // =========================================================================
    // DELETE VERTEX: RETRIANGULATION AND TOPOLOGY
    // =========================================================================

    #[test]
    fn test_delete_vertex_retriangulates_cavity_2d() {
        // Build 2D triangulation with 4 vertices, delete one, verify valid.
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![0.5, 0.5].unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let initial_simplices = dt.number_of_simplices();
        let vertex_key = dt
            .vertices()
            .find(|(_, v)| {
                let c = v.point().coords();
                (c[0] - 0.5).abs() < 1e-10 && (c[1] - 0.5).abs() < 1e-10
            })
            .map(|(k, _)| k)
            .unwrap();

        let removed = dt.delete_vertex(vertex_key).unwrap();
        assert!(removed > 0, "Should have removed at least 1 simplex");
        assert!(dt.number_of_simplices() <= initial_simplices);
        assert_eq!(dt.number_of_vertices(), 3);
    }

    #[test]
    fn test_delete_vertex_entire_triangulation_2d() {
        // When we delete a vertex from a single-simplex triangulation,
        // the empty boundary case triggers Tds::remove_vertex fallback.
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let initial_vertices = dt.number_of_vertices();
        let initial_simplices = dt.number_of_simplices();
        let vertex_key = dt.vertices().next().unwrap().0;

        let error = dt.delete_vertex(vertex_key).unwrap_err();

        assert!(
            matches!(
                error,
                crate::DeleteVertexError::InvariantViolation {
                    ref source
                } if matches!(
                    source.as_ref(),
                    InvariantError::Triangulation(
                        TriangulationValidationError::IsolatedVertex { .. }
                    )
                )
            ),
            "expected isolated-vertex invariant failure, got {error:?}"
        );
        assert_eq!(dt.number_of_vertices(), initial_vertices);
        assert_eq!(dt.number_of_simplices(), initial_simplices);
        assert!(dt.tds().contains_vertex_key(vertex_key));
    }

    // =========================================================================
    // FAN FILL CAVITY: ERROR CASE
    // =========================================================================

    #[test]
    fn test_fan_fill_cavity_errors_when_no_simplices_produced() {
        // If the apex is on every boundary facet, fan_fill_cavity should error.
        let (mut tri, vkeys, ck) = build_single_tet();

        // Use vkeys[0] as apex; construct boundary facets that ALL include vkeys[0].
        // In a tet, facet 0 is opposite vkeys[0] (does NOT include it),
        // but facets 1,2,3 each include vkeys[0].
        let boundary_facets: CavityBoundaryBuffer = (1..=3)
            .map(|i| FacetHandle::from_validated(ck, i))
            .collect();

        let result = tri.fan_fill_cavity(vkeys[0], &boundary_facets);
        // All facets include vkeys[0], so no simplices should be created.
        assert!(result.is_err());
    }

    // =========================================================================
    // REPAIR LOCAL FACET ISSUES: NON-EMPTY ISSUES MAP
    // =========================================================================

    #[test]
    fn test_repair_local_facet_issues_handles_duplicate_simplex_fixture_transactionally() {
        // Build 2D triangulation with enough simplices to have interior facets,
        // then artificially create an over-shared facet by duplicating a simplex.
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![1.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tri = dt.as_triangulation().clone();

        // Add a duplicate simplex with the same vertices as an existing simplex.
        let (_, existing_simplex) = tri.tds.simplices().next().unwrap();
        let vkeys: Vec<_> = existing_simplex.vertices().to_vec();
        let dup_simplex = Simplex::try_new_with_data(vkeys, None).unwrap();
        let _ = tri
            .tds
            .insert_simplex_bypassing_topology_checks_for_test(dup_simplex)
            .unwrap();

        // Now detect issues.
        let all_simplices: Vec<_> = tri.tds.simplex_keys().collect();
        let issues = tri.detect_local_facet_issues(&all_simplices).unwrap();
        assert!(issues.is_some(), "Should detect over-shared facet");
        let original_simplex_count = tri.tds.number_of_simplices();

        match tri.repair_local_facet_issues(&issues.unwrap(), usize::MAX) {
            Ok(removed) => {
                assert!(removed > 0, "repair should remove at least one simplex");
                tri.validate()
                    .expect("successful public repair should leave valid topology");
            }
            Err(InsertionError::TopologyValidation(TdsError::DuplicateSimplices { .. })) => {
                assert_eq!(tri.tds.number_of_simplices(), original_simplex_count);
            }
            Err(error) => panic!("unexpected duplicate-simplex repair error: {error:?}"),
        }
    }

    /// Return the facet index opposite the vertex not on the tested shared edge.
    fn shared_edge_facet_index(simplex: &Simplex<(), 2>, v_a: VertexKey, v_b: VertexKey) -> usize {
        simplex
            .vertices()
            .iter()
            .position(|&vertex_key| vertex_key != v_a && vertex_key != v_b)
            .expect("test simplices should contain the shared edge")
    }

    /// Read the neighbor slot across the tested shared edge in a 2D repair fixture.
    fn neighbor_across_shared_edge(
        tri: &Triangulation<FastKernel<f64>, (), (), 2>,
        simplex_key: SimplexKey,
        v_a: VertexKey,
        v_b: VertexKey,
    ) -> Option<SimplexKey> {
        let simplex = tri.tds.simplex(simplex_key).unwrap();
        let facet_idx = shared_edge_facet_index(simplex, v_a, v_b);
        simplex.neighbor_key(facet_idx).flatten()
    }

    #[test]
    fn test_local_repair_uses_removal_frontier() {
        let (mut tri, original_simplices, v_a, v_b) = build_overshared_edge_fixture();
        let issues = tri
            .detect_local_facet_issues(&original_simplices)
            .unwrap()
            .expect("three simplices sharing one edge should be detected as over-shared");

        let repair = tri
            .repair_local_facet_issues_with_frontier(&issues, usize::MAX)
            .unwrap();
        assert_eq!(repair.removed_count, 1);
        assert!(
            !repair.frontier_simplices.is_empty(),
            "removed-simplex neighbors should seed the local repair frontier"
        );

        let survivors: Vec<_> = original_simplices
            .into_iter()
            .filter(|simplex_key| tri.tds.contains_simplex(*simplex_key))
            .collect();
        assert_eq!(survivors.len(), 2);
        let [first_survivor, second_survivor] = survivors.as_slice() else {
            panic!("fixture should leave exactly two surviving simplices");
        };
        for &survivor in &survivors {
            assert!(
                repair.frontier_simplices.contains(&survivor),
                "facet-issue survivors should seed the local repair frontier"
            );
        }
        let survivor_pairs = [
            (*first_survivor, *second_survivor),
            (*second_survivor, *first_survivor),
        ];

        let missing_shared_slots_before = survivor_pairs
            .iter()
            .filter(|&&(simplex_key, other)| {
                neighbor_across_shared_edge(&tri, simplex_key, v_a, v_b) != Some(other)
            })
            .count();
        assert!(
            missing_shared_slots_before > 0,
            "simplex removal should leave at least one survivor slot needing local repair"
        );

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.extend(original_simplices);
        let repaired = tri
            .repair_neighbors_after_local_simplex_removal(
                &new_simplices,
                &repair.frontier_simplices,
            )
            .unwrap();

        assert!(repaired > 0);
        for (simplex_key, other) in survivor_pairs {
            assert_eq!(
                neighbor_across_shared_edge(&tri, simplex_key, v_a, v_b),
                Some(other),
                "surviving simplices should be rewired across the formerly over-shared edge"
            );
        }
        assert!(tri.tds.validate_facet_sharing().is_ok());
        assert!(tri.detect_local_facet_issues(&survivors).unwrap().is_none());
    }

    #[test]
    fn test_local_repair_budget_failure_does_not_remove_simplices() {
        let (mut tri, original_simplices, _, _) = build_overshared_edge_fixture();
        let issues = tri
            .detect_local_facet_issues(&original_simplices)
            .unwrap()
            .expect("three simplices sharing one edge should be detected as over-shared");
        let original_simplex_count = tri.tds.number_of_simplices();

        let result = tri.repair_local_facet_issues_with_frontier(&issues, 0);

        assert_matches!(
            result,
            Err(InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 0,
                attempted
            }) if attempted > 0
        );
        assert_eq!(tri.tds.number_of_simplices(), original_simplex_count);
        for simplex_key in original_simplices {
            assert!(
                tri.tds.contains_simplex(simplex_key),
                "budget failure should happen before simplex removal"
            );
        }
    }

    #[test]
    fn test_vertex_removal_facet_repair_reports_budget_exhaustion() {
        let (mut tri, original_simplices, _, _) = build_overshared_edge_fixture();
        let original_simplex_count = tri.tds.number_of_simplices();
        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.extend(original_simplices);
        let mut simplices_removed = 0;
        let mut post_repair_frontier = SimplexKeyBuffer::new();

        let result = tri.repair_vertex_removal_facet_issues(
            &new_simplices,
            &mut simplices_removed,
            &mut post_repair_frontier,
            0,
        );

        assert_matches!(
            result,
            Err(InvariantError::Tds(TdsError::InconsistentDataStructure { ref message }))
                if message.contains("Local facet repair after vertex removal failed")
                    && message.contains("Local facet repair removal budget exceeded")
        );
        assert_eq!(simplices_removed, 0);
        assert!(post_repair_frontier.is_empty());
        assert_eq!(tri.tds.number_of_simplices(), original_simplex_count);
        for simplex_key in original_simplices {
            assert!(tri.tds.contains_simplex(simplex_key));
        }
    }

    #[test]
    fn test_vertex_removal_facet_repair_restores_facet_sharing() {
        let (mut tri, original_simplices, _, _) = build_overshared_edge_fixture();
        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.extend(original_simplices);
        let mut simplices_removed = 0;
        let mut post_repair_frontier = SimplexKeyBuffer::new();

        tri.repair_vertex_removal_facet_issues(
            &new_simplices,
            &mut simplices_removed,
            &mut post_repair_frontier,
            usize::MAX,
        )
        .unwrap();

        assert_eq!(simplices_removed, 1);
        assert!(
            !post_repair_frontier.is_empty(),
            "removed-simplex neighbors should seed the post-repair frontier"
        );
        assert!(tri.tds.validate_facet_sharing().is_ok());
        let survivors: Vec<_> = original_simplices
            .into_iter()
            .filter(|simplex_key| tri.tds.contains_simplex(*simplex_key))
            .collect();
        assert_eq!(survivors.len(), 2);
        assert!(tri.detect_local_facet_issues(&survivors).unwrap().is_none());
    }

    #[test]
    fn test_repair_local_facet_issues_rolls_back_invalid_public_repair() {
        let (mut tri, original_simplices, _, _) = build_overshared_edge_fixture();
        let issues = tri
            .detect_local_facet_issues(&original_simplices)
            .unwrap()
            .expect("three simplices sharing one edge should be detected as over-shared");
        let original_simplex_count = tri.tds.number_of_simplices();
        let original_vertex_count = tri.tds.number_of_vertices();

        let result = tri.repair_local_facet_issues(&issues, usize::MAX);

        assert!(
            result.is_err(),
            "public repair should reject an end state that fails full validation"
        );
        assert_eq!(tri.tds.number_of_simplices(), original_simplex_count);
        assert_eq!(tri.tds.number_of_vertices(), original_vertex_count);
        for simplex_key in original_simplices {
            assert!(
                tri.tds.contains_simplex(simplex_key),
                "rollback should restore every pre-repair simplex"
            );
        }
    }

    #[test]
    fn test_repair_local_facet_issues_respects_removal_budget() {
        let (mut tri, original_simplices, _, _) = build_overshared_edge_fixture();
        let issues = tri
            .detect_local_facet_issues(&original_simplices)
            .unwrap()
            .expect("three simplices sharing one edge should be detected as over-shared");
        let original_simplex_count = tri.tds.number_of_simplices();

        let result = tri.repair_local_facet_issues(&issues, 0);

        assert_matches!(
            result,
            Err(InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 0,
                attempted
            }) if attempted > 0
        );
        assert_eq!(tri.tds.number_of_simplices(), original_simplex_count);
    }
}
