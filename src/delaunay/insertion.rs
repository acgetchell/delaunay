//! Incremental insertion operations for Delaunay triangulations.
//!
//! This module owns post-construction vertex insertion, insertion-cache
//! maintenance, and policy-controlled local Delaunay repair after insertion.
//!
//! The insertion workflow follows the Bowyer-Watson cavity model \[2]\[3]:
//! point location and cavity identification depend on exact orientation and
//! in-sphere predicates \[1], then the TDS fills the cavity and locally rewires
//! neighbors. Flip-based repair uses regular-triangulation flip theory \[4].
//! Robust fallback rebuilds and fan retriangulation are bounded recovery paths:
//! they still validate with exact predicates where they identify cavities or
//! test in-sphere/Delaunay violations, but their local repair ordering is
//! heuristic and must be followed by topology and Delaunay validation \[5].
//! See `REFERENCES.md` for the numbered bibliography.

#![forbid(unsafe_code)]

#[cfg(test)]
use crate::construction::test_hooks;
use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairRun, repair_delaunay_with_flips_k2_k3_run,
};
use crate::core::algorithms::incremental_insertion::{
    DelaunayRepairFailureContext, InsertionError, InsertionTopologyValidationContext,
};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{FastHashSet, SimplexKeyBuffer};
use crate::core::operations::{InsertionOutcome, InsertionStatistics};
use crate::core::tds::{SimplexKey, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::validation::{TopologyGuarantee, TriangulationValidationError};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::repair::DelaunayRepairPolicy;
use crate::topology::manifold::{ManifoldError, validate_ridge_links_for_simplices};
use crate::triangulation::DelaunayTriangulation;
#[cfg(test)]
use crate::validation::DelaunayTriangulationCandidate;
use std::env;

fn ridge_link_repair_validation_error(err: ManifoldError) -> InsertionError {
    match TriangulationValidationError::try_from(err) {
        Ok(source) => InsertionError::TopologyValidationFailed {
            context: InsertionTopologyValidationContext::DelaunayRepair,
            source,
        },
        Err(source) => InsertionError::TopologyValidation(source),
    }
}

// =============================================================================
// MUTATION (Requires Numeric Scalar Bounds)
// =============================================================================
//
// Incremental insertion, deletion, and post-insertion repair/check helpers.
// These require an f64-backed kernel for spatial-index construction,
// Triangulation-layer insertion, and Triangulation-layer deletion.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Lazily seeds the spatial index from existing vertices so incremental
    /// insertion can start from deserialized or manually constructed TDS state.
    fn ensure_spatial_index_seeded(&mut self) -> Result<(), InsertionError> {
        if self.spatial_index.is_some() {
            return Ok(());
        }

        let duplicate_tolerance = 1e-10_f64;
        let mut index = HashGridIndex::try_new(duplicate_tolerance)?;

        for (vkey, vertex) in self.tri.tds.vertices() {
            index.insert_vertex(vkey, vertex.point().coords());
        }

        self.spatial_index = Some(index);
        Ok(())
    }

    /// Insert a vertex into the Delaunay triangulation using incremental cavity-based algorithm.
    ///
    /// This method handles all stages of triangulation construction:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating simplices
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-simplex
    /// - **Incremental (> D+1 vertices)**: Uses cavity-based insertion with point location
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate simplex containing the point
    /// 4. Find conflict region (simplices whose circumspheres contain the point)
    /// 5. Extract cavity boundary
    /// 6. Fill cavity (create new simplices)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict simplices
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails (when reaching D+1 vertices)
    /// - Point is on a facet, edge, or vertex (degenerate cases not yet implemented)
    /// - Spatial index construction fails while seeding duplicate-detection and locate hints
    /// - Conflict region computation fails
    /// - Cavity boundary extraction fails
    /// - Cavity filling or neighbor wiring fails
    ///
    /// Note: Points outside the convex hull are handled automatically via hull extension.
    ///
    /// # Examples
    ///
    /// Incremental insertion from empty triangulation:
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulation, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_simplices(), 0);
    ///
    /// // Insert vertices one by one - bootstrap phase (no simplices yet)
    /// dt.insert_vertex(vertex![0.0, 0.0, 0.0]?)?;
    /// dt.insert_vertex(vertex![1.0, 0.0, 0.0]?)?;
    /// dt.insert_vertex(vertex![0.0, 1.0, 0.0]?)?;
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 0); // Still no simplices
    ///
    /// // 4th vertex triggers initial simplex creation
    /// dt.insert_vertex(vertex![0.0, 0.0, 1.0]?)?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_simplices(), 1); // First simplex created!
    ///
    /// // Further insertions use cavity-based algorithm
    /// dt.insert_vertex(vertex![0.2, 0.2, 0.2]?)?;
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// assert!(dt.number_of_simplices() > 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Using batch construction (traditional approach):
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Create initial triangulation with 5 vertices (4-simplex)
    /// let vertices = vec![
    ///     vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 5);
    ///
    /// // Insert additional interior vertex
    /// dt.insert_vertex(vertex![0.2, 0.2, 0.2, 0.2]?)?;
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// assert!(dt.number_of_simplices() > 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_vertex(&mut self, vertex: Vertex<U, D>) -> Result<VertexKey, InsertionError> {
        self.ensure_spatial_index_seeded()?;

        // Fully delegate to Triangulation layer
        // Triangulation handles:
        // - Manifold maintenance (conflict simplices, cavity, repairs)
        // - Bootstrap and initial simplex
        // - Location and conflict region computation
        //
        // DelaunayTriangulation adds:
        // - Kernel (provides in-sphere predicate for Delaunay property)
        // - Hint caching for performance
        // - Future: Delaunay property restoration after deletion
        //
        // Transactional guard: post-steps (flip repair and/or global Delaunay checks) can fail.
        // If they do, rollback to leave the triangulation unchanged.
        let next_insertion_count = self
            .insertion_state
            .delaunay_repair_insertion_count
            .saturating_add(1);
        let could_have_simplices_after_insertion = self.tri.tds.number_of_simplices() > 0
            || self.tri.tds.number_of_vertices().saturating_add(1) > D;
        let snapshot_needed = could_have_simplices_after_insertion
            && (self.insertion_state.delaunay_repair_policy != DelaunayRepairPolicy::Never
                || self
                    .insertion_state
                    .delaunay_check_policy
                    .should_check(next_insertion_count));
        let snapshot =
            snapshot_needed.then(|| (self.tri.tds.clone_for_rollback(), self.insertion_state));

        let insertion_result = (|| {
            let hint = self.insertion_state.last_inserted_simplex;
            let insert_detail = {
                let (tri, spatial_index) = (&mut self.tri, &mut self.spatial_index);
                tri.insert_with_statistics_seeded_indexed_detailed(
                    vertex,
                    None,
                    hint,
                    0,
                    spatial_index.as_mut(),
                    None,
                )?
            };
            let repair_seed_simplices = insert_detail.repair_seed_simplices;
            let delaunay_repair_required = insert_detail.delaunay_repair_required;

            match insert_detail.outcome {
                InsertionOutcome::Inserted {
                    vertex_key: v_key,
                    hint,
                } => {
                    self.insertion_state.last_inserted_simplex = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    if delaunay_repair_required {
                        self.maybe_repair_after_insertion(v_key, hint, &repair_seed_simplices)?;
                    }
                    self.maybe_check_after_insertion()?;
                    Ok(v_key)
                }
                InsertionOutcome::Skipped { error } => Err(error),
            }
        })();

        match insertion_result {
            Ok(v_key) => Ok(v_key),
            Err(err) => {
                if let Some((tds, insertion_state)) = snapshot {
                    self.spatial_index = None;
                    self.tri.tds = tds;
                    self.insertion_state = insertion_state;
                }
                Err(err)
            }
        }
    }

    /// Insert a vertex and return the insertion outcome plus statistics.
    ///
    /// This is a convenience wrapper around the triangulation-layer insertion-with-statistics
    /// implementation that also updates the internal `insertion_state.last_inserted_simplex` hint cache.
    /// Unlike [`insert_best_effort_with_statistics`](Self::insert_best_effort_with_statistics),
    /// skipped insertions are reported as [`InsertionError`] values so callers using `?`
    /// cannot accidentally ignore a duplicate or retry-exhausted degeneracy.
    ///
    /// # Errors
    ///
    /// Returns [`InsertionError`] for structural failures, duplicate coordinates,
    /// retryable geometric degeneracies that exhaust all attempts, and spatial
    /// index construction failures while seeding duplicate-detection and locate hints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulation};
    /// use delaunay::prelude::insertion::{InsertionError, InsertionOutcome};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// let vertex = delaunay::vertex![0.0, 0.0, 0.0]?;
    /// let (outcome, stats) = dt.insert_with_statistics(vertex)?;
    ///
    /// assert!(stats.success());
    /// std::assert_matches!(outcome, InsertionOutcome::Inserted { .. });
    ///
    /// let duplicate_vertex = delaunay::vertex![0.0, 0.0, 0.0]?;
    /// let duplicate = dt.insert_with_statistics(duplicate_vertex);
    /// std::assert_matches!(
    ///     duplicate,
    ///     Err(InsertionError::DuplicateCoordinates { .. })
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError> {
        match self.insert_best_effort_with_statistics(vertex)? {
            (outcome @ InsertionOutcome::Inserted { .. }, stats) => Ok((outcome, stats)),
            (InsertionOutcome::Skipped { error }, _stats) => Err(error),
        }
    }

    /// Insert a vertex and return telemetry even when the vertex is skipped.
    ///
    /// This best-effort API is intended for diagnostics, bulk-style ingestion,
    /// and workloads that deliberately continue after duplicate coordinates or
    /// retry-exhausted geometric degeneracies. A skipped insertion returns
    /// `Ok((InsertionOutcome::Skipped { .. }, stats))`; the triangulation is
    /// left unchanged for that vertex.
    ///
    /// # Errors
    ///
    /// Returns [`InsertionError`] for non-skip structural failures that cannot
    /// be represented as an [`InsertionOutcome::Skipped`] value, including
    /// spatial index construction failures while seeding duplicate-detection
    /// and locate hints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulation};
    /// use delaunay::prelude::insertion::InsertionOutcome;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// let (outcome, stats) = dt
    ///     .insert_best_effort_with_statistics(delaunay::vertex![0.0, 0.0, 0.0]?)?;
    ///
    /// assert!(stats.success());
    /// std::assert_matches!(outcome, InsertionOutcome::Inserted { .. });
    ///
    /// let vertices_before_duplicate = dt.number_of_vertices();
    /// let (duplicate_outcome, duplicate_stats) = dt
    ///     .insert_best_effort_with_statistics(delaunay::vertex![0.0, 0.0, 0.0]?)?;
    ///
    /// std::assert_matches!(duplicate_outcome, InsertionOutcome::Skipped { .. });
    /// assert!(duplicate_stats.skipped_duplicate());
    /// assert_eq!(dt.number_of_vertices(), vertices_before_duplicate);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_best_effort_with_statistics(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError> {
        self.ensure_spatial_index_seeded()?;

        // Transactional guard: post-steps (flip repair and/or global Delaunay checks) can fail.
        // If they do, rollback to leave the triangulation unchanged.
        let next_insertion_count = self
            .insertion_state
            .delaunay_repair_insertion_count
            .saturating_add(1);
        let could_have_simplices_after_insertion = self.tri.tds.number_of_simplices() > 0
            || self.tri.tds.number_of_vertices().saturating_add(1) > D;
        let snapshot_needed = could_have_simplices_after_insertion
            && (self.insertion_state.delaunay_repair_policy != DelaunayRepairPolicy::Never
                || self
                    .insertion_state
                    .delaunay_check_policy
                    .should_check(next_insertion_count));
        let snapshot =
            snapshot_needed.then(|| (self.tri.tds.clone_for_rollback(), self.insertion_state));

        let insertion_result = (|| {
            let hint = self.insertion_state.last_inserted_simplex;
            let insert_detail = {
                let (tri, spatial_index) = (&mut self.tri, &mut self.spatial_index);
                tri.insert_with_statistics_seeded_indexed_detailed(
                    vertex,
                    None,
                    hint,
                    0,
                    spatial_index.as_mut(),
                    None,
                )?
            };
            let stats = insert_detail.stats;
            let repair_seed_simplices = insert_detail.repair_seed_simplices;
            let delaunay_repair_required = insert_detail.delaunay_repair_required;

            let outcome = match insert_detail.outcome {
                InsertionOutcome::Inserted { vertex_key, hint } => {
                    self.insertion_state.last_inserted_simplex = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    if delaunay_repair_required {
                        self.maybe_repair_after_insertion(
                            vertex_key,
                            hint,
                            &repair_seed_simplices,
                        )?;
                    }
                    self.maybe_check_after_insertion()?;
                    InsertionOutcome::Inserted { vertex_key, hint }
                }
                other @ InsertionOutcome::Skipped { .. } => other,
            };

            Ok((outcome, stats))
        })();

        match insertion_result {
            Ok((outcome, stats)) => Ok((outcome, stats)),
            Err(err) => {
                if let Some((tds, insertion_state)) = snapshot {
                    self.spatial_index = None;
                    self.tri.tds = tds;
                    self.insertion_state = insertion_state;
                }
                Err(err)
            }
        }
    }

    /// Keeps the default insertion path on the same repair helper as capped
    /// debug and heuristic paths.
    fn maybe_repair_after_insertion(
        &mut self,
        vertex_key: VertexKey,
        hint: Option<SimplexKey>,
        extra_seed_simplices: &[SimplexKey],
    ) -> Result<(), InsertionError> {
        self.maybe_repair_after_insertion_capped(vertex_key, hint, extra_seed_simplices, None)
    }

    /// Like [`maybe_repair_after_insertion`](Self::maybe_repair_after_insertion) but
    /// forwards an optional per-attempt flip cap to the underlying repair functions.
    ///
    /// `extra_seed_simplices` widens the local repair frontier beyond the inserted vertex
    /// star. This is used when cavity reduction shrinks simplices out of the conflict
    /// region: those simplices stay in the triangulation and may still need a local
    /// Delaunay revisit even though they are no longer adjacent to the new vertex.
    pub(crate) fn maybe_repair_after_insertion_capped(
        &mut self,
        vertex_key: VertexKey,
        hint: Option<SimplexKey>,
        extra_seed_simplices: &[SimplexKey],
        max_flips: Option<usize>,
    ) -> Result<(), InsertionError> {
        let topology = self.tri.topology_guarantee();
        if !self.should_run_delaunay_repair_for(
            topology,
            self.insertion_state.delaunay_repair_insertion_count,
        ) {
            return Ok(());
        }

        // Prefer the merged local frontier when we have one; otherwise fall back to the
        // validated locate hint so repair can still start from the inserted star.
        let seed_simplices =
            self.collect_local_repair_seed_simplices(vertex_key, extra_seed_simplices);
        let hint_seed = hint.and_then(|ck| {
            if !self.tri.tds.contains_simplex(ck) {
                if env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
                    tracing::debug!(
                        "[repair] insertion seed hint missing (simplex={ck:?}={vertex_key:?})"
                    );
                }
                return None;
            }

            let contains_vertex = self
                .tri
                .tds
                .simplex(ck)
                .is_some_and(|simplex| simplex.contains_vertex(vertex_key));
            if !contains_vertex && env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
                tracing::debug!(
                    "[repair] insertion seed hint does not contain vertex (simplex={ck:?}={vertex_key:?})"
                );
            }

            contains_vertex.then_some(ck)
        });

        let seed_ref = if seed_simplices.is_empty() {
            hint_seed.as_ref().map(std::slice::from_ref)
        } else {
            Some(seed_simplices.as_slice())
        };

        let repair_result = {
            self.invalidate_locate_hint_cache();
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_with_flips_k2_k3_run(tds, kernel, seed_ref, topology, max_flips)
        };

        #[cfg(test)]
        let repair_result = if test_hooks::force_repair_nonconvergent_enabled() {
            Err(test_hooks::synthetic_nonconvergent_error())
        } else {
            repair_result
        };

        match repair_result {
            Ok(run) => {
                self.validate_ridge_links_after_repair(topology, &run)?;
            }
            Err(
                e @ (DelaunayRepairError::NonConvergent { .. }
                | DelaunayRepairError::PostconditionFailed { .. }),
            ) => {
                // Robust fallback: retry with `RobustKernel` which guarantees exact
                // predicate evaluation. This covers 99.9%+ of repair failures.
                //
                // If the robust pass also fails, return an error. Callers that need
                // the full heuristic rebuild (shuffled re-insertion) can invoke
                // `repair_delaunay_with_flips_advanced()` explicitly.
                let robust_run = self
                    .repair_delaunay_with_flips_robust_run(seed_ref, max_flips)
                    .map_err(|robust_err| InsertionError::DelaunayRepairFailed {
                        source: Box::new(robust_err),
                        context: DelaunayRepairFailureContext::LocalRepairRobustFallback {
                            initial: Box::new(e),
                        },
                    })?;
                self.validate_ridge_links_after_repair(topology, &robust_run)?;
            }
            Err(e) => {
                return Err(InsertionError::DelaunayRepairFailed {
                    source: Box::new(e),
                    context: DelaunayRepairFailureContext::LocalRepairNonRecoverable,
                });
            }
        }

        // Flip-based repair mutates simplex orderings; restore canonical positive geometric
        // orientation before exposing the updated triangulation state.
        self.tri.normalize_and_promote_positive_orientation()?;
        self.tri
            .validate_geometric_simplex_orientation()
            .map_err(InsertionError::TopologyValidation)?;
        Ok(())
    }

    /// Validates PL ridge links after a repair pass that actually performed flips.
    ///
    /// Ridge-link topology only changes where flips created replacement simplices,
    /// so validation follows that mutation frontier even if the repair queues
    /// were seeded from the full triangulation.  If a repair reports flips
    /// without a mutation frontier, fall back to a full simplex list defensively.
    fn validate_ridge_links_after_repair(
        &self,
        topology: TopologyGuarantee,
        run: &DelaunayRepairRun,
    ) -> Result<(), InsertionError> {
        if !topology.requires_ridge_links() || run.stats.flips_performed == 0 {
            return Ok(());
        }

        if !run.touched_simplices.is_empty() {
            if run.used_full_reseed && env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
                tracing::debug!(
                    "[repair] validating ridge links on {} flip-created simplices after full reseed",
                    run.touched_simplices.len()
                );
            }
            return validate_ridge_links_for_simplices(
                &self.tri.tds,
                run.touched_simplices.iter().copied(),
            )
            .map_err(ridge_link_repair_validation_error);
        }

        validate_ridge_links_for_simplices(&self.tri.tds, self.tri.tds.simplex_keys())
            .map_err(ridge_link_repair_validation_error)
    }

    /// Merge the inserted vertex star with any simplices that cavity reduction touched and
    /// left in place. Stale simplices are ignored so callers can pass raw cavity-trace sets.
    fn collect_local_repair_seed_simplices(
        &self,
        vertex_key: VertexKey,
        extra_seed_simplices: &[SimplexKey],
    ) -> SimplexKeyBuffer {
        let mut seen: FastHashSet<SimplexKey> = FastHashSet::default();
        let mut seed_simplices = SimplexKeyBuffer::new();

        // Keep the inserted vertex star first because it is the hottest local region and
        // the best chance of fixing ordinary post-insertion violations cheaply.
        for simplex_key in self.tri.adjacent_simplices(vertex_key) {
            if seen.insert(simplex_key) {
                seed_simplices.push(simplex_key);
            }
        }

        // Then widen the frontier with simplices touched by cavity shaping that survived in
        // the triangulation; deduping here lets callers pass raw trace buffers safely.
        for &simplex_key in extra_seed_simplices {
            if self.tri.tds.contains_simplex(simplex_key) && seen.insert(simplex_key) {
                seed_simplices.push(simplex_key);
            }
        }

        seed_simplices
    }

    /// Runs policy-controlled global validation after insertion so expensive
    /// Delaunay checks stay opt-in for incremental workflows.
    pub(crate) fn maybe_check_after_insertion(&self) -> Result<(), InsertionError> {
        if self.tri.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        let policy = self.insertion_state.delaunay_check_policy;
        let insertion_count = self.insertion_state.delaunay_repair_insertion_count;
        if !policy.should_check(insertion_count) {
            return Ok(());
        }

        self.is_valid()
            .map_err(|e| InsertionError::DelaunayValidationFailed { source: e })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::DelaunayRepairStats;
    use crate::core::simplex::Simplex;
    use crate::core::tds::{Tds, TdsError};
    use crate::geometry::kernel::{AdaptiveKernel, RobustKernel};
    use crate::topology::traits::topological_space::GlobalTopology;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::sync::Once;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    struct ForceRepairNonconvergentGuard {
        previous: bool,
    }

    impl ForceRepairNonconvergentGuard {
        fn enable() -> Self {
            Self {
                previous: test_hooks::set_force_repair_nonconvergent(true),
            }
        }
    }

    impl Drop for ForceRepairNonconvergentGuard {
        fn drop(&mut self) {
            let _ = test_hooks::set_force_repair_nonconvergent(self.previous);
        }
    }

    #[test]
    fn test_ridge_link_repair_validation_error_routes_tds_errors_to_tds_layer() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "unit test".to_string(),
        };

        match ridge_link_repair_validation_error(ManifoldError::Tds(tds_err.clone())) {
            InsertionError::TopologyValidation(source) => assert_eq!(source, tds_err),
            other => panic!("expected TopologyValidation, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_repair_validation_error_routes_manifold_errors_to_triangulation_layer() {
        let ridge_key = 0x1234_u64;
        let error = ridge_link_repair_validation_error(ManifoldError::BoundaryRidgeMultiplicity {
            ridge_key,
            boundary_facet_count: 3,
        });

        match error {
            InsertionError::TopologyValidationFailed { context, source } => {
                assert_eq!(context, InsertionTopologyValidationContext::DelaunayRepair);
                assert_matches!(
                    source,
                    TriangulationValidationError::BoundaryRidgeMultiplicity {
                        ridge_key: observed_ridge_key,
                        boundary_facet_count: 3
                    } if observed_ridge_key == ridge_key
                );
            }
            other => panic!("expected TopologyValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_insert_single_interior_point_2d() {
        init_tracing();
        let vertices = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_simplices(), 1);

        let v_key = dt.insert_vertex(vertex![0.3, 0.3].unwrap()).unwrap();

        // Verify insertion succeeded
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_simplices(), 3);

        // Verify the returned key can access the vertex
        assert!(dt.tri.tds.vertex(v_key).is_some());
    }

    #[test]
    fn test_insert_multiple_sequential_points_2d() {
        init_tracing();
        let vertices = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Insert 3 interior points sequentially
        dt.insert_vertex(vertex![0.3, 0.3].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 4);

        dt.insert_vertex(vertex![0.5, 0.2].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert_vertex(vertex![0.2, 0.5].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        // All vertices should be present
        assert!(dt.number_of_simplices() > 1);
    }

    #[test]
    fn test_insert_multiple_sequential_points_3d() {
        init_tracing();
        let vertices = vec![
            vertex![0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0].unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Insert 3 interior points sequentially (well inside the tetrahedron)
        dt.insert_vertex(vertex![0.1, 0.1, 0.1].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert_vertex(vertex![0.15, 0.15, 0.1].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        dt.insert_vertex(vertex![0.1, 0.15, 0.15].unwrap()).unwrap();
        assert_eq!(dt.number_of_vertices(), 7);

        assert!(dt.number_of_simplices() > 1);
    }

    #[test]
    fn test_insert_updates_last_inserted_simplex() {
        init_tracing();
        let vertices = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

        // Initially no last_inserted_simplex
        assert!(dt.insertion_state.last_inserted_simplex.is_none());

        // After insertion, should have a cached simplex
        dt.insert_vertex(vertex![0.3, 0.3].unwrap()).unwrap();
        assert!(dt.insertion_state.last_inserted_simplex.is_some());
    }

    #[test]
    fn test_bootstrap_with_custom_kernel() {
        init_tracing();
        // Verify bootstrap works with RobustKernel
        let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel(RobustKernel::new());

        assert_eq!(dt.number_of_vertices(), 0);

        // Bootstrap with robust predicates
        dt.insert_vertex(vertex![0.0, 0.0, 0.0].unwrap()).unwrap();
        dt.insert_vertex(vertex![1.0, 0.0, 0.0].unwrap()).unwrap();
        dt.insert_vertex(vertex![0.0, 1.0, 0.0].unwrap()).unwrap();
        assert_eq!(dt.number_of_simplices(), 0); // Still bootstrapping

        dt.insert_vertex(vertex![0.0, 0.0, 1.0].unwrap()).unwrap();
        assert_eq!(dt.number_of_simplices(), 1); // Initial simplex created

        assert!(dt.is_valid().is_ok());
    }

    /// When the primary per-insertion repair returns `NonConvergent`, the robust
    /// fallback in `maybe_repair_after_insertion` should rescue the insertion.
    #[test]
    fn test_maybe_repair_after_insertion_robust_fallback_on_forced_nonconvergent() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let _guard = ForceRepairNonconvergentGuard::enable();
        let result = dt.insert_vertex(vertex![0.5, 0.5].unwrap());
        let inserted_key = result
            .as_ref()
            .copied()
            .expect("Insertion should succeed via robust fallback");
        assert!(
            result.is_ok(),
            "Insertion should succeed via robust fallback: {result:?}"
        );
        let spatial_index = dt
            .spatial_index
            .as_ref()
            .expect("topology-only repair should preserve the duplicate-detection index");
        let mut found_inserted_key = false;
        assert!(
            spatial_index.for_each_candidate_vertex_key(&[0.5, 0.5], |candidate| {
                found_inserted_key |= candidate == inserted_key;
                true
            })
        );
        assert!(found_inserted_key);
        assert!(dt.validate().is_ok());
    }

    fn wedge_two_spheres_share_vertex_tds_2d() -> (Tds<(), (), 2>, SimplexKey, SimplexKey) {
        // Two closed 2D spheres (boundaries of tetrahedra) sharing one vertex are
        // pseudomanifold but not PL-manifold: the shared vertex has a disconnected link.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex![0.0, 0.0].unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex![1.0, 0.0].unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex![0.0, 1.0].unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex![1.0, 1.0].unwrap())
            .unwrap();

        let incident = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let nonincident = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let v4 = tds
            .insert_vertex_with_mapping(vertex![10.0, 10.0].unwrap())
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex![11.0, 10.0].unwrap())
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex![10.0, 11.0].unwrap())
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v5], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v6], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v5, v6], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v4, v5, v6], None).unwrap(),
            )
            .unwrap();

        (tds, incident, nonincident)
    }

    #[test]
    fn test_collect_local_repair_seed_simplices_merges_adjacent_extra_and_ignores_stale() {
        let vertices = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![1.0, 1.0].unwrap(),
            vertex![0.5, 0.5].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let all_simplices: Vec<SimplexKey> =
            dt.simplices().map(|(simplex_key, _)| simplex_key).collect();

        let (vertex_key, adjacent, extra_simplex) = dt
            .vertices()
            .find_map(|(vertex_key, _)| {
                let adjacent: Vec<SimplexKey> = dt.tri.adjacent_simplices(vertex_key).collect();
                all_simplices
                    .iter()
                    .copied()
                    .find(|simplex_key| !adjacent.contains(simplex_key))
                    .map(|extra_simplex| (vertex_key, adjacent, extra_simplex))
            })
            .expect("fixture should contain a simplex outside at least one vertex star");

        let stale_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let seeds = dt.collect_local_repair_seed_simplices(
            vertex_key,
            &[adjacent[0], extra_simplex, extra_simplex, stale_simplex],
        );

        assert_eq!(seeds.len(), adjacent.len() + 1);
        assert_eq!(&seeds[..adjacent.len()], adjacent.as_slice());
        assert_eq!(seeds[adjacent.len()], extra_simplex);
        assert!(!seeds.contains(&stale_simplex));
    }

    #[test]
    fn test_validate_ridge_links_after_full_reseed_repair_uses_mutation_frontier() {
        init_tracing();
        let (tds, incident_to_invalid_ridge, nonincident) = wedge_two_spheres_share_vertex_tds_2d();
        let dt = DelaunayTriangulationCandidate::assemble(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::PLManifold,
            GlobalTopology::DEFAULT,
        )
        .into_repairable_delaunay_for_test();
        let stats = DelaunayRepairStats {
            flips_performed: 1,
            ..DelaunayRepairStats::default()
        };

        let local_run = DelaunayRepairRun {
            stats: stats.clone(),
            touched_simplices: std::iter::once(nonincident).collect(),
            used_full_reseed: true,
        };
        assert!(
            dt.validate_ridge_links_after_repair(TopologyGuarantee::PLManifold, &local_run)
                .is_ok()
        );

        let invalid_scope_run = DelaunayRepairRun {
            stats,
            touched_simplices: std::iter::once(incident_to_invalid_ridge).collect(),
            used_full_reseed: true,
        };
        assert!(
            dt.validate_ridge_links_after_repair(
                TopologyGuarantee::PLManifold,
                &invalid_scope_run,
            )
            .is_err()
        );
    }
}
