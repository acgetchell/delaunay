//! Incremental insertion and vertex-removal operations for Delaunay triangulations.
//!
//! This module owns post-construction mutation APIs: inserting vertices, removing
//! vertices, maintaining insertion caches, and running policy-controlled local
//! Delaunay repair after those mutations.
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
    DelaunayRepairError, DelaunayRepairRun, FlipError, apply_bistellar_flip_k1_inverse,
    repair_delaunay_with_flips_k2_k3, repair_delaunay_with_flips_k2_k3_run,
};
use crate::core::algorithms::incremental_insertion::{
    DelaunayRepairFailureContext, InsertionError, InsertionTopologyValidationContext,
};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{FastHashSet, SimplexKeyBuffer};
use crate::core::operations::{InsertionOutcome, InsertionStatistics};
use crate::core::tds::{InvariantError, NeighborValidationError, SimplexKey, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::validation::{
    TopologyGuarantee, TriangulationValidationError, insertion_error_to_invariant_error,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::repair::{DelaunayRepairOperation, DelaunayRepairPolicy};
use crate::topology::manifold::{ManifoldError, validate_ridge_links_for_simplices};
use crate::triangulation::DelaunayTriangulation;
#[cfg(test)]
use crate::validation::DelaunayTriangulationCandidate;
use crate::validation::DelaunayTriangulationValidationError;
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
// Incremental insertion, removal, and post-insertion repair/check helpers.
// These require an f64-backed kernel for spatial-index construction,
// Triangulation-layer insertion, and Triangulation-layer removal.

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
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulation};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_simplices(), 0);
    ///
    /// // Insert vertices one by one - bootstrap phase (no simplices yet)
    /// dt.insert(delaunay::vertex![0.0, 0.0, 0.0]?)?;
    /// dt.insert(delaunay::vertex![1.0, 0.0, 0.0]?)?;
    /// dt.insert(delaunay::vertex![0.0, 1.0, 0.0]?)?;
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 0); // Still no simplices
    ///
    /// // 4th vertex triggers initial simplex creation
    /// dt.insert(delaunay::vertex![0.0, 0.0, 1.0]?)?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_simplices(), 1); // First simplex created!
    ///
    /// // Further insertions use cavity-based algorithm
    /// dt.insert(delaunay::vertex![0.2, 0.2, 0.2]?)?;
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// assert!(dt.number_of_simplices() > 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Using batch construction (traditional approach):
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Create initial triangulation with 5 vertices (4-simplex)
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 5);
    ///
    /// // Insert additional interior vertex
    /// dt.insert(delaunay::vertex![0.2, 0.2, 0.2, 0.2]?)?;
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// assert!(dt.number_of_simplices() > 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert(&mut self, vertex: Vertex<U, D>) -> Result<VertexKey, InsertionError> {
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
        // - Future: Delaunay property restoration after removal
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

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation delegates to the core triangulation layer, which:
    /// 1. Finds all simplices containing the vertex
    /// 2. Removes those simplices (creating a cavity)
    /// 3. Fills the cavity with fan triangulation
    /// 4. Wires neighbors and rebuilds vertex-simplex incidence
    /// 5. Removes the vertex
    ///
    /// Fast-path: if the vertex star is a simplex (exactly D+1 incident simplices with
    /// consistent adjacency), this method collapses it via the **inverse k=1** bistellar
    /// flip. Otherwise it falls back to fan triangulation.
    ///
    /// This operation is topology-preserving on success: it returns `Ok` only after the
    /// post-removal triangulation satisfies the required manifold and topology invariants. A
    /// candidate removal that would collapse the mesh to a lower-dimensional remnant or isolate
    /// remaining vertices is rejected as an [`InvariantError::Triangulation`] failure, and the
    /// pre-removal state is restored. Both the inverse k=1 fast-path and fan triangulation may
    /// temporarily violate the Delaunay property in some cases. If the [`DelaunayRepairPolicy`]
    /// allows it, a flip-based repair pass is run automatically after removal.
    ///
    /// The post-removal repair and orientation canonicalization steps are
    /// transactional: if either step fails, this method restores the triangulation
    /// and insertion state to their pre-removal state before returning the error.
    /// The spatial index is retained across rollback because its keys are
    /// validated against the live TDS before use. On successful removal,
    /// topology-dependent locate hints are invalidated and the removed vertex key
    /// is pruned from the spatial index.
    ///
    /// **Future Enhancement**: Delaunay-aware cavity retriangulation will be added for
    /// removals. For now, occasional Delaunay violations after removal are expected and
    /// can be addressed by running flip-based repair (e.g., [`repair_delaunay_with_flips`](Self::repair_delaunay_with_flips))
    /// or by leaving automatic repair enabled via [`DelaunayRepairPolicy`].
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - Key of the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of simplices that were removed along with the vertex. Returns `Ok(0)` if
    /// `vertex_key` does not refer to a vertex in the triangulation (e.g. a stale key from
    /// a previously removed vertex or a key that was never inserted). This is a successful
    /// no-op, not an error.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] if:
    /// - The inverse k=1 flip encounters a neighbor-wiring failure ([`InvariantError::Tds`]).
    /// - Fan retriangulation fails ([`InvariantError::Tds`]).
    /// - Post-removal topology validation fails, for example because removal would leave
    ///   isolated vertices or a lower-dimensional remnant
    ///   ([`InvariantError::Triangulation`]).
    /// - Delaunay flip-based repair fails after removal
    ///   ([`InvariantError::Delaunay`] wrapping
    ///   [`DelaunayTriangulationValidationError::RepairOperationFailed`]).
    /// - Orientation canonicalization fails after repair ([`InvariantError::Tds`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Invariant(#[from] delaunay::tds::InvariantError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let interior = delaunay::vertex![0.3, 0.3]?;
    /// let interior_uuid = interior.uuid();
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    ///     interior,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Find the key of a known interior vertex.
    /// let Some((vertex_key, _)) = dt.vertices().find(|(_, v)| v.uuid() == interior_uuid) else {
    ///     return Ok(());
    /// };
    ///
    /// // Remove the vertex and all simplices containing it
    /// let simplices_removed = dt.remove_vertex(vertex_key)?;
    /// println!("Removed {} simplices along with the vertex", simplices_removed);
    ///
    /// // Vertex removal preserves topology; automatic repair is attempted when enabled.
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Removals that would leave a non-manifold remnant fail and roll back:
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    /// use delaunay::prelude::tds::InvariantError;
    /// use delaunay::prelude::triangulation::TriangulationValidationError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((vertex_key, _)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let err = dt
    ///     .remove_vertex(vertex_key)
    ///     .expect_err("removal should leave an isolated vertex");
    /// std::assert_matches!(
    ///     err,
    ///     InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex { .. })
    /// );
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_vertex(&mut self, vertex_key: VertexKey) -> Result<usize, InvariantError> {
        let Some(removed_vertex) = self.tri.tds.vertex(vertex_key) else {
            return Ok(0);
        };
        let removed_vertex_coords = *removed_vertex.point().coords();
        let snapshot = (self.tri.tds.clone_for_rollback(), self.insertion_state);

        let result = (|| {
            // Fast path: inverse k=1 flip when the vertex star is a simplex.
            let mut seed_simplices: Option<SimplexKeyBuffer> = None;
            let simplices_removed =
                match apply_bistellar_flip_k1_inverse(&mut self.tri.tds, vertex_key) {
                    Ok(info) => {
                        seed_simplices = Some(info.new_simplices);
                        info.removed_simplices.len()
                    }
                    Err(FlipError::NeighborWiring { reason }) => {
                        return Err(TdsError::InvalidNeighbors {
                            reason: NeighborValidationError::FlipNeighborWiring { reason },
                        }
                        .into());
                    }
                    Err(_) => self.tri.remove_vertex(vertex_key)?,
                };

            let topology = self.tri.topology_guarantee();
            if self.should_run_delaunay_repair_after_mutation(topology) {
                let seed_ref = seed_simplices.as_deref();
                let repair_result = {
                    self.invalidate_locate_hint_cache();
                    let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                    repair_delaunay_with_flips_k2_k3(tds, kernel, seed_ref, topology, None)
                };

                #[cfg(test)]
                let repair_result = if test_hooks::force_repair_nonconvergent_enabled() {
                    Err(test_hooks::synthetic_nonconvergent_error())
                } else {
                    repair_result
                };

                repair_result.map_err(|source| {
                    InvariantError::Delaunay(
                        DelaunayTriangulationValidationError::RepairOperationFailed {
                            operation: DelaunayRepairOperation::VertexRemoval,
                            source: Box::new(source),
                        },
                    )
                })?;

                // Re-canonicalize geometric orientation (#258): flip repair may leave
                // the global sign negative.
                self.tri
                    .normalize_and_promote_positive_orientation()
                    .map_err(|e| {
                        insertion_error_to_invariant_error(
                            e,
                            "Orientation canonicalization failed after vertex removal",
                        )
                    })?;
            }

            Ok(simplices_removed)
        })();

        match result {
            Ok(simplices_removed) => {
                self.insertion_state.last_inserted_simplex = None;
                if let Some(index) = self.spatial_index.as_mut() {
                    index.remove_vertex(&vertex_key, &removed_vertex_coords);
                }
                Ok(simplices_removed)
            }
            Err(err) => {
                let (tds, insertion_state) = snapshot;
                self.tri.tds = tds;
                self.insertion_state = insertion_state;
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::DelaunayRepairStats;
    use crate::core::simplex::Simplex;
    use crate::core::tds::Tds;
    use crate::flips::BistellarFlips;
    use crate::geometry::kernel::{AdaptiveKernel, RobustKernel};
    use crate::geometry::util::safe_usize_to_scalar;
    use crate::topology::traits::topological_space::GlobalTopology;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::sync::Once;
    use uuid::Uuid;

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

    fn simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(crate::core::vertex::Vertex::<(), _>::try_new([0.0; D]).unwrap());
        for axis in 0..D {
            let mut coords = [0.0; D];
            coords[axis] = 1.0;
            vertices.push(crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap());
        }
        vertices
    }

    fn interior_vertex_for_k1_insert<const D: usize>() -> Vertex<(), D> {
        let denominator = safe_usize_to_scalar(D + 2)
            .expect("D + 2 should convert exactly for rollback test dimensions");
        let coord = 1.0 / denominator;
        crate::core::vertex::Vertex::<(), _>::try_new([coord; D]).unwrap()
    }

    fn rollback_probe_vertex<const D: usize>(point_index: usize) -> Vertex<(), D> {
        let dimension = safe_usize_to_scalar(D).expect("test dimensions should convert exactly");
        let point_index_scalar =
            safe_usize_to_scalar(point_index).expect("point index should convert exactly");
        let mut coords = [0.2 / dimension; D];
        let axis = point_index % D;
        coords[axis] += point_index_scalar.mul_add(0.005, 0.02);
        crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap()
    }

    fn incident_simplex_count<const D: usize>(
        dt: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
        vertex_key: VertexKey,
    ) -> usize {
        dt.simplices()
            .filter(|(_, simplex)| simplex.vertices().contains(&vertex_key))
            .count()
    }

    fn assert_forced_remove_vertex_rolls_back<const D: usize>(
        dt: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
        vertex_key: VertexKey,
        inserted_uuid: Uuid,
    ) {
        let vertex_count_before = dt.number_of_vertices();
        let simplex_count_before = dt.number_of_simplices();
        let hint_simplex_before = dt.simplices().next().map(|(key, _)| key);
        dt.insertion_state.last_inserted_simplex = hint_simplex_before;
        let mut spatial_index = HashGridIndex::<D>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);
        let last_inserted_simplex_before = dt.insertion_state.last_inserted_simplex;
        let spatial_index_before = dt
            .spatial_index
            .as_ref()
            .map(HashGridIndex::<D>::debug_snapshot);

        let _guard = ForceRepairNonconvergentGuard::enable();
        let result = dt.remove_vertex(vertex_key);
        let err = result.expect_err("forced repair failure should make removal fail");
        match err {
            InvariantError::Delaunay(
                DelaunayTriangulationValidationError::RepairOperationFailed {
                    operation: DelaunayRepairOperation::VertexRemoval,
                    source,
                },
            ) if matches!(
                source.as_ref(),
                DelaunayRepairError::NonConvergent { max_flips: 0, .. }
            ) => {}
            InvariantError::Triangulation(
                TriangulationValidationError::OrientationPromotionNonConvergence { .. },
            )
            | InvariantError::Tds(TdsError::FacetSharingViolation { .. }) => {}
            other => panic!(
                "expected vertex-removal rollback error from forced repair path, got {other:?}"
            ),
        }

        assert_eq!(dt.number_of_vertices(), vertex_count_before);
        assert_eq!(dt.number_of_simplices(), simplex_count_before);
        assert_eq!(
            dt.insertion_state.last_inserted_simplex, last_inserted_simplex_before,
            "remove_vertex rollback should restore last_inserted_simplex"
        );
        assert_eq!(
            dt.spatial_index
                .as_ref()
                .map(HashGridIndex::<D>::debug_snapshot),
            spatial_index_before,
            "remove_vertex rollback should restore spatial_index"
        );
        assert!(dt.vertices().any(|(_, v)| v.uuid() == inserted_uuid));
        assert!(dt.as_triangulation().validate().is_ok());
    }

    fn assert_remove_vertex_rollback<const D: usize>() {
        init_tracing();
        let vertices = simplex_vertices::<D>();

        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let simplex_key = dt.simplices().next().unwrap().0;
        let inserted_vertex = interior_vertex_for_k1_insert::<D>();
        let inserted_uuid = inserted_vertex.uuid();
        dt.flip_k1_insert(simplex_key, inserted_vertex).unwrap();

        let vertex_key = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(k, _)| k)
            .expect("Inserted vertex not found");

        assert_forced_remove_vertex_rolls_back(&mut dt, vertex_key, inserted_uuid);
    }

    fn assert_remove_vertex_fallback_rollback<const D: usize>() {
        init_tracing();
        let vertices = simplex_vertices::<D>();

        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let mut inserted_vertices = Vec::new();
        for point_index in 0..(D + 3) {
            let inserted_vertex = rollback_probe_vertex::<D>(point_index);
            let inserted_uuid = inserted_vertex.uuid();
            let vertex_key = dt
                .insert(inserted_vertex)
                .expect("rollback fallback fixture insertion should succeed");
            inserted_vertices.push((vertex_key, inserted_uuid));
        }

        let (vertex_key, inserted_uuid, incident_simplices) = inserted_vertices
            .iter()
            .find_map(|&(vertex_key, inserted_uuid)| {
                let incident_simplices = incident_simplex_count(&dt, vertex_key);
                (incident_simplices != D + 1).then_some((
                    vertex_key,
                    inserted_uuid,
                    incident_simplices,
                ))
            })
            .expect("expected at least one inserted vertex with a non-simplex star");
        assert_ne!(
            incident_simplices,
            D + 1,
            "fallback rollback fixture must avoid the inverse-k=1 simplex-star path"
        );

        assert_forced_remove_vertex_rolls_back(&mut dt, vertex_key, inserted_uuid);
    }

    macro_rules! gen_remove_vertex_rollback_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<remove_vertex_rollback_ $dim d>]() {
                    assert_remove_vertex_rollback::<$dim>();
                }

                #[test]
                fn [<remove_vertex_fallback_rollback_ $dim d>]() {
                    assert_remove_vertex_fallback_rollback::<$dim>();
                }
            }
        };
    }

    gen_remove_vertex_rollback_tests!(2);
    gen_remove_vertex_rollback_tests!(3);
    gen_remove_vertex_rollback_tests!(4);
    gen_remove_vertex_rollback_tests!(5);

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
    fn test_remove_vertex_fast_path_inverse_k1() {
        init_tracing();
        let vertices: Vec<Vertex<(), 3>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);
        let original_vertex_count = dt.number_of_vertices();
        let original_simplex_count = dt.number_of_simplices();

        let simplex_key = dt.simplices().next().unwrap().0;
        let inserted_vertex =
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2]).unwrap();
        let inserted_uuid = inserted_vertex.uuid();
        dt.flip_k1_insert(simplex_key, inserted_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), original_vertex_count + 1);
        assert_eq!(dt.number_of_simplices(), original_simplex_count + 3);

        let vertex_key = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(k, _)| k)
            .expect("Inserted vertex not found");

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_simplices = dt.remove_vertex(vertex_key).unwrap();

        assert_eq!(removed_simplices, 4);
        assert_eq!(dt.number_of_vertices(), original_vertex_count);
        assert_eq!(dt.number_of_simplices(), original_simplex_count);
        assert!(dt.as_triangulation().validate().is_ok());
        assert!(dt.vertices().all(|(_, v)| v.uuid() != inserted_uuid));
    }

    #[test]
    fn remove_vertex_invalidates_locate_hint_and_prunes_spatial_index() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let vertex_key = dt
            .insert(crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25]).unwrap())
            .unwrap();
        let hint_simplex = dt.simplices().next().map(|(key, _)| key);
        dt.insertion_state.last_inserted_simplex = hint_simplex;
        let mut spatial_index = HashGridIndex::<2>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);
        assert!(dt.insertion_state.last_inserted_simplex.is_some());
        assert!(dt.spatial_index.is_some());

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_simplices = dt.remove_vertex(vertex_key).unwrap();

        assert!(removed_simplices > 0);
        assert!(dt.insertion_state.last_inserted_simplex.is_none());
        let spatial_index = dt
            .spatial_index
            .as_ref()
            .expect("successful vertex removal should retain the spatial index");
        let mut found_removed_key = false;
        assert!(
            spatial_index.for_each_candidate_vertex_key(&[0.25, 0.25], |candidate| {
                found_removed_key |= candidate == vertex_key;
                true
            })
        );
        assert!(!found_removed_key);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn test_insert_single_interior_point_2d() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_simplices(), 1);

        let v_key = dt
            .insert(crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3]).unwrap())
            .unwrap();

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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Insert 3 interior points sequentially
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 4);

        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.2]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.5]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        // All vertices should be present
        assert!(dt.number_of_simplices() > 1);
    }

    #[test]
    fn test_insert_multiple_sequential_points_3d() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Insert 3 interior points sequentially (well inside the tetrahedron)
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.1, 0.1, 0.1]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.15, 0.15, 0.1]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.1, 0.15, 0.15]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 7);

        assert!(dt.number_of_simplices() > 1);
    }

    #[test]
    fn test_insert_updates_last_inserted_simplex() {
        init_tracing();
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

        // Initially no last_inserted_simplex
        assert!(dt.insertion_state.last_inserted_simplex.is_none());

        // After insertion, should have a cached simplex
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3]).unwrap())
            .unwrap();
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
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_simplices(), 0); // Still bootstrapping

        dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        assert_eq!(dt.number_of_simplices(), 1); // Initial simplex created

        assert!(dt.is_valid().is_ok());
    }

    /// When the primary per-insertion repair returns `NonConvergent`, the robust
    /// fallback in `maybe_repair_after_insertion` should rescue the insertion.
    #[test]
    fn test_maybe_repair_after_insertion_robust_fallback_on_forced_nonconvergent() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let _guard = ForceRepairNonconvergentGuard::enable();
        let result = dt.insert(crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap());
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
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
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 10.0]).unwrap(),
            )
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([11.0, 10.0]).unwrap(),
            )
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 11.0]).unwrap(),
            )
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
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
