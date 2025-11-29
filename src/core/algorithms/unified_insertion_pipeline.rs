use crate::core::algorithms::{
    bowyer_watson::IncrementalBowyerWatson, robust_bowyer_watson::RobustBowyerWatson,
};
use crate::core::collections::{FastHashSet, fast_hash_set_with_capacity};
use crate::core::traits::data_type::DataType;
use crate::core::traits::insertion_algorithm::{
    DelaunayCheckPolicy, InitialSimplexSearchStats, InsertionAlgorithm, InsertionError,
    InsertionInfo, InsertionStatistics, UnifiedPerVertexInsertionOutcome,
    UnsalvageableVertexReport, VertexClassification, find_initial_simplex,
    unified_insert_vertex_fast_robust_or_skip,
};
use crate::core::triangulation_data_structure::{
    Tds, TriangulationConstructionError, TriangulationConstructionState, TriangulationStatistics,
};
use crate::core::vertex::Vertex;
use crate::geometry::traits::coordinate::CoordinateScalar;
use num_traits::NumCast;
use std::iter::Sum;
use std::ops::{AddAssign, SubAssign};

/// Internal unified insertion pipeline that orchestrates fast → robust → skip per vertex.
///
/// This type is intentionally `pub(crate)` and not exposed as part of the public API. It
/// owns both the fast incremental algorithm and the robust algorithm and delegates
/// per-vertex insertion decisions to the Stage 2 helper
/// `unified_insert_vertex_fast_robust_or_skip`.
pub(crate) struct UnifiedInsertionPipeline<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    pub(crate) fast: IncrementalBowyerWatson<T, U, V, D>,
    pub(crate) robust: RobustBowyerWatson<T, U, V, D>,
    pub(crate) unsalvageable_reports: Vec<UnsalvageableVertexReport<T, U, D>>,
    pub(crate) delaunay_check_policy: DelaunayCheckPolicy,
    /// Total number of successful insertions (fast + robust). Used to drive
    /// [`DelaunayCheckPolicy`] scheduling.
    pub(crate) successful_insertions: usize,
    /// Number of vertices for which the fast path was attempted.
    pub(crate) fast_path_attempts: usize,
    /// Number of vertices for which the robust path was attempted.
    pub(crate) robust_path_attempts: usize,
    /// Number of vertices that were successfully inserted via the fast path.
    pub(crate) successful_fast_insertions: usize,
    /// Number of vertices that were successfully inserted via the robust path.
    pub(crate) successful_robust_insertions: usize,
    /// Number of times global Delaunay validation was invoked for this pipeline.
    pub(crate) global_delaunay_validation_runs: usize,
    /// Stage 1 initial simplex search statistics for the last triangulation run.
    pub(crate) initial_simplex_stats: Option<InitialSimplexSearchStats>,
}

impl<T, U, V, const D: usize> UnifiedInsertionPipeline<T, U, V, D>
where
    T: CoordinateScalar + NumCast + AddAssign<T> + SubAssign<T> + Sum,
    U: DataType,
    V: DataType,
{
    /// Create a new pipeline instance with an explicit global Delaunay
    /// validation policy.
    #[must_use]
    pub(crate) fn with_policy(policy: DelaunayCheckPolicy) -> Self {
        Self {
            fast: IncrementalBowyerWatson::new(),
            // Use the degenerate-robust configuration for the robust side of the pipeline,
            // since reaching it indicates that the fast path encountered a difficult case.
            robust: RobustBowyerWatson::for_degenerate_cases(),
            unsalvageable_reports: Vec::new(),
            delaunay_check_policy: policy,
            successful_insertions: 0,
            fast_path_attempts: 0,
            robust_path_attempts: 0,
            successful_fast_insertions: 0,
            successful_robust_insertions: 0,
            global_delaunay_validation_runs: 0,
            initial_simplex_stats: None,
        }
    }

    // Aggregate statistics from the fast and robust algorithms together with
    // pipeline-level counters into a single [`TriangulationStatistics`] value.
    pub(crate) fn unified_statistics(&self) -> TriangulationStatistics {
        // Aggregate per-algorithm statistics.
        let fast_stats = self.fast.statistics();
        let robust_stats = self.robust.statistics();

        let mut insertion = InsertionStatistics::new();
        insertion.vertices_processed =
            fast_stats.vertices_processed + robust_stats.vertices_processed;
        insertion.total_cells_created =
            fast_stats.total_cells_created + robust_stats.total_cells_created;
        insertion.total_cells_removed =
            fast_stats.total_cells_removed + robust_stats.total_cells_removed;
        insertion.fallback_strategies_used =
            fast_stats.fallback_strategies_used + robust_stats.fallback_strategies_used;
        insertion.skipped_vertices = fast_stats.skipped_vertices + robust_stats.skipped_vertices;
        insertion.degenerate_cases_handled =
            fast_stats.degenerate_cases_handled + robust_stats.degenerate_cases_handled;
        insertion.cavity_boundary_failures =
            fast_stats.cavity_boundary_failures + robust_stats.cavity_boundary_failures;
        insertion.cavity_boundary_recoveries =
            fast_stats.cavity_boundary_recoveries + robust_stats.cavity_boundary_recoveries;
        insertion.hull_extensions = fast_stats.hull_extensions + robust_stats.hull_extensions;
        insertion.vertex_perturbations =
            fast_stats.vertex_perturbations + robust_stats.vertex_perturbations;

        // Derive duplicate vs unsalvageable counts from per-vertex reports.
        let mut duplicate_vertices = 0usize;
        let mut unsalvageable_vertices = 0usize;
        for report in &self.unsalvageable_reports {
            match report.classification() {
                VertexClassification::DuplicateExact
                | VertexClassification::DuplicateWithinTolerance { .. } => {
                    duplicate_vertices += 1;
                }
                _ => {
                    unsalvageable_vertices += 1;
                }
            }
        }

        let skipped_vertices = self.unsalvageable_reports.len();

        TriangulationStatistics {
            insertion,
            fast_path_attempts: self.fast_path_attempts,
            robust_path_attempts: self.robust_path_attempts,
            fast_path_successes: self.successful_fast_insertions,
            robust_path_successes: self.successful_robust_insertions,
            skipped_vertices,
            duplicate_vertices,
            unsalvageable_vertices,
            global_delaunay_validation_runs: self.global_delaunay_validation_runs,
            initial_simplex: self.initial_simplex_stats.clone().unwrap_or_default(),
        }
    }

    /// Consume and return all unsalvageable vertex reports accumulated so far.
    #[must_use]
    pub(crate) fn take_unsalvageable_reports(&mut self) -> Vec<UnsalvageableVertexReport<T, U, D>> {
        std::mem::take(&mut self.unsalvageable_reports)
    }

    /// Called after each successful vertex insertion (fast or robust path).
    ///
    /// For `DelaunayCheckPolicy::EveryN(k)`, this runs a global Delaunay
    /// validation every k successful insertions by delegating to the shared
    /// policy-aware validator on the `InsertionAlgorithm` trait.
    pub(crate) fn on_successful_insertion(
        &mut self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<(), InsertionError> {
        self.successful_insertions = self.successful_insertions.saturating_add(1);

        if let DelaunayCheckPolicy::EveryN(k) = self.delaunay_check_policy
            && self.successful_insertions.is_multiple_of(k.get())
        {
            <Self as InsertionAlgorithm<T, U, V, D>>::run_global_delaunay_validation_with_policy(
                tds,
                self.delaunay_check_policy,
            )
            .map_err(InsertionError::TriangulationConstruction)?;

            self.global_delaunay_validation_runs =
                self.global_delaunay_validation_runs.saturating_add(1);
        }

        Ok(())
    }
}

impl<T, U, V, const D: usize> InsertionAlgorithm<T, U, V, D>
    for UnifiedInsertionPipeline<T, U, V, D>
where
    T: CoordinateScalar + NumCast + AddAssign<T> + SubAssign<T> + Sum,
    U: DataType,
    V: DataType,
{
    fn insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError> {
        // Bypass the default duplicate detection in `InsertionAlgorithm::insert_vertex`.
        // The unified Stage 2 helper performs its own classification and records
        // duplicate and unsalvageable vertices in `unsalvageable_reports`.
        self.insert_vertex_impl(tds, vertex)
    }

    fn insert_vertex_impl(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError> {
        let outcome = unified_insert_vertex_fast_robust_or_skip(
            tds,
            &mut self.fast,
            &mut self.robust,
            vertex,
        );

        match outcome {
            UnifiedPerVertexInsertionOutcome::FastSuccess {
                classification,
                info,
            } => {
                // Fast path was attempted once and succeeded.
                self.fast_path_attempts = self.fast_path_attempts.saturating_add(1);
                debug_assert!(
                    matches!(
                        classification,
                        VertexClassification::Unique
                            | VertexClassification::DegenerateCollinear
                            | VertexClassification::DegenerateCoplanar
                            | VertexClassification::DegenerateOther
                    ),
                    "Unexpected vertex classification for fast-path success: {classification:?}",
                );
                if info.success {
                    self.successful_fast_insertions =
                        self.successful_fast_insertions.saturating_add(1);
                    self.on_successful_insertion(tds)?;
                }
                Ok(info)
            }
            UnifiedPerVertexInsertionOutcome::RobustSuccess {
                classification,
                fast_error,
                info,
            } => {
                // Both fast and robust paths were attempted; robust succeeded.
                self.fast_path_attempts = self.fast_path_attempts.saturating_add(1);
                self.robust_path_attempts = self.robust_path_attempts.saturating_add(1);
                debug_assert!(
                    matches!(
                        classification,
                        VertexClassification::Unique
                            | VertexClassification::DegenerateCollinear
                            | VertexClassification::DegenerateCoplanar
                            | VertexClassification::DegenerateOther
                    ),
                    "Unexpected vertex classification for robust-path success: {classification:?}",
                );
                debug_assert!(
                    fast_error.is_some(),
                    "RobustSuccess is expected to follow a recoverable fast-path error",
                );
                if info.success {
                    self.successful_robust_insertions =
                        self.successful_robust_insertions.saturating_add(1);
                    self.on_successful_insertion(tds)?;
                }
                Ok(info)
            }
            UnifiedPerVertexInsertionOutcome::Skipped(report) => {
                // For skipped vertices, count fast/robust attempts based on the
                // recorded strategy chain. Duplicate classifications never attempt
                // either algorithm.
                match report.classification() {
                    VertexClassification::DuplicateExact
                    | VertexClassification::DuplicateWithinTolerance { .. } => {}
                    _ => {
                        let strategies = report.attempted_strategies();
                        if !strategies.is_empty() {
                            self.fast_path_attempts = self.fast_path_attempts.saturating_add(1);
                        }
                        if strategies.len() >= 2 {
                            self.robust_path_attempts = self.robust_path_attempts.saturating_add(1);
                        }
                    }
                }

                // Per-vertex failure: record diagnostics and treat as a soft failure.
                // The underlying algorithms must guarantee transactional semantics on
                // error, so the TDS should be unchanged here.
                self.unsalvageable_reports.push(report);

                Ok(InsertionInfo {
                    strategy: crate::core::traits::insertion_algorithm::InsertionStrategy::Skip,
                    cells_removed: 0,
                    cells_created: 0,
                    success: false,
                    degenerate_case_handled: true,
                })
            }
        }
    }

    fn triangulate(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertices: &[Vertex<T, U, D>],
    ) -> Result<(), TriangulationConstructionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
    {
        if vertices.is_empty() {
            return Ok(());
        }

        if vertices.len() < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Stage 1: robust initial simplex search with duplicate/degenerate handling.
        let search_result = find_initial_simplex::<T, U, D>(vertices);
        self.initial_simplex_stats = Some(search_result.stats.clone());

        if let Some(initial_simplex_vertices) = search_result.simplex_vertices {
            // Successful initial simplex: create the initial cell set.
            <Self as InsertionAlgorithm<T, U, V, D>>::create_initial_simplex(
                tds,
                initial_simplex_vertices.clone(),
            )?;

            // Update statistics for initial simplex creation (one cell, no removals).
            self.update_statistics(1, 0);

            // Build a UUID set so we can avoid re-inserting initial simplex vertices.
            let mut simplex_uuids: FastHashSet<uuid::Uuid> =
                fast_hash_set_with_capacity(initial_simplex_vertices.len());
            for v in &initial_simplex_vertices {
                simplex_uuids.insert(v.uuid());
            }

            // Stage 2: Insert remaining vertices incrementally using the unified
            // fast 9 robust 9 skip pipeline.
            for vertex in vertices {
                if simplex_uuids.contains(&vertex.uuid()) {
                    continue;
                }

                self.insert_vertex(tds, *vertex).map_err(|e| match e {
                    InsertionError::TriangulationConstruction(tc_err) => tc_err,
                    other => TriangulationConstructionError::FailedToAddVertex {
                        message: format!("Vertex insertion failed during triangulation: {other}",),
                    },
                })?;
            }

            // Stage 3: Structural finalization (duplicates, facet sharing,
            // neighbors, incident cells).
            <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds)
                .map_err(TriangulationConstructionError::ValidationError)?;

            // Stage 4: Global Delaunay repair/validation using the robust
            // algorithm. This ensures the final triangulation satisfies the
            // empty circumsphere property even for vertices inserted via
            // robust fallback paths.
            self.robust.repair_global_delaunay_violations(tds)?;

            // Final global validation according to the configured policy. This
            // increments the test-only GLOBAL_DELAUNAY_VALIDATION_CALLS counter
            // and ensures no residual Delaunay violations remain after the
            // robust repair pass.
            <Self as InsertionAlgorithm<T, U, V, D>>::run_global_delaunay_validation_with_policy(
                tds,
                self.delaunay_check_policy,
            )?;
            self.global_delaunay_validation_runs =
                self.global_delaunay_validation_runs.saturating_add(1);

            Ok(())
        } else {
            // No non-degenerate simplex could be constructed from the available vertices.
            // If the TDS has no cells yet, populate it with the unique vertex subset to
            // leave a valid zero-cell triangulation that callers can recover from.
            if tds.number_of_cells() == 0 {
                for vertex in &search_result.unique_vertices {
                    if tds.vertex_key_from_uuid(&vertex.uuid()).is_none() {
                        tds.insert_vertex_with_mapping(*vertex).map_err(|e| {
                            TriangulationConstructionError::FailedToAddVertex {
                                message: format!(
                                    "Failed to insert vertex while handling degenerate input: {e}",
                                ),
                            }
                        })?;
                    }
                }

                let vertex_count = tds.number_of_vertices();
                tds.construction_state = TriangulationConstructionState::Incomplete(vertex_count);
            }

            let stats = search_result.stats;
            let message = format!(
                "Could not construct an initial {dim}D simplex from input vertices. \
                 {unique} unique vertices after duplicate filtering, \
                 {dup_exact} exact duplicates skipped, \
                 {dup_near} near-duplicates (within tolerance) skipped, \
                 {degenerate} candidate simplices were degenerate or numerically unstable.",
                dim = D,
                unique = stats.unique_vertices,
                dup_exact = stats.duplicate_exact,
                dup_near = stats.duplicate_within_tolerance,
                degenerate = stats.degenerate_candidates,
            );

            Err(TriangulationConstructionError::GeometricDegeneracy { message })
        }
    }

    fn get_statistics(&self) -> (usize, usize, usize) {
        // Aggregate statistics from both fast and robust algorithms.
        let (fp, fc, fr) = self.fast.get_statistics();
        let (rp, rc, rr) = self.robust.get_statistics();
        (fp + rp, fc + rc, fr + rr)
    }

    fn reset(&mut self) {
        self.fast.reset();
        self.robust.reset();
        self.unsalvageable_reports.clear();
        self.successful_insertions = 0;
        self.fast_path_attempts = 0;
        self.robust_path_attempts = 0;
        self.successful_fast_insertions = 0;
        self.successful_robust_insertions = 0;
        self.global_delaunay_validation_runs = 0;
        self.initial_simplex_stats = None;
        // Policy remains unchanged across resets; callers can construct a new
        // pipeline if they need to change it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::insertion_algorithm::{DelaunayCheckPolicy, InsertionAlgorithm};
    use crate::core::triangulation_data_structure::Tds;
    use crate::core::util::debug_print_first_delaunay_violation;
    use crate::core::vertex::Vertex;
    use crate::vertex;

    fn run_stepwise_unified_insertion_debug<const D: usize>(
        base_vertices: &[Vertex<f64, (), D>],
        new_vertex: Vertex<f64, (), D>,
        context: &str,
    ) {
        let mut tds: Tds<f64, (), (), D> =
            Tds::new(base_vertices).expect("base triangulation should construct");

        assert!(
            tds.is_valid().is_ok(),
            "{context}: base triangulation from {} vertices should be structurally valid",
            base_vertices.len(),
        );

        // At the moment we do *not* assert global Delaunay for the base; if this
        // fails, it indicates the issue lies earlier than the insertion of the
        // new vertex.
        let base_delaunay = tds.validate_delaunay();
        if base_delaunay.is_err() {
            debug_print_first_delaunay_violation(&tds, None);
            panic!(
                "{context}: base triangulation is already non-Delaunay; fix this first before attributing failures to insertion of the new vertex: {base_delaunay:?}",
            );
        }

        // Insert the new vertex via the unified pipeline with EndOnly policy,
        // mirroring the behavior of `Tds::new`'s Bowyer–Watson stage.
        let mut pipeline: UnifiedInsertionPipeline<f64, (), (), D> =
            UnifiedInsertionPipeline::with_policy(DelaunayCheckPolicy::EndOnly);

        let insert_info = <UnifiedInsertionPipeline<f64, (), (), D> as InsertionAlgorithm<
            f64,
            (),
            (),
            D,
        >>::insert_vertex(&mut pipeline, &mut tds, new_vertex);

        // After tightening unified-pipeline semantics, hard geometric failures from
        // the fast path (e.g., stalled cavity refinement) are treated as
        // unsalvageable and cause the vertex to be skipped. We still expect the
        // insertion call itself to succeed at the API level, but it may report a
        // `Skip` strategy with `success = false`.
        let info = insert_info.unwrap_or_else(|_| {
            panic!(
                "{}",
                format!(
                    "{context}: unified pipeline insertion call should succeed at the API level"
                )
            )
        });
        assert!(
            !info.success,
            "{context}: for this canonical 5D configuration the 7th vertex is expected to be skipped, not inserted successfully: {info:?}",
        );
        assert!(
            matches!(
                info.strategy,
                crate::core::traits::insertion_algorithm::InsertionStrategy::Skip
            ),
            "{context}: expected unified pipeline to report a Skip strategy for the unsalvageable 7th vertex, got {info:?}",
        );

        // The triangulation must remain globally Delaunay immediately after the
        // attempted insertion, since the unsalvageable vertex is fully skipped.
        let pre_finalize_delaunay = tds.validate_delaunay();
        if let Err(err) = pre_finalize_delaunay {
            eprintln!(
                "[{context}] Unexpected Delaunay violation detected immediately after insertion attempt: {err:?}",
            );
            debug_print_first_delaunay_violation(&tds, None);
            panic!(
                "{context}: unified insertion pipeline must not leave a non-Delaunay triangulation after skipping an unsalvageable vertex",
            );
        }

        // Run the same finalization step that `triangulate` would invoke at the end.
        let finalize_result = <UnifiedInsertionPipeline<f64, (), (), D> as InsertionAlgorithm<
            f64,
            (),
            (),
            D,
        >>::finalize_triangulation(&mut tds);

        if let Err(err) = finalize_result {
            eprintln!(
                "[{context}] finalize_triangulation unexpectedly failed after skipping unsalvageable vertex: {err:?}",
            );
            debug_print_first_delaunay_violation(&tds, None);
            panic!(
                "{context}: finalize_triangulation must succeed when the per-vertex pipeline skips an unsalvageable vertex",
            );
        }

        // After finalization the triangulation must still be globally Delaunay.
        if let Err(err) = tds.validate_delaunay() {
            eprintln!(
                "[{context}] Delaunay violation detected after finalize_triangulation: {err:?}",
            );
            debug_print_first_delaunay_violation(&tds, None);
            panic!(
                "{context}: finalize_triangulation must not introduce Delaunay violations for this configuration",
            );
        }
    }

    /// Stepwise debug/regression: construct a base 5D triangulation from the
    /// first 6 points of the canonical configuration, then attempt to insert the
    /// 7th point via the unified insertion pipeline.
    ///
    /// The expected behavior is now:
    /// - The unified pipeline reports a `Skip` strategy for the 7th vertex
    ///   (unsalvageable for this configuration).
    /// - The triangulation remains globally Delaunay both before and after
    ///   `finalize_triangulation`.
    #[test]
    fn debug_5d_stepwise_insertion_of_seventh_vertex() {
        // Reconstruct the full 7-point configuration.
        let v1 = vertex!([
            61.994_906_139_357_86,
            66.880_064_158_234_8,
            62.542_871_273_730_91,
            -27.857_784_980_103_375,
            -78.369_282_526_711_23,
        ]);
        let v2 = vertex!([
            -31.430_765_957_270_268,
            50.418_208_939_604_746,
            88.657_219_404_750_96,
            47.248_786_623_931_88,
            -81.163_199_600_681_14,
        ]);
        let v3 = vertex!([
            -89.902_834_998_758_96,
            93.719_989_121_636_87,
            64.524_277_928_893_98,
            40.001_314_184_454_05,
            14.196_053_554_411_321,
        ]);
        let v4 = vertex!([
            2.625_958_385_925_883,
            48.251_155_688_054_36,
            3.491_542_746_106_750_5,
            97.241_732_043_079_37,
            -27.107_939_334_194_757,
        ]);
        let v5 = vertex!([
            62.628_856_831_188_11,
            -18.181_728_263_486_345,
            -32.153_141_689_537_584,
            25.692_809_519_458_7,
            26.369_541_091_117_114,
        ]);
        let v6 = vertex!([
            -41.886_149_523_644_406,
            -54.537_563_736_672_65,
            -54.555_379_092_740_964,
            75.499_924_758_912_23,
            16.127_546_041_675_355,
        ]);
        let v7 = vertex!([
            -77.161_459_173_963_2,
            -59.065_517_574_769_37,
            -19.652_689_679_369_03,
            -51.622_382_706_243_18,
            -26.000_263_271_298_543,
        ]);

        // Base triangulation built from the first 6 vertices.
        let base_vertices = vec![v1, v2, v3, v4, v5, v6];

        run_stepwise_unified_insertion_debug::<5>(
            &base_vertices,
            v7,
            "debug_5d_stepwise_insertion_of_seventh_vertex",
        );
    }
}
