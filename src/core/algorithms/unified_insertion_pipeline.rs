use crate::core::algorithms::{
    bowyer_watson::IncrementalBowyerWatson, robust_bowyer_watson::RobustBowyerWatson,
};
use crate::core::collections::{FastHashSet, fast_hash_set_with_capacity};
use crate::core::traits::data_type::DataType;
use crate::core::traits::insertion_algorithm::{
    DelaunayCheckPolicy, InsertionAlgorithm, InsertionError, InsertionInfo,
    UnifiedPerVertexInsertionOutcome, UnsalvageableVertexReport, VertexClassification,
    find_initial_simplex, unified_insert_vertex_fast_robust_or_skip,
};
use crate::core::triangulation_data_structure::{
    Tds, TriangulationConstructionError, TriangulationConstructionState,
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
    pub(crate) successful_insertions: usize,
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
                    self.on_successful_insertion(tds)?;
                }
                Ok(info)
            }
            UnifiedPerVertexInsertionOutcome::RobustSuccess {
                classification,
                fast_error,
                info,
            } => {
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
                    self.on_successful_insertion(tds)?;
                }
                Ok(info)
            }
            UnifiedPerVertexInsertionOutcome::Skipped(report) => {
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
        base_vertices: &[Vertex<f64, Option<()>, D>],
        new_vertex: Vertex<f64, Option<()>, D>,
        context: &str,
    ) {
        let mut tds: Tds<f64, Option<()>, Option<()>, D> =
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
        let mut pipeline: UnifiedInsertionPipeline<f64, Option<()>, Option<()>, D> =
            UnifiedInsertionPipeline::with_policy(DelaunayCheckPolicy::EndOnly);

        let insert_info =
            <UnifiedInsertionPipeline<f64, Option<()>, Option<()>, D> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                D,
            >>::insert_vertex(&mut pipeline, &mut tds, new_vertex);

        // Current behavior (for known failing configurations): insertion itself is
        // expected to succeed (i.e., local cavity/hull refinement completes), but
        // the final global validation in `finalize_triangulation` reports
        // violations.
        assert!(
            insert_info.is_ok(),
            "{context}: unified pipeline insertion of the new vertex is expected to succeed; if this fails instead, update this debug test accordingly: {insert_info:?}",
        );

        // Check global Delaunay property immediately after insertion, before running
        // the final `finalize_triangulation` pass. This helps us determine whether
        // the violation is already present after the per-insertion pipeline, or if
        // it arises only during the final global cleanup/validation step.
        let pre_finalize_delaunay = tds.validate_delaunay();
        if pre_finalize_delaunay.is_err() {
            eprintln!(
                "[{context}] Delaunay violation detected immediately after insertion, before finalize_triangulation",
            );
            debug_print_first_delaunay_violation(&tds, None);
        } else {
            eprintln!(
                "[{context}] Triangulation is Delaunay immediately after insertion; violation arises during finalize_triangulation",
            );
        }

        // Run the same finalization step that `triangulate` would invoke at the end.
        let finalize_result =
            <UnifiedInsertionPipeline<f64, Option<()>, Option<()>, D> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                D,
            >>::finalize_triangulation(&mut tds);

        assert!(
            finalize_result.is_err(),
            "{context}: expected finalize_triangulation to currently fail with a Delaunay validation error for this configuration; once fixed, flip this expectation and remove #[ignore]",
        );

        // Print detailed diagnostics for the post-insertion state.
        debug_print_first_delaunay_violation(&tds, None);
    }

    /// Stepwise debug: construct a base 5D triangulation from the first 6 points
    /// of the canonical configuration, then insert the 7th point via the unified
    /// insertion pipeline and observe where the Delaunay violation arises.
    ///
    /// This test is ignored for now because it documents current failing behavior.
    /// Once the insertion pipeline is fixed for this configuration, the expectations
    /// should be flipped and the `ignore` removed.
    #[test]
    #[ignore = "documents current failing unified insertion behavior; remove once fixed"]
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
