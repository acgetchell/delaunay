use crate::core::algorithms::{
    bowyer_watson::IncrementalBowyerWatson, robust_bowyer_watson::RobustBowyerWatson,
};
use crate::core::traits::data_type::DataType;
use crate::core::traits::insertion_algorithm::{
    DelaunayCheckPolicy, InsertionAlgorithm, InsertionError, InsertionInfo,
    UnifiedPerVertexInsertionOutcome, UnsalvageableVertexReport, VertexClassification,
    unified_insert_vertex_fast_robust_or_skip,
};
use crate::core::triangulation_data_structure::Tds;
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
