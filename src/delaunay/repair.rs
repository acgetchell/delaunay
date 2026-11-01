//! Repair policies and outcomes for Delaunay triangulations.
//!
//! This module separates mutating Delaunay repair policy from validation-only
//! checking. [`DelaunayRepairPolicy`] controls when construction and editing
//! paths may run local flip repair, while [`DelaunayCheckPolicy`] controls
//! global Level 5 Delaunay validation cadence without mutating topology.
//!
//! Import these APIs through [`crate::prelude::repair`]
//! for downstream examples, tests, and applications.

#![forbid(unsafe_code)]

use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairRun, repair_delaunay_with_flips_k2_k3_run,
};
use crate::core::operations::{RepairDecision, TopologicalOperation};
use crate::core::tds::SimplexKey;
use crate::core::traits::data_type::DataType;
use crate::delaunay_model::DelaunayTriangulation;
use crate::geometry::kernel::{Kernel, RobustKernel};
use crate::triangulation::validation::TopologyGuarantee;
use std::{fmt, num::NonZeroUsize};

/// Mutating Delaunay operation that can invoke flip-based repair internally.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayRepairOperation;
///
/// assert_eq!(DelaunayRepairOperation::VertexRemoval.to_string(), "vertex removal");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairOperation {
    /// Repair after removing a vertex.
    VertexRemoval,
}

impl fmt::Display for DelaunayRepairOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexRemoval => f.write_str("vertex removal"),
        }
    }
}

/// Policy controlling automatic flip-based Delaunay repair during unpublished
/// construction and internal mutation transactions.
///
/// This policy schedules **local flip-based repairs** after successful insertions
/// (and removals that modify topology). Public construction options can tune
/// the cadence while the candidate remains unpublished; final Level 5
/// certification is still mandatory before a `DelaunayTriangulation` is
/// returned.
/// It is separate from any *validation-only* policy to allow checking the Delaunay
/// property without mutating topology when needed.
///
/// During batch construction, [`DelaunayRepairPolicy::EveryN`] is a scheduled
/// cadence rather than a hard lower bound on repair frequency: construction may
/// run an additional local repair earlier when the accumulated seed frontier
/// grows large. [`DelaunayRepairPolicy::Never`] disables those automatic batch
/// repairs. For an already published [`DelaunayTriangulation`], skipping a
/// repair never weakens the owner: an insertion that reports repair is needed
/// must instead pass Level 5 validation or be rolled back.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayRepairPolicy;
/// use std::num::NonZeroUsize;
///
/// # fn main() {
/// let Some(every_four) = NonZeroUsize::new(4) else {
///     return;
/// };
/// let policy = DelaunayRepairPolicy::EveryN(every_four);
/// assert!(!policy.should_repair(0));
/// assert!(!policy.should_repair(3));
/// assert!(policy.should_repair(4));
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelaunayRepairPolicy {
    /// Disable automatic Delaunay repairs.
    Never,
    /// Run local flip-based repair after every successful insertion.
    EveryInsertion,
    /// Run local flip-based repair after every N successful insertions.
    EveryN(NonZeroUsize),
}

impl Default for DelaunayRepairPolicy {
    #[inline]
    fn default() -> Self {
        Self::EveryInsertion
    }
}

impl DelaunayRepairPolicy {
    /// Returns true if a repair pass should run after the given insertion count.
    #[inline]
    #[must_use]
    pub const fn should_repair(self, insertion_count: usize) -> bool {
        match self {
            Self::Never => false,
            Self::EveryInsertion => insertion_count != 0,
            Self::EveryN(n) => insertion_count != 0 && insertion_count.is_multiple_of(n.get()),
        }
    }
}

/// Policy controlling when **global** Delaunay validation runs.
///
/// This policy is **validation-only** (non-mutating) and is distinct from
/// [`DelaunayRepairPolicy`], which performs flip-based repairs.
///
/// # ⚠️ Performance Warning
///
/// For fixed dimension, complete Euclidean point-set owners can pass an
/// O(simplices) robust local certificate. Otherwise, Euclidean validation uses
/// an O(simplices × vertices) all-vertices fallback. Non-Euclidean owners use
/// topology-specific verification. Choose the check cadence with the possible
/// fallback cost in mind; see [`DelaunayTriangulation::is_valid_delaunay`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayCheckPolicy;
/// use std::num::NonZeroUsize;
///
/// # fn main() {
/// let Some(every_three) = NonZeroUsize::new(3) else {
///     return;
/// };
/// let policy = DelaunayCheckPolicy::EveryN(every_three);
/// assert!(!policy.should_check(2));
/// assert!(policy.should_check(3));
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaunayCheckPolicy {
    /// Run global Delaunay validation once after batch construction
    /// (the builder's `build()` / `build_with_kernel()`).
    ///
    /// Incremental insertion does not automatically run a final global check because there is no
    /// intrinsic “end” signal; call
    /// [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay)
    /// or
    /// [`DelaunayTriangulation::validate`](crate::DelaunayTriangulation::validate)
    /// when you are done inserting.
    #[default]
    EndOnly,
    /// Run global Delaunay validation after every N successful insertions.
    EveryN(NonZeroUsize),
}

impl DelaunayCheckPolicy {
    /// Returns true if a global Delaunay validation pass should run after the given insertion count.
    #[inline]
    #[must_use]
    pub const fn should_check(self, insertion_count: usize) -> bool {
        match self {
            Self::EndOnly => false,
            Self::EveryN(n) => insertion_count != 0 && insertion_count.is_multiple_of(n.get()),
        }
    }
}

// =============================================================================
// REPAIR (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Replays repair with an exact-predicate kernel and returns the validation frontier.
    pub(crate) fn repair_delaunay_with_flips_robust_run(
        &mut self,
        seed_simplices: Option<&[SimplexKey]>,
        max_flips: Option<usize>,
    ) -> Result<DelaunayRepairRun, DelaunayRepairError>
    where
        U: DataType,
        V: DataType,
    {
        let topology = self.tri.topology_guarantee();
        let global_topology = self.tri.global_topology();
        let kernel = RobustKernel::<f64>::new();
        self.invalidate_locate_hint_cache();
        let (tds, kernel) = (&mut self.tri.tds, &kernel);
        repair_delaunay_with_flips_k2_k3_run(
            tds,
            kernel,
            seed_simplices,
            topology,
            global_topology,
            max_flips,
        )
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D> {
    // =============================================================================
    // REPAIR POLICY GATES (No Kernel Bounds)
    // =============================================================================

    /// Returns whether insertion cadence schedules a Delaunay repair decision.
    ///
    /// This intentionally does not inspect topology admissibility: callers
    /// still need rollback protection when the cadence fires but facet flips
    /// are inadmissible, because that path must validate and reject a Level 5
    /// violation instead of silently publishing it.
    pub(crate) const fn delaunay_repair_scheduled_for(&self, insertion_count: usize) -> bool {
        D >= 2
            && self
                .insertion_state
                .delaunay_repair_policy
                .should_repair(insertion_count)
    }

    /// Applies the repair schedule only when stored simplices and topology can
    /// support bistellar flips.
    pub(crate) const fn should_run_delaunay_repair_for(
        &self,
        topology: TopologyGuarantee,
        insertion_count: usize,
    ) -> bool {
        if self.tri.tds.number_of_simplices() == 0 {
            return false;
        }
        if !self.delaunay_repair_scheduled_for(insertion_count) {
            return false;
        }

        matches!(
            self.insertion_state.delaunay_repair_policy.decide(
                insertion_count,
                topology,
                TopologicalOperation::FacetFlip,
            ),
            RepairDecision::Proceed
        )
    }

    /// Applies repair-policy and topology gates to non-insertion mutating operations.
    ///
    /// These operations do not have a meaningful insertion cadence, so every enabled
    /// repair policy permits the post-mutation repair attempt.
    pub(crate) fn should_run_delaunay_repair_after_mutation(
        &self,
        topology: TopologyGuarantee,
    ) -> bool {
        if D < 2 {
            return false;
        }
        if self.tri.tds.number_of_simplices() == 0 {
            return false;
        }
        if self.insertion_state.delaunay_repair_policy == DelaunayRepairPolicy::Never {
            return false;
        }

        TopologicalOperation::FacetFlip.is_admissible_under(topology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairHeuristicRebuildFailure,
        DelaunayRepairPostconditionFailure, FlipError, RepairQueueOrder,
    };
    use crate::core::vertex::Vertex;
    use crate::delaunay_model::DelaunayTriangulation;
    use crate::delaunay_test_support::init_tracing;
    use crate::triangulation::validation::TopologyGuarantee;
    use crate::vertex;
    use std::num::NonZeroUsize;

    // =========================================================================
    // Delaunay repair helper methods
    // =========================================================================

    #[test]
    fn test_should_run_delaunay_repair_for_skips_for_dimension_lt_2() {
        init_tracing();
        let vertices: Vec<Vertex<(), 1>> = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt: DelaunayTriangulation<_, (), (), 1> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        assert_eq!(dt.number_of_simplices(), 1);
        assert_eq!(
            dt.insertion_state.delaunay_repair_policy,
            DelaunayRepairPolicy::EveryInsertion
        );
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_skips_when_no_simplices() {
        init_tracing();
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_simplices(), 0);
        assert_eq!(
            dt.insertion_state.delaunay_repair_policy,
            DelaunayRepairPolicy::EveryInsertion
        );
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_skips_when_policy_never() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        assert_eq!(dt.number_of_simplices(), 1);
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_respects_every_n_schedule() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        dt.insertion_state.delaunay_repair_policy =
            DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap());
        let topology = dt.topology_guarantee();

        assert!(!dt.should_run_delaunay_repair_for(topology, 0));
        assert!(!dt.should_run_delaunay_repair_for(topology, 1));
        assert!(dt.should_run_delaunay_repair_for(topology, 2));
    }

    #[test]
    fn repair_schedule_is_distinct_from_flip_admissibility() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        dt.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .unwrap();
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::EveryInsertion;

        assert!(dt.delaunay_repair_scheduled_for(1));
        assert!(!dt.should_run_delaunay_repair_for(TopologyGuarantee::Pseudomanifold, 1));
    }

    #[test]
    fn test_non_insertion_mutation_repair_gate_ignores_insertion_cadence() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let topology = dt.topology_guarantee();

        dt.insertion_state.delaunay_repair_policy =
            DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap());
        assert!(dt.should_run_delaunay_repair_after_mutation(topology));

        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;
        assert!(!dt.should_run_delaunay_repair_after_mutation(topology));
    }

    /// Preserve the public typed fallback-chain error's diagnostic contract.
    #[test]
    fn test_advanced_repair_fallback_error_preserves_full_chain_context() {
        // The public diagnostic remains inspectable independently of the retired
        // private strategy that originally produced this combination.
        let primary_err = DelaunayRepairError::NonConvergent {
            max_flips: 1000,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 50,
                flips_performed: 1000,
                max_queue_len: 42,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: RepairQueueOrder::Fifo,
            }),
        };
        let robust_err = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let combined = DelaunayRepairError::HeuristicRebuildFailed {
            reason: Box::new(DelaunayRepairHeuristicRebuildFailure::FallbackChainFailed {
                primary: Box::new(primary_err),
                robust: Box::new(robust_err),
                heuristic: Box::new(DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts {
                    attempts: 3,
                    last_failure: Box::new(DelaunayRepairHeuristicRebuildFailure::AttemptFailed {
                        attempt: 3,
                        max_attempts: 3,
                        shuffle_seed: 1,
                        perturbation_seed: 2,
                        source: Box::new(DelaunayRepairError::from(FlipError::DegenerateSimplex)),
                    }),
                }),
            }),
        };

        let msg = combined.to_string();
        assert!(
            msg.contains("primary repair failed"),
            "error should mention primary failure: {msg}"
        );
        assert!(
            msg.contains("robust fallback failed"),
            "error should mention robust failure: {msg}"
        );
        assert!(
            msg.contains("disconnected the triangulation"),
            "error should include robust failure details: {msg}"
        );
        assert!(
            msg.contains("heuristic rebuild failed after 3 attempts"),
            "error should include heuristic rebuild details: {msg}"
        );
    }
    #[test]
    fn repair_operation_display_describes_mutation() {
        assert_eq!(
            DelaunayRepairOperation::VertexRemoval.to_string(),
            "vertex removal"
        );
    }

    #[test]
    fn check_policy_end_only_never_checks_during_insertion() {
        assert!(!DelaunayCheckPolicy::EndOnly.should_check(0));
        assert!(!DelaunayCheckPolicy::EndOnly.should_check(1));
    }

    #[test]
    fn check_policy_every_n_checks_on_multiples() {
        let every_2 = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(2).unwrap());

        assert!(!every_2.should_check(0));
        assert!(!every_2.should_check(1));
        assert!(every_2.should_check(2));
        assert!(!every_2.should_check(3));
        assert!(every_2.should_check(4));
    }

    #[test]
    fn repair_policy_zero_insertions_never_repairs() {
        assert!(!DelaunayRepairPolicy::EveryInsertion.should_repair(0));
        assert!(!DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap()).should_repair(0));
        assert!(!DelaunayRepairPolicy::Never.should_repair(0));
    }

    #[test]
    fn repair_policy_every_insertion_skips_never_and_repairs_after_first_insertion() {
        assert!(!DelaunayRepairPolicy::Never.should_repair(1));
        assert!(DelaunayRepairPolicy::EveryInsertion.should_repair(1));
        assert!(DelaunayRepairPolicy::EveryInsertion.should_repair(17));
    }

    #[test]
    fn repair_policy_every_n_repairs_only_on_nonzero_multiples() {
        let every_3 = DelaunayRepairPolicy::EveryN(NonZeroUsize::new(3).unwrap());

        assert!(!every_3.should_repair(1));
        assert!(!every_3.should_repair(2));
        assert!(every_3.should_repair(3));
        assert!(!every_3.should_repair(4));
        assert!(every_3.should_repair(6));
    }
}
