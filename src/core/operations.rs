//! Semantic classification and telemetry for triangulation operations.
//!
//! This module is intentionally **not** about implementation mechanics. It defines:
//! - What operation occurred (taxonomy)
//! - What the outcome was
//! - Lightweight telemetry/flags describing suspicious paths
//!
//! The actual algorithms live under `core::algorithms`.

use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::delaunay_triangulation::DelaunayRepairPolicy;
use crate::core::triangulation::TopologyGuarantee;
use crate::core::triangulation_data_structure::CellKey;

/// Semantic classification of topological modifications to a triangulation.
///
/// These correspond to bistellar (Pachner) move classes, but are not required
/// to be implemented as single atomic flips.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TopologicalOperation {
    /// k = 1 forward: vertex insertion (1 → d+1).
    InsertVertex,
    /// k = 1 inverse: vertex deletion (d+1 → 1).
    DeleteVertex,
    /// k = 2: facet bistellar flip.
    FacetFlip,
    /// k ≥ 3: higher-order cavity flip (typically in higher dimensions).
    CavityFlip,
}

/// Decision outcome for a flip-based Delaunay repair attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairDecision {
    /// Proceed with flip-based repair.
    Proceed,
    /// Skip repair, with a structured reason.
    Skip {
        /// Reason the repair was skipped.
        reason: RepairSkipReason,
    },
}

/// Reason why flip-based repair was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairSkipReason {
    /// Repair policy is disabled for this insertion count.
    PolicyDisabled,
    /// The requested operation is inadmissible under the current topology guarantee.
    Inadmissible {
        /// The operation that was attempted.
        operation: TopologicalOperation,
        /// Required topology guarantee for this operation.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
    },
}

impl DelaunayRepairPolicy {
    /// Decide whether a flip-based repair should run, given topology and operation constraints.
    #[must_use]
    pub const fn decide(
        self,
        insertion_count: usize,
        topology: TopologyGuarantee,
        operation: TopologicalOperation,
    ) -> RepairDecision {
        if !self.should_repair(insertion_count) {
            return RepairDecision::Skip {
                reason: RepairSkipReason::PolicyDisabled,
            };
        }

        if !operation.is_admissible_under(topology) {
            return RepairDecision::Skip {
                reason: RepairSkipReason::Inadmissible {
                    operation,
                    required: operation.required_topology(),
                    found: topology,
                },
            };
        }

        RepairDecision::Proceed
    }
}

impl TopologicalOperation {
    /// Returns `true` if this operation requires a PL-manifold topology guarantee.
    #[must_use]
    pub const fn requires_pl_manifold(self) -> bool {
        // Higher-order cavity flips rely on stronger local topology guarantees.
        //
        // Note: k=2/k=3 flips used for Delaunay repair are admissible under the weaker
        // pseudomanifold invariants (facet degree + closed boundary). Callers that need
        // strict PL-manifold guarantees should select `TopologyGuarantee::PLManifold`.
        matches!(self, Self::CavityFlip)
    }

    /// Returns `true` if this operation is admissible under the given topology guarantee.
    #[must_use]
    pub const fn is_admissible_under(self, topology: TopologyGuarantee) -> bool {
        match topology {
            TopologyGuarantee::PLManifold => true,
            TopologyGuarantee::Pseudomanifold => !self.requires_pl_manifold(),
        }
    }

    /// Returns the minimum topology guarantee required for this operation.
    #[must_use]
    pub const fn required_topology(self) -> TopologyGuarantee {
        if self.requires_pl_manifold() {
            TopologyGuarantee::PLManifold
        } else {
            TopologyGuarantee::Pseudomanifold
        }
    }
}

/// Result of an insertion attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InsertionResult {
    /// The vertex was successfully inserted.
    #[default]
    Inserted,
    /// The vertex was skipped due to duplicate coordinates.
    SkippedDuplicate,
    /// The vertex was skipped due to geometric degeneracy after retries.
    SkippedDegeneracy,
}

/// Statistics about a vertex insertion operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct InsertionStatistics {
    /// Number of insertion attempts (1 = success on first try, >1 = needed perturbation)
    pub attempts: usize,
    /// Number of cells removed during repair
    pub cells_removed_during_repair: usize,
    /// Result of the insertion attempt
    pub result: InsertionResult,
}

impl InsertionStatistics {
    /// Returns true if perturbation was applied (attempts > 1).
    #[must_use]
    pub const fn used_perturbation(&self) -> bool {
        self.attempts > 1
    }

    /// Returns true if the insertion succeeded.
    #[must_use]
    pub const fn success(&self) -> bool {
        matches!(self.result, InsertionResult::Inserted)
    }

    /// Returns true if the vertex was skipped (any reason).
    #[must_use]
    pub const fn skipped(&self) -> bool {
        matches!(
            self.result,
            InsertionResult::SkippedDuplicate | InsertionResult::SkippedDegeneracy
        )
    }

    /// Returns true if the vertex was skipped due to duplicate coordinates.
    #[must_use]
    pub const fn skipped_duplicate(&self) -> bool {
        matches!(self.result, InsertionResult::SkippedDuplicate)
    }
}

/// Ephemeral insertion state used by Delaunay triangulations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DelaunayInsertionState {
    /// Hint for the next `locate()` call (last inserted cell).
    pub last_inserted_cell: Option<CellKey>,
    /// Policy controlling automatic Delaunay repair (flip-based).
    pub delaunay_repair_policy: DelaunayRepairPolicy,
    /// Count of successful insertions (used to schedule repairs).
    pub delaunay_repair_insertion_count: usize,
}

impl DelaunayInsertionState {
    /// Create a fresh insertion state with default repair policy.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_inserted_cell: None,
            delaunay_repair_policy: DelaunayRepairPolicy::EveryInsertion,
            delaunay_repair_insertion_count: 0,
        }
    }
}

/// Outcome of a single-vertex insertion attempt.
///
/// This distinguishes between:
/// - A successful insertion (`Inserted`)
/// - An intentionally skipped insertion (`Skipped`) where the triangulation is left unchanged
///   for this vertex (transactional rollback). This can happen for example when:
///   - The input vertex is a duplicate/near-duplicate (skipped immediately)
///   - A retryable geometric degeneracy exhausts all perturbation attempts
///
/// Other non-recoverable structural failures are returned as `Err(InsertionError)` instead
/// (e.g. duplicate UUID).
#[derive(Debug, Clone)]
pub enum InsertionOutcome {
    /// The vertex was inserted successfully.
    Inserted {
        /// Key of the inserted vertex.
        vertex_key: crate::core::triangulation_data_structure::VertexKey,
        /// Optional cell key that can be used as a hint for subsequent insertions.
        hint: Option<crate::core::triangulation_data_structure::CellKey>,
    },
    /// The vertex was intentionally not inserted.
    ///
    /// This covers both immediate skips (e.g. duplicate/near-duplicate coordinates) and skips
    /// after exhausting retry attempts for geometric degeneracies.
    ///
    /// The triangulation is left unchanged for this vertex (transactional rollback).
    Skipped {
        /// The reason the vertex was skipped.
        ///
        /// This may be non-retryable (e.g. [`InsertionError::DuplicateCoordinates`]) or, for
        /// retry-based skips, the last error encountered.
        error: InsertionError,
    },
}

/// Adaptive error-checking on suspicious operations.
#[derive(Clone, Copy, Debug, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "A small set of boolean flags is clearer here than bitflags or an enum"
)]
pub struct SuspicionFlags {
    /// A perturbation retry was required to resolve a geometric degeneracy.
    pub perturbation_used: bool,

    /// A conflict-region computation returned an empty set for an interior point.
    pub empty_conflict_region: bool,

    /// The insertion fell back to splitting the containing cell (star-split) to avoid
    /// creating a dangling vertex.
    pub fallback_star_split: bool,

    /// The non-manifold repair loop was entered after insertion/hull extension.
    pub repair_loop_entered: bool,

    /// One or more cells were removed during non-manifold repair.
    pub cells_removed: bool,

    /// Neighbor pointers were rebuilt (facet-matched) after topology repair.
    pub neighbor_pointers_rebuilt: bool,
}

impl SuspicionFlags {
    /// Returns `true` if any suspicious condition was observed.
    #[inline]
    #[must_use]
    pub const fn is_suspicious(&self) -> bool {
        self.perturbation_used
            || self.empty_conflict_region
            || self.fallback_star_split
            || self.repair_loop_entered
            || self.cells_removed
            || self.neighbor_pointers_rebuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_operation_admissibility() {
        assert!(!TopologicalOperation::FacetFlip.requires_pl_manifold());
        assert!(TopologicalOperation::CavityFlip.requires_pl_manifold());
        assert!(!TopologicalOperation::InsertVertex.requires_pl_manifold());
        assert!(!TopologicalOperation::DeleteVertex.requires_pl_manifold());

        assert!(TopologicalOperation::FacetFlip.is_admissible_under(TopologyGuarantee::PLManifold));
        assert!(
            TopologicalOperation::FacetFlip.is_admissible_under(TopologyGuarantee::Pseudomanifold)
        );

        assert!(
            TopologicalOperation::InsertVertex
                .is_admissible_under(TopologyGuarantee::Pseudomanifold)
        );
    }

    #[test]
    fn test_repair_policy_decide_respects_policy_and_topology() {
        let op = TopologicalOperation::FacetFlip;

        let decision =
            DelaunayRepairPolicy::EveryInsertion.decide(1, TopologyGuarantee::PLManifold, op);
        assert!(matches!(decision, RepairDecision::Proceed));

        let decision =
            DelaunayRepairPolicy::EveryInsertion.decide(1, TopologyGuarantee::Pseudomanifold, op);
        assert!(matches!(decision, RepairDecision::Proceed));

        let decision = DelaunayRepairPolicy::Never.decide(1, TopologyGuarantee::PLManifold, op);
        assert!(matches!(
            decision,
            RepairDecision::Skip {
                reason: RepairSkipReason::PolicyDisabled
            }
        ));
    }
}
