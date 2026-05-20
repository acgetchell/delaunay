//! Repair policies and outcomes for Delaunay triangulations.
//!
//! This module separates mutating Delaunay repair policy from validation-only
//! checking. [`DelaunayRepairPolicy`] controls when construction and editing
//! paths may run local flip repair, while [`DelaunayCheckPolicy`] controls
//! global Level 4 validation cadence without mutating topology.
//!
//! Import these APIs through [`delaunay::prelude::repair`](crate::prelude::repair)
//! for downstream examples, tests, and applications.

#![forbid(unsafe_code)]

#[cfg(test)]
use crate::construction::test_hooks;
use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairRun, DelaunayRepairStats, repair_delaunay_with_flips_k2_k3,
    repair_delaunay_with_flips_k2_k3_run,
};
use crate::core::collections::FastHasher;
use crate::core::operations::{InsertionOutcome, RepairDecision, TopologicalOperation};
use crate::core::tds::SimplexKey;
use crate::core::traits::data_type::DataType;
use crate::core::util::stable_hash_u64_slice;
use crate::core::validation::TopologyGuarantee;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{ExactPredicates, Kernel, RobustKernel};
use crate::triangulation::DelaunayTriangulation;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::{
    fmt,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
};

// Heuristic rebuild attempts must be consistent across build profiles to avoid
// release-only construction failures (see #306).
const HEURISTIC_REBUILD_ATTEMPTS: usize = 6;
const MAX_HEURISTIC_REBUILD_DEPTH: usize = 1;

thread_local! {
    static HEURISTIC_REBUILD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct HeuristicRebuildRecursionGuard {
    prior_depth: usize,
}

impl HeuristicRebuildRecursionGuard {
    /// Tracks nested heuristic rebuilds so fallback construction cannot recurse
    /// indefinitely through repair hooks.
    fn enter() -> Result<Self, DelaunayRepairError> {
        let prior_depth = HEURISTIC_REBUILD_DEPTH.with(|depth| {
            let prior = depth.get();
            if prior < MAX_HEURISTIC_REBUILD_DEPTH {
                depth.set(prior.saturating_add(1));
            }
            prior
        });
        if prior_depth >= MAX_HEURISTIC_REBUILD_DEPTH {
            return Err(DelaunayRepairError::HeuristicRebuildFailed {
                message: format!(
                    "heuristic rebuild recursion depth exceeded {MAX_HEURISTIC_REBUILD_DEPTH}"
                ),
            });
        }
        Ok(Self { prior_depth })
    }
}

impl Drop for HeuristicRebuildRecursionGuard {
    fn drop(&mut self) {
        HEURISTIC_REBUILD_DEPTH.with(|depth| depth.set(self.prior_depth));
    }
}

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

/// Policy controlling automatic flip-based Delaunay repair.
///
/// This policy schedules **local flip-based repairs** after successful insertions
/// (and removals that modify topology).
/// It is separate from any *validation-only* policy to allow checking the Delaunay
/// property without mutating topology when needed.
///
/// During batch construction, [`DelaunayRepairPolicy::EveryN`] is a scheduled
/// cadence rather than a hard lower bound on repair frequency: construction may
/// run an additional local repair earlier when the accumulated seed frontier
/// grows large. [`DelaunayRepairPolicy::Never`] disables those automatic batch
/// repairs.
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
/// Configuration for the optional heuristic rebuild fallback in Delaunay repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayRepairHeuristicConfig;
///
/// let mut config = DelaunayRepairHeuristicConfig::default();
/// config.shuffle_seed = Some(7);
/// config.perturbation_seed = Some(11);
/// assert_eq!(config.shuffle_seed, Some(7));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DelaunayRepairHeuristicConfig {
    /// Optional RNG seed used to shuffle vertex insertion order.
    pub shuffle_seed: Option<u64>,
    /// Optional seed used to vary the deterministic perturbation pattern.
    pub perturbation_seed: Option<u64>,
    /// Optional per-attempt flip budget cap.
    ///
    /// When set, each repair attempt is limited to at most this many flips.
    /// `None` (the default) uses the dimension-dependent internal budget
    /// computed from the triangulation size.
    ///
    /// This is primarily useful for debug harnesses that want to study
    /// repair convergence behavior at different budgets without disabling
    /// repair entirely.
    pub max_flips: Option<usize>,
}

impl DelaunayRepairHeuristicConfig {
    /// Fills omitted seeds from a stable base so heuristic rebuilds are
    /// repeatable even when callers only configure one axis of randomness.
    pub(crate) fn resolve_seeds(self, base_seed: u64) -> DelaunayRepairHeuristicSeeds {
        // Derive deterministic defaults when the caller does not provide explicit seeds.
        const SHUFFLE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
        const PERTURB_SALT: u64 = 0xD1B5_4A32_D192_ED03;

        let mut shuffle_seed = self
            .shuffle_seed
            .unwrap_or_else(|| base_seed.wrapping_add(SHUFFLE_SALT));
        if self.shuffle_seed.is_none() && shuffle_seed == 0 {
            shuffle_seed = 1;
        }

        let mut perturbation_seed = self
            .perturbation_seed
            .unwrap_or_else(|| base_seed.rotate_left(17) ^ PERTURB_SALT);
        if self.perturbation_seed.is_none() && perturbation_seed == 0 {
            perturbation_seed = 1;
        }

        DelaunayRepairHeuristicSeeds {
            shuffle_seed,
            perturbation_seed,
        }
    }
}

/// Seeds used for a heuristic rebuild.
///
/// If the caller does not provide explicit seeds, deterministic defaults are derived from a stable
/// hash of the current vertex set.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::DelaunayRepairHeuristicSeeds;
///
/// let seeds = DelaunayRepairHeuristicSeeds {
///     shuffle_seed: 1,
///     perturbation_seed: 2,
/// };
/// assert_eq!(seeds.shuffle_seed, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelaunayRepairHeuristicSeeds {
    /// RNG seed used to shuffle vertex insertion order.
    pub shuffle_seed: u64,
    /// Seed used to vary the perturbation pattern during retries.
    pub perturbation_seed: u64,
}

/// Result of a flip-based repair attempt, including heuristic fallback metadata.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairOutcome, DelaunayRepairStats,
/// };
///
/// let outcome = DelaunayRepairOutcome {
///     stats: DelaunayRepairStats::default(),
///     heuristic: None,
/// };
/// assert!(!outcome.used_heuristic());
/// ```
#[derive(Debug, Clone)]
pub struct DelaunayRepairOutcome {
    /// Statistics from the final flip-based repair pass.
    pub stats: DelaunayRepairStats,
    /// Heuristic rebuild seeds, if a fallback was used.
    pub heuristic: Option<DelaunayRepairHeuristicSeeds>,
}

impl DelaunayRepairOutcome {
    /// Returns `true` if a heuristic rebuild fallback was used.
    #[must_use]
    pub const fn used_heuristic(&self) -> bool {
        self.heuristic.is_some()
    }
}

/// Policy controlling when **global** Delaunay validation runs.
///
/// This policy is **validation-only** (non-mutating) and is distinct from
/// [`DelaunayRepairPolicy`], which performs flip-based repairs.
///
/// # ⚠️ Performance Warning
///
/// Global Delaunay validation is **extremely expensive**: O(simplices × vertices). Use this policy
/// primarily when you need correctness guarantees and are willing to pay the cost.
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
    /// Run global Delaunay validation once after batch construction (e.g. `new()` / `with_kernel()`).
    ///
    /// Incremental insertion does not automatically run a final global check because there is no
    /// intrinsic “end” signal; call
    /// [`DelaunayTriangulation::is_valid`](crate::DelaunayTriangulation::is_valid)
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
            Self::EveryN(n) => insertion_count.is_multiple_of(n.get()),
        }
    }
}

// =============================================================================
// REPAIR (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Runs flip-based Delaunay repair over the full triangulation.
    ///
    /// This is a manual entrypoint that performs a global scan of interior facets
    /// and applies k=2/k=3 bistellar flips until locally Delaunay or until the flip
    /// budget is exhausted. On success, geometric orientation is re-canonicalized
    /// to the positive sign.
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayRepairError`] if the repair fails to converge, an underlying
    /// flip operation fails, or post-repair orientation canonicalization fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// use delaunay::prelude::repair::DelaunayRepairStats;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Repair(#[from] delaunay::flips::DelaunayRepairError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let stats = dt.repair_delaunay_with_flips()?;
    /// assert!(stats.facets_checked >= stats.flips_performed);
    /// # Ok(())
    /// # }
    /// ```
    pub fn repair_delaunay_with_flips(&mut self) -> Result<DelaunayRepairStats, DelaunayRepairError>
    where
        K: ExactPredicates<D>,
        U: DataType,
        V: DataType,
    {
        self.repair_delaunay_with_flips_capped(None)
    }

    /// Runs flip-based repair with an optional per-attempt cap so public repair
    /// and heuristic harnesses share one mutation path.
    fn repair_delaunay_with_flips_capped(
        &mut self,
        max_flips: Option<usize>,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError>
    where
        K: ExactPredicates<D>,
        U: DataType,
        V: DataType,
    {
        #[cfg(test)]
        if test_hooks::force_repair_nonconvergent_enabled() {
            return Err(test_hooks::synthetic_nonconvergent_error());
        }
        let operation = TopologicalOperation::FacetFlip;
        let topology = self.tri.topology_guarantee();
        if !operation.is_admissible_under(topology) {
            return Err(DelaunayRepairError::InvalidTopology {
                required: operation.required_topology(),
                found: topology,
                message: "Bistellar flips require a PL-manifold (vertex-link validation)",
            });
        }
        self.invalidate_locate_hint_cache();
        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
        let stats = repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology, max_flips)?;

        // Re-canonicalize geometric orientation (#258): flip repair may leave
        // the global sign negative.
        self.ensure_positive_orientation()?;

        Ok(stats)
    }

    /// Canonicalize geometric orientation to the positive sign, preserving
    /// canonicalization failures as their own repair error variant.
    fn ensure_positive_orientation(&mut self) -> Result<(), DelaunayRepairError>
    where
        U: DataType,
        V: DataType,
    {
        self.tri
            .normalize_and_promote_positive_orientation()
            .map_err(|e| DelaunayRepairError::OrientationCanonicalizationFailed {
                message: format!("after flip repair: {e}"),
            })
    }

    /// Replays repair with an exact-predicate kernel before escalating to
    /// heuristic rebuild.
    fn repair_delaunay_with_flips_robust(
        &mut self,
        seed_simplices: Option<&[SimplexKey]>,
        max_flips: Option<usize>,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError>
    where
        U: DataType,
        V: DataType,
    {
        self.repair_delaunay_with_flips_robust_run(seed_simplices, max_flips)
            .map(|run| run.stats)
    }

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
        let kernel = RobustKernel::<K::Scalar>::new();
        self.invalidate_locate_hint_cache();
        let (tds, kernel) = (&mut self.tri.tds, &kernel);
        repair_delaunay_with_flips_k2_k3_run(tds, kernel, seed_simplices, topology, max_flips)
    }

    /// Applies the repair policy only when the dimension and topology can
    /// support bistellar flips.
    pub(crate) fn should_run_delaunay_repair_for(
        &self,
        topology: TopologyGuarantee,
        insertion_count: usize,
    ) -> bool {
        if D < 2 {
            return false;
        }
        if self.tri.tds.number_of_simplices() == 0 {
            return false;
        }

        let policy = self.insertion_state.delaunay_repair_policy;
        if policy == DelaunayRepairPolicy::Never {
            return false;
        }

        matches!(
            policy.decide(insertion_count, topology, TopologicalOperation::FacetFlip),
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

    /// Enables test-only repair fallback paths without exposing a public knob.
    #[cfg_attr(
        not(test),
        expect(
            clippy::missing_const_for_fn,
            reason = "runtime feature and environment checks should remain ordinary functions"
        )
    )]
    fn force_heuristic_rebuild_enabled() -> bool {
        #[cfg(test)]
        {
            test_hooks::force_heuristic_rebuild_enabled()
        }
        #[cfg(not(test))]
        {
            false
        }
    }
}

// =============================================================================
// ADVANCED REPAIR & HEURISTIC REBUILD (Requires Numeric Scalar Bounds)
// =============================================================================
//
// `repair_delaunay_with_flips_advanced` can fall back to `rebuild_with_heuristic`,
// which constructs a new triangulation and therefore adds `NumCast` on top of
// the scalar requirements guaranteed by `Kernel`.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: NumCast,
    U: DataType,
    V: DataType,
{
    /// Runs flip-based Delaunay repair
    ///
    /// This first attempts the standard two-pass flip repair. If it fails to converge (or if
    /// the result cannot be verified as Delaunay), it rebuilds the triangulation from the
    /// current vertex set using a shuffled insertion order and a perturbation seed, then runs
    /// a final flip-repair pass. On success, geometric orientation is re-canonicalized
    /// to the positive sign.
    ///
    /// The returned outcome marks whether the heuristic fallback was used and records
    /// the seeds needed to reproduce it (if desired).
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayRepairError`] if the flip-based repair fails, the heuristic
    /// rebuild fallback cannot construct a valid triangulation, or post-repair
    /// orientation canonicalization fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// use delaunay::prelude::repair::DelaunayRepairHeuristicConfig;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Repair(#[from] delaunay::flips::DelaunayRepairError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let outcome = dt
    ///     .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
    ///     ?;
    /// assert!(outcome.stats.facets_checked >= outcome.stats.flips_performed);
    /// # Ok(())
    /// # }
    /// ```
    pub fn repair_delaunay_with_flips_advanced(
        &mut self,
        config: DelaunayRepairHeuristicConfig,
    ) -> Result<DelaunayRepairOutcome, DelaunayRepairError>
    where
        K: ExactPredicates<D>,
    {
        if Self::force_heuristic_rebuild_enabled() {
            let base_seed = self.heuristic_rebuild_base_seed();
            let seeds = config.resolve_seeds(base_seed);
            let (candidate, stats, used_seeds) =
                self.rebuild_with_heuristic(seeds, config.max_flips)?;
            *self = candidate;
            return Ok(DelaunayRepairOutcome {
                stats,
                heuristic: Some(used_seeds),
            });
        }
        let max_flips = config.max_flips;
        match self.repair_delaunay_with_flips_capped(max_flips) {
            Ok(stats) => Ok(DelaunayRepairOutcome {
                stats,
                heuristic: None,
            }),
            Err(
                primary_err @ (DelaunayRepairError::NonConvergent { .. }
                | DelaunayRepairError::PostconditionFailed { .. }),
            ) => {
                match self.repair_delaunay_with_flips_robust(None, max_flips) {
                    Ok(stats) => {
                        // Re-canonicalize geometric orientation (#258): robust flip
                        // repair may leave the global sign negative.
                        self.ensure_positive_orientation()?;
                        Ok(DelaunayRepairOutcome {
                            stats,
                            heuristic: None,
                        })
                    }
                    Err(robust_err) => {
                        let base_seed = self.heuristic_rebuild_base_seed();
                        let seeds = config.resolve_seeds(base_seed);
                        let (candidate, stats, used_seeds) = self
                            .rebuild_with_heuristic(seeds, max_flips)
                            .map_err(|heuristic_err| {
                                let heuristic_message = match heuristic_err {
                                    DelaunayRepairError::HeuristicRebuildFailed { message } => {
                                        message
                                    }
                                    other => other.to_string(),
                                };
                                DelaunayRepairError::HeuristicRebuildFailed {
                                    message: format!(
                                        "primary repair failed ({primary_err}); robust fallback failed ({robust_err}); {heuristic_message}"
                                    ),
                                }
                            })?;
                        *self = candidate;
                        Ok(DelaunayRepairOutcome {
                            stats,
                            heuristic: Some(used_seeds),
                        })
                    }
                }
            }
            Err(err) => Err(err),
        }
    }

    /// Rebuilds from the current vertex set with varied deterministic seeds when
    /// flip repair cannot converge directly.
    #[expect(
        clippy::too_many_lines,
        reason = "heuristic rebuild keeps point extraction, reconstruction, and validation together"
    )]
    fn rebuild_with_heuristic(
        &self,
        base_seeds: DelaunayRepairHeuristicSeeds,
        max_flips_override: Option<usize>,
    ) -> Result<(Self, DelaunayRepairStats, DelaunayRepairHeuristicSeeds), DelaunayRepairError>
    where
        K: ExactPredicates<D>,
    {
        let base_vertices = self.collect_vertices_for_rebuild();

        let mut last_error: Option<String> = None;

        for attempt in 0..HEURISTIC_REBUILD_ATTEMPTS {
            let seeds = if attempt == 0 {
                base_seeds
            } else {
                // Vary the deterministic shuffle and perturbation patterns across attempts.
                const SHUFFLE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
                const PERTURB_SALT: u64 = 0xD1B5_4A32_D192_ED03;

                let attempt_u64 = attempt as u64;

                let mut shuffle_seed = base_seeds
                    .shuffle_seed
                    .wrapping_add(attempt_u64.wrapping_mul(SHUFFLE_SALT));
                if shuffle_seed == 0 {
                    shuffle_seed = 1;
                }

                let mut perturbation_seed =
                    base_seeds.perturbation_seed ^ attempt_u64.wrapping_mul(PERTURB_SALT);
                if perturbation_seed == 0 {
                    perturbation_seed = 1;
                }

                DelaunayRepairHeuristicSeeds {
                    shuffle_seed,
                    perturbation_seed,
                }
            };

            let rebuild_attempt = (|| {
                let _guard = HeuristicRebuildRecursionGuard::enter()?;

                // Shuffle vertices for this attempt.
                let mut vertices = base_vertices.clone();
                let mut rng = rand::rngs::StdRng::seed_from_u64(seeds.shuffle_seed);
                vertices.shuffle(&mut rng);

                // Heuristic rebuild is a last-resort fallback when global repair fails. Prefer an
                // insertion schedule that keeps the triangulation near-Delaunay (local repairs on
                // each insertion) so we do not get stuck in a non-regular configuration that flip
                // repair cannot escape.
                let topology_guarantee = self.tri.topology_guarantee();
                let global_topology = self.tri.global_topology();
                let mut candidate = Self::with_empty_kernel_and_topology_guarantee(
                    self.tri.kernel.clone(),
                    topology_guarantee,
                );
                candidate.set_global_topology(global_topology);

                // During rebuild, force local repair after every insertion. The caller's
                // policies are copied onto the finished candidate below.
                candidate.insertion_state.delaunay_repair_policy =
                    DelaunayRepairPolicy::EveryInsertion;
                candidate.insertion_state.delaunay_check_policy = DelaunayCheckPolicy::EndOnly;

                for (idx, vertex) in vertices.into_iter().enumerate() {
                    let uuid = vertex.uuid();
                    let coords = *vertex.point().coords();

                    let hint = candidate.insertion_state.last_inserted_simplex;
                    let insert_detail = {
                        let (tri, spatial_index) =
                            (&mut candidate.tri, &mut candidate.spatial_index);
                        tri.insert_with_statistics_seeded_indexed_detailed(
                            vertex,
                            None,
                            hint,
                            seeds.perturbation_seed,
                            spatial_index.as_mut(),
                            Some(idx),
                        )
                        .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                            message: format!(
                                "heuristic rebuild insertion failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                            ),
                        })?
                    };
                    let repair_seed_simplices = insert_detail.repair_seed_simplices;
                    let delaunay_repair_required = insert_detail.delaunay_repair_required;

                    match insert_detail.outcome {
                        InsertionOutcome::Inserted { vertex_key, hint } => {
                            candidate.insertion_state.last_inserted_simplex = hint;
                            candidate.insertion_state.delaunay_repair_insertion_count = candidate
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);

                            if delaunay_repair_required {
                                candidate
                                    .maybe_repair_after_insertion_capped(
                                        vertex_key,
                                        hint,
                                        &repair_seed_simplices,
                                        max_flips_override,
                                    )
                                    .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                                        message: format!(
                                            "heuristic rebuild repair failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                                        ),
                                    })?;
                            }

                            candidate
                                .maybe_check_after_insertion()
                                .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                                    message: format!(
                                        "heuristic rebuild Delaunay check failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                                    ),
                                })?;
                        }
                        InsertionOutcome::Skipped { error } => {
                            return Err(DelaunayRepairError::HeuristicRebuildFailed {
                                message: format!(
                                    "heuristic rebuild skipped vertex at idx={idx} uuid={uuid} coords={coords:?}: {error}"
                                ),
                            });
                        }
                    }
                }

                candidate.tri.validation_policy = self.tri.validation_policy;
                candidate.insertion_state.delaunay_repair_policy =
                    self.insertion_state.delaunay_repair_policy;
                candidate.insertion_state.delaunay_check_policy =
                    self.insertion_state.delaunay_check_policy;
                candidate.insertion_state.delaunay_repair_insertion_count =
                    self.insertion_state.delaunay_repair_insertion_count;
                candidate.insertion_state.last_inserted_simplex = None;

                let topology = candidate.tri.topology_guarantee();
                candidate.invalidate_locate_hint_cache();
                let (tds, kernel) = (&mut candidate.tri.tds, &candidate.tri.kernel);
                let stats = repair_delaunay_with_flips_k2_k3(
                    tds,
                    kernel,
                    None,
                    topology,
                    max_flips_override,
                )?;

                // Re-canonicalize geometric orientation (#258): the final flip
                // repair may leave the global sign negative.
                candidate.ensure_positive_orientation()?;

                Ok::<_, DelaunayRepairError>((candidate, stats))
            })();

            match rebuild_attempt {
                Ok((candidate, stats)) => return Ok((candidate, stats, seeds)),
                Err(err) => {
                    last_error = Some(format!(
                        "attempt {}/{} (shuffle_seed={} perturbation_seed={}): {err}",
                        attempt + 1,
                        HEURISTIC_REBUILD_ATTEMPTS,
                        seeds.shuffle_seed,
                        seeds.perturbation_seed,
                    ));
                }
            }
        }

        Err(DelaunayRepairError::HeuristicRebuildFailed {
            message: format!(
                "heuristic rebuild failed after {HEURISTIC_REBUILD_ATTEMPTS} attempts: {}",
                last_error.unwrap_or_else(|| "unknown error".to_string())
            ),
        })
    }

    /// Preserves vertex UUIDs and data so heuristic rebuilds remain an internal
    /// repair strategy, not a user-visible remapping.
    fn collect_vertices_for_rebuild(&self) -> Vec<Vertex<K::Scalar, U, D>> {
        self.tri
            .tds
            .vertices()
            .map(|(_, vertex)| Vertex::new_with_uuid(*vertex.point(), vertex.uuid(), vertex.data))
            .collect()
    }

    /// Derives rebuild seeds from the vertex set so fallback behavior is
    /// reproducible regardless of slotmap iteration accidents.
    fn heuristic_rebuild_base_seed(&self) -> u64 {
        let mut vertex_hashes = Vec::with_capacity(self.tri.tds.number_of_vertices());
        for (_, vertex) in self.tri.tds.vertices() {
            let mut hasher = FastHasher::default();
            vertex.hash(&mut hasher);
            vertex_hashes.push(hasher.finish());
        }
        vertex_hashes.sort_unstable();
        stable_hash_u64_slice(&vertex_hashes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::test_hooks;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, FlipError, RepairQueueOrder, verify_delaunay_via_flip_predicates,
    };
    use crate::core::simplex::Simplex;
    use crate::core::tds::{Tds, TriangulationConstructionState};
    use crate::core::validation::TopologyGuarantee;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::{AdaptiveKernel, RobustKernel};
    use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use std::{num::NonZeroUsize, sync::Once};

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

    struct ForceHeuristicRebuildGuard {
        prior: bool,
    }

    impl ForceHeuristicRebuildGuard {
        fn enable() -> Self {
            let prior = test_hooks::set_force_heuristic_rebuild(true);
            Self { prior }
        }
    }

    impl Drop for ForceHeuristicRebuildGuard {
        fn drop(&mut self) {
            test_hooks::restore_force_heuristic_rebuild(self.prior);
        }
    }

    struct ForceRepairNonconvergentGuard {
        prior: bool,
    }

    impl ForceRepairNonconvergentGuard {
        fn enable() -> Self {
            let prior = test_hooks::set_force_repair_nonconvergent(true);
            Self { prior }
        }
    }

    impl Drop for ForceRepairNonconvergentGuard {
        fn drop(&mut self) {
            test_hooks::restore_force_repair_nonconvergent(self.prior);
        }
    }

    fn non_delaunay_quad_tds() -> Tds<f64, (), (), 2> {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([4.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([4.0, 2.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 2.0])).unwrap();

        tds.insert_simplex_with_mapping(Simplex::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        tds.insert_simplex_with_mapping(Simplex::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds
    }

    // =========================================================================
    // Delaunay repair helper methods
    // =========================================================================

    #[test]
    fn test_should_run_delaunay_repair_for_skips_for_dimension_lt_2() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 1>> = vec![vertex!([0.0]), vertex!([1.0])];
        let dt: DelaunayTriangulation<_, (), (), 1> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_simplices(), 1);
        assert_eq!(
            dt.delaunay_repair_policy(),
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
            dt.delaunay_repair_policy(),
            DelaunayRepairPolicy::EveryInsertion
        );
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_skips_when_policy_never() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_simplices(), 1);
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_respects_every_n_schedule() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap()));
        let topology = dt.topology_guarantee();

        assert!(!dt.should_run_delaunay_repair_for(topology, 0));
        assert!(!dt.should_run_delaunay_repair_for(topology, 1));
        assert!(dt.should_run_delaunay_repair_for(topology, 2));
    }

    #[test]
    fn test_non_insertion_mutation_repair_gate_ignores_insertion_cadence() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let topology = dt.topology_guarantee();

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap()));
        assert!(dt.should_run_delaunay_repair_after_mutation(topology));

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        assert!(!dt.should_run_delaunay_repair_after_mutation(topology));
    }

    #[test]
    fn test_vertex_key_valid_after_explicit_heuristic_rebuild() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Insert a vertex normally (no heuristic rebuild during insert).
        let inserted = vertex!([0.25, 0.25]);
        let inserted_uuid = inserted.uuid();

        let (outcome, _stats) = dt.insert_with_statistics(inserted).unwrap();
        let InsertionOutcome::Inserted { vertex_key, .. } = outcome else {
            panic!("Expected successful insertion outcome");
        };

        // Force a heuristic rebuild via the public repair API.
        let _guard = ForceHeuristicRebuildGuard::enable();
        let outcome = dt
            .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
            .unwrap();
        assert!(
            outcome.used_heuristic(),
            "Expected heuristic rebuild to be used"
        );

        // Verify the vertex is still findable by UUID after heuristic rebuild.
        let remapped = dt
            .tri
            .tds
            .vertex_key_from_uuid(&inserted_uuid)
            .expect("Inserted vertex UUID missing after heuristic rebuild");

        // The vertex key may have changed after heuristic rebuild, but the
        // vertex should still be present and accessible.
        assert!(dt.tri.tds.vertex(remapped).is_some());
        assert!(dt.validate().is_ok());
        // Original vertex_key may be stale after heuristic rebuild; that is
        // expected. The important invariant is that the UUID lookup works.
        let _ = vertex_key;
    }

    #[test]
    fn test_heuristic_rebuild_preserves_global_topology() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let global_topology = GlobalTopology::Toroidal {
            domain: [1.0, 1.0],
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        };
        dt.set_global_topology(global_topology);

        let _guard = ForceHeuristicRebuildGuard::enable();
        let outcome = dt
            .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
            .unwrap();

        assert!(
            outcome.used_heuristic(),
            "Expected forced heuristic rebuild to be used"
        );
        assert_eq!(dt.global_topology(), global_topology);
        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    }

    #[test]
    fn test_repair_delaunay_with_flips_allows_pl_manifold() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let result = dt.repair_delaunay_with_flips();
        assert!(
            !matches!(result, Err(DelaunayRepairError::InvalidTopology { .. })),
            "Flip-based repair should be admissible under PLManifold topology"
        );
    }

    /// When the primary flip repair returns `NonConvergent`, the advanced repair
    /// method falls back to `repair_delaunay_with_flips_robust`.  On a valid
    /// triangulation the robust pass succeeds, so the outcome reports no
    /// heuristic rebuild.
    #[test]
    fn test_repair_delaunay_with_flips_advanced_robust_fallback_succeeds() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let _guard = ForceRepairNonconvergentGuard::enable();
        let outcome = dt
            .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
            .unwrap();
        assert!(
            !outcome.used_heuristic(),
            "Robust fallback should succeed without needing heuristic rebuild"
        );
    }

    /// Verifies that `DelaunayRepairHeuristicConfig::max_flips` caps the repair budget
    /// when called through the public `repair_delaunay_with_flips_advanced` API.
    ///
    /// Sub-case 1: A budget of 0 on a triangulation that is already Delaunay should succeed
    /// (the initial repair pass finds no violations).
    ///
    /// Sub-case 2: A budget of 0 on a forced-non-convergent state should hit the
    /// robust fallback path (the primary pass returns `NonConvergent`, the robust
    /// pass succeeds because the triangulation is actually Delaunay).
    #[test]
    fn test_repair_advanced_max_flips_zero_on_valid_triangulation_succeeds() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Sub-case 1: Already Delaunay — max_flips=0 should succeed (no flips needed).
        let config = DelaunayRepairHeuristicConfig {
            max_flips: Some(0),
            ..DelaunayRepairHeuristicConfig::default()
        };
        let outcome = dt.repair_delaunay_with_flips_advanced(config).unwrap();
        assert_eq!(outcome.stats.flips_performed, 0);
        assert!(
            !outcome.used_heuristic(),
            "Already-Delaunay triangulation should not trigger heuristic rebuild"
        );
    }

    /// Sub-case 2 of the `max_flips` budget test: force the primary repair to fail
    /// (via `ForceRepairNonconvergentGuard`) with `max_flips=0`, then verify the
    /// robust fallback succeeds (the triangulation is actually valid).
    #[test]
    fn test_repair_advanced_max_flips_zero_forced_nonconvergent_hits_robust_fallback() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let _guard = ForceRepairNonconvergentGuard::enable();
        let config = DelaunayRepairHeuristicConfig {
            max_flips: Some(0),
            ..DelaunayRepairHeuristicConfig::default()
        };
        // The primary repair is forced to fail; the robust fallback should succeed
        // because the triangulation is actually Delaunay.
        let outcome = dt.repair_delaunay_with_flips_advanced(config).unwrap();
        assert_eq!(
            outcome.stats.flips_performed, 0,
            "max_flips=0 should prevent any flips even on the robust fallback path"
        );
        assert!(
            !outcome.used_heuristic(),
            "Robust fallback should succeed without heuristic rebuild"
        );
    }

    /// Sub-case 3:
    /// verify `max_flips=0` returns `NonConvergent`, then retry with a sufficient budget
    /// and verify repair succeeds with flips performed.
    #[test]
    fn test_repair_advanced_max_flips_on_non_delaunay_triangulation() {
        init_tracing();

        // Reuse the explicit non-Delaunay quadrilateral fixture so the primary
        // and robust fallback kernels both see a real flip-repair site.
        let kernel = AdaptiveKernel::<f64>::new();
        let robust_kernel = RobustKernel::<f64>::new();
        let tds = non_delaunay_quad_tds();
        assert!(verify_delaunay_via_flip_predicates(&tds, &kernel).is_err());
        assert!(verify_delaunay_via_flip_predicates(&tds, &robust_kernel).is_err());
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds_with_topology_guarantee(
                tds,
                kernel,
                TopologyGuarantee::PLManifold,
            );
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        // max_flips=0 should fail (flips are needed but budget is zero).
        let config_zero = DelaunayRepairHeuristicConfig {
            max_flips: Some(0),
            ..DelaunayRepairHeuristicConfig::default()
        };
        // The advanced path tries primary (fails at budget=0), then robust fallback.
        // The robust fallback also respects the budget, so it should also fail at 0,
        // then the heuristic rebuild fires. The key assertion: it should not silently
        // succeed with 0 flips on the primary path.
        let outcome_zero = dt.repair_delaunay_with_flips_advanced(config_zero);
        // Either heuristic rebuild succeeds or we get an error — both are acceptable.
        // What would be wrong is a silent Ok with 0 flips on a non-Delaunay input.
        if let Ok(ref outcome) = outcome_zero {
            assert!(
                outcome.used_heuristic() || outcome.stats.flips_performed > 0,
                "max_flips=0 on non-Delaunay input must not silently succeed with 0 flips and no heuristic"
            );
        }

        // Now retry with a generous budget — should succeed.
        let config_generous = DelaunayRepairHeuristicConfig {
            max_flips: Some(100),
            ..DelaunayRepairHeuristicConfig::default()
        };
        // Reconstruct dt from the same raw TDS in case the previous attempt mutated it.
        let tds2 = non_delaunay_quad_tds();
        let mut dt2: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds_with_topology_guarantee(
                tds2,
                AdaptiveKernel::new(),
                TopologyGuarantee::PLManifold,
            );
        dt2.set_topology_guarantee(TopologyGuarantee::PLManifold);
        let outcome_generous = dt2
            .repair_delaunay_with_flips_advanced(config_generous)
            .unwrap();
        assert!(
            outcome_generous.stats.flips_performed > 0,
            "Generous budget should allow flips to repair the non-Delaunay triangulation"
        );
    }

    /// `repair_delaunay_with_flips` delegates to `repair_delaunay_with_flips_k2_k3`
    /// which requires D ≥ 2.  On a 1D triangulation the inner function returns
    /// `FlipError::UnsupportedDimension`, surfaced as `DelaunayRepairError::Flip`.
    #[test]
    fn test_repair_delaunay_with_flips_returns_flip_error_for_1d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 1>> = vec![vertex!([0.0]), vertex!([1.0])];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 1> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let result = dt.repair_delaunay_with_flips();
        assert!(
            matches!(
                result,
                Err(DelaunayRepairError::Flip { ref source })
                    if matches!(
                        source.as_ref(),
                        FlipError::UnsupportedDimension { dimension: 1 }
                    )
            ),
            "Expected Flip(UnsupportedDimension {{ dimension: 1 }}) for D=1, got: {result:?}"
        );
    }

    /// `repair_delaunay_with_flips_advanced` passes through non-retryable errors
    /// (anything other than `NonConvergent` / `PostconditionFailed`) from the
    /// inner `repair_delaunay_with_flips` call.  A 1D triangulation triggers
    /// `UnsupportedDimension` which must hit the `Err(err) => Err(err)` arm.
    #[test]
    fn test_repair_delaunay_with_flips_advanced_passes_through_non_retryable_error() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 1>> = vec![vertex!([0.0]), vertex!([1.0])];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 1> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let result =
            dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default());
        assert!(
            matches!(
                result,
                Err(DelaunayRepairError::Flip { ref source })
                    if matches!(
                        source.as_ref(),
                        FlipError::UnsupportedDimension { dimension: 1 }
                    )
            ),
            "Expected non-retryable Flip(UnsupportedDimension) pass-through for D=1, got: {result:?}"
        );
    }

    // ---- advanced repair fallback-chain error context tests ----

    /// Verify that the `HeuristicRebuildFailed` error from
    /// `repair_delaunay_with_flips_advanced` includes the full fallback
    /// chain context (primary, robust, and heuristic failures) when all
    /// three stages fail.
    #[test]
    fn test_advanced_repair_fallback_error_preserves_full_chain_context() {
        // Construct the error exactly the way `repair_delaunay_with_flips_advanced`
        // builds it when all three stages fail.
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
            message: "robust postcondition failure".to_string(),
        };
        let heuristic_inner = DelaunayRepairError::HeuristicRebuildFailed {
            message: "heuristic rebuild failed after 3 attempts: attempt 3/3 (shuffle_seed=1 perturbation_seed=2): inner".to_string(),
        };

        // Simulate the map_err closure in repair_delaunay_with_flips_advanced.
        let heuristic_message = match heuristic_inner {
            DelaunayRepairError::HeuristicRebuildFailed { message } => message,
            other => other.to_string(),
        };
        let combined = DelaunayRepairError::HeuristicRebuildFailed {
            message: format!(
                "primary repair failed ({primary_err}); robust fallback failed ({robust_err}); {heuristic_message}"
            ),
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
            msg.contains("robust postcondition failure"),
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

        assert!(every_2.should_check(0));
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

    #[test]
    fn heuristic_config_resolves_missing_seeds_deterministically() {
        let config = DelaunayRepairHeuristicConfig {
            shuffle_seed: None,
            perturbation_seed: Some(11),
            max_flips: Some(7),
        };

        let seeds = config.resolve_seeds(5);

        assert_ne!(seeds.shuffle_seed, 0);
        assert_eq!(seeds.perturbation_seed, 11);
    }

    #[test]
    fn heuristic_config_keeps_explicit_zero_seeds() {
        let config = DelaunayRepairHeuristicConfig {
            shuffle_seed: Some(0),
            perturbation_seed: Some(0),
            max_flips: None,
        };

        let seeds = config.resolve_seeds(0);

        assert_eq!(seeds.shuffle_seed, 0);
        assert_eq!(seeds.perturbation_seed, 0);
    }

    #[test]
    fn repair_outcome_reports_whether_heuristic_was_used() {
        let without_heuristic = DelaunayRepairOutcome {
            stats: DelaunayRepairStats::default(),
            heuristic: None,
        };
        let with_heuristic = DelaunayRepairOutcome {
            stats: DelaunayRepairStats::default(),
            heuristic: Some(DelaunayRepairHeuristicSeeds {
                shuffle_seed: 1,
                perturbation_seed: 2,
            }),
        };

        assert!(!without_heuristic.used_heuristic());
        assert!(with_heuristic.used_heuristic());
    }
}
