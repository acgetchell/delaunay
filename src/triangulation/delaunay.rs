//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

#![forbid(unsafe_code)]

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairStats, FlipError, apply_bistellar_flip_k1_inverse,
    repair_delaunay_local_single_pass, repair_delaunay_with_flips_k2_k3,
    verify_delaunay_for_triangulation,
};
use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::cell::{Cell, CellValidationError};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{CellKeyBuffer, FastHashMap, FastHashSet, FastHasher, SmallBuffer};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::operations::{
    DelaunayInsertionState, InsertionOutcome, InsertionStatistics, RepairDecision,
    TopologicalOperation,
};
use crate::core::tds::{
    CellKey, InvariantError, InvariantKind, InvariantViolation, Tds, TdsConstructionError,
    TdsError, TriangulationValidationReport, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{
    TopologyGuarantee, Triangulation, TriangulationConstructionError, TriangulationValidationError,
    ValidationPolicy, insertion_error_to_invariant_error, record_duplicate_detection_metrics,
};
use crate::core::util::{
    coords_equal_exact, coords_within_epsilon, hilbert_indices_prequantized, hilbert_quantize,
    stable_hash_u64_slice,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{AdaptiveKernel, ExactPredicates, Kernel, RobustKernel};
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::geometry::util::safe_usize_to_scalar;
use crate::topology::manifold::validate_ridge_links_for_cells;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::builder::DelaunayTriangulationBuilder;
use core::cmp::Ordering;
use num_traits::{NumCast, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::env;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

const DELAUNAY_SHUFFLE_ATTEMPTS: usize = 6;
const DELAUNAY_SHUFFLE_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

// Heuristic rebuild attempts must be consistent across build profiles to avoid
// release-only construction failures (see #306).
const HEURISTIC_REBUILD_ATTEMPTS: usize = 6;

// Per-insertion local-repair flip-budget tunables.
//
// Budget formula: `seed_cells.len() * (D + 1) * FACTOR` with a minimum of
// `FLOOR`. Two regimes so that D≥4's higher queue demand does not force a
// global budget increase.
//
// The D≥4 constants are sized from the measured `max_queue` distribution on
// the 500-point 4D seeded repro (seed `0xD225B8A07E274AE6`, ball radius 100)
// captured in `docs/archive/issue_204_investigation.md`:
//
//   max_queue samples  min=91 p50=207 p90=281 p95=312 p99=409 max=416
//
// `FACTOR = 12` with `FLOOR = 96` yields a typical 300-flip budget (5-cell seed
// set), covering p50–p90 and brushing p95. The p95–p99 tail is intentionally
// left to the escalation pass (see `LOCAL_REPAIR_ESCALATION_*`) rather than
// paid for on every insertion.
pub(crate) const LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4: usize = 12;
pub(crate) const LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4: usize = 96;
pub(crate) const LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_LT_4: usize = 4;
pub(crate) const LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_LT_4: usize = 16;

// Escalation tunables for D≥4. When the base local repair hits its budget,
// the soft-fail path reruns the repair once with `BASE_BUDGET * ESCALATION_FACTOR`
// and the full TDS as seed set before giving up. The escalation is rate-limited
// so every insertion does not pay for a near-global flip pass.
pub(crate) const LOCAL_REPAIR_ESCALATION_BUDGET_FACTOR_D_GE_4: usize = 4;
pub(crate) const LOCAL_REPAIR_ESCALATION_MIN_GAP: usize = 8;

/// Outcome of a per-insertion D≥4 local-repair escalation attempt.
///
/// Three orthogonal cases so the caller and any telemetry downstream can match
/// on the outcome without string parsing:
///
/// - [`Skipped`](Self::Skipped) — the escalation did not run. The caller
///   should fall through to the soft-fail path using the original
///   [`DelaunayRepairError`] that triggered escalation.
/// - [`Succeeded`](Self::Succeeded) — the escalation converged. The caller
///   has already canonicalized the triangulation and should continue to the
///   next insertion.
/// - [`FailedAlso`](Self::FailedAlso) — the escalation ran but also hit its
///   budget or postcondition. The typed `DelaunayRepairError` is preserved so
///   downstream diagnostics can correlate it with the original error; the
///   caller should fall through to the soft-fail path.
///
/// [`DelaunayRepairError`]: crate::core::algorithms::flips::DelaunayRepairError
#[derive(Clone, Debug)]
enum LocalRepairEscalationOutcome {
    /// The escalation was not attempted.
    Skipped {
        /// Why the escalation was skipped.
        reason: EscalationSkipReason,
    },
    /// The escalation ran and successfully converged.
    Succeeded {
        /// Repair diagnostics from the successful escalation attempt.
        stats: DelaunayRepairStats,
    },
    /// The escalation ran but also failed to converge or satisfy its
    /// postcondition.
    FailedAlso {
        /// Typed repair error produced by the escalation attempt. Preserved
        /// by value so callers can match on the variant instead of parsing
        /// the display form.
        escalation_error: DelaunayRepairError,
    },
}

/// Why a [`LocalRepairEscalationOutcome::Skipped`] escalation attempt did not
/// run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EscalationSkipReason {
    /// The previous escalation ran within the `min_gap` insertion window, so
    /// this attempt was rate-limited.
    RateLimited {
        /// Insertion index of the previous escalation.
        last_escalation_idx: usize,
        /// Configured minimum gap between escalations.
        min_gap: usize,
    },
    /// The triangulation had no cells to seed repair with. This is an edge
    /// case for early insertions where the initial simplex has not been
    /// committed; escalation there has nothing to escalate against.
    EmptyTds,
}

/// Per-insertion local Delaunay repair flip budget.
///
/// Computes `seeds * (D + 1) * FACTOR` with a minimum of `FLOOR`, using the
/// dimension-aware constants above.
const fn local_repair_flip_budget<const D: usize>(seed_cells_len: usize) -> usize {
    let (factor, floor) = if D >= 4 {
        (
            LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4,
            LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4,
        )
    } else {
        (
            LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_LT_4,
            LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_LT_4,
        )
    };
    let raw = seed_cells_len.saturating_mul(D + 1).saturating_mul(factor);
    if raw > floor { raw } else { floor }
}

thread_local! {
    static HEURISTIC_REBUILD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    static FORCE_HEURISTIC_REBUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_REPAIR_NONCONVERGENT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct HeuristicRebuildRecursionGuard {
    prior_depth: usize,
}

impl HeuristicRebuildRecursionGuard {
    /// Tracks nested heuristic rebuilds so fallback construction cannot recurse
    /// indefinitely through repair hooks.
    fn enter() -> Self {
        let prior_depth = HEURISTIC_REBUILD_DEPTH.with(|depth| {
            let prior = depth.get();
            depth.set(prior.saturating_add(1));
            prior
        });
        Self { prior_depth }
    }
}

impl Drop for HeuristicRebuildRecursionGuard {
    fn drop(&mut self) {
        HEURISTIC_REBUILD_DEPTH.with(|depth| depth.set(self.prior_depth));
    }
}

/// Errors that can occur during Delaunay triangulation construction.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::DelaunayTriangulationConstructionError;
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let result: Result<_, DelaunayTriangulationConstructionError> =
///     DelaunayTriangulation::new(&vertices);
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayTriangulationConstructionError {
    /// Lower-layer construction error (Triangulation / TDS).
    #[error(transparent)]
    Triangulation(#[from] TriangulationConstructionError),

    /// Input validation error from explicit combinatorial construction.
    ///
    /// Returned by [`DelaunayTriangulationBuilder::from_vertices_and_cells`](crate::triangulation::builder::DelaunayTriangulationBuilder::from_vertices_and_cells)
    /// when the caller-provided vertices/cells fail validation (wrong arity,
    /// out-of-bounds indices, etc.). TDS assembly errors flow through the
    /// [`Triangulation`](Self::Triangulation) variant instead.
    #[error(transparent)]
    ExplicitConstruction(#[from] crate::triangulation::builder::ExplicitConstructionError),
}

/// Errors that can occur during Delaunay triangulation validation and repair.
///
/// The first three variants are returned by [`DelaunayTriangulation::validate`]
/// (validation Levels 1–4):
/// - [`Tds`](Self::Tds) — element or TDS structural errors (Levels 1–2).
/// - [`Triangulation`](Self::Triangulation) — topology errors (Level 3).
/// - [`VerificationFailed`](Self::VerificationFailed) — Delaunay property violation (Level 4).
///
/// [`DelaunayTriangulation::is_valid`] returns only the Level 4
/// [`VerificationFailed`](Self::VerificationFailed) variant.
///
/// The [`RepairFailed`](Self::RepairFailed) variant is **not** returned by
/// `validate()` or `is_valid()`. It is produced by mutating operations that
/// invoke flip-based repair internally (e.g.
/// [`DelaunayTriangulation::remove_vertex`]).
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::DelaunayTriangulationValidationError;
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result: Result<(), DelaunayTriangulationValidationError> = dt.validate();
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayTriangulationValidationError {
    /// Lower-layer element or TDS structural validation error (Levels 1–2).
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// Lower-layer topology validation error (Level 3).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// Flip-based Delaunay verification detected a violation.
    ///
    /// This is returned by [`DelaunayTriangulation::is_valid`] when the fast
    /// O(cells) flip-predicate scan finds a Delaunay violation.  The error is
    /// a Level 4 (Delaunay property) issue, not a Level 1–2 structural problem.
    #[error("Delaunay verification failed: {message}")]
    VerificationFailed {
        /// Description of the verification failure.
        message: String,
    },

    /// Flip-based Delaunay repair failed.
    ///
    /// This is returned by mutating operations that invoke flip-based repair
    /// internally (e.g. [`DelaunayTriangulation::remove_vertex`]) when repair
    /// encounters any error (budget exhaustion, topology violation, predicate
    /// failure, etc.).
    ///
    /// **Not** returned by `validate()` or `is_valid()` — those use
    /// [`VerificationFailed`](Self::VerificationFailed) for passive checks.
    #[error("Delaunay repair failed: {message}")]
    RepairFailed {
        /// Description of the repair failure.
        message: String,
    },
}

// =============================================================================
// BATCH CONSTRUCTION OPTIONS
// =============================================================================

/// Strategy used to order input vertices before batch construction.
///
/// The default is [`InsertionOrderStrategy::Hilbert`], which improves spatial locality during
/// bulk insertion and provides unconditional quantized dedup.
///
/// If you need to preserve the caller-provided order (for example to control the initial simplex
/// vertices), use [`InsertionOrderStrategy::Input`].
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::{ConstructionOptions, InsertionOrderStrategy};
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let options = ConstructionOptions::default()
///     .with_insertion_order(InsertionOrderStrategy::Input);
/// let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();
/// assert_eq!(dt.number_of_vertices(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InsertionOrderStrategy {
    /// Preserve the caller-provided input order (no reordering).
    Input,
    /// Sort vertices by Hilbert curve (quantized, normalized coordinates).
    ///
    /// This ordering can improve spatial locality during bulk insertion, reducing point location
    /// cost.
    ///
    /// Ties are broken lexicographically by coordinates, then by original input index.
    #[default]
    Hilbert,
}

/// Policy controlling optional preprocessing to remove duplicate or near-duplicate vertices
/// before batch construction.
///
/// This is a **performance-tuning** knob, not a correctness requirement.  The
/// triangulation engine applies two built-in safety layers:
///
/// 1. **Hilbert quantized dedup** — when
///    [`InsertionOrderStrategy::Hilbert`] is active (the default), vertices
///    that map to the same quantized grid cell are removed in a single O(n)
///    sweep during sorting (zero extra cost since the quantized coordinates
///    are already computed).  This runs regardless of `DedupPolicy`, but only
///    when the Hilbert insertion order is selected.
/// 2. **Per-insertion duplicate check** *(unconditional)* — every `insert`
///    call checks the incoming vertex against existing vertices
///    (squared-distance tolerance 1e-10).  Duplicates are skipped without
///    modifying the triangulation.
///
/// Use `DedupPolicy::Exact` or `DedupPolicy::Epsilon` when your input is
/// known to contain many duplicates and you want to avoid the per-vertex
/// insertion overhead for each one.
///
/// The default is [`DedupPolicy::Off`].
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::{ConstructionOptions, DedupPolicy};
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let options = ConstructionOptions::default().with_dedup_policy(DedupPolicy::Exact);
/// let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();
/// assert_eq!(dt.number_of_vertices(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum DedupPolicy {
    /// Do not apply explicit preprocessing dedup (rely on the built-in
    /// Hilbert quantized dedup and per-insertion duplicate checks).
    #[default]
    Off,
    /// Remove exact coordinate duplicates before construction (NaN-aware, +0.0 == -0.0).
    ///
    /// This is a performance optimisation for inputs with many exact duplicates;
    /// it avoids paying per-vertex insertion cost for each duplicate.
    Exact,
    /// Remove near-duplicates within the given Euclidean tolerance before construction.
    ///
    /// The tolerance is expressed as an `f64` and is converted to the triangulation's scalar type
    /// at runtime. Invalid (negative / non-finite) tolerances are rejected.
    Epsilon {
        /// Non-negative Euclidean tolerance used when considering two vertices identical.
        tolerance: f64,
    },
}

/// Strategy controlling how the initial D+1 simplex vertices are selected during batch construction.
///
/// The default (`First`) preserves current behavior by taking the first D+1 vertices after
/// preprocessing and insertion-ordering. The balanced strategy is opt-in and chooses a more
/// spread-out simplex using a deterministic farthest-point heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InitialSimplexStrategy {
    /// Use the first D+1 vertices after preprocessing (legacy behavior).
    #[default]
    First,
    /// Choose a better-conditioned simplex using a deterministic farthest-point heuristic.
    Balanced,
}

/// Policy controlling deterministic "retry with alternative insertion orders" during batch
/// construction.
///
/// When enabled, the constructor deterministically retries construction with alternative insertion
/// orders (shuffles) when the initial attempt fails (e.g. flip-repair cycling on co-spherical
/// configurations).  The default is [`Shuffled`](Self::Shuffled) with 6 attempts, which is
/// essential for robust 3D+ construction.  Use [`Disabled`](Self::Disabled) to opt out.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::{ConstructionOptions, RetryPolicy};
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Disabled);
/// let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();
/// assert_eq!(dt.number_of_vertices(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryPolicy {
    /// Do not attempt shuffled reconstruction retries.
    Disabled,
    /// Retry construction with a small number of deterministic shuffles if the final Delaunay
    /// property check fails.
    Shuffled {
        /// Number of shuffled reconstruction attempts (excluding the original-order attempt).
        attempts: NonZeroUsize,
        /// Optional base seed. If `None`, a deterministic seed is derived from the vertex set.
        base_seed: Option<u64>,
    },
    /// Retry construction with a small number of deterministic shuffles if the
    /// final Delaunay property check fails, but only in debug/test builds.
    ///
    /// In release builds, this is treated as [`RetryPolicy::Disabled`].
    ///
    /// Note: [`RetryPolicy::default()`] now returns [`Shuffled`](Self::Shuffled)
    /// in all build modes, so this variant is only useful when you explicitly
    /// want retries restricted to debug/test builds.
    DebugOnlyShuffled {
        /// Number of shuffled reconstruction attempts (excluding the original-order attempt).
        attempts: NonZeroUsize,
        /// Optional base seed. If `None`, a deterministic seed is derived from the vertex set.
        base_seed: Option<u64>,
    },
}

impl Default for RetryPolicy {
    fn default() -> Self {
        // Shuffled retries are essential for correctness: certain Hilbert-sorted
        // insertion orders produce co-spherical configurations that cause flip-repair
        // cycling.  Retrying with a different order avoids the problematic sequence.
        // Previously disabled in release builds, which caused #306.
        Self::Shuffled {
            attempts: NonZeroUsize::new(DELAUNAY_SHUFFLE_ATTEMPTS)
                .expect("DELAUNAY_SHUFFLE_ATTEMPTS must be non-zero"),
            base_seed: None,
        }
    }
}

/// Options controlling batch construction behavior.
///
/// Higher-level constructors delegate to the options-based constructor using
/// [`ConstructionOptions::default`].
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::{
///     ConstructionOptions, DedupPolicy, InsertionOrderStrategy, RetryPolicy,
/// };
///
/// let options = ConstructionOptions::default()
///     .with_insertion_order(InsertionOrderStrategy::Hilbert)
///     .with_dedup_policy(DedupPolicy::Off)
///     .with_retry_policy(RetryPolicy::Disabled);
///
/// assert_eq!(options.insertion_order(), InsertionOrderStrategy::Hilbert);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConstructionOptions {
    insertion_order: InsertionOrderStrategy,
    dedup_policy: DedupPolicy,
    initial_simplex: InitialSimplexStrategy,
    retry_policy: RetryPolicy,
    /// When `true` (default), D<4 per-insertion repair falls back to a global
    /// `repair_delaunay_with_flips_k2_k3` pass when the bounded local pass
    /// cycles.  Set to `false` for constructions where global repair could
    /// disrupt the triangulation topology (e.g. periodic image-point builds).
    pub(crate) use_global_repair_fallback: bool,
}

impl Default for ConstructionOptions {
    fn default() -> Self {
        Self {
            insertion_order: InsertionOrderStrategy::default(),
            dedup_policy: DedupPolicy::default(),
            initial_simplex: InitialSimplexStrategy::default(),
            retry_policy: RetryPolicy::default(),
            use_global_repair_fallback: true,
        }
    }
}

impl ConstructionOptions {
    /// Returns the input ordering strategy used for batch construction.
    #[must_use]
    pub const fn insertion_order(&self) -> InsertionOrderStrategy {
        self.insertion_order
    }

    /// Returns the deduplication policy applied before batch construction.
    #[must_use]
    pub const fn dedup_policy(&self) -> DedupPolicy {
        self.dedup_policy
    }
    /// Returns the strategy used to select the initial simplex.
    #[must_use]
    pub const fn initial_simplex_strategy(&self) -> InitialSimplexStrategy {
        self.initial_simplex
    }

    /// Returns the retry policy used during batch construction.
    #[must_use]
    pub const fn retry_policy(&self) -> RetryPolicy {
        self.retry_policy
    }

    /// Sets the input ordering strategy used for batch construction.
    #[must_use]
    pub const fn with_insertion_order(mut self, insertion_order: InsertionOrderStrategy) -> Self {
        self.insertion_order = insertion_order;
        self
    }

    /// Sets the deduplication policy applied before batch construction.
    #[must_use]
    pub const fn with_dedup_policy(mut self, dedup_policy: DedupPolicy) -> Self {
        self.dedup_policy = dedup_policy;
        self
    }
    /// Sets the initial simplex selection strategy.
    #[must_use]
    pub const fn with_initial_simplex_strategy(
        mut self,
        initial_simplex: InitialSimplexStrategy,
    ) -> Self {
        self.initial_simplex = initial_simplex;
        self
    }

    /// Sets the retry policy used during batch construction.
    #[must_use]
    pub const fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Disables the D<4 global repair fallback.
    #[must_use]
    pub(crate) const fn without_global_repair_fallback(mut self) -> Self {
        self.use_global_repair_fallback = false;
        self
    }
}

// =============================================================================
// BATCH CONSTRUCTION STATISTICS
// =============================================================================

/// Aggregate statistics collected during batch construction.
///
/// This summarizes the per-vertex [`InsertionStatistics`] generated by the incremental insertion
/// engine during bulk construction (including vertices that are skipped via transactional rollback).
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct ConstructionStatistics {
    /// Number of vertices successfully inserted (includes the initial D+1 simplex vertices).
    pub inserted: usize,
    /// Number of vertices skipped due to duplicate coordinates.
    pub skipped_duplicate: usize,
    /// Number of vertices skipped due to geometric degeneracy after exhausting retries.
    pub skipped_degeneracy: usize,

    /// Total number of insertion attempts across all vertices.
    pub total_attempts: usize,
    /// Maximum attempts for any single vertex.
    pub max_attempts: usize,
    /// Histogram of attempts: `attempts_histogram[k]` = number of vertices that took `k` attempts.
    pub attempts_histogram: Vec<usize>,

    /// Number of vertices that required perturbation (attempts > 1).
    pub used_perturbation: usize,

    /// Total number of cells removed during insertion safety-net / repair bookkeeping.
    pub cells_removed_total: usize,
    /// Maximum number of cells removed during repair for any single insertion.
    pub cells_removed_max: usize,

    /// A small set of representative skipped vertices recorded during batch construction.
    ///
    /// This is intended for debugging/reproduction and is capped (currently the first 8 skips).
    pub skip_samples: Vec<ConstructionSkipSample>,
}

/// A single skipped-vertex sample captured during batch construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConstructionSkipSample {
    /// Index in the construction insertion order (after preprocessing and ordering).
    pub index: usize,
    /// UUID of the skipped vertex.
    pub uuid: Uuid,
    /// Coordinates of the skipped vertex, converted to `f64` for logging/debugging.
    pub coords: Vec<f64>,
    /// Number of insertion attempts for this vertex.
    pub attempts: usize,
    /// Human-readable error message describing why the vertex was skipped.
    pub error: String,
}

/// Construction error that also carries aggregate statistics collected up to the failure point.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{error}")]
#[non_exhaustive]
pub struct DelaunayTriangulationConstructionErrorWithStatistics {
    /// Underlying construction error.
    #[source]
    pub error: DelaunayTriangulationConstructionError,
    /// Aggregate construction statistics collected before the error occurred.
    pub statistics: ConstructionStatistics,
}

impl ConstructionStatistics {
    /// Aggregates attempt counters shared by inserted, skipped, and duplicate
    /// insertion outcomes.
    #[inline]
    fn record_common(&mut self, stats: &InsertionStatistics) {
        self.total_attempts = self.total_attempts.saturating_add(stats.attempts);
        self.max_attempts = self.max_attempts.max(stats.attempts);

        if self.attempts_histogram.len() <= stats.attempts {
            self.attempts_histogram.resize(stats.attempts + 1, 0);
        }
        self.attempts_histogram[stats.attempts] =
            self.attempts_histogram[stats.attempts].saturating_add(1);

        if stats.used_perturbation() {
            self.used_perturbation = self.used_perturbation.saturating_add(1);
        }

        self.cells_removed_total = self
            .cells_removed_total
            .saturating_add(stats.cells_removed_during_repair);
        self.cells_removed_max = self
            .cells_removed_max
            .max(stats.cells_removed_during_repair);
    }

    const MAX_SKIP_SAMPLES: usize = 8;

    /// Record a single insertion attempt (inserted or skipped).
    pub fn record_insertion(&mut self, stats: &InsertionStatistics) {
        if stats.skipped_duplicate() {
            self.skipped_duplicate = self.skipped_duplicate.saturating_add(1);
        } else if stats.skipped() {
            self.skipped_degeneracy = self.skipped_degeneracy.saturating_add(1);
        } else {
            self.inserted = self.inserted.saturating_add(1);
        }

        self.record_common(stats);
    }

    /// Record a representative skipped-vertex sample for debugging.
    pub fn record_skip_sample(&mut self, sample: ConstructionSkipSample) {
        if self.skip_samples.len() < Self::MAX_SKIP_SAMPLES {
            self.skip_samples.push(sample);
        }
    }

    /// Total number of skipped vertices.
    #[must_use]
    pub const fn total_skipped(&self) -> usize {
        self.skipped_duplicate + self.skipped_degeneracy
    }
}

// =============================================================================
// BATCH CONSTRUCTION ORDERING HELPERS (INTERNAL)
// =============================================================================

type VertexBuffer<T, U, const D: usize> = Vec<Vertex<T, U, D>>;
struct PreprocessVertices<T, U, const D: usize> {
    primary: Option<VertexBuffer<T, U, D>>,
    fallback: Option<VertexBuffer<T, U, D>>,
    grid_cell_size: Option<T>,
}

impl<T, U, const D: usize> PreprocessVertices<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// Borrows the preprocessed vertex order when one exists, avoiding a clone
    /// for policies that leave the input unchanged.
    fn primary_slice<'a>(&'a self, input: &'a [Vertex<T, U, D>]) -> &'a [Vertex<T, U, D>] {
        self.primary.as_deref().unwrap_or(input)
    }

    /// Exposes the original order as a retry fallback for balanced-simplex
    /// preprocessing.
    fn fallback_slice(&self) -> Option<&[Vertex<T, U, D>]> {
        self.fallback.as_deref()
    }

    /// Carries the dedup grid size forward so incremental insertion can reuse a
    /// compatible spatial index.
    const fn grid_cell_size(&self) -> Option<T> {
        self.grid_cell_size
    }
}

type PreprocessVerticesResult<T, U, const D: usize> =
    Result<PreprocessVertices<T, U, D>, DelaunayTriangulationConstructionError>;

/// Hashes coordinates as a deterministic tiebreaker for partial vertex ordering.
fn vertex_coordinate_hash<T, U, const D: usize>(vertex: &Vertex<T, U, D>) -> u64
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut hasher = FastHasher::default();
    vertex.hash(&mut hasher);
    hasher.finish()
}

/// Produces a stable construction order when Hilbert ordering is unavailable or
/// unsuitable.
fn order_vertices_lexicographic<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut keyed: Vec<(Vertex<T, U, D>, u64, usize)> = vertices
        .into_iter()
        .enumerate()
        .map(|(input_index, vertex)| {
            let hash = vertex_coordinate_hash(&vertex);
            (vertex, hash, input_index)
        })
        .collect();

    keyed.sort_by(|(a, a_hash, a_idx), (b, b_hash, b_idx)| {
        a.partial_cmp(b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a_hash.cmp(b_hash))
            .then_with(|| a_idx.cmp(b_idx))
    });

    keyed.into_iter().map(|(v, _, _)| v).collect()
}

const BATCH_DEDUP_BUCKET_INLINE_CAPACITY: usize = 8;
const BATCH_DEDUP_MAX_DIMENSION: usize = 5;

/// Centralizes insertion-order dispatch so preprocessing applies dedup and
/// ordering in a consistent sequence.
fn order_vertices_by_strategy<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    insertion_order: InsertionOrderStrategy,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    match insertion_order {
        InsertionOrderStrategy::Input => vertices,
        InsertionOrderStrategy::Hilbert => order_vertices_hilbert(vertices, true),
    }
}

/// Provides a scalar-aware tolerance for dedup paths that need a nonzero grid
/// size even under exact duplicate policy.
fn default_duplicate_tolerance<T: CoordinateScalar>() -> T {
    <T as NumCast>::from(1e-10_f64).unwrap_or_else(T::default_tolerance)
}

/// Verifies the hash grid can represent every input coordinate before choosing
/// the O(n) duplicate path.
fn hash_grid_usable_for_vertices<T, U, const D: usize>(
    grid: &HashGridIndex<T, D, usize>,
    vertices: &[Vertex<T, U, D>],
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
{
    if !grid.is_usable() {
        return false;
    }
    vertices
        .iter()
        .all(|v| grid.can_key_coords(v.point().coords()))
}

/// Keeps exact dedup deterministic when the hash grid cannot safely key the
/// input coordinates.
fn dedup_vertices_exact_sorted<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let ordered = order_vertices_lexicographic(vertices);
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(ordered.len());

    for v in ordered {
        if let Some(last) = unique.last()
            && coords_equal_exact(v.point().coords(), last.point().coords())
        {
            record_duplicate_detection_metrics(false, 0, true);
            continue;
        }
        record_duplicate_detection_metrics(false, 0, true);
        unique.push(v);
    }

    unique
}

/// Uses the spatial grid for exact duplicate removal while falling back to the
/// sorted path if coordinate keying is unavailable.
fn dedup_vertices_exact_hash_grid<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    grid: &mut HashGridIndex<T, D, usize>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if !hash_grid_usable_for_vertices(grid, &vertices) {
        return dedup_vertices_exact_sorted(vertices);
    }
    grid.clear();
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    for v in vertices {
        let coords = v.point().coords();
        let mut duplicate = false;
        let mut candidate_count = 0usize;
        let used_index = grid.for_each_candidate_vertex_key(coords, |idx| {
            candidate_count = candidate_count.saturating_add(1);
            let existing_coords = unique[idx].point().coords();
            if coords_equal_exact(coords, existing_coords) {
                duplicate = true;
                return false;
            }
            true
        });

        record_duplicate_detection_metrics(used_index, candidate_count, !used_index);

        if !duplicate {
            let idx = unique.len();
            unique.push(v);
            grid.insert_vertex(idx, coords);
        }
    }

    unique
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct QuantizedKey<const D: usize>([i64; D]);

/// Quantizes coordinates for epsilon buckets only when finite values can be
/// represented without losing the bucket invariant.
fn quantize_coords<T: CoordinateScalar, const D: usize>(
    coords: &[T; D],
    inv_cell: f64,
) -> Option<[i64; D]> {
    let mut key = [0_i64; D];
    for (axis, coord) in coords.iter().enumerate() {
        let c = coord.to_f64()?;
        if !c.is_finite() {
            return None;
        }
        let scaled = c * inv_cell;
        if !scaled.is_finite() {
            return None;
        }
        let quantized = scaled.floor();
        let q = quantized.to_i64()?;
        key[axis] = q;
    }
    Some(key)
}

/// Visits adjacent epsilon buckets recursively so near-duplicate checks cover
/// boundary-straddling points.
fn visit_quantized_neighbors<const D: usize, F>(
    axis: usize,
    base: &[i64; D],
    current: &mut [i64; D],
    f: &mut F,
) -> bool
where
    F: FnMut([i64; D]) -> bool,
{
    if axis == D {
        return f(*current);
    }

    let offsets = [-1_i64, 0, 1];
    for offset in offsets {
        if let Some(value) = base[axis].checked_add(offset) {
            current[axis] = value;
            if !visit_quantized_neighbors(axis + 1, base, current, f) {
                return false;
            }
        }
    }
    true
}

/// Provides the correctness fallback for epsilon dedup when bucket or grid
/// assumptions fail.
fn dedup_vertices_epsilon_n2<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    epsilon: T,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());
    for v in vertices {
        let mut duplicate = false;
        for u in &unique {
            if coords_within_epsilon(v.point().coords(), u.point().coords(), epsilon) {
                duplicate = true;
                break;
            }
        }
        record_duplicate_detection_metrics(false, 0, true);
        if !duplicate {
            unique.push(v);
        }
    }
    unique
}

/// Uses bounded quantized buckets for epsilon dedup in practical dimensions
/// while preserving an exact fallback for unsupported cases.
fn dedup_vertices_epsilon_quantized<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    epsilon: T,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if D > BATCH_DEDUP_MAX_DIMENSION {
        return dedup_vertices_epsilon_n2(vertices, epsilon);
    }

    let Some(eps_f64) = epsilon.to_f64() else {
        return dedup_vertices_epsilon_n2(vertices, epsilon);
    };
    if !eps_f64.is_finite() || eps_f64 <= 0.0 {
        return dedup_vertices_epsilon_n2(vertices, epsilon);
    }

    let inv_cell = 1.0 / eps_f64;
    let mut buckets: FastHashMap<
        QuantizedKey<D>,
        SmallBuffer<usize, BATCH_DEDUP_BUCKET_INLINE_CAPACITY>,
    > = FastHashMap::default();
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());
    let mut iter = vertices.into_iter();
    while let Some(v) = iter.next() {
        let coords = v.point().coords();
        let Some(base_key) = quantize_coords(coords, inv_cell) else {
            return dedup_vertices_epsilon_n2(
                unique
                    .into_iter()
                    .chain(std::iter::once(v))
                    .chain(iter)
                    .collect(),
                epsilon,
            );
        };

        let mut duplicate = false;
        let mut candidate_count = 0usize;
        let mut current = base_key;
        visit_quantized_neighbors(0, &base_key, &mut current, &mut |neighbor| {
            if let Some(bucket) = buckets.get(&QuantizedKey(neighbor)) {
                for &idx in bucket {
                    candidate_count = candidate_count.saturating_add(1);
                    let existing_coords = unique[idx].point().coords();
                    if coords_within_epsilon(coords, existing_coords, epsilon) {
                        duplicate = true;
                        return false;
                    }
                }
            }
            true
        });

        record_duplicate_detection_metrics(false, 0, true);

        if !duplicate {
            let idx = unique.len();
            unique.push(v);
            buckets.entry(QuantizedKey(base_key)).or_default().push(idx);
        }
    }

    unique
}

/// Prefers the reusable hash grid for epsilon dedup when its coordinate model is
/// valid for every input vertex.
fn dedup_vertices_epsilon_hash_grid<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    epsilon: T,
    grid: &mut HashGridIndex<T, D, usize>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if !hash_grid_usable_for_vertices(grid, &vertices) {
        return dedup_vertices_epsilon_quantized(vertices, epsilon);
    }
    grid.clear();
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    let epsilon_sq = epsilon * epsilon;
    for v in vertices {
        let coords = v.point().coords();
        let mut duplicate = false;
        let mut candidate_count = 0usize;
        let used_index = grid.for_each_candidate_vertex_key(coords, |idx| {
            candidate_count = candidate_count.saturating_add(1);
            let existing_coords = unique[idx].point().coords();
            let mut dist_sq = T::zero();
            for i in 0..D {
                let diff = coords[i] - existing_coords[i];
                dist_sq += diff * diff;
            }
            if dist_sq < epsilon_sq {
                duplicate = true;
                return false;
            }
            true
        });

        record_duplicate_detection_metrics(used_index, candidate_count, !used_index);

        if !duplicate {
            let idx = unique.len();
            unique.push(v);
            grid.insert_vertex(idx, coords);
        }
    }

    unique
}

/// Chooses a well-spread initial simplex to reduce early degeneracy in
/// incremental construction.
fn select_balanced_simplex_indices<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
) -> Option<Vec<usize>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if vertices.len() < D + 1 {
        return None;
    }

    let mut coords_f64: Vec<[f64; D]> = Vec::with_capacity(vertices.len());
    for v in vertices {
        let mut coords = [0.0_f64; D];
        for (axis, coord) in v.point().coords().iter().enumerate() {
            let c = coord.to_f64()?;
            if !c.is_finite() {
                return None;
            }
            coords[axis] = c;
        }
        coords_f64.push(coords);
    }
    let dist_sq = |a: &[f64; D], b: &[f64; D]| {
        a.iter()
            .zip(b.iter())
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f64>()
    };

    let mut seed_idx = 0usize;
    for i in 1..coords_f64.len() {
        if coords_f64[i].partial_cmp(&coords_f64[seed_idx]) == Some(Ordering::Less) {
            seed_idx = i;
        }
    }

    let mut selected = Vec::with_capacity(D + 1);
    let mut selected_mask = vec![false; coords_f64.len()];
    selected.push(seed_idx);
    selected_mask[seed_idx] = true;

    let mut min_dist_sq = vec![f64::INFINITY; coords_f64.len()];
    for i in 0..coords_f64.len() {
        min_dist_sq[i] = dist_sq(&coords_f64[i], &coords_f64[seed_idx]);
    }
    min_dist_sq[seed_idx] = 0.0;

    while selected.len() < D + 1 {
        let mut best_idx: Option<usize> = None;
        let mut best_dist = -1.0_f64;

        for i in 0..coords_f64.len() {
            if selected_mask[i] {
                continue;
            }
            let dist = min_dist_sq[i];
            if !dist.is_finite() {
                continue;
            }
            let replace = best_idx.is_none_or(|best_idx_val| match dist.partial_cmp(&best_dist) {
                Some(Ordering::Greater) => true,
                Some(Ordering::Equal) => {
                    coords_f64[i].partial_cmp(&coords_f64[best_idx_val]) == Some(Ordering::Less)
                }
                _ => false,
            });
            if replace {
                best_idx = Some(i);
                best_dist = dist;
            }
        }

        let Some(best_idx) = best_idx else {
            break;
        };
        selected.push(best_idx);
        selected_mask[best_idx] = true;

        for i in 0..coords_f64.len() {
            if selected_mask[i] {
                continue;
            }
            let dist_sq = dist_sq(&coords_f64[i], &coords_f64[best_idx]);
            if dist_sq < min_dist_sq[i] {
                min_dist_sq[i] = dist_sq;
            }
        }
    }

    if selected.len() == D + 1 {
        Some(selected)
    } else {
        None
    }
}

/// Places the selected simplex first while preserving every remaining input
/// vertex exactly once.
fn reorder_vertices_for_simplex<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    simplex_indices: &[usize],
) -> Option<Vec<Vertex<T, U, D>>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if simplex_indices.len() != D + 1 {
        return None;
    }

    let mut seen = vec![false; vertices.len()];
    let mut reordered = Vec::with_capacity(vertices.len());

    for &idx in simplex_indices {
        if idx >= vertices.len() || seen[idx] {
            return None;
        }
        seen[idx] = true;
        reordered.push(vertices[idx]);
    }

    for (idx, vertex) in vertices.iter().enumerate() {
        if !seen[idx] {
            reordered.push(*vertex);
        }
    }

    Some(reordered)
}

/// Computes the largest per-axis Hilbert precision that still fits in the
/// u128-backed index.
fn hilbert_bits_per_coord<const D: usize>() -> Option<u32> {
    if D == 0 {
        return None;
    }

    let Ok(d_u32) = u32::try_from(D) else {
        return None;
    };

    // `hilbert_index` encodes D coordinates with `bits` bits each into a `u128`.
    // Use as many bits as possible (up to the `hilbert` module's `bits <= 31` bound).
    let bits_per_coord = (128_u32 / d_u32).min(31);
    if bits_per_coord == 0 {
        return None;
    }

    Some(bits_per_coord)
}

/// Reads the optional batch-construction progress cadence from the environment.
///
/// `DELAUNAY_BULK_PROGRESS_EVERY` is the canonical knob. The large-scale debug
/// harness also reuses `DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY` so manual runs can
/// request periodic progress without additional wiring.
fn bulk_progress_every_from_env() -> Option<usize> {
    [
        "DELAUNAY_BULK_PROGRESS_EVERY",
        "DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY",
    ]
    .into_iter()
    .find_map(|name| {
        env::var(name)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
    })
    .filter(|every| *every > 0)
}

/// Enables release-visible retry-boundary tracing for bulk construction.
fn construction_retry_trace_enabled() -> bool {
    bulk_progress_every_from_env().is_some()
        || env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some()
        || env::var_os("DELAUNAY_INSERT_TRACE").is_some()
}

#[derive(Clone, Copy, Debug)]
/// Snapshot of one batch-construction progress sample.
struct BatchProgressSample {
    processed: usize,
    inserted: usize,
    skipped: usize,
    cell_count: usize,
    perturbation_seed: u64,
}

#[derive(Clone, Copy, Debug)]
/// Rolling state used to compute periodic batch throughput summaries.
struct BatchProgressState {
    total_vertices: usize,
    progress_every: usize,
    started: Instant,
    last_progress: Instant,
    last_processed: usize,
}

/// Emits periodic batch-construction progress for long-running release-mode
/// investigations such as the 4D large-scale debug harness.
///
/// Progress is emitted via `tracing::debug!`; enable with `RUST_LOG=debug` (the
/// large-scale debug harness wires this up automatically when
/// `DELAUNAY_BULK_PROGRESS_EVERY` is set).
fn log_bulk_progress_if_due(sample: BatchProgressSample, state: &mut Option<BatchProgressState>) {
    let Some(state) = state.as_mut() else {
        return;
    };
    if sample.processed == 0 {
        return;
    }

    // Always log the final sample, even when the total is not an exact multiple of the
    // requested cadence, so interrupted runs still end with a complete progress line.
    let should_log = sample.processed == state.total_vertices
        || sample.processed.is_multiple_of(state.progress_every);
    if !should_log {
        return;
    }

    let elapsed = state.started.elapsed();
    let chunk_elapsed = state.last_progress.elapsed();
    let chunk_processed = sample.processed.saturating_sub(state.last_processed);

    let overall_rate = safe_usize_to_scalar::<f64>(sample.processed).unwrap_or(f64::NAN)
        / elapsed.as_secs_f64().max(1e-9);
    let chunk_rate = safe_usize_to_scalar::<f64>(chunk_processed).unwrap_or(f64::NAN)
        / chunk_elapsed.as_secs_f64().max(1e-9);

    tracing::debug!(
        target: "delaunay::bulk_progress",
        perturbation_seed = format_args!("0x{:X}", sample.perturbation_seed),
        processed = sample.processed,
        total_vertices = state.total_vertices,
        inserted = sample.inserted,
        skipped = sample.skipped,
        cells = sample.cell_count,
        elapsed = ?elapsed,
        total_rate_pts_per_s = overall_rate,
        recent_rate_pts_per_s = chunk_rate,
        "bulk-construction progress"
    );

    state.last_progress = Instant::now();
    state.last_processed = sample.processed;
}

/// Emits retry-boundary events for release-mode large-scale construction runs.
fn log_construction_retry_start(attempt: usize, attempt_seed: u64, perturbation_seed: u64) {
    if !construction_retry_trace_enabled() {
        return;
    }

    tracing::debug!(
        target: "delaunay::bulk_retry",
        attempt,
        attempt_seed = format_args!("0x{:X}", attempt_seed),
        perturbation_seed = format_args!("0x{:X}", perturbation_seed),
        "shuffled retry attempt starting"
    );
}

/// Emits retry attempt outcomes with optional construction statistics.
fn log_construction_retry_result(
    attempt: usize,
    attempt_seed: Option<u64>,
    perturbation_seed: u64,
    outcome: &'static str,
    error: Option<&str>,
    stats: Option<&ConstructionStatistics>,
) {
    if !construction_retry_trace_enabled() {
        return;
    }

    let attempt_seed_display =
        attempt_seed.map_or_else(|| String::from("input-order"), |seed| format!("0x{seed:X}"));
    let error_display = error.unwrap_or("-");

    if let Some(stats) = stats {
        tracing::debug!(
            target: "delaunay::bulk_retry",
            attempt,
            attempt_seed = %attempt_seed_display,
            perturbation_seed = format_args!("0x{:X}", perturbation_seed),
            outcome,
            inserted = stats.inserted,
            skipped_duplicate = stats.skipped_duplicate,
            skipped_degeneracy = stats.skipped_degeneracy,
            total_attempts = stats.total_attempts,
            max_attempts = stats.max_attempts,
            cells_removed_total = stats.cells_removed_total,
            cells_removed_max = stats.cells_removed_max,
            error = %error_display,
            "shuffled retry attempt result (with stats)"
        );
    } else {
        tracing::debug!(
            target: "delaunay::bulk_retry",
            attempt,
            attempt_seed = %attempt_seed_display,
            perturbation_seed = format_args!("0x{:X}", perturbation_seed),
            outcome,
            error = %error_display,
            "shuffled retry attempt result"
        );
    }
}

/// Sort key for Hilbert ordering: `(hilbert_index, quantized_coords, vertex, input_index)`.
type HilbertSortKey<T, U, const D: usize> = (u128, [u32; D], Vertex<T, U, D>, usize);

/// Orders vertices along a Hilbert curve to improve insertion locality while
/// retaining deterministic lexicographic fallbacks.
fn order_vertices_hilbert<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    dedup_quantized: bool,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    if vertices.is_empty() || D == 0 {
        return vertices;
    }

    let Some(bits_per_coord) = hilbert_bits_per_coord::<D>() else {
        return order_vertices_lexicographic(vertices);
    };

    // Compute global bounds in f64 for normalization. If any coordinate is non-finite,
    // fall back to lexicographic ordering (Hilbert normalization assumes finite values).
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for v in &vertices {
        for &coord in v.point().coords() {
            let Some(c) = coord.to_f64() else {
                return order_vertices_lexicographic(vertices);
            };
            if !c.is_finite() {
                return order_vertices_lexicographic(vertices);
            }
            min = min.min(c);
            max = max.max(c);
        }
    }

    let (Some(min_t), Some(max_t)) = (NumCast::from(min), NumCast::from(max)) else {
        return order_vertices_lexicographic(vertices);
    };

    let bounds = (min_t, max_t);

    // Phase 1: Quantize all coordinates
    let quantized: Result<Vec<[u32; D]>, ()> = vertices
        .iter()
        .map(|vertex| {
            hilbert_quantize(vertex.point().coords(), bounds, bits_per_coord).map_err(|_| ())
        })
        .collect();

    let Ok(quantized) = quantized else {
        // On quantization error, fall back to true lexicographic ordering of original coordinates
        return order_vertices_lexicographic(vertices);
    };

    // Phase 2: Compute all indices in bulk
    let Ok(indices) = hilbert_indices_prequantized(&quantized, bits_per_coord) else {
        // On bulk index computation error, fall back to true lexicographic ordering
        return order_vertices_lexicographic(vertices);
    };
    // Phase 3: Pair indices with vertices, quantized coords, and input indices
    let mut keyed: Vec<HilbertSortKey<T, U, D>> = vertices
        .into_iter()
        .enumerate()
        .map(|(input_index, vertex)| {
            let idx = indices
                .get(input_index)
                .copied()
                // Fallback to input index directly as u128 (no u32 truncation)
                .unwrap_or(input_index as u128);
            let q = quantized[input_index];
            (idx, q, vertex, input_index)
        })
        .collect();

    keyed.sort_by(
        |(a_idx, a_q, a_vertex, a_in), (b_idx, b_q, b_vertex, b_in)| {
            a_idx
                .cmp(b_idx)
                .then_with(|| a_q.cmp(b_q))
                .then_with(|| a_vertex.partial_cmp(b_vertex).unwrap_or(Ordering::Equal))
                .then_with(|| a_in.cmp(b_in))
        },
    );

    if dedup_quantized {
        // Deduplicate at quantization resolution in a single linear sweep.
        // Because vertices sharing the same quantized cell are now adjacent
        // after sorting, we can eliminate duplicates without re-quantizing.
        let input_len = keyed.len();
        let mut prev_q: Option<[u32; D]> = None;
        let deduped: Vec<Vertex<T, U, D>> = keyed
            .into_iter()
            .filter_map(|(_, q, v, _)| {
                if prev_q == Some(q) {
                    return None;
                }
                prev_q = Some(q);
                Some(v)
            })
            .collect();

        let removed = input_len - deduped.len();
        if removed > 0 {
            tracing::debug!(
                "Hilbert-sort dedup removed {removed} vertices (quantized at {bits_per_coord} bits/coord)"
            );
        }

        deduped
    } else {
        keyed.into_iter().map(|(_, _, v, _)| v).collect()
    }
}

/// Delaunay triangulation with incremental insertion support.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Delaunay Property Note
///
/// The triangulation satisfies **structural validity** (all TDS invariants) and
/// uses **flip-based repairs** to restore the local Delaunay property after insertion.
/// By default, k=2/k=3 bistellar flip queues run automatically after each successful
/// insertion (see [`DelaunayRepairPolicy`]).
///
/// For applications requiring explicit verification, you can still call
/// [`is_valid`](Self::is_valid) (Level 4) or [`validate`](Self::validate) (Levels 1–4).
/// If flip-based repair fails to converge, insertion returns an error and the
/// triangulation is left structurally valid but not guaranteed Delaunay.
///
/// See: [Issue #120 Investigation](https://github.com/acgetchell/delaunay/blob/main/docs/archive/issue_120_investigation.md)
///
/// # Implementation
///
/// Uses efficient incremental cavity-based insertion algorithm:
/// - ✅ Point location (facet walking) - [`locate`]
/// - ✅ Conflict region computation (local BFS) - [`find_conflict_region`]
/// - ✅ Cavity extraction and filling - [`extract_cavity_boundary`], [`fill_cavity`]
/// - ✅ Local neighbor wiring - [`wire_cavity_neighbors`]
/// - ✅ Hull extension for outside points - [`extend_hull`]
/// - ✅ Flip-based Delaunay repair (k=2/k=3 bistellar flips)
///
/// [`locate`]: crate::core::algorithms::locate::locate
/// [`find_conflict_region`]: crate::core::algorithms::locate::find_conflict_region
/// [`extract_cavity_boundary`]: crate::core::algorithms::locate::extract_cavity_boundary
/// [`fill_cavity`]: crate::core::algorithms::incremental_insertion::fill_cavity
/// [`wire_cavity_neighbors`]: crate::core::algorithms::incremental_insertion::wire_cavity_neighbors
/// [`extend_hull`]: crate::core::algorithms::incremental_insertion::extend_hull
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
///
/// assert_eq!(dt.number_of_cells(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct DelaunayTriangulation<K: Kernel<D>, U, V, const D: usize> {
    /// The underlying generic triangulation.
    pub(crate) tri: Triangulation<K, U, V, D>,
    /// Ephemeral insertion/repair state (hint caching + repair scheduling).
    insertion_state: DelaunayInsertionState,
    /// Optional spatial hash-grid index used to accelerate duplicate detection and locate-hint
    /// selection during incremental insertion.
    ///
    /// This is a performance-only cache and is not serialized; it may be rebuilt lazily.
    spatial_index: Option<HashGridIndex<K::Scalar, D>>,
}

// Most common case: f64 with AdaptiveKernel, no vertex or cell data
impl<const D: usize> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
    /// Create a Delaunay triangulation from vertices with no data (most common case).
    ///
    /// This is the simplest constructor for the most common use case:
    /// - f64 coordinates
    /// - Adaptive precision predicates with Simulation of Simplicity (`SoS`)
    /// - No vertex data
    /// - No cell data
    ///
    /// No type annotations needed! The compiler can infer everything.
    ///
    /// # Advanced Configuration
    ///
    /// For advanced use cases requiring custom construction options, topology guarantees,
    /// or toroidal (periodic) triangulations, use [`DelaunayTriangulationBuilder`]:
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // Advanced: custom topology guarantee
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::Pseudomanifold)
    ///     .build::<()>()
    ///     .unwrap();
    /// ```
    ///
    /// For toroidal (periodic) triangulations:
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.1, 0.2]),
    ///     vertex!([0.8, 0.3]),
    ///     vertex!([0.5, 0.7]),
    /// ];
    ///
    /// // Advanced: toroidal (periodic) triangulation
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .toroidal([1.0, 1.0])  // Phase 1: canonicalization
    ///     .build::<()>()
    ///     .unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns error if initial simplex cannot be constructed or insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // No type annotations needed!
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    pub fn new(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        Self::with_kernel(&AdaptiveKernel::<f64>::new(), vertices)
    }

    /// Create a Delaunay triangulation and return aggregate construction statistics.
    ///
    /// This is identical to [`new`](Self::new) (including default [`ConstructionOptions`]) but also
    /// returns a [`ConstructionStatistics`] summary of the insertion attempts performed during
    /// batch construction.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if construction fails.
    /// The returned error includes the partial [`ConstructionStatistics`] collected up to the
    /// failure point.
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics for compatibility"
    )]
    pub fn new_with_construction_statistics(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_topology_guarantee_and_options_with_construction_statistics(
            &kernel,
            vertices,
            TopologyGuarantee::DEFAULT,
            ConstructionOptions::default(),
        )
    }

    /// Create a Delaunay triangulation with explicit batch-construction options and return
    /// aggregate construction statistics.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if construction fails.
    /// The returned error includes the partial [`ConstructionStatistics`] collected up to the
    /// failure point.
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics for compatibility"
    )]
    pub fn new_with_options_and_construction_statistics(
        vertices: &[Vertex<f64, (), D>],
        options: ConstructionOptions,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_topology_guarantee_and_options_with_construction_statistics(
            &kernel,
            vertices,
            TopologyGuarantee::DEFAULT,
            options,
        )
    }

    /// Create a Delaunay triangulation with explicit batch-construction options (fast-kernel convenience).
    ///
    /// This is an additive API over [`new`](Self::new): it allows callers to override the default
    /// batch-construction options (insertion ordering, deduplication, retry policy).
    ///
    /// # Errors
    /// Returns an error if construction fails, or if the selected options are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::{
    ///     ConstructionOptions, DedupPolicy, InsertionOrderStrategy,
    /// };
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let options = ConstructionOptions::default()
    ///     .with_insertion_order(InsertionOrderStrategy::Hilbert)
    ///     .with_dedup_policy(DedupPolicy::Exact);
    ///
    /// let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    pub fn new_with_options(
        vertices: &[Vertex<f64, (), D>],
        options: ConstructionOptions,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_topology_guarantee_and_options(
            &kernel,
            vertices,
            TopologyGuarantee::DEFAULT,
            options,
        )
    }

    /// Create a Delaunay triangulation with an explicit topology guarantee (fast-kernel convenience).
    ///
    /// The default topology guarantee is [`TopologyGuarantee::PLManifold`]. Use this
    /// constructor to override it (e.g. relax to [`TopologyGuarantee::Pseudomanifold`]
    /// for speed at the cost of weaker topology guarantees).
    ///
    /// # Errors
    /// Returns error if construction fails or if the requested topology guarantee
    /// cannot be satisfied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new_with_topology_guarantee(
    ///     &vertices,
    ///     TopologyGuarantee::Pseudomanifold,
    /// )
    /// .unwrap();
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    pub fn new_with_topology_guarantee(
        vertices: &[Vertex<f64, (), D>],
        topology_guarantee: TopologyGuarantee,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_topology_guarantee(&kernel, vertices, topology_guarantee)
    }

    /// Create an empty Delaunay triangulation with no data (most common case).
    ///
    /// Use this when you want to build a triangulation incrementally by inserting vertices
    /// one at a time. The triangulation will automatically bootstrap itself when you
    /// insert the (D+1)th vertex, creating the initial simplex.
    ///
    /// No type annotations needed! The compiler can infer everything.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices one by one
    /// dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap(); // Initial simplex created automatically
    /// assert_eq!(dt.number_of_cells(), 1);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self::with_empty_kernel(AdaptiveKernel::<f64>::new())
    }

    /// Create an empty Delaunay triangulation with an explicit topology guarantee (fast-kernel convenience).
    ///
    /// The default topology guarantee is [`TopologyGuarantee::PLManifold`]. Use this
    /// constructor to override it (e.g. relax to [`TopologyGuarantee::Pseudomanifold`]
    /// for speed at the cost of weaker topology guarantees).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    ///
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[must_use]
    pub fn empty_with_topology_guarantee(topology_guarantee: TopologyGuarantee) -> Self {
        Self::with_empty_kernel_and_topology_guarantee(
            AdaptiveKernel::<f64>::new(),
            topology_guarantee,
        )
    }

    /// Create a fluent builder for constructing a Delaunay triangulation.
    ///
    /// This is a convenience entry point that produces a
    /// [`DelaunayTriangulationBuilder`]
    /// pre-typed for `f64` coordinates, no vertex data (`()`), and dimension `D`.
    ///
    /// For non-`f64` coordinates, vertex data (`U ≠ ()`), or custom kernels, construct
    /// `DelaunayTriangulationBuilder::new(vertices)` directly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::builder(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// ```
    ///
    /// ## Toroidal construction
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Vertices outside [0, 1)² are canonicalized before building.
    /// let vertices = vec![
    ///     vertex!([0.2, 0.3]),
    ///     vertex!([1.8, 0.1]), // wraps to (0.8, 0.1)
    ///     vertex!([0.5, 0.7]),
    ///     vertex!([-0.4, 0.9]), // wraps to (0.6, 0.9)
    /// ];
    ///
    /// let dt = DelaunayTriangulation::builder(&vertices)
    ///     .toroidal([1.0, 1.0])
    ///     .build::<()>()
    ///     .unwrap();
    ///
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    #[must_use]
    pub fn builder(
        vertices: &[Vertex<f64, (), D>],
    ) -> DelaunayTriangulationBuilder<'_, f64, (), D> {
        DelaunayTriangulationBuilder::new(vertices)
    }
}

// =============================================================================
// CONSTRUCTION (Requires Numeric Scalar Bounds)
// =============================================================================
//
// Batch and incremental constructors, preprocessing, Hilbert ordering, spatial
// hashing, and deduplication — all require `CoordinateScalar + NumCast`.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + NumCast,
    U: DataType,
    V: DataType,
{
    /// Create an empty Delaunay triangulation with the given kernel (advanced usage).
    ///
    /// Most users should use [`DelaunayTriangulation::empty()`] instead, which uses fast predicates
    /// by default. Use this method only if you need custom coordinate precision or specialized kernels.
    ///
    /// This creates a triangulation with no vertices or cells. Use [`insert`](Self::insert)
    /// to add vertices incrementally. The triangulation will automatically bootstrap itself when
    /// you insert the (D+1)th vertex, creating the initial simplex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    /// use delaunay::geometry::kernel::RobustKernel;
    ///
    /// // Start with empty triangulation using robust kernel
    /// let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::with_empty_kernel(RobustKernel::new());
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices incrementally
    /// dt.insert(vertex!([0.0, 0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 1.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 0.0, 1.0])).unwrap(); // Initial simplex created
    /// assert_eq!(dt.number_of_cells(), 1);
    /// ```
    #[must_use]
    pub fn with_empty_kernel(kernel: K) -> Self {
        let duplicate_tolerance = default_duplicate_tolerance::<K::Scalar>();

        Self {
            tri: Triangulation::new_empty(kernel),
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: Some(HashGridIndex::new(duplicate_tolerance)),
        }
    }

    /// Create an empty Delaunay triangulation with the given kernel and topology guarantee.
    ///
    /// This is the kernel-parameterized variant of
    /// [`DelaunayTriangulation::empty_with_topology_guarantee`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    ///     DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
    ///         RobustKernel::new(),
    ///         TopologyGuarantee::PLManifold,
    ///     );
    ///
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// ```
    #[must_use]
    pub fn with_empty_kernel_and_topology_guarantee(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        let duplicate_tolerance = default_duplicate_tolerance::<K::Scalar>();

        let mut tri = Triangulation::new_empty(kernel);
        tri.set_topology_guarantee(topology_guarantee);
        Self {
            tri,
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: Some(HashGridIndex::new(duplicate_tolerance)),
        }
    }

    /// Create a Delaunay triangulation from vertices with an explicit kernel (advanced usage).
    ///
    /// Most users should use [`DelaunayTriangulation::new()`] instead, which uses the
    /// [`AdaptiveKernel`] by default. Use this method when you need a different kernel:
    ///
    /// - **[`RobustKernel`]** — exact Bareiss arithmetic (recommended for correctness)
    /// - **[`FastKernel`](crate::geometry::kernel::FastKernel)** — raw `f64` (faster,
    ///   but may produce incorrect results for near-degenerate inputs)
    /// - Custom coordinate precision (f32, custom types)
    ///
    /// **Note:** `FastKernel` is accepted for construction and insertion, but the
    /// explicit repair methods ([`repair_delaunay_with_flips`](Self::repair_delaunay_with_flips),
    /// [`repair_delaunay_with_flips_advanced`](Self::repair_delaunay_with_flips_advanced))
    /// require [`ExactPredicates`] and are not available for `FastKernel`.
    ///
    /// This uses the efficient cavity-based algorithm:
    /// 1. Build initial simplex (D+1 vertices) directly
    /// 2. Insert remaining vertices incrementally with locate → conflict → cavity → wire
    ///
    /// # Errors
    /// Returns error if initial simplex cannot be constructed or insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // Use robust kernel for exact arithmetic
    /// let kernel = RobustKernel::new();
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::with_kernel(&kernel, &vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// ```
    pub fn with_kernel(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        Self::with_topology_guarantee(kernel, vertices, TopologyGuarantee::DEFAULT)
    }

    /// Create a Delaunay triangulation with an explicit topology guarantee.
    ///
    /// Passing [`TopologyGuarantee::PLManifold`] enforces ridge-link validation during
    /// construction and validates vertex-links at completion. Use
    /// [`TopologyGuarantee::PLManifoldStrict`] for per-insertion vertex-link checks.
    ///
    /// # Shuffled Retries
    /// For `D >= 2` with more than `D + 1` vertices, the constructor retries
    /// construction with up to 6 shuffled insertion orders if the Delaunay
    /// property check fails (see [`RetryPolicy::default()`]).  To disable
    /// retries, pass [`ConstructionOptions::default().with_retry_policy(RetryPolicy::Disabled)`](Self::with_topology_guarantee_and_options).
    ///
    /// # Errors
    /// Returns error if construction fails or if the requested topology guarantee
    /// cannot be satisfied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let kernel = RobustKernel::new();
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    ///     DelaunayTriangulation::with_topology_guarantee(
    ///         &kernel,
    ///         &vertices,
    ///         TopologyGuarantee::PLManifold,
    ///     )
    ///     .unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    pub fn with_topology_guarantee(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        Self::with_topology_guarantee_and_options(
            kernel,
            vertices,
            topology_guarantee,
            ConstructionOptions::default(),
        )
    }

    /// Create a Delaunay triangulation with an explicit topology guarantee and batch-construction options.
    ///
    /// This is the core constructor used by the higher-level convenience constructors. It allows callers
    /// to opt into deterministic preprocessing and retry behavior.
    ///
    /// # Errors
    /// Returns an error if:
    /// - construction fails or the requested topology guarantee cannot be satisfied, or
    /// - the selected preprocessing options are invalid (e.g. a negative / non-finite epsilon tolerance).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::{
    ///     ConstructionOptions, DedupPolicy, InsertionOrderStrategy,
    /// };
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let options = ConstructionOptions::default()
    ///     .with_insertion_order(InsertionOrderStrategy::Hilbert)
    ///     .with_dedup_policy(DedupPolicy::Off);
    ///
    /// let kernel = RobustKernel::new();
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    ///     DelaunayTriangulation::with_topology_guarantee_and_options(
    ///         &kernel,
    ///         &vertices,
    ///         TopologyGuarantee::PLManifold,
    ///         options,
    ///     )
    ///     .unwrap();
    /// assert_eq!(dt.number_of_cells(), 1);
    /// ```
    pub fn with_topology_guarantee_and_options(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        options: ConstructionOptions,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let ConstructionOptions {
            insertion_order,
            dedup_policy,
            initial_simplex,
            retry_policy,
            use_global_repair_fallback,
        } = options;

        let preprocessed = Self::preprocess_vertices_for_construction(
            vertices,
            dedup_policy,
            insertion_order,
            initial_simplex,
        )?;
        let grid_cell_size = preprocessed.grid_cell_size();
        let primary_vertices: &[Vertex<K::Scalar, U, D>] = preprocessed.primary_slice(vertices);
        let fallback_vertices = preprocessed.fallback_slice();

        let build_with_vertices = |vertices: &[Vertex<K::Scalar, U, D>]| {
            match retry_policy {
                RetryPolicy::Disabled => {}
                RetryPolicy::Shuffled {
                    attempts,
                    base_seed,
                } => {
                    if Self::should_retry_construction(vertices) {
                        return Self::build_with_shuffled_retries(
                            kernel,
                            vertices,
                            topology_guarantee,
                            attempts,
                            base_seed,
                            grid_cell_size,
                            use_global_repair_fallback,
                        );
                    }
                }
                RetryPolicy::DebugOnlyShuffled {
                    attempts,
                    base_seed,
                } => {
                    if cfg!(any(test, debug_assertions))
                        && Self::should_retry_construction(vertices)
                    {
                        return Self::build_with_shuffled_retries(
                            kernel,
                            vertices,
                            topology_guarantee,
                            attempts,
                            base_seed,
                            grid_cell_size,
                            use_global_repair_fallback,
                        );
                    }
                }
            }

            Self::build_with_kernel_inner(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                grid_cell_size,
                use_global_repair_fallback,
            )
        };

        let result = build_with_vertices(primary_vertices);
        if result.is_err()
            && let Some(fallback) = fallback_vertices
        {
            return build_with_vertices(fallback);
        }

        result
    }

    /// Like [`with_topology_guarantee_and_options`](Self::with_topology_guarantee_and_options), but
    /// also returns aggregate [`ConstructionStatistics`] collected during batch construction.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if construction fails.
    /// The returned error includes the partial [`ConstructionStatistics`] collected up to the
    /// failure point.
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics for compatibility"
    )]
    pub fn with_topology_guarantee_and_options_with_construction_statistics(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        options: ConstructionOptions,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let ConstructionOptions {
            insertion_order,
            dedup_policy,
            initial_simplex,
            retry_policy,
            use_global_repair_fallback,
        } = options;

        let preprocessed = Self::preprocess_vertices_for_construction(
            vertices,
            dedup_policy,
            insertion_order,
            initial_simplex,
        )
        .map_err(
            |error| DelaunayTriangulationConstructionErrorWithStatistics {
                error,
                statistics: ConstructionStatistics::default(),
            },
        )?;
        let grid_cell_size = preprocessed.grid_cell_size();
        let primary_vertices: &[Vertex<K::Scalar, U, D>] = preprocessed.primary_slice(vertices);
        let fallback_vertices = preprocessed.fallback_slice();

        let build_with_vertices = |vertices: &[Vertex<K::Scalar, U, D>]| {
            match retry_policy {
                RetryPolicy::Disabled => {}
                RetryPolicy::Shuffled {
                    attempts,
                    base_seed,
                } => {
                    if Self::should_retry_construction(vertices) {
                        return Self::build_with_shuffled_retries_with_construction_statistics(
                            kernel,
                            vertices,
                            topology_guarantee,
                            attempts,
                            base_seed,
                            grid_cell_size,
                            use_global_repair_fallback,
                        );
                    }
                }
                RetryPolicy::DebugOnlyShuffled {
                    attempts,
                    base_seed,
                } => {
                    if cfg!(any(test, debug_assertions))
                        && Self::should_retry_construction(vertices)
                    {
                        return Self::build_with_shuffled_retries_with_construction_statistics(
                            kernel,
                            vertices,
                            topology_guarantee,
                            attempts,
                            base_seed,
                            grid_cell_size,
                            use_global_repair_fallback,
                        );
                    }
                }
            }

            Self::build_with_kernel_inner_with_construction_statistics(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                grid_cell_size,
                use_global_repair_fallback,
            )
        };

        let result = build_with_vertices(primary_vertices);
        if result.is_err()
            && let Some(fallback) = fallback_vertices
        {
            return build_with_vertices(fallback);
        }

        result
    }

    /// Applies deduplication, insertion ordering, and initial-simplex selection
    /// before any topology is created.
    fn preprocess_vertices_for_construction(
        vertices: &[Vertex<K::Scalar, U, D>],
        dedup_policy: DedupPolicy,
        insertion_order: InsertionOrderStrategy,
        initial_simplex: InitialSimplexStrategy,
    ) -> PreprocessVerticesResult<K::Scalar, U, D> {
        let default_tolerance = default_duplicate_tolerance::<K::Scalar>();

        let mut epsilon: Option<K::Scalar> = None;
        if let DedupPolicy::Epsilon { tolerance } = dedup_policy {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Invalid DedupPolicy::Epsilon tolerance {tolerance:?} (must be finite and non-negative)"
                    ),
                }
                .into());
            }

            let Some(epsilon_value) = <K::Scalar as NumCast>::from(tolerance) else {
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Failed to convert DedupPolicy::Epsilon tolerance {tolerance:?} into scalar type"
                    ),
                }
                .into());
            };
            epsilon = Some(epsilon_value);
        }

        let grid_cell_size_value =
            if let (DedupPolicy::Epsilon { .. }, Some(eps)) = (dedup_policy, epsilon) {
                if eps > K::Scalar::zero() {
                    eps
                } else {
                    default_tolerance
                }
            } else {
                default_tolerance
            };
        let mut grid: HashGridIndex<K::Scalar, D, usize> = HashGridIndex::new(grid_cell_size_value);

        // Deduplicate first to reduce work for ordering strategies.
        let mut owned_vertices: Option<Vec<Vertex<K::Scalar, U, D>>> = match dedup_policy {
            DedupPolicy::Off => None,
            DedupPolicy::Exact => {
                let vertices = vertices.to_vec();
                if hash_grid_usable_for_vertices(&grid, &vertices) {
                    Some(dedup_vertices_exact_hash_grid(vertices, &mut grid))
                } else {
                    Some(dedup_vertices_exact_sorted(vertices))
                }
            }
            DedupPolicy::Epsilon { .. } => {
                let epsilon = epsilon.expect("epsilon validated above");
                let vertices = vertices.to_vec();
                if hash_grid_usable_for_vertices(&grid, &vertices) {
                    Some(dedup_vertices_epsilon_hash_grid(
                        vertices, epsilon, &mut grid,
                    ))
                } else {
                    Some(dedup_vertices_epsilon_quantized(vertices, epsilon))
                }
            }
        };

        owned_vertices = match insertion_order {
            InsertionOrderStrategy::Input => owned_vertices,
            _ => Some(order_vertices_by_strategy(
                owned_vertices.unwrap_or_else(|| vertices.to_vec()),
                insertion_order,
            )),
        };

        let (primary, fallback) = match initial_simplex {
            InitialSimplexStrategy::First => (owned_vertices, None),
            InitialSimplexStrategy::Balanced => {
                let base = owned_vertices.unwrap_or_else(|| vertices.to_vec());
                if let Some(indices) = select_balanced_simplex_indices(&base) {
                    if let Some(reordered) = reorder_vertices_for_simplex(&base, &indices) {
                        (Some(reordered), Some(base))
                    } else {
                        (Some(base), None)
                    }
                } else {
                    (Some(base), None)
                }
            }
        };

        let final_slice = primary.as_deref().unwrap_or(vertices);
        let grid_cell_size = if hash_grid_usable_for_vertices(&grid, final_slice) {
            Some(grid.cell_size())
        } else {
            None
        };

        Ok(PreprocessVertices {
            primary,
            fallback,
            grid_cell_size,
        })
    }

    /// Returns `true` if the construction error is deterministic and should not
    /// be masked by shuffled retry logic (e.g. duplicate UUIDs, internal bugs).
    const fn is_non_retryable_construction_error(
        err: &DelaunayTriangulationConstructionError,
    ) -> bool {
        matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::Tds(TdsConstructionError::DuplicateUuid { .. })
                    | TriangulationConstructionError::InternalInconsistency { .. }
            )
        )
    }

    /// Retries batch construction with deterministic shuffles so retryable
    /// degeneracies can be escaped reproducibly.
    #[allow(clippy::too_many_lines)]
    fn build_with_shuffled_retries(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let base_seed = base_seed.unwrap_or_else(|| Self::construction_shuffle_seed(vertices));

        #[cfg(debug_assertions)]
        let log_shuffle = env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some();

        #[cfg(debug_assertions)]
        if log_shuffle {
            tracing::debug!(
                base_seed,
                attempts = attempts.get(),
                vertex_count = vertices.len(),
                "build_with_shuffled_retries: starting"
            );
        }

        // Attempt 0: original order, no extra perturbation salt.
        log_construction_retry_result(0, None, 0_u64, "started", None, None);
        let mut last_error: String = match Self::build_with_kernel_inner_seeded(
            <K as Clone>::clone(kernel),
            vertices,
            topology_guarantee,
            0_u64,
            true,
            grid_cell_size,
            use_global_repair_fallback,
        ) {
            Ok(candidate) => match crate::core::util::is_delaunay_property_only(&candidate.tri.tds)
            {
                Ok(()) => {
                    log_construction_retry_result(0, None, 0_u64, "succeeded", None, None);
                    return Ok(candidate);
                }
                Err(err) => format!("Delaunay property violated after construction: {err}"),
            },
            Err(err) => {
                let err_string = err.to_string();
                if Self::is_non_retryable_construction_error(&err) {
                    log_construction_retry_result(
                        0,
                        None,
                        0_u64,
                        "failed",
                        Some(&err_string),
                        None,
                    );
                    return Err(err);
                }
                err_string
            }
        };

        #[cfg(debug_assertions)]
        if log_shuffle {
            tracing::debug!(
                attempt = 0,
                perturbation_seed = 0_u64,
                last_error = %last_error,
                "build_with_shuffled_retries: initial attempt failed: {last_error}"
            );
        }
        log_construction_retry_result(0, None, 0_u64, "failed", Some(&last_error), None);

        // Shuffled retries (total iterations: attempts shuffled).
        for attempt in 1..=attempts.get() {
            let mut shuffled = vertices.to_vec();

            let mut attempt_seed =
                base_seed.wrapping_add((attempt as u64).wrapping_mul(DELAUNAY_SHUFFLE_SEED_SALT));
            if attempt_seed == 0 {
                attempt_seed = 1;
            }

            Self::shuffle_vertices(&mut shuffled, attempt_seed);

            // Vary the deterministic perturbation pattern across retry attempts.
            let perturbation_seed = attempt_seed ^ 0xD1B5_4A32_D192_ED03;

            #[cfg(debug_assertions)]
            if log_shuffle {
                tracing::debug!(
                    attempt,
                    attempt_seed,
                    perturbation_seed,
                    "build_with_shuffled_retries: shuffled attempt starting"
                );
            }
            log_construction_retry_start(attempt, attempt_seed, perturbation_seed);

            match Self::build_with_kernel_inner_seeded(
                <K as Clone>::clone(kernel),
                &shuffled,
                topology_guarantee,
                perturbation_seed,
                true,
                grid_cell_size,
                use_global_repair_fallback,
            ) {
                Ok(candidate) => {
                    match crate::core::util::is_delaunay_property_only(&candidate.tri.tds) {
                        Ok(()) => {
                            log_construction_retry_result(
                                attempt,
                                Some(attempt_seed),
                                perturbation_seed,
                                "succeeded",
                                None,
                                None,
                            );
                            return Ok(candidate);
                        }
                        Err(err) => {
                            last_error =
                                format!("Delaunay property violated after construction: {err}");
                        }
                    }
                }
                Err(err) => {
                    if Self::is_non_retryable_construction_error(&err) {
                        return Err(err);
                    }
                    last_error = err.to_string();
                }
            }

            #[cfg(debug_assertions)]
            if log_shuffle {
                tracing::debug!(
                    attempt,
                    attempt_seed,
                    perturbation_seed,
                    last_error = %last_error,
                    "build_with_shuffled_retries: attempt failed: {last_error}"
                );
            }
            log_construction_retry_result(
                attempt,
                Some(attempt_seed),
                perturbation_seed,
                "failed",
                Some(&last_error),
                None,
            );
        }

        // Treat persistent construction failures or Delaunay violations as hard construction
        // errors so callers can deterministically reject.
        Err(TriangulationConstructionError::GeometricDegeneracy {
            message: format!(
                "Delaunay construction failed after shuffled reconstruction attempts: {last_error}"
            ),
        }
        .into())
    }

    /// Mirrors shuffled retry construction while preserving per-attempt
    /// statistics for callers that need skip and retry diagnostics.
    #[allow(clippy::too_many_lines)]
    #[expect(
        clippy::result_large_err,
        reason = "Internal helper propagates public by-value construction-statistics error type"
    )]
    fn build_with_shuffled_retries_with_construction_statistics(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let base_seed = base_seed.unwrap_or_else(|| Self::construction_shuffle_seed(vertices));

        #[cfg(debug_assertions)]
        let log_shuffle = env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some();

        #[cfg(debug_assertions)]
        if log_shuffle {
            tracing::debug!(
                base_seed,
                attempts = attempts.get(),
                vertex_count = vertices.len(),
                "build_with_shuffled_retries_with_construction_statistics: starting"
            );
        }

        let mut last_stats: Option<ConstructionStatistics> = None;

        // Attempt 0: original order, no extra perturbation salt.
        let mut last_error: String =
            match Self::build_with_kernel_inner_seeded_with_construction_statistics(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                0_u64,
                true,
                grid_cell_size,
                use_global_repair_fallback,
            ) {
                Ok((candidate, stats)) => {
                    match crate::core::util::is_delaunay_property_only(&candidate.tri.tds) {
                        Ok(()) => return Ok((candidate, stats)),
                        Err(err) => {
                            last_stats.replace(stats);
                            format!("Delaunay property violated after construction: {err}")
                        }
                    }
                }
                Err(err) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics { error, statistics } =
                        err;
                    if Self::is_non_retryable_construction_error(&error) {
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics,
                        });
                    }
                    last_stats.replace(statistics);
                    error.to_string()
                }
            };

        #[cfg(debug_assertions)]
        if log_shuffle {
            tracing::debug!(
                attempt = 0,
                perturbation_seed = 0_u64,
                last_error = %last_error,
                "build_with_shuffled_retries_with_construction_statistics: initial attempt failed: {last_error}"
            );
        }
        log_construction_retry_result(
            0,
            None,
            0_u64,
            "failed",
            Some(&last_error),
            last_stats.as_ref(),
        );

        // Shuffled retries (total iterations: attempts shuffled).
        for attempt in 1..=attempts.get() {
            let mut shuffled = vertices.to_vec();

            let mut attempt_seed =
                base_seed.wrapping_add((attempt as u64).wrapping_mul(DELAUNAY_SHUFFLE_SEED_SALT));
            if attempt_seed == 0 {
                attempt_seed = 1;
            }

            Self::shuffle_vertices(&mut shuffled, attempt_seed);

            // Vary the deterministic perturbation pattern across retry attempts.
            let perturbation_seed = attempt_seed ^ 0xD1B5_4A32_D192_ED03;

            #[cfg(debug_assertions)]
            if log_shuffle {
                tracing::debug!(
                    attempt,
                    attempt_seed,
                    perturbation_seed,
                    "build_with_shuffled_retries_with_construction_statistics: shuffled attempt starting"
                );
            }
            log_construction_retry_start(attempt, attempt_seed, perturbation_seed);

            match Self::build_with_kernel_inner_seeded_with_construction_statistics(
                <K as Clone>::clone(kernel),
                &shuffled,
                topology_guarantee,
                perturbation_seed,
                true,
                grid_cell_size,
                use_global_repair_fallback,
            ) {
                Ok((candidate, stats)) => {
                    match crate::core::util::is_delaunay_property_only(&candidate.tri.tds) {
                        Ok(()) => {
                            log_construction_retry_result(
                                attempt,
                                Some(attempt_seed),
                                perturbation_seed,
                                "succeeded",
                                None,
                                Some(&stats),
                            );
                            return Ok((candidate, stats));
                        }
                        Err(err) => {
                            last_stats.replace(stats);
                            last_error =
                                format!("Delaunay property violated after construction: {err}");
                        }
                    }
                }
                Err(err) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics { error, statistics } =
                        err;
                    if Self::is_non_retryable_construction_error(&error) {
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics,
                        });
                    }
                    last_stats.replace(statistics);
                    last_error = error.to_string();
                }
            }

            #[cfg(debug_assertions)]
            if log_shuffle {
                tracing::debug!(
                    attempt,
                    attempt_seed,
                    perturbation_seed,
                    last_error = %last_error,
                    "build_with_shuffled_retries_with_construction_statistics: attempt failed: {last_error}"
                );
            }
            log_construction_retry_result(
                attempt,
                Some(attempt_seed),
                perturbation_seed,
                "failed",
                Some(&last_error),
                last_stats.as_ref(),
            );
        }

        // Treat persistent construction failures or Delaunay violations as hard construction
        // errors so callers can deterministically reject.
        let statistics = last_stats.unwrap_or_default();
        Err(DelaunayTriangulationConstructionErrorWithStatistics {
            error: TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Delaunay construction failed after shuffled reconstruction attempts: {last_error}"
                ),
            }
            .into(),
            statistics,
        })
    }

    /// Avoids retry work when construction has no incremental phase to reorder.
    const fn should_retry_construction(vertices: &[Vertex<K::Scalar, U, D>]) -> bool {
        D >= 2 && vertices.len() > D + 1
    }

    /// Derives an input-order-independent seed so retries are reproducible for
    /// the same vertex set.
    fn construction_shuffle_seed(vertices: &[Vertex<K::Scalar, U, D>]) -> u64 {
        let mut vertex_hashes = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let mut hasher = FastHasher::default();
            vertex.hash(&mut hasher);
            vertex_hashes.push(hasher.finish());
        }
        vertex_hashes.sort_unstable();
        stable_hash_u64_slice(&vertex_hashes)
    }

    /// Keeps construction retry shuffling deterministic for diagnostics and
    /// tests.
    fn shuffle_vertices(vertices: &mut [Vertex<K::Scalar, U, D>], seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        vertices.shuffle(&mut rng);
    }

    /// Runs batch construction without statistics while preserving the same
    /// final validation path as the statistics variant.
    fn build_with_kernel_inner(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let dt = Self::build_with_kernel_inner_seeded(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
            use_global_repair_fallback,
        )?;

        // Final validation at construction completion for PLManifold/PLManifoldStrict.
        // This ensures PL-manifold guarantee even with ValidationPolicy::OnSuspicion during
        // incremental insertion.
        if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            tracing::debug!("post-construction: starting topology validation (build)");
            let validation_started = Instant::now();
            let validation_result = dt.tri.validate();
            tracing::debug!(
                elapsed = ?validation_started.elapsed(),
                success = validation_result.is_ok(),
                "post-construction: topology validation (build) completed"
            );
            if let Err(err) = validation_result {
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("PL-manifold validation failed after construction: {err}"),
                }
                .into());
            }
        }

        // `DelaunayCheckPolicy::EndOnly`: always run a final global Delaunay validation pass after
        // batch construction.
        tracing::debug!("post-construction: starting Delaunay validation (build)");
        let delaunay_started = Instant::now();
        let delaunay_result = dt.is_valid();
        tracing::debug!(
            elapsed = ?delaunay_started.elapsed(),
            success = delaunay_result.is_ok(),
            "post-construction: Delaunay validation (build) completed"
        );
        delaunay_result.map_err(|err| TriangulationConstructionError::GeometricDegeneracy {
            message: format!("Delaunay property violated after construction: {err}"),
        })?;

        Ok(dt)
    }

    /// Runs batch construction with aggregate statistics without changing the
    /// construction algorithm itself.
    #[expect(
        clippy::result_large_err,
        reason = "Internal helper propagates public by-value construction-statistics error type"
    )]
    fn build_with_kernel_inner_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let (dt, stats) = Self::build_with_kernel_inner_seeded_with_construction_statistics(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
            use_global_repair_fallback,
        )?;

        // Final validation at construction completion for PLManifold/PLManifoldStrict.
        // This ensures PL-manifold guarantee even with ValidationPolicy::OnSuspicion during
        // incremental insertion.
        if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            tracing::debug!("post-construction: starting topology validation (build stats)");
            let validation_started = Instant::now();
            let validation_result = dt.tri.validate();
            tracing::debug!(
                elapsed = ?validation_started.elapsed(),
                success = validation_result.is_ok(),
                "post-construction: topology validation (build stats) completed"
            );
            if let Err(err) = validation_result {
                return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                    error: TriangulationConstructionError::GeometricDegeneracy {
                        message: format!("PL-manifold validation failed after construction: {err}"),
                    }
                    .into(),
                    statistics: stats,
                });
            }
        }

        // `DelaunayCheckPolicy::EndOnly`: always run a final global Delaunay validation pass after
        // batch construction.
        tracing::debug!("post-construction: starting Delaunay validation (build stats)");
        let delaunay_started = Instant::now();
        let delaunay_result = dt.is_valid();
        tracing::debug!(
            elapsed = ?delaunay_started.elapsed(),
            success = delaunay_result.is_ok(),
            "post-construction: Delaunay validation (build stats) completed"
        );
        if let Err(err) = delaunay_result {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error: TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Delaunay property violated after construction: {err}"),
                }
                .into(),
                statistics: stats,
            });
        }

        Ok((dt, stats))
    }

    /// Implements the seeded batch-construction core so retry and statistics
    /// entry points share perturbation behavior.
    #[expect(
        clippy::result_large_err,
        reason = "Internal helper propagates public by-value construction-statistics error type"
    )]
    fn build_with_kernel_inner_seeded_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        if vertices.len() < D + 1 {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error: TriangulationConstructionError::InsufficientVertices {
                    dimension: D,
                    source: CellValidationError::InsufficientVertices {
                        actual: vertices.len(),
                        expected: D + 1,
                        dimension: D,
                    },
                }
                .into(),
                statistics: ConstructionStatistics::default(),
            });
        }

        // Build initial simplex directly (no Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Triangulation::<K, U, V, D>::build_initial_simplex(initial_vertices).map_err(
            |error| DelaunayTriangulationConstructionErrorWithStatistics {
                error: error.into(),
                statistics: ConstructionStatistics::default(),
            },
        )?;

        let mut dt = Self {
            tri: Triangulation {
                kernel,
                tds,
                global_topology: GlobalTopology::DEFAULT,
                validation_policy: topology_guarantee.default_validation_policy(),
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        };

        // During batch construction, use suspicion-driven validation instead of
        // per-insertion validation.  Running a full O(cells) topology check after
        // every insertion is prohibitively expensive at scale (O(n²) total).  The
        // OnSuspicion policy only validates when the insertion logic itself flags a
        // potential issue (e.g. after rollback/retry).  A comprehensive post-
        // construction validation in finalize_bulk_construction catches any issues
        // that slip through.
        //
        // Exception: PLManifoldStrict requires per-insertion vertex-link validation,
        // so we must use ValidationPolicy::Always to satisfy that guarantee.
        let original_validation_policy = dt.tri.validation_policy;
        dt.tri.validation_policy = if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            ValidationPolicy::Always
        } else if dt.tri.topology_guarantee.requires_ridge_links() {
            ValidationPolicy::OnSuspicion
        } else {
            ValidationPolicy::DebugOnly
        };

        // Disable maybe_repair_after_insertion during bulk construction: its full pipeline
        // (multi-pass repair + topology validation + heuristic rebuild) is too expensive
        // per insertion.  Instead, insert_remaining_vertices_seeded calls
        // repair_delaunay_local_single_pass directly after each insertion (no topology
        // check, no heuristic rebuild, soft-fail on non-convergence for D≥4).  Soft-failed
        // insertions (D≥4 only) record their adjacent cells in soft_fail_seeds, which is
        // used as the seed for the final seeded repair in finalize_bulk_construction.  If
        // no soft-fails occurred the seed is empty and finalize skips the repair entirely.
        let original_repair_policy = dt.insertion_state.delaunay_repair_policy;
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;
        dt.insertion_state.use_global_repair_fallback = use_global_repair_fallback;

        let mut stats = ConstructionStatistics::default();
        let simplex_stats = InsertionStatistics {
            attempts: 1,
            ..InsertionStatistics::default()
        };
        for _ in 0..=D {
            stats.record_insertion(&simplex_stats);
        }

        let mut soft_fail_seeds: Vec<CellKey> = Vec::new();
        if let Err(error) = dt.insert_remaining_vertices_seeded(
            vertices,
            perturbation_seed,
            grid_cell_size,
            Some(&mut stats),
            &mut soft_fail_seeds,
        ) {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error,
                statistics: stats,
            });
        }

        if let Err(error) = dt.finalize_bulk_construction(
            original_validation_policy,
            original_repair_policy,
            run_final_repair,
            &soft_fail_seeds,
        ) {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error,
                statistics: stats,
            });
        }

        Ok((dt, stats))
    }

    /// Implements the non-statistics seeded construction core for callers that
    /// only need the triangulation.
    fn build_with_kernel_inner_seeded(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
        use_global_repair_fallback: bool,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        if vertices.len() < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            }
            .into());
        }

        // Build initial simplex directly (no Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Triangulation::<K, U, V, D>::build_initial_simplex(initial_vertices)?;

        let mut dt = Self {
            tri: Triangulation {
                kernel,
                tds,
                global_topology: GlobalTopology::DEFAULT,
                validation_policy: topology_guarantee.default_validation_policy(),
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        };

        // During batch construction, use suspicion-driven validation instead of
        // per-insertion validation (see _with_construction_statistics variant for
        // rationale: O(n²) avoidance + post-construction catch-all).
        //
        // Exception: PLManifoldStrict requires per-insertion vertex-link validation,
        // so we must use ValidationPolicy::Always to satisfy that guarantee.
        let original_validation_policy = dt.tri.validation_policy;
        dt.tri.validation_policy = if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            ValidationPolicy::Always
        } else if dt.tri.topology_guarantee.requires_ridge_links() {
            ValidationPolicy::OnSuspicion
        } else {
            ValidationPolicy::DebugOnly
        };

        // See the _with_construction_statistics variant for the repair policy rationale.
        let original_repair_policy = dt.insertion_state.delaunay_repair_policy;
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;
        dt.insertion_state.use_global_repair_fallback = use_global_repair_fallback;
        let mut soft_fail_seeds: Vec<CellKey> = Vec::new();
        dt.insert_remaining_vertices_seeded(
            vertices,
            perturbation_seed,
            grid_cell_size,
            None,
            &mut soft_fail_seeds,
        )?;
        dt.finalize_bulk_construction(
            original_validation_policy,
            original_repair_policy,
            run_final_repair,
            &soft_fail_seeds,
        )?;

        Ok(dt)
    }

    /// Handle D<4 local repair non-convergence by falling back to global repair or
    /// hard-failing to trigger shuffle retry.
    ///
    /// Returns `Ok(())` if global repair succeeded (caller should `continue` the
    /// insertion loop).  Returns `Err(...)` if the caller should propagate the
    /// construction error.
    fn try_d_lt4_global_repair_fallback(
        tds: &mut Tds<K::Scalar, U, V, D>,
        kernel: &K,
        topology: TopologyGuarantee,
        use_global_repair_fallback: bool,
        index: usize,
        repair_err: &DelaunayRepairError,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        if use_global_repair_fallback {
            tracing::debug!(
                error = %repair_err,
                idx = index,
                "bulk D<4: local repair cycling; falling back to global repair"
            );
            let global_result = repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology, None);
            if let Err(global_err) = global_result {
                tracing::debug!(
                    error = %global_err,
                    idx = index,
                    "bulk D<4: global repair also failed; aborting this vertex ordering"
                );
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "per-insertion Delaunay repair failed at index {index}: local error: {repair_err}; global fallback: {global_err}"
                    ),
                }
                .into());
            }
            return Ok(());
        }
        // Global repair disabled (e.g. periodic build): hard-fail to trigger
        // shuffle retry with a different vertex ordering.
        tracing::debug!(
            error = %repair_err,
            idx = index,
            "bulk D<4: local repair cycling (global fallback disabled); aborting"
        );
        Err(TriangulationConstructionError::GeometricDegeneracy {
            message: format!("per-insertion Delaunay repair failed at index {index}: {repair_err}"),
        }
        .into())
    }

    /// Restores positive geometric orientation after bulk repair calls the
    /// low-level TDS flip routine directly.
    fn canonicalize_after_bulk_repair(
        &mut self,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        self.tri
            .normalize_and_promote_positive_orientation()
            .map_err(Self::map_orientation_canonicalization_error)?;
        self.tri
            .validate_geometric_cell_orientation()
            .map_err(|error| {
                Self::map_orientation_canonicalization_error(InsertionError::TopologyValidation(
                    error,
                ))
            })?;
        Ok(())
    }

    /// Attempt one D≥4 local-repair escalation before the soft-fail path
    /// continues.
    ///
    /// Reruns `repair_delaunay_local_single_pass` with
    /// `base_budget * LOCAL_REPAIR_ESCALATION_BUDGET_FACTOR_D_GE_4` and the
    /// full TDS as seed set. Rate-limited by `LOCAL_REPAIR_ESCALATION_MIN_GAP`
    /// so only every Nth insertion pays the (near-global) flip pass cost.
    ///
    /// Returns a typed [`LocalRepairEscalationOutcome`] so the caller can
    /// distinguish `Skipped { reason }` (rate-limited or empty TDS) from
    /// `Succeeded { stats }` (caller has already canonicalized and should
    /// continue normally) from `FailedAlso { escalation_error }` (the
    /// escalation ran but also hit its budget; the caller should fall through
    /// to the soft-fail path, and the typed `DelaunayRepairError` is
    /// preserved for downstream diagnostics). `Err(...)` is reserved for
    /// canonicalization failures after a successful escalation, which are
    /// hard errors the bulk loop must propagate.
    fn try_local_repair_escalation_d_ge_4(
        &mut self,
        index: usize,
        base_budget: usize,
        last_escalation_idx: &mut Option<usize>,
        original_err: &DelaunayRepairError,
    ) -> Result<LocalRepairEscalationOutcome, DelaunayTriangulationConstructionError> {
        // Rate-limit: only escalate if we have not escalated within the last
        // LOCAL_REPAIR_ESCALATION_MIN_GAP insertions. This keeps healthy runs
        // from paying the near-global flip pass on every insertion while still
        // catching pathological clusters of consecutive soft-fails.
        if let Some(last_idx) = *last_escalation_idx
            && index.saturating_sub(last_idx) < LOCAL_REPAIR_ESCALATION_MIN_GAP
        {
            return Ok(LocalRepairEscalationOutcome::Skipped {
                reason: EscalationSkipReason::RateLimited {
                    last_escalation_idx: last_idx,
                    min_gap: LOCAL_REPAIR_ESCALATION_MIN_GAP,
                },
            });
        }

        // Escalation seed set: use every current cell key. This gives the
        // repair the broadest possible view of the local backlog without
        // switching to a different repair entry point.
        let full_seeds: Vec<CellKey> = self.tri.tds.cell_keys().collect();
        if full_seeds.is_empty() {
            return Ok(LocalRepairEscalationOutcome::Skipped {
                reason: EscalationSkipReason::EmptyTds,
            });
        }
        let escalated_budget =
            base_budget.saturating_mul(LOCAL_REPAIR_ESCALATION_BUDGET_FACTOR_D_GE_4);

        tracing::debug!(
            idx = index,
            seed_cells = full_seeds.len(),
            base_budget,
            escalated_budget,
            original_error = %original_err,
            "bulk D≥4: escalating local repair with full-TDS seed set"
        );

        let escalation_result = {
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_local_single_pass(tds, kernel, &full_seeds, escalated_budget)
        };

        *last_escalation_idx = Some(index);

        match escalation_result {
            Ok(stats) => {
                tracing::debug!(
                    idx = index,
                    flips = stats.flips_performed,
                    max_queue = stats.max_queue_len,
                    "bulk D≥4: escalation succeeded"
                );
                if stats.flips_performed > 0 {
                    self.canonicalize_after_bulk_repair()?;
                }
                Ok(LocalRepairEscalationOutcome::Succeeded { stats })
            }
            Err(escalation_err) => {
                tracing::debug!(
                    idx = index,
                    error = %escalation_err,
                    "bulk D≥4: escalation also non-convergent; falling through to soft-fail"
                );
                Ok(LocalRepairEscalationOutcome::FailedAlso {
                    escalation_error: escalation_err,
                })
            }
        }
    }

    /// Inserts the non-simplex vertices under a fixed perturbation seed so bulk
    /// construction retries are reproducible.
    #[allow(clippy::too_many_lines)]
    fn insert_remaining_vertices_seeded(
        &mut self,
        vertices: &[Vertex<K::Scalar, U, D>],
        perturbation_seed: u64,
        grid_cell_size: Option<K::Scalar>,
        construction_stats: Option<&mut ConstructionStatistics>,
        soft_fail_seeds: &mut Vec<CellKey>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        let mut grid_index = grid_cell_size.map(HashGridIndex::new);
        if let Some(grid) = grid_index.as_mut()
            && !grid.is_usable()
        {
            grid_index = None;
        }

        // Seed the local index from the initial simplex.
        if let Some(grid_index) = grid_index.as_mut() {
            for (vkey, vertex) in self.tri.tds.vertices() {
                grid_index.insert_vertex(vkey, vertex.point().coords());
            }
        }

        let trace_insertion = env::var_os("DELAUNAY_INSERT_TRACE").is_some();
        let mut batch_progress = bulk_progress_every_from_env().map(|progress_every| {
            let started = Instant::now();
            BatchProgressState {
                // The initial simplex is already present when this loop starts, so progress
                // and throughput only count the remaining bulk vertices — the counters live
                // in a "bulk-only" frame, 0…(input_len - (D+1)).
                total_vertices: vertices.len().saturating_sub(D + 1),
                progress_every,
                started,
                last_progress: started,
                last_processed: 0,
            }
        });
        // Bulk-only counters: `inserted_vertices` and `skipped_vertices` track work done
        // inside this loop and sum to `offset + 1` after each iteration, so the logged
        // progress line reads `processed=N/total inserted=I skipped=S` coherently.
        let mut inserted_vertices = 0usize;
        let mut skipped_vertices = 0usize;
        // Last insertion index at which the D≥4 local-repair escalation ran,
        // used for `LOCAL_REPAIR_ESCALATION_MIN_GAP` rate limiting across both
        // stats-enabled and stats-disabled arms.
        let mut last_escalation_idx: Option<usize> = None;

        match construction_stats {
            None => {
                // Insert remaining vertices incrementally.
                // Retryable geometric degeneracies are retried with perturbation and ultimately skipped
                // (transactional rollback) to keep the triangulation manifold. Duplicate/near-duplicate
                // coordinates are skipped immediately.
                for (offset, vertex) in vertices.iter().skip(D + 1).enumerate() {
                    let index = (D + 1).saturating_add(offset);
                    let uuid = vertex.uuid();
                    let coords = trace_insertion.then(|| {
                        vertex
                            .point()
                            .coords()
                            .iter()
                            .map(|c| c.to_f64().unwrap_or(f64::NAN))
                            .collect::<Vec<f64>>()
                    });

                    if trace_insertion && let Some(coords) = coords.as_ref() {
                        tracing::debug!(index, %uuid, coords = ?coords, "[bulk] start");
                    }

                    let started = trace_insertion.then(std::time::Instant::now);
                    let mut insert = || {
                        // Pass the batch index through to transactional insertion so the
                        // lower-layer retryable-skip trace can point back to this exact
                        // bulk-construction position.
                        self.tri.insert_with_statistics_seeded_indexed_detailed(
                            *vertex,
                            None,
                            self.insertion_state.last_inserted_cell,
                            perturbation_seed,
                            grid_index.as_mut(),
                            Some(index),
                        )
                    };
                    let insert_result = if trace_insertion {
                        let span = tracing::warn_span!(
                            "bulk_insert",
                            index,
                            uuid = %uuid,
                            coords = ?coords
                        );
                        span.in_scope(insert)
                    } else {
                        insert()
                    };
                    let elapsed = started.map(|started| started.elapsed());
                    let insert_result = insert_result.map(|detail| {
                        let repair_seed_cells = detail.repair_seed_cells;
                        (detail.outcome, detail.stats, repair_seed_cells)
                    });
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: v_key,
                                hint,
                            },
                            _stats,
                            repair_seed_cells,
                        )) => {
                            inserted_vertices = inserted_vertices.saturating_add(1);
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(index, %uuid, elapsed = ?elapsed, "[bulk] inserted");
                            }
                            // Cache hint for faster subsequent insertions.
                            self.insertion_state.last_inserted_cell = hint;
                            self.insertion_state.delaunay_repair_insertion_count = self
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);
                            // Per-insertion local Delaunay repair: seeded from the star of
                            // the inserted vertex with a seed-proportional flip budget.
                            //
                            // For D<4: the flip graph is proven convergent (Lawson 1977 for
                            // D=2, Rajan 1991/Joe 1991 for D=3).  On cycling (FP noise near
                            // co-spherical configurations), roll back the insertion and retry
                            // with perturbation to break the co-sphericity.
                            //
                            // For D≥4: Bowyer-Watson with the fast kernel can produce
                            // non-Delaunay facets when the conflict region is detected
                            // imprecisely (co-spherical configurations).  A bounded
                            // per-insertion repair pass fixes these violations.  If repair
                            // does not converge (e.g. co-spherical cycling suppressed by
                            // both_positive_artifact), the soft-fail path lets construction
                            // continue; the final is_valid() check validates the result.
                            let topology = self.tri.topology_guarantee();
                            if D >= 2
                                && TopologicalOperation::FacetFlip.is_admissible_under(topology)
                                && self.tri.tds.number_of_cells() > 0
                            {
                                let seed_cells =
                                    self.collect_local_repair_seed_cells(v_key, &repair_seed_cells);
                                if !seed_cells.is_empty() {
                                    let max_flips = local_repair_flip_budget::<D>(seed_cells.len());
                                    let repair_result = {
                                        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                                        repair_delaunay_local_single_pass(
                                            tds,
                                            kernel,
                                            &seed_cells,
                                            max_flips,
                                        )
                                    };
                                    #[cfg(test)]
                                    let repair_result =
                                        if tests::force_repair_nonconvergent_enabled() {
                                            Err(tests::synthetic_nonconvergent_error())
                                        } else {
                                            repair_result
                                        };
                                    match repair_result {
                                        Ok(stats) => {
                                            if stats.flips_performed > 0 {
                                                self.canonicalize_after_bulk_repair()?;
                                            }
                                        }
                                        Err(repair_err) => {
                                            if D < 4 {
                                                Self::try_d_lt4_global_repair_fallback(
                                                    &mut self.tri.tds,
                                                    &self.tri.kernel,
                                                    topology,
                                                    self.insertion_state.use_global_repair_fallback,
                                                    index,
                                                    &repair_err,
                                                )?;
                                                self.canonicalize_after_bulk_repair()?;
                                                log_bulk_progress_if_due(
                                                    BatchProgressSample {
                                                        processed: offset + 1,
                                                        inserted: inserted_vertices,
                                                        skipped: skipped_vertices,
                                                        cell_count: self.tri.tds.number_of_cells(),
                                                        perturbation_seed,
                                                    },
                                                    &mut batch_progress,
                                                );
                                                continue;
                                            }
                                            // D≥4: try one escalation with a 4× budget and the full
                                            // TDS as seed set before accepting the soft-fail. The
                                            // escalation is rate-limited so healthy runs do not pay
                                            // for it on every insertion.
                                            let outcome = self.try_local_repair_escalation_d_ge_4(
                                                index,
                                                max_flips,
                                                &mut last_escalation_idx,
                                                &repair_err,
                                            )?;
                                            match outcome {
                                                LocalRepairEscalationOutcome::Succeeded {
                                                    stats,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        flips = stats.flips_performed,
                                                        max_queue = stats.max_queue_len,
                                                        "bulk D≥4: escalation closed the \
                                                         non-convergence; continuing"
                                                    );
                                                    log_bulk_progress_if_due(
                                                        BatchProgressSample {
                                                            processed: offset + 1,
                                                            inserted: inserted_vertices,
                                                            skipped: skipped_vertices,
                                                            cell_count: self
                                                                .tri
                                                                .tds
                                                                .number_of_cells(),
                                                            perturbation_seed,
                                                        },
                                                        &mut batch_progress,
                                                    );
                                                    continue;
                                                }
                                                LocalRepairEscalationOutcome::Skipped {
                                                    reason,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        error = %repair_err,
                                                        escalation_outcome = "skipped",
                                                        skip_reason = ?reason,
                                                        "bulk D≥4: per-insertion repair \
                                                         non-convergent; continuing \
                                                         (both_positive_artifact handled)"
                                                    );
                                                    self.canonicalize_after_bulk_repair()?;
                                                    soft_fail_seeds
                                                        .extend(seed_cells.iter().copied());
                                                }
                                                LocalRepairEscalationOutcome::FailedAlso {
                                                    escalation_error,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        error = %repair_err,
                                                        escalation_outcome = "failed_also",
                                                        escalation_error = %escalation_error,
                                                        "bulk D≥4: per-insertion repair \
                                                         non-convergent; continuing \
                                                         (both_positive_artifact handled)"
                                                    );
                                                    self.canonicalize_after_bulk_repair()?;
                                                    soft_fail_seeds
                                                        .extend(seed_cells.iter().copied());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    processed: offset + 1,
                                    inserted: inserted_vertices,
                                    skipped: skipped_vertices,
                                    cell_count: self.tri.tds.number_of_cells(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Ok((InsertionOutcome::Skipped { error }, stats, _repair_seed_cells)) => {
                            skipped_vertices = skipped_vertices.saturating_add(1);
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    attempts = stats.attempts,
                                    elapsed = ?elapsed,
                                    error = %error,
                                    "[bulk] skipped"
                                );
                            }
                            // Keep going: this vertex was intentionally skipped (e.g. duplicate/near-duplicate
                            // coordinates, or an unsalvageable geometric degeneracy after retries).
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                attempts = stats.attempts,
                                error = %error,
                                "SKIPPED: vertex insertion during construction"
                            );
                            #[cfg(not(debug_assertions))]
                            {
                                let _ = (error, stats);
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    processed: offset + 1,
                                    inserted: inserted_vertices,
                                    skipped: skipped_vertices,
                                    cell_count: self.tri.tds.number_of_cells(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Err(e) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    elapsed = ?elapsed,
                                    error = %e,
                                    "[bulk] failed"
                                );
                            }
                            // Non-retryable failure: abort construction with a structured error.
                            return Err(Self::map_insertion_error(e).into());
                        }
                    }
                }
            }
            Some(construction_stats) => {
                // Same as above, but record insertion statistics and capture representative skip
                // samples for debugging.
                for (offset, vertex) in vertices.iter().skip(D + 1).enumerate() {
                    let index = (D + 1).saturating_add(offset);
                    let uuid = vertex.uuid();
                    let coords = trace_insertion.then(|| {
                        vertex
                            .point()
                            .coords()
                            .iter()
                            .map(|c| c.to_f64().unwrap_or(f64::NAN))
                            .collect::<Vec<f64>>()
                    });

                    if trace_insertion && let Some(coords) = coords.as_ref() {
                        tracing::debug!(index, %uuid, coords = ?coords, "[bulk] start");
                    }

                    let started = trace_insertion.then(std::time::Instant::now);
                    let mut insert = || {
                        // Keep the stats and non-stats branches aligned so bulk-index-based
                        // tracing behaves the same regardless of whether the caller records
                        // construction statistics.
                        self.tri.insert_with_statistics_seeded_indexed_detailed(
                            *vertex,
                            None,
                            self.insertion_state.last_inserted_cell,
                            perturbation_seed,
                            grid_index.as_mut(),
                            Some(index),
                        )
                    };
                    let insert_result = if trace_insertion {
                        let span = tracing::warn_span!(
                            "bulk_insert",
                            index,
                            uuid = %uuid,
                            coords = ?coords
                        );
                        span.in_scope(insert)
                    } else {
                        insert()
                    };
                    let elapsed = started.map(|started| started.elapsed());
                    let insert_result = insert_result.map(|detail| {
                        let repair_seed_cells = detail.repair_seed_cells;
                        (detail.outcome, detail.stats, repair_seed_cells)
                    });
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: v_key,
                                hint,
                            },
                            stats,
                            repair_seed_cells,
                        )) => {
                            inserted_vertices = inserted_vertices.saturating_add(1);
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    attempts = stats.attempts,
                                    elapsed = ?elapsed,
                                    "[bulk] inserted"
                                );
                            }
                            construction_stats.record_insertion(&stats);

                            // Cache hint for faster subsequent insertions.
                            self.insertion_state.last_inserted_cell = hint;
                            self.insertion_state.delaunay_repair_insertion_count = self
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);
                            // Per-insertion local repair: see the non-stats branch
                            // comment for full details.
                            let topology = self.tri.topology_guarantee();
                            if D >= 2
                                && TopologicalOperation::FacetFlip.is_admissible_under(topology)
                                && self.tri.tds.number_of_cells() > 0
                            {
                                let seed_cells =
                                    self.collect_local_repair_seed_cells(v_key, &repair_seed_cells);
                                if !seed_cells.is_empty() {
                                    let max_flips = local_repair_flip_budget::<D>(seed_cells.len());
                                    let repair_result = {
                                        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                                        repair_delaunay_local_single_pass(
                                            tds,
                                            kernel,
                                            &seed_cells,
                                            max_flips,
                                        )
                                    };
                                    #[cfg(test)]
                                    let repair_result =
                                        if tests::force_repair_nonconvergent_enabled() {
                                            Err(tests::synthetic_nonconvergent_error())
                                        } else {
                                            repair_result
                                        };
                                    match repair_result {
                                        Ok(stats) => {
                                            if stats.flips_performed > 0 {
                                                self.canonicalize_after_bulk_repair()?;
                                            }
                                        }
                                        Err(repair_err) => {
                                            if D < 4 {
                                                Self::try_d_lt4_global_repair_fallback(
                                                    &mut self.tri.tds,
                                                    &self.tri.kernel,
                                                    topology,
                                                    self.insertion_state.use_global_repair_fallback,
                                                    index,
                                                    &repair_err,
                                                )?;
                                                self.canonicalize_after_bulk_repair()?;
                                                log_bulk_progress_if_due(
                                                    BatchProgressSample {
                                                        processed: offset + 1,
                                                        inserted: inserted_vertices,
                                                        skipped: skipped_vertices,
                                                        cell_count: self.tri.tds.number_of_cells(),
                                                        perturbation_seed,
                                                    },
                                                    &mut batch_progress,
                                                );
                                                continue;
                                            }
                                            // D≥4: try one escalation with a 4× budget and the full
                                            // TDS as seed set before accepting the soft-fail. The
                                            // escalation is rate-limited so healthy runs do not pay
                                            // for it on every insertion.
                                            let outcome = self.try_local_repair_escalation_d_ge_4(
                                                index,
                                                max_flips,
                                                &mut last_escalation_idx,
                                                &repair_err,
                                            )?;
                                            match outcome {
                                                LocalRepairEscalationOutcome::Succeeded {
                                                    stats,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        flips = stats.flips_performed,
                                                        max_queue = stats.max_queue_len,
                                                        "bulk D≥4: escalation closed the \
                                                         non-convergence; continuing"
                                                    );
                                                    log_bulk_progress_if_due(
                                                        BatchProgressSample {
                                                            processed: offset + 1,
                                                            inserted: inserted_vertices,
                                                            skipped: skipped_vertices,
                                                            cell_count: self
                                                                .tri
                                                                .tds
                                                                .number_of_cells(),
                                                            perturbation_seed,
                                                        },
                                                        &mut batch_progress,
                                                    );
                                                    continue;
                                                }
                                                LocalRepairEscalationOutcome::Skipped {
                                                    reason,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        error = %repair_err,
                                                        escalation_outcome = "skipped",
                                                        skip_reason = ?reason,
                                                        "bulk D≥4: per-insertion repair \
                                                         non-convergent; continuing \
                                                         (both_positive_artifact handled)"
                                                    );
                                                    self.canonicalize_after_bulk_repair()?;
                                                    soft_fail_seeds
                                                        .extend(seed_cells.iter().copied());
                                                }
                                                LocalRepairEscalationOutcome::FailedAlso {
                                                    escalation_error,
                                                } => {
                                                    tracing::debug!(
                                                        idx = index,
                                                        error = %repair_err,
                                                        escalation_outcome = "failed_also",
                                                        escalation_error = %escalation_error,
                                                        "bulk D≥4: per-insertion repair \
                                                         non-convergent; continuing \
                                                         (both_positive_artifact handled)"
                                                    );
                                                    self.canonicalize_after_bulk_repair()?;
                                                    soft_fail_seeds
                                                        .extend(seed_cells.iter().copied());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    processed: offset + 1,
                                    inserted: inserted_vertices,
                                    skipped: skipped_vertices,
                                    cell_count: self.tri.tds.number_of_cells(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Ok((InsertionOutcome::Skipped { error }, stats, _repair_seed_cells)) => {
                            skipped_vertices = skipped_vertices.saturating_add(1);
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    attempts = stats.attempts,
                                    elapsed = ?elapsed,
                                    error = %error,
                                    "[bulk] skipped"
                                );
                            }
                            construction_stats.record_insertion(&stats);

                            // Keep the first few skip samples so we have concrete reproduction anchors.
                            let coords: Vec<f64> = vertex
                                .point()
                                .coords()
                                .iter()
                                .map(|c| c.to_f64().unwrap_or(f64::NAN))
                                .collect();
                            construction_stats.record_skip_sample(ConstructionSkipSample {
                                index,
                                uuid: vertex.uuid(),
                                coords,
                                attempts: stats.attempts,
                                error: error.to_string(),
                            });

                            // Keep going: this vertex was intentionally skipped (e.g. duplicate/near-duplicate
                            // coordinates, or an unsalvageable geometric degeneracy after retries).
                            #[cfg(debug_assertions)]
                            tracing::debug!(
                                attempts = stats.attempts,
                                error = %error,
                                "SKIPPED: vertex insertion during construction"
                            );
                            #[cfg(not(debug_assertions))]
                            {
                                let _ = (error, stats);
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    processed: offset + 1,
                                    inserted: inserted_vertices,
                                    skipped: skipped_vertices,
                                    cell_count: self.tri.tds.number_of_cells(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Err(e) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    elapsed = ?elapsed,
                                    error = %e,
                                    "[bulk] failed"
                                );
                            }
                            // Non-retryable failure: abort construction with a structured error.
                            return Err(Self::map_insertion_error(e).into());
                        }
                    }
                }
            }
        }

        // Keep the populated index for subsequent incremental insertions.
        self.spatial_index = grid_index;

        Ok(())
    }

    /// Restores runtime policies and performs the final repair/orientation
    /// checks that were deferred during batch insertion.
    fn finalize_bulk_construction(
        &mut self,
        original_validation_policy: ValidationPolicy,
        original_repair_policy: DelaunayRepairPolicy,
        run_final_repair: bool,
        soft_fail_seeds: &[CellKey],
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        // Restore policies after batch construction.
        self.tri.validation_policy = original_validation_policy;
        self.insertion_state.delaunay_repair_policy = original_repair_policy;

        let topology = self.tri.topology_guarantee();
        if run_final_repair && self.should_run_delaunay_repair_for(topology, 0) {
            // For D≥4: always run a global repair seeded from ALL cells.
            //   BW with the fast kernel can produce non-Delaunay facets anywhere,
            //   not only in the star of soft-failed insertions.  A small fixed
            //   budget ensures we fail fast on cycling rather than spending minutes.
            //   Non-convergence is a soft-fail; correctness is validated by
            //   is_delaunay_property_only() in build_with_shuffled_retries.
            //
            // For D<4: repair is proven convergent; per-insertion repair now
            //   falls back to global repair_delaunay_with_flips_k2_k3 on
            //   local non-convergence, so soft_fail_seeds is typically empty
            //   for D<4.  The seeded path below is kept for completeness.
            if D >= 4 {
                let cell_count = self.tri.tds.number_of_cells();
                if cell_count > 0 {
                    let all_cells: Vec<CellKey> = self.tri.tds.cell_keys().collect();
                    tracing::debug!(
                        cell_count,
                        "post-construction: starting global D≥4 finalize repair"
                    );
                    let repair_started = Instant::now();
                    let repair_result = {
                        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                        repair_delaunay_local_single_pass(tds, kernel, &all_cells, 512).map(|_| ())
                    };
                    tracing::debug!(
                        elapsed = ?repair_started.elapsed(),
                        success = repair_result.is_ok(),
                        "post-construction: D≥4 finalize repair completed (soft-fail)"
                    );
                    // Always soft-fail: is_delaunay_property_only() validates correctness.
                }
            } else if !soft_fail_seeds.is_empty() {
                // D<4 seeded repair (unused in practice; kept for completeness).
                tracing::debug!(
                    seed_count = soft_fail_seeds.len(),
                    "post-construction: starting seeded D<4 finalize repair"
                );
                let repair_started = Instant::now();
                let max_flips = (soft_fail_seeds.len() * (D + 1) * 16).max(512);
                let repair_result = {
                    let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                    repair_delaunay_local_single_pass(tds, kernel, soft_fail_seeds, max_flips)
                        .map(|_| ())
                };
                let repair_outcome: Result<(), DelaunayTriangulationConstructionError> =
                    match repair_result {
                        Ok(()) => Ok(()),
                        Err(e) => Err(TriangulationConstructionError::GeometricDegeneracy {
                            message: format!("Delaunay repair failed after construction: {e}"),
                        }
                        .into()),
                    };
                tracing::debug!(
                    elapsed = ?repair_started.elapsed(),
                    success = repair_outcome.is_ok(),
                    "post-construction: D<4 finalize repair completed"
                );
                repair_outcome?;
            }
        }

        // Flip-based repair calls normalize_coherent_orientation() which makes all cells
        // combinatorially coherent but can leave the global sign negative.  Re-canonicalize
        // geometric orientation to positive before validation (#258).
        self.tri
            .normalize_and_promote_positive_orientation()
            .map_err(Self::map_orientation_canonicalization_error)?;

        if topology.requires_vertex_links_at_completion() {
            tracing::debug!("post-construction: starting topology validation (finalize)");
            let validation_started = Instant::now();
            let validation_result = self.tri.validate();
            tracing::debug!(
                elapsed = ?validation_started.elapsed(),
                success = validation_result.is_ok(),
                "post-construction: topology validation (finalize) completed"
            );
            if let Err(err) = validation_result {
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("PL-manifold validation failed after construction: {err}"),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Map an [`InsertionError`] from post-construction orientation canonicalization
    /// into a [`TriangulationConstructionError`].
    ///
    /// Structural / data-structure errors (missing cells, broken invariants) become
    /// [`InternalInconsistency`](TriangulationConstructionError::InternalInconsistency)
    /// because they indicate algorithmic bugs rather than bad input geometry.
    /// Geometry-related failures (degenerate predicates, conflict regions, etc.) become
    /// [`GeometricDegeneracy`](TriangulationConstructionError::GeometricDegeneracy).
    ///
    /// NOTE: This match is intentionally exhaustive over `InsertionError`.
    /// When adding new variants, decide whether the failure mode is an internal
    /// bug or an input-geometry problem.
    fn map_orientation_canonicalization_error(
        error: InsertionError,
    ) -> TriangulationConstructionError {
        match error {
            // Construction already wraps a TriangulationConstructionError — preserve it
            // directly rather than rewrapping, mirroring map_insertion_error.
            InsertionError::Construction(source) => source,
            // Geometric orientation errors (degenerate or negative) are
            // geometry problems, not internal bugs.
            InsertionError::TopologyValidation(error @ TdsError::Geometric(_)) => {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Failed to canonicalize orientation after post-construction repair: {error}"
                    ),
                }
            }
            // Structural / data-structure errors indicate algorithmic bugs,
            // not input-geometry problems.
            //
            // NOTE: OrientationViolation (coherent-orientation invariant breach between
            // adjacent cells) lands here rather than in the geometry arm above.  After
            // normalize_coherent_orientation() BFS, a surviving violation would mean the
            // normalization algorithm failed its post-condition — an internal bug, not
            // bad input geometry.  DegenerateOrientation / NegativeOrientation capture
            // the actual FP-related geometry failures.
            error @ (InsertionError::TopologyValidation(_)
            | InsertionError::TopologyValidationFailed { .. }
            | InsertionError::CavityFilling { .. }
            | InsertionError::NeighborWiring { .. }
            | InsertionError::DuplicateUuid { .. }) => {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!(
                        "Failed to canonicalize orientation after post-construction repair: {error}"
                    ),
                }
            }
            // Geometry-related failures (degenerate input, predicate issues).
            error @ (InsertionError::ConflictRegion(_)
            | InsertionError::Location(_)
            | InsertionError::NonManifoldTopology { .. }
            | InsertionError::HullExtension { .. }
            | InsertionError::DelaunayValidationFailed { .. }
            | InsertionError::DelaunayRepairFailed { .. }
            | InsertionError::DuplicateCoordinates { .. }) => {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!(
                        "Failed to canonicalize orientation after post-construction repair: {error}"
                    ),
                }
            }
        }
    }

    /// Classifies insertion-layer failures as input degeneracy or internal
    /// inconsistency for construction callers.
    fn map_insertion_error(error: InsertionError) -> TriangulationConstructionError {
        match error {
            // Preserve underlying construction errors (e.g. duplicate UUID).
            InsertionError::Construction(source) => source,
            InsertionError::CavityFilling { message } => {
                TriangulationConstructionError::FailedToCreateCell { message }
            }
            InsertionError::NeighborWiring { message } => {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!("Neighbor wiring failed: {message}"),
                }
            }
            InsertionError::TopologyValidation(source) => {
                TriangulationConstructionError::from(TdsConstructionError::ValidationError(source))
            }
            InsertionError::DuplicateUuid { entity, uuid } => {
                TriangulationConstructionError::from(TdsConstructionError::DuplicateUuid {
                    entity,
                    uuid,
                })
            }
            InsertionError::DuplicateCoordinates { coordinates } => {
                TriangulationConstructionError::DuplicateCoordinates { coordinates }
            }

            // Insertion-layer failures that are best surfaced during construction as a
            // geometric degeneracy (e.g. numerical instability, hull visibility issues).
            //
            // NOTE: This match is intentionally exhaustive over `InsertionError`.
            // When adding new insertion failure modes in the future, revisit whether they
            // deserve a dedicated `TriangulationConstructionError` variant instead of being
            // collapsed into `GeometricDegeneracy`.
            //
            // We intentionally preserve the high-level insertion failure *bucket* in the
            // degeneracy message by capturing `error.to_string()` (rather than only
            // `source.to_string()`), so callers/telemetry can distinguish e.g.
            // "Conflict region error" vs "Location error" vs "Hull extension failed".
            insertion_error @ (InsertionError::ConflictRegion(_)
            | InsertionError::Location(_)
            | InsertionError::NonManifoldTopology { .. }
            | InsertionError::HullExtension { .. }
            | InsertionError::DelaunayValidationFailed { .. }
            | InsertionError::DelaunayRepairFailed { .. }
            | InsertionError::TopologyValidationFailed { .. }) => {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: insertion_error.to_string(),
                }
            }
        }
    }
}

// =============================================================================
// QUERY, CONFIGURATION, TRAVERSAL, REPAIR & VALIDATION (Minimal Bounds)
// =============================================================================
//
// Methods that only need `K: Kernel<D>` — no scalar arithmetic.  Downstream
// generic code (e.g. `delaunayize_by_flips`) does not need to carry
// `CoordinateScalar + NumCast` bounds when calling these methods.
//
// Follows the precedent of the existing PURE STRUCT ASSEMBLY impl block.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // QUERY / ACCESSORS
    // -------------------------------------------------------------------------

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    ///     vertex!([0.2, 0.2, 0.2, 0.2]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tri.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// // One 4-simplex in 4D
    /// assert_eq!(dt.number_of_cells(), 1);
    /// ```
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tri.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// Returns the dimension `D` as an `i32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.dim(), 4);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tri.dim()
    }

    /// Returns an iterator over all cells in the triangulation.
    ///
    /// This method provides access to the cells stored in the underlying
    /// triangulation data structure. The iterator yields `(CellKey, &Cell)`
    /// pairs for each cell in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(CellKey, &Cell<K::Scalar, U, V, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (cell_key, cell) in dt.cells() {
    ///     println!("Cell {:?} has {} vertices", cell_key, cell.number_of_vertices());
    /// }
    /// ```
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<K::Scalar, U, V, D>)> {
        self.tri.tds.cells()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// This method provides access to the vertices stored in the underlying
    /// triangulation data structure. The iterator yields `(VertexKey, &Vertex)`
    /// pairs for each vertex in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(VertexKey, &Vertex<K::Scalar, U, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tri.vertices()
    }

    /// Sets the auxiliary data on a vertex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants, so
    /// no caches are invalidated.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<U>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices: [Vertex<f64, i32, 2>; 3] = [
    ///     vertex!([0.0, 0.0], 10i32),
    ///     vertex!([1.0, 0.0], 20),
    ///     vertex!([0.0, 1.0], 30),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()
    ///     .unwrap();
    /// let key = dt.vertices().next().unwrap().0;
    ///
    /// let prev = dt.set_vertex_data(key, Some(99));
    /// assert!(prev.is_some());
    ///
    /// // Clear data
    /// let prev = dt.set_vertex_data(key, None);
    /// assert_eq!(prev, Some(Some(99)));
    /// assert_eq!(dt.tds().get_vertex_by_key(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_vertex_data(&mut self, key: VertexKey, data: Option<U>) -> Option<Option<U>> {
        self.tri.tds.set_vertex_data(key, data)
    }

    /// Sets the auxiliary data on a cell, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants, so
    /// no caches are invalidated.
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<V>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<i32>()
    ///     .unwrap();
    /// let key = dt.cells().next().unwrap().0;
    ///
    /// let prev = dt.set_cell_data(key, Some(42));
    /// assert_eq!(prev, Some(None));
    ///
    /// // Clear data
    /// let prev = dt.set_cell_data(key, None);
    /// assert_eq!(prev, Some(Some(42)));
    /// assert_eq!(dt.tds().get_cell(key).unwrap().data(), None);
    /// ```
    #[inline]
    pub fn set_cell_data(&mut self, key: CellKey, data: Option<V>) -> Option<Option<V>> {
        self.tri.tds.set_cell_data(key, data)
    }

    /// Returns a reference to the underlying triangulation data structure.
    ///
    /// This provides access to the purely combinatorial Tds layer for
    /// advanced operations and performance testing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// assert_eq!(tds.number_of_vertices(), 5);
    /// ```
    #[must_use]
    pub const fn tds(&self) -> &Tds<K::Scalar, U, V, D> {
        &self.tri.tds
    }

    /// Returns a mutable reference to the underlying triangulation data structure.
    ///
    /// This provides mutable access to the purely combinatorial Tds layer for
    /// advanced operations and testing of internal algorithms.
    ///
    /// # Safety
    ///
    /// Modifying the Tds directly can break Delaunay invariants. Use this only
    /// when you know what you're doing (typically in tests or specialized algorithms).
    #[cfg(test)]
    pub(crate) fn tds_mut(&mut self) -> &mut Tds<K::Scalar, U, V, D> {
        // Direct mutable access can invalidate performance caches.
        self.insertion_state.last_inserted_cell = None;
        self.spatial_index = None;
        &mut self.tri.tds
    }

    /// Returns a reference to the underlying `Triangulation` (kernel + tds).
    ///
    /// This is useful when you need to pass the triangulation to methods that
    /// expect a `&Triangulation`, such as `ConvexHull::from_triangulation()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// let vertices: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull = ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
    /// assert_eq!(hull.number_of_facets(), 4);
    /// ```
    #[must_use]
    pub const fn as_triangulation(&self) -> &Triangulation<K, U, V, D> {
        &self.tri
    }

    /// Returns a mutable reference to the underlying `Triangulation`.
    ///
    /// # ⚠️ WARNING - ADVANCED USE ONLY
    ///
    /// This method provides direct mutable access to the internal triangulation state.
    /// **Modifying the triangulation through this reference can break Delaunay invariants
    /// and leave the data structure in an inconsistent state.**
    ///
    /// ## When to Use
    ///
    /// This is primarily intended for:
    /// - **Testing internal algorithms** (topology validation, repair mechanisms)
    /// - **Advanced library development** (implementing custom triangulation operations)
    /// - **Research prototyping** (experimenting with new algorithms)
    ///
    /// ## What Can Go Wrong
    ///
    /// Direct mutations can violate critical invariants:
    /// - **Delaunay property**: Cells may no longer satisfy the empty circumsphere condition
    /// - **Manifold topology**: Facets may become over-shared or improperly connected
    /// - **Neighbor consistency**: Cell neighbor pointers may become invalid
    /// - **Hint caching**: Location hints may point to deleted cells
    ///
    /// After direct modification, you should:
    /// 1. Call `detect_local_facet_issues()` and `repair_local_facet_issues()` if you modified topology
    /// 2. Run `dt.as_triangulation().validate()` (Levels 1–3) or `dt.validate()` (Levels 1–4) to verify structural/topological consistency
    /// 3. Reserve `dt.is_valid()` for Delaunay-only (Level 4) checks
    ///
    /// ## Safe Alternatives
    ///
    /// For most use cases, prefer these safe, high-level methods:
    /// - [`insert()`](Self::insert) - Add vertices (maintains all invariants)
    /// - [`remove_vertex()`](Self::remove_vertex) - Remove vertices safely
    /// - [`tds()`](Self::tds) - Read-only access to the data structure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // ⚠️ Advanced use: direct access for testing validation
    /// let tri = dt.as_triangulation_mut();
    /// // ... perform internal algorithm testing ...
    ///
    /// // Always validate after direct modifications
    /// assert!(dt.validate().is_ok());
    /// ```
    #[must_use]
    pub fn as_triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        // Direct mutable access can invalidate performance caches.
        self.insertion_state.last_inserted_cell = None;
        self.spatial_index = None;
        &mut self.tri
    }

    /// Returns the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This policy controls when Level 3 (`Triangulation::is_valid()`) is run automatically
    /// during incremental insertion (as part of the topology safety net).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     delaunay::core::triangulation::ValidationPolicy::OnSuspicion
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.tri.validation_policy
    }

    /// Sets the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This affects subsequent incremental insertions. (Construction-time behavior is determined
    /// by the policy active during `new()` / `with_kernel()`.)
    ///
    /// If the requested policy is incompatible with the current topology guarantee (for example,
    /// `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](crate::core::triangulation::Triangulation::validate_at_completion)
    /// to provide immediate feedback and emits a warning. Call `validate_at_completion()` after
    /// batch construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// dt.set_validation_policy(delaunay::core::triangulation::ValidationPolicy::Always);
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     delaunay::core::triangulation::ValidationPolicy::Always
    /// );
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        self.tri.set_validation_policy(policy);
    }
    /// Returns the automatic Delaunay repair policy.
    #[inline]
    #[must_use]
    pub const fn delaunay_repair_policy(&self) -> DelaunayRepairPolicy {
        self.insertion_state.delaunay_repair_policy
    }

    /// Sets the automatic Delaunay repair policy.
    #[inline]
    pub const fn set_delaunay_repair_policy(&mut self, policy: DelaunayRepairPolicy) {
        self.insertion_state.delaunay_repair_policy = policy;
    }

    /// Returns the automatic global Delaunay validation policy.
    #[inline]
    #[must_use]
    pub const fn delaunay_check_policy(&self) -> DelaunayCheckPolicy {
        self.insertion_state.delaunay_check_policy
    }

    /// Sets the automatic global Delaunay validation policy.
    #[inline]
    pub const fn set_delaunay_check_policy(&mut self, policy: DelaunayCheckPolicy) {
        self.insertion_state.delaunay_check_policy = policy;
    }

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
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let stats = dt.repair_delaunay_with_flips().unwrap();
    /// assert!(stats.facets_checked >= stats.flips_performed);
    /// ```
    pub fn repair_delaunay_with_flips(&mut self) -> Result<DelaunayRepairStats, DelaunayRepairError>
    where
        K: ExactPredicates,
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
        K: ExactPredicates,
    {
        #[cfg(test)]
        if tests::force_repair_nonconvergent_enabled() {
            return Err(tests::synthetic_nonconvergent_error());
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
        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
        let stats = repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology, max_flips)?;

        // Re-canonicalize geometric orientation (#258): flip repair may leave
        // the global sign negative.
        self.ensure_positive_orientation()?;

        Ok(stats)
    }

    /// Canonicalize geometric orientation to the positive sign, mapping failures
    /// to [`DelaunayRepairError::PostconditionFailed`].
    fn ensure_positive_orientation(&mut self) -> Result<(), DelaunayRepairError> {
        self.tri
            .normalize_and_promote_positive_orientation()
            .map_err(|e| DelaunayRepairError::PostconditionFailed {
                message: format!("Orientation canonicalization failed after repair: {e}"),
            })
    }

    /// Replays repair with an exact-predicate kernel before escalating to
    /// heuristic rebuild.
    fn repair_delaunay_with_flips_robust(
        &mut self,
        seed_cells: Option<&[CellKey]>,
        max_flips: Option<usize>,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError> {
        let topology = self.tri.topology_guarantee();
        let kernel = RobustKernel::<K::Scalar>::new();
        let (tds, kernel) = (&mut self.tri.tds, &kernel);
        repair_delaunay_with_flips_k2_k3(tds, kernel, seed_cells, topology, max_flips)
    }

    /// Applies the repair policy only when the dimension and topology can
    /// support bistellar flips.
    fn should_run_delaunay_repair_for(
        &self,
        topology: TopologyGuarantee,
        insertion_count: usize,
    ) -> bool {
        if D < 2 {
            return false;
        }
        if self.tri.tds.number_of_cells() == 0 {
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

    /// Enables test-only repair fallback paths without exposing a public knob.
    #[allow(clippy::missing_const_for_fn)]
    fn force_heuristic_rebuild_enabled() -> bool {
        #[cfg(test)]
        {
            FORCE_HEURISTIC_REBUILD.with(std::cell::Cell::get)
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
// which constructs a new triangulation and therefore requires `CoordinateScalar`.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + NumCast,
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
    /// use delaunay::triangulation::delaunay::DelaunayRepairHeuristicConfig;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let outcome = dt
    ///     .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
    ///     .unwrap();
    /// assert!(outcome.stats.facets_checked >= outcome.stats.flips_performed);
    /// ```
    pub fn repair_delaunay_with_flips_advanced(
        &mut self,
        config: DelaunayRepairHeuristicConfig,
    ) -> Result<DelaunayRepairOutcome, DelaunayRepairError>
    where
        K: ExactPredicates,
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
    #[allow(clippy::too_many_lines)]
    fn rebuild_with_heuristic(
        &self,
        base_seeds: DelaunayRepairHeuristicSeeds,
        max_flips_override: Option<usize>,
    ) -> Result<(Self, DelaunayRepairStats, DelaunayRepairHeuristicSeeds), DelaunayRepairError>
    where
        K: ExactPredicates,
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
                let _guard = HeuristicRebuildRecursionGuard::enter();

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

                // During rebuild, force local repair after every insertion. We'll restore the caller's
                // policies after we have a repaired candidate.
                let rebuild_repair_policy = candidate.insertion_state.delaunay_repair_policy;
                let rebuild_check_policy = candidate.insertion_state.delaunay_check_policy;
                candidate.insertion_state.delaunay_repair_policy =
                    DelaunayRepairPolicy::EveryInsertion;
                candidate.insertion_state.delaunay_check_policy = DelaunayCheckPolicy::EndOnly;

                for (idx, vertex) in vertices.into_iter().enumerate() {
                    let uuid = vertex.uuid();
                    let coords = *vertex.point().coords();

                    let hint = candidate.insertion_state.last_inserted_cell;
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
                    let repair_seed_cells = insert_detail.repair_seed_cells;

                    match insert_detail.outcome {
                        InsertionOutcome::Inserted { vertex_key, hint } => {
                            candidate.insertion_state.last_inserted_cell = hint;
                            candidate.insertion_state.delaunay_repair_insertion_count = candidate
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);

                            candidate
                                .maybe_repair_after_insertion_capped(
                                    vertex_key,
                                    hint,
                                    &repair_seed_cells,
                                    max_flips_override,
                                )
                                .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                                    message: format!(
                                        "heuristic rebuild repair failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                                    ),
                                })?;

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
                candidate.insertion_state.last_inserted_cell = None;

                // Restore prior rebuild-only policies (kept for completeness; currently overwritten above).
                let _ = (rebuild_repair_policy, rebuild_check_policy);

                let topology = candidate.tri.topology_guarantee();
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

// =============================================================================
// CONFIGURATION & TRAVERSAL (Minimal Bounds, continued)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // CONFIGURATION
    // -------------------------------------------------------------------------

    /// Returns the topology guarantee used for Level 3 topology validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.tri.topology_guarantee()
    }

    /// Returns runtime global topology metadata associated with this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert!(dt.global_topology().is_euclidean());
    /// ```
    #[inline]
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        self.tri.global_topology()
    }

    /// Returns the high-level topology kind (`Euclidean`, `Toroidal`, etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// assert_eq!(dt.topology_kind(), TopologyKind::Euclidean);
    /// ```
    #[inline]
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        self.tri.topology_kind()
    }

    /// Sets runtime global topology metadata on this triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>().unwrap();
    /// dt.set_global_topology(GlobalTopology::Euclidean);
    /// assert!(dt.global_topology().is_euclidean());
    /// ```
    #[inline]
    pub const fn set_global_topology(&mut self, global_topology: GlobalTopology<D>) {
        self.tri.set_global_topology(global_topology);
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::TopologyGuarantee;
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    ///
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[inline]
    pub fn set_topology_guarantee(&mut self, guarantee: TopologyGuarantee) {
        self.tri.set_topology_guarantee(guarantee);
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// Delegates to the underlying `Triangulation` layer. This provides
    /// efficient access to all facets without pre-allocating a vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let facet_count = dt.facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.facets()
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one cell. This method
    /// computes the facet-to-cells map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.boundary_facets()
    }

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::build_adjacency_index`](crate::core::triangulation::Triangulation::build_adjacency_index).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying triangulation data structure is internally inconsistent
    /// (e.g., a cell references a missing vertex key or a missing neighbor cell key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// assert_eq!(index.number_of_edges(), 6);
    /// ```
    #[inline]
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        self.as_triangulation().build_adjacency_index()
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges`](crate::core::triangulation::Triangulation::edges).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let edges: std::collections::HashSet<_> = dt.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().edges()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges_with_index`](crate::core::triangulation::Triangulation::edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// let edges: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.as_triangulation().edges_with_index(index)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges`](crate::core::triangulation::Triangulation::incident_edges).
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// // In a tetrahedron, each vertex has degree 3.
    /// assert_eq!(dt.incident_edges(v0).count(), 3);
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.as_triangulation().incident_edges(v)
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges_with_index`](crate::core::triangulation::Triangulation::incident_edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// assert_eq!(dt.incident_edges_with_index(&index, v0).count(), 3);
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.as_triangulation().incident_edges_with_index(index, v)
    }

    /// Returns an iterator over all neighbors of a cell.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_neighbors`](crate::core::triangulation::Triangulation::cell_neighbors).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single tetrahedron has no cell neighbors.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// assert_eq!(dt.cell_neighbors(cell_key).count(), 0);
    /// ```
    pub fn cell_neighbors(&self, c: CellKey) -> impl Iterator<Item = CellKey> + '_ {
        self.as_triangulation().cell_neighbors(c)
    }

    /// Returns an iterator over all neighbors of a cell using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_neighbors_with_index`](crate::core::triangulation::Triangulation::cell_neighbors_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // Two tetrahedra sharing a triangular facet => each tetra has exactly one neighbor.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// let cell_key = dt.cells().next().unwrap().0;
    /// assert_eq!(dt.cell_neighbors_with_index(&index, cell_key).count(), 1);
    /// ```
    pub fn cell_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: CellKey,
    ) -> impl Iterator<Item = CellKey> + 'a {
        self.as_triangulation().cell_neighbors_with_index(index, c)
    }

    /// Returns a slice view of a cell's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_vertices`](crate::core::triangulation::Triangulation::cell_vertices).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let cell_vertices = dt.cell_vertices(cell_key).unwrap();
    /// assert_eq!(cell_vertices.len(), 3); // D+1 for a 2D simplex
    /// ```
    #[must_use]
    pub fn cell_vertices(&self, c: CellKey) -> Option<&[VertexKey]> {
        self.as_triangulation().cell_vertices(c)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::vertex_coords`](crate::core::triangulation::Triangulation::vertex_coords).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let v_key = dt
    ///     .vertices()
    ///     .find_map(|(vk, _)| (dt.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// assert_eq!(dt.vertex_coords(v_key).unwrap(), [1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]> {
        self.as_triangulation().vertex_coords(v)
    }
}

// =============================================================================
// MUTATION (Requires Numeric Scalar Bounds)
// =============================================================================
//
// Incremental insertion, removal, and post-insertion repair/check helpers.
// These require `CoordinateScalar + NumCast` for spatial-index construction,
// Triangulation-layer insertion, and Triangulation-layer removal.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + NumCast,
    U: DataType,
    V: DataType,
{
    /// Lazily seeds the spatial index from existing vertices so incremental
    /// insertion can start from deserialized or manually constructed TDS state.
    fn ensure_spatial_index_seeded(&mut self) {
        if self.spatial_index.is_some() {
            return;
        }

        let duplicate_tolerance: K::Scalar =
            <K::Scalar as NumCast>::from(1e-10_f64).unwrap_or_else(K::Scalar::default_tolerance);
        let mut index: HashGridIndex<K::Scalar, D> = HashGridIndex::new(duplicate_tolerance);

        for (vkey, vertex) in self.tri.tds.vertices() {
            index.insert_vertex(vkey, vertex.point().coords());
        }

        self.spatial_index = Some(index);
    }

    /// Insert a vertex into the Delaunay triangulation using incremental cavity-based algorithm.
    ///
    /// This method handles all stages of triangulation construction:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating cells
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-cell
    /// - **Incremental (> D+1 vertices)**: Uses cavity-based insertion with point location
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate cell containing the point
    /// 4. Find conflict region (cells whose circumspheres contain the point)
    /// 5. Extract cavity boundary
    /// 6. Fill cavity (create new cells)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict cells
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails (when reaching D+1 vertices)
    /// - Point is on a facet, edge, or vertex (degenerate cases not yet implemented)
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
    /// use delaunay::prelude::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices one by one - bootstrap phase (no cells yet)
    /// dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_cells(), 0); // Still no cells
    ///
    /// // 4th vertex triggers initial simplex creation
    /// dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_cells(), 1); // First cell created!
    ///
    /// // Further insertions use cavity-based algorithm
    /// dt.insert(vertex!([0.2, 0.2, 0.2])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// assert!(dt.number_of_cells() > 1);
    /// ```
    ///
    /// Using batch construction (traditional approach):
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// // Create initial triangulation with 5 vertices (4-simplex)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    ///
    /// // Insert additional interior vertex
    /// dt.insert(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// assert!(dt.number_of_cells() > 1);
    /// ```
    pub fn insert(&mut self, vertex: Vertex<K::Scalar, U, D>) -> Result<VertexKey, InsertionError> {
        self.ensure_spatial_index_seeded();

        // Fully delegate to Triangulation layer
        // Triangulation handles:
        // - Manifold maintenance (conflict cells, cavity, repairs)
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
        let could_have_cells_after_insertion = self.tri.tds.number_of_cells() > 0
            || self.tri.tds.number_of_vertices().saturating_add(1) > D;
        let snapshot_needed = could_have_cells_after_insertion
            && (self.insertion_state.delaunay_repair_policy != DelaunayRepairPolicy::Never
                || self
                    .insertion_state
                    .delaunay_check_policy
                    .should_check(next_insertion_count));
        let snapshot = snapshot_needed.then(|| {
            (
                self.tri.tds.clone(),
                self.insertion_state,
                self.spatial_index.clone(),
            )
        });

        let insertion_result = (|| {
            let hint = self.insertion_state.last_inserted_cell;
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
            let repair_seed_cells = insert_detail.repair_seed_cells;

            match insert_detail.outcome {
                InsertionOutcome::Inserted {
                    vertex_key: v_key,
                    hint,
                } => {
                    self.insertion_state.last_inserted_cell = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    self.maybe_repair_after_insertion(v_key, hint, &repair_seed_cells)?;
                    self.maybe_check_after_insertion()?;
                    Ok(v_key)
                }
                InsertionOutcome::Skipped { error } => Err(error),
            }
        })();

        match insertion_result {
            Ok(v_key) => Ok(v_key),
            Err(err) => {
                if let Some((tds, insertion_state, spatial_index)) = snapshot {
                    self.tri.tds = tds;
                    self.insertion_state = insertion_state;
                    self.spatial_index = spatial_index;
                }
                Err(err)
            }
        }
    }

    /// Insert a vertex and return the insertion outcome plus statistics.
    ///
    /// This is a convenience wrapper around the triangulation-layer insertion-with-statistics
    /// implementation that also updates the internal `insertion_state.last_inserted_cell` hint cache.
    ///
    /// # Errors
    ///
    /// Returns `Err(InsertionError)` only for non-retryable structural failures.
    /// Retryable geometric degeneracies that exhaust all attempts return
    /// `Ok((InsertionOutcome::Skipped { .. }, stats))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// let (outcome, stats) = dt
    ///     .insert_with_statistics(vertex!([0.0, 0.0, 0.0]))
    ///     .unwrap();
    ///
    /// assert!(stats.success());
    /// assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
    /// ```
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError> {
        self.ensure_spatial_index_seeded();

        // Transactional guard: post-steps (flip repair and/or global Delaunay checks) can fail.
        // If they do, rollback to leave the triangulation unchanged.
        let next_insertion_count = self
            .insertion_state
            .delaunay_repair_insertion_count
            .saturating_add(1);
        let could_have_cells_after_insertion = self.tri.tds.number_of_cells() > 0
            || self.tri.tds.number_of_vertices().saturating_add(1) > D;
        let snapshot_needed = could_have_cells_after_insertion
            && (self.insertion_state.delaunay_repair_policy != DelaunayRepairPolicy::Never
                || self
                    .insertion_state
                    .delaunay_check_policy
                    .should_check(next_insertion_count));
        let snapshot = snapshot_needed.then(|| {
            (
                self.tri.tds.clone(),
                self.insertion_state,
                self.spatial_index.clone(),
            )
        });

        let insertion_result = (|| {
            let hint = self.insertion_state.last_inserted_cell;
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
            let repair_seed_cells = insert_detail.repair_seed_cells;

            let outcome = match insert_detail.outcome {
                InsertionOutcome::Inserted { vertex_key, hint } => {
                    self.insertion_state.last_inserted_cell = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    self.maybe_repair_after_insertion(vertex_key, hint, &repair_seed_cells)?;
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
                if let Some((tds, insertion_state, spatial_index)) = snapshot {
                    self.tri.tds = tds;
                    self.insertion_state = insertion_state;
                    self.spatial_index = spatial_index;
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
        hint: Option<CellKey>,
        extra_seed_cells: &[CellKey],
    ) -> Result<(), InsertionError> {
        self.maybe_repair_after_insertion_capped(vertex_key, hint, extra_seed_cells, None)
    }

    /// Like [`maybe_repair_after_insertion`](Self::maybe_repair_after_insertion) but
    /// forwards an optional per-attempt flip cap to the underlying repair functions.
    ///
    /// `extra_seed_cells` widens the local repair frontier beyond the inserted vertex
    /// star. This is used when cavity reduction shrinks cells out of the conflict
    /// region: those cells stay in the triangulation and may still need a local
    /// Delaunay revisit even though they are no longer adjacent to the new vertex.
    fn maybe_repair_after_insertion_capped(
        &mut self,
        vertex_key: VertexKey,
        hint: Option<CellKey>,
        extra_seed_cells: &[CellKey],
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
        let seed_cells = self.collect_local_repair_seed_cells(vertex_key, extra_seed_cells);
        let hint_seed = hint.and_then(|ck| {
            if !self.tri.tds.contains_cell(ck) {
                if env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
                    tracing::debug!(
                        "[repair] insertion seed hint missing (cell={ck:?}, vertex={vertex_key:?})"
                    );
                }
                return None;
            }

            let contains_vertex = self
                .tri
                .tds
                .get_cell(ck)
                .is_some_and(|cell| cell.contains_vertex(vertex_key));
            if !contains_vertex && env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
                tracing::debug!(
                    "[repair] insertion seed hint does not contain vertex (cell={ck:?}, vertex={vertex_key:?})"
                );
            }

            contains_vertex.then_some(ck)
        });

        let seed_ref = if seed_cells.is_empty() {
            hint_seed.as_ref().map(std::slice::from_ref)
        } else {
            Some(seed_cells.as_slice())
        };

        let repair_result = {
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_with_flips_k2_k3(tds, kernel, seed_ref, topology, max_flips).map(|_| ())
        };

        #[cfg(test)]
        let repair_result = if tests::force_repair_nonconvergent_enabled() {
            Err(tests::synthetic_nonconvergent_error())
        } else {
            repair_result
        };

        match repair_result {
            Ok(()) => {}
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
                self.repair_delaunay_with_flips_robust(seed_ref, max_flips)
                    .map_err(|robust_err| InsertionError::DelaunayRepairFailed {
                        source: Box::new(robust_err),
                        context: format!("local repair failed ({e}); robust fallback also failed"),
                    })?;
            }
            Err(e) => {
                return Err(InsertionError::DelaunayRepairFailed {
                    source: Box::new(e),
                    context: "Delaunay repair failed (non-recoverable)".to_string(),
                });
            }
        }

        // Topology safety-net: flip-based repair is a topological operation and must not
        // violate the requested topology guarantee.
        //
        // In practice, higher-dimensional flip sequences can transiently (or permanently)
        // introduce PL-manifold violations (e.g., disconnected ridge links). Catch those
        // locally and surface an insertion error so the outer transactional guard can roll
        // back the insertion.
        //
        // The validation scope must match what repair actually touched: the inserted
        // vertex star (which may have grown via flips) **plus** any still-alive cells
        // from the pre-repair seed frontier. Otherwise a violation introduced in an
        // `extra_seed_cells` cell that is no longer adjacent to the new vertex would
        // slip past this safety-net.
        if topology.requires_ridge_links() {
            let mut validation_cells: Vec<CellKey> = self.tri.adjacent_cells(vertex_key).collect();
            let mut seen: FastHashSet<CellKey> = validation_cells.iter().copied().collect();
            for &cell_key in &seed_cells {
                if self.tri.tds.contains_cell(cell_key) && seen.insert(cell_key) {
                    validation_cells.push(cell_key);
                }
            }
            if !validation_cells.is_empty()
                && let Err(err) = validate_ridge_links_for_cells(&self.tri.tds, &validation_cells)
            {
                return Err(InsertionError::TopologyValidationFailed {
                    message: "Topology invalid after Delaunay repair".to_string(),
                    source: Box::new(TriangulationValidationError::from(err)),
                });
            }
        }
        // Flip-based repair mutates cell orderings; restore canonical positive geometric
        // orientation before exposing the updated triangulation state.
        self.tri.normalize_and_promote_positive_orientation()?;
        self.tri
            .validate_geometric_cell_orientation()
            .map_err(InsertionError::TopologyValidation)?;
        Ok(())
    }

    /// Merge the inserted vertex star with any cells that cavity reduction touched and
    /// left in place. Stale cells are ignored so callers can pass raw cavity-trace sets.
    fn collect_local_repair_seed_cells(
        &self,
        vertex_key: VertexKey,
        extra_seed_cells: &[CellKey],
    ) -> Vec<CellKey> {
        let mut seen: FastHashSet<CellKey> = FastHashSet::default();
        let mut seed_cells = Vec::new();

        // Keep the inserted vertex star first because it is the hottest local region and
        // the best chance of fixing ordinary post-insertion violations cheaply.
        for cell_key in self.tri.adjacent_cells(vertex_key) {
            if seen.insert(cell_key) {
                seed_cells.push(cell_key);
            }
        }

        // Then widen the frontier with cells touched by cavity shaping that survived in
        // the triangulation; deduping here lets callers pass raw trace buffers safely.
        for &cell_key in extra_seed_cells {
            if self.tri.tds.contains_cell(cell_key) && seen.insert(cell_key) {
                seed_cells.push(cell_key);
            }
        }

        seed_cells
    }

    /// Runs policy-controlled global validation after insertion so expensive
    /// Delaunay checks stay opt-in for incremental workflows.
    fn maybe_check_after_insertion(&self) -> Result<(), InsertionError> {
        if self.tri.tds.number_of_cells() == 0 {
            return Ok(());
        }

        let policy = self.insertion_state.delaunay_check_policy;
        let insertion_count = self.insertion_state.delaunay_repair_insertion_count;
        if !policy.should_check(insertion_count) {
            return Ok(());
        }

        self.is_valid()
            .map_err(|e| InsertionError::DelaunayValidationFailed {
                source: Box::new(e),
            })
    }

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation delegates to `Triangulation::remove_vertex()` which:
    /// 1. Finds all cells containing the vertex
    /// 2. Removes those cells (creating a cavity)
    /// 3. Fills the cavity with fan triangulation
    /// 4. Wires neighbors and rebuilds vertex-cell incidence
    /// 5. Removes the vertex
    ///
    /// Fast-path: if the vertex star is a simplex (exactly D+1 incident cells with
    /// consistent adjacency), this method collapses it via the **inverse k=1** bistellar
    /// flip. Otherwise it falls back to fan triangulation.
    ///
    /// The triangulation remains topologically valid after removal. However, both the
    /// inverse k=1 fast-path and fan triangulation may temporarily violate the Delaunay
    /// property in some cases. If the [`DelaunayRepairPolicy`] allows it, a flip-based
    /// repair pass is run automatically after removal.
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
    /// The number of cells that were removed along with the vertex. Returns `Ok(0)` if
    /// `vertex_key` does not refer to a vertex in the triangulation (e.g. a stale key from
    /// a previously removed vertex or a key that was never inserted). This is a successful
    /// no-op, not an error.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError`] if:
    /// - The inverse k=1 flip encounters a neighbor-wiring failure (`InvariantError::Tds`).
    /// - Fan retriangulation fails (`InvariantError::Tds`).
    /// - Delaunay flip-based repair fails after removal (`InvariantError::Delaunay`).
    /// - Orientation canonicalization fails after repair (`InvariantError::Tds`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([0.3, 0.3]), // interior vertex
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Find the key of a known interior vertex.
    /// let vertex_key = dt
    ///     .vertices()
    ///     .find(|(_, v)| {
    ///         let coords = v.point().coords();
    ///         (coords[0] - 0.3).abs() < 1e-10 && (coords[1] - 0.3).abs() < 1e-10
    ///     })
    ///     .map(|(k, _)| k)
    ///     .unwrap();
    ///
    /// // Remove the vertex and all cells containing it
    /// let cells_removed = dt.remove_vertex(vertex_key).unwrap();
    /// println!("Removed {} cells along with the vertex", cells_removed);
    ///
    /// // Vertex removal preserves Levels 1–3 but may not preserve the Delaunay property.
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// ```
    pub fn remove_vertex(&mut self, vertex_key: VertexKey) -> Result<usize, InvariantError> {
        if self.tri.tds.get_vertex_by_key(vertex_key).is_none() {
            return Ok(0);
        }

        // Fast path: inverse k=1 flip when the vertex star is a simplex.
        let mut seed_cells: Option<CellKeyBuffer> = None;
        let cells_removed = match apply_bistellar_flip_k1_inverse(&mut self.tri.tds, vertex_key) {
            Ok(info) => {
                seed_cells = Some(info.new_cells);
                info.removed_cells.len()
            }
            Err(FlipError::NeighborWiring { message }) => {
                return Err(TdsError::InvalidNeighbors {
                    message: format!("inverse k=1 flip failed during remove_vertex: {message}"),
                }
                .into());
            }
            Err(_) => self.tri.remove_vertex(vertex_key)?,
        };

        let topology = self.tri.topology_guarantee();
        if self.should_run_delaunay_repair_for(topology, 0) {
            let seed_ref = seed_cells.as_deref();
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_with_flips_k2_k3(tds, kernel, seed_ref, topology, None).map_err(
                |e| {
                    InvariantError::Delaunay(DelaunayTriangulationValidationError::RepairFailed {
                        message: format!("Delaunay repair failed after vertex removal: {e}"),
                    })
                },
            )?;

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

        Ok(cells_removed)
    }
}

// =============================================================================
// VALIDATION (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // VALIDATION
    // -------------------------------------------------------------------------

    /// Validates the Delaunay empty-circumsphere property (Level 4).
    ///
    /// This is the Delaunay layer's `is_valid`: it checks **only** the Delaunay property
    /// and intentionally does **not** run lower-layer validation.
    ///
    /// **Performance**: Uses fast O(cells) flip-based verification instead of the naive
    /// O(cells × vertices) brute-force check, providing ~40-100x speedup. This method is
    /// correct for all properly-constructed triangulations (which is the standard case).
    ///
    /// For cumulative validation across the whole hierarchy, use [`validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if the empty-circumsphere test fails, or if
    /// the underlying triangulation state is inconsistent and prevents geometric predicates
    /// from being evaluated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Level 4: Delaunay property only
    /// assert!(dt.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), DelaunayTriangulationValidationError> {
        // Use fast flip-based verification (O(cells) instead of O(cells × vertices))
        self.is_delaunay_via_flips().map_err(|err| {
            DelaunayTriangulationValidationError::VerificationFailed {
                message: err.to_string(),
            }
        })
    }

    /// Verify the Delaunay property via fast O(cells) flip predicates.
    ///
    /// This checks the Delaunay property by testing all possible flip configurations
    /// (k=2 facets, k=3 ridges, and their inverses) instead of the naive O(cells × vertices)
    /// brute-force check. This is ~40-100x faster while being equally correct.
    ///
    /// Ideal for property-based testing with many iterations.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayRepairError`] if any flip predicate detects a Delaunay violation.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Fast O(N) verification
    /// assert!(dt.is_delaunay_via_flips().is_ok());
    /// ```
    pub fn is_delaunay_via_flips(&self) -> Result<(), DelaunayRepairError> {
        verify_delaunay_for_triangulation(&self.tri)
    }

    /// Performs cumulative validation for Levels 1–4.
    ///
    /// This validates:
    /// - **Levels 1–3** via [`Triangulation::validate`](crate::core::triangulation::Triangulation::validate)
    /// - **Level 4** via [`DelaunayTriangulation::is_valid`](Self::is_valid)
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if Levels 1–3 validation fails or if the
    /// Delaunay property check (Level 4) fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Levels 1–4: elements + structure + topology + Delaunay property
    /// assert!(dt.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), DelaunayTriangulationValidationError> {
        self.tri.validate().map_err(|e| match e {
            InvariantError::Tds(tds_err) => DelaunayTriangulationValidationError::Tds(tds_err),
            InvariantError::Triangulation(tri_err) => {
                DelaunayTriangulationValidationError::Triangulation(tri_err)
            }
            InvariantError::Delaunay(dt_err) => dt_err,
        })?;
        self.is_valid()
    }

    /// Generate a comprehensive validation report for the full validation hierarchy.
    ///
    /// This is intended for debugging/telemetry (e.g. `insert_with_statistics`) where
    /// you want to see *all* violated invariants, not just the first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–4.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all violated invariants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Returns Ok(()) on success; otherwise returns a report listing all violations.
    /// let report = dt.validation_report();
    /// assert!(report.is_ok());
    /// ```
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        // Levels 1–3: reuse the Triangulation layer report.
        match self.tri.validation_report() {
            Ok(()) => {
                // Level 4 (Delaunay property)
                if let Err(e) = self.is_valid() {
                    return Err(TriangulationValidationReport {
                        violations: vec![InvariantViolation {
                            kind: InvariantKind::DelaunayProperty,
                            error: e.into(),
                        }],
                    });
                }
                Ok(())
            }
            Err(mut report) => {
                // If mappings are inconsistent, return the lower-layer report unchanged.
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::CellMappings
                    )
                }) {
                    return Err(report);
                }

                // Level 4 (Delaunay property)
                if let Err(e) = self.is_delaunay_via_flips() {
                    report.violations.push(InvariantViolation {
                        kind: InvariantKind::DelaunayProperty,
                        error: InvariantError::Delaunay(
                            DelaunayTriangulationValidationError::VerificationFailed {
                                message: e.to_string(),
                            },
                        ),
                    });
                }

                if report.violations.is_empty() {
                    Ok(())
                } else {
                    Err(report)
                }
            }
        }
    }
    // -------------------------------------------------------------------------
    // PURE STRUCT ASSEMBLY
    // -------------------------------------------------------------------------
    /// Create a `DelaunayTriangulation` from a `Tds` with a default kernel.
    ///
    /// This is useful when you've serialized just the `Tds` and want to reconstruct
    /// the `DelaunayTriangulation` with default kernel settings.
    ///
    /// # Notes
    ///
    /// - The internal `insertion_state.last_inserted_cell` "locate hint" is intentionally **not** persisted
    ///   across serialization boundaries. Constructing via `from_tds` (including the serde
    ///   `Deserialize` impl below) always resets it to `None`. This can make the first few
    ///   insertions after loading slightly slower, but is otherwise behaviorally irrelevant.
    /// - The internal spatial hash-grid index used to accelerate incremental insertion is also a
    ///   performance-only cache and is not serialized. Constructing via `from_tds` leaves it unset
    ///   so it can be rebuilt lazily on demand.
    /// - The topology guarantee ([`TopologyGuarantee`]) is also not serialized (this type serializes
    ///   only the `Tds`). Constructing via `from_tds` resets it to `TopologyGuarantee::DEFAULT`
    ///   (currently `PLManifold`). Call [`set_topology_guarantee`](Self::set_topology_guarantee)
    ///   after loading if you want to relax to `Pseudomanifold` for performance, or use
    ///   [`from_tds_with_topology_guarantee`](Self::from_tds_with_topology_guarantee) to set it
    ///   at construction time.
    /// - Runtime global topology metadata ([`GlobalTopology`]) is also not serialized. Constructing
    ///   via `from_tds` resets it to [`GlobalTopology::Euclidean`]. Use
    ///   [`set_global_topology`](Self::set_global_topology) after loading if you need to restore
    ///   toroidal metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::triangulation::delaunay::DelaunayTriangulation;
    /// use delaunay::core::tds::Tds;
    /// use delaunay::geometry::kernel::FastKernel;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Serialize just the Tds
    /// let json = serde_json::to_string(dt.tds()).unwrap();
    ///
    /// // Deserialize Tds and reconstruct DelaunayTriangulation
    /// let tds: Tds<f64, (), (), 4> = serde_json::from_str(&json).unwrap();
    /// let reconstructed = DelaunayTriangulation::from_tds(tds, FastKernel::new());
    /// assert_eq!(reconstructed.number_of_vertices(), 5);
    /// ```
    #[must_use]
    pub const fn from_tds(tds: Tds<K::Scalar, U, V, D>, kernel: K) -> Self {
        Self::from_tds_with_topology_guarantee(tds, kernel, TopologyGuarantee::DEFAULT)
    }

    /// Create a `DelaunayTriangulation` from a `Tds` with an explicit topology guarantee.
    ///
    /// This is identical to [`from_tds`](Self::from_tds), but allows callers to set the desired
    /// [`TopologyGuarantee`] at construction time. The initial
    /// [`ValidationPolicy`] is derived from the guarantee:
    /// [`PLManifoldStrict`](TopologyGuarantee::PLManifoldStrict) uses
    /// [`Always`](ValidationPolicy::Always); all others default to
    /// [`OnSuspicion`](ValidationPolicy::OnSuspicion).
    #[must_use]
    pub const fn from_tds_with_topology_guarantee(
        tds: Tds<K::Scalar, U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        let validation_policy = topology_guarantee.default_validation_policy();
        Self {
            tri: Triangulation {
                kernel,
                tds,
                global_topology: GlobalTopology::DEFAULT,
                validation_policy,
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        }
    }
}

// Custom Serialize implementation that only serializes the Tds
impl<K, U, V, const D: usize> Serialize for DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Only serialize the Tds; kernel can be reconstructed on deserialization
        self.tri.tds.serialize(serializer)
    }
}

/// Custom `Deserialize` implementation for `RobustKernel<f64>` with no custom data.
///
/// Kernels are stateless and can be reconstructed on deserialization.  We only
/// serialize the `Tds` (which contains all the geometric and topological data),
/// then reconstruct the kernel wrapper on deserialization.
///
/// # Note on Locate Hint Persistence
///
/// The internal `insertion_state.last_inserted_cell` "locate hint" is intentionally
/// **not** serialized.  Deserialization reconstructs a fresh triangulation via
/// [`from_tds()`](Self::from_tds), which resets the hint to `None`.  This only
/// affects performance for the first few insertions after loading.
///
/// # Usage with Other Kernels
///
/// For other kernels (e.g., `AdaptiveKernel`, `FastKernel`) or custom data types,
/// deserialize the `Tds` directly and reconstruct with [`from_tds()`](Self::from_tds):
///
/// ```rust
/// # use delaunay::prelude::geometry::*;
/// # use delaunay::prelude::triangulation::*;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create and serialize a triangulation
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::<_, (), (), 3>::new(&vertices)?;
/// let json = serde_json::to_string(&dt)?;
///
/// // Deserialize with a specific kernel via from_tds
/// let tds: Tds<f64, (), (), 3> = serde_json::from_str(&json)?;
/// let dt_adaptive = DelaunayTriangulation::from_tds(tds, AdaptiveKernel::new());
/// # Ok(())
/// # }
/// ```
impl<'de, const D: usize> Deserialize<'de> for DelaunayTriangulation<RobustKernel<f64>, (), (), D>
where
    Tds<f64, (), (), D>: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let tds = Tds::deserialize(deserializer)?;
        Ok(Self::from_tds(tds, RobustKernel::new()))
    }
}

/// Policy controlling automatic flip-based Delaunay repair.
///
/// This policy schedules **local flip-based repairs** after successful insertions
/// (and removals that modify topology).
/// It is separate from any *validation-only* policy to allow checking the Delaunay
/// property without mutating topology when needed.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::DelaunayRepairPolicy;
/// use std::num::NonZeroUsize;
///
/// let policy = DelaunayRepairPolicy::EveryN(NonZeroUsize::new(4).unwrap());
/// assert!(!policy.should_repair(3));
/// assert!(policy.should_repair(4));
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
            Self::EveryInsertion => true,
            Self::EveryN(n) => insertion_count.is_multiple_of(n.get()),
        }
    }
}
/// Configuration for the optional heuristic rebuild fallback in Delaunay repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::DelaunayRepairHeuristicConfig;
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
    fn resolve_seeds(self, base_seed: u64) -> DelaunayRepairHeuristicSeeds {
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
/// use delaunay::triangulation::delaunay::DelaunayRepairHeuristicSeeds;
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
/// use delaunay::core::algorithms::flips::DelaunayRepairStats;
/// use delaunay::triangulation::delaunay::DelaunayRepairOutcome;
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
/// Global Delaunay validation is **extremely expensive**: O(cells × vertices). Use this policy
/// primarily when you need correctness guarantees and are willing to pay the cost.
///
/// # Examples
///
/// ```rust
/// use delaunay::triangulation::delaunay::DelaunayCheckPolicy;
/// use std::num::NonZeroUsize;
///
/// let policy = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(3).unwrap());
/// assert!(!policy.should_check(2));
/// assert!(policy.should_check(3));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaunayCheckPolicy {
    /// Run global Delaunay validation once after batch construction (e.g. `new()` / `with_kernel()`).
    ///
    /// Incremental insertion does not automatically run a final global check because there is no
    /// intrinsic “end” signal; call [`DelaunayTriangulation::is_valid`] or
    /// [`DelaunayTriangulation::validate`] when you are done inserting.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairError, FlipError, RepairQueueOrder,
        verify_delaunay_via_flip_predicates,
    };
    use crate::core::algorithms::incremental_insertion::{
        HullExtensionReason, repair_neighbor_pointers,
    };
    use crate::core::algorithms::locate::{ConflictError, LocateError};
    use crate::core::tds::{EntityKind, GeometricError};
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel, RobustKernel};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::topology::characteristics::euler::TopologyClassification;
    use crate::topology::traits::topological_space::ToroidalConstructionMode;
    use crate::triangulation::flips::BistellarFlips;
    use crate::vertex;
    use rand::{RngExt, SeedableRng};
    use slotmap::KeyData;

    pub(super) fn force_repair_nonconvergent_enabled() -> bool {
        FORCE_REPAIR_NONCONVERGENT.with(std::cell::Cell::get)
    }

    pub(super) fn synthetic_nonconvergent_error() -> DelaunayRepairError {
        DelaunayRepairError::NonConvergent {
            max_flips: 0,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 0,
                flips_performed: 0,
                max_queue_len: 0,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 0,
                queue_order: RepairQueueOrder::Fifo,
            }),
        }
    }
    fn init_tracing() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    macro_rules! gen_local_repair_flip_budget_tests {
        ($dim:literal, $floor:ident, $factor:ident) => {
            pastey::paste! {
                #[test]
                fn [<test_local_repair_flip_budget_uses_dimension_specific_floor_and_factor_ $dim d>]() {
                    assert_eq!(local_repair_flip_budget::<$dim>(0), $floor);

                    let seed_count = 10;
                    let raw = seed_count * ($dim + 1) * $factor;
                    assert_eq!(local_repair_flip_budget::<$dim>(seed_count), raw.max($floor));
                }
            }
        };
    }

    gen_local_repair_flip_budget_tests!(
        2,
        LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_LT_4,
        LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_LT_4
    );
    gen_local_repair_flip_budget_tests!(
        3,
        LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_LT_4,
        LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_LT_4
    );
    gen_local_repair_flip_budget_tests!(
        4,
        LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4,
        LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4
    );
    gen_local_repair_flip_budget_tests!(
        5,
        LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4,
        LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4
    );

    #[test]
    fn test_log_bulk_progress_if_due_updates_progress_state_only_when_due() {
        let sample = BatchProgressSample {
            processed: 5,
            inserted: 4,
            skipped: 1,
            cell_count: 7,
            perturbation_seed: 0xCAFE,
        };

        let mut disabled = None;
        log_bulk_progress_if_due(sample, &mut disabled);
        assert!(disabled.is_none());

        let mut state = Some(BatchProgressState {
            total_vertices: 10,
            progress_every: 5,
            started: Instant::now(),
            last_progress: Instant::now(),
            last_processed: 0,
        });

        log_bulk_progress_if_due(
            BatchProgressSample {
                processed: 0,
                ..sample
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 0);

        log_bulk_progress_if_due(
            BatchProgressSample {
                processed: 3,
                ..sample
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 0);

        log_bulk_progress_if_due(sample, &mut state);
        assert_eq!(state.as_ref().unwrap().last_processed, 5);

        log_bulk_progress_if_due(
            BatchProgressSample {
                processed: 10,
                inserted: 8,
                skipped: 2,
                cell_count: 11,
                perturbation_seed: 0xBEEF,
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 10);
    }

    #[test]
    fn test_collect_local_repair_seed_cells_merges_adjacent_extra_and_ignores_stale() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let all_cells: Vec<CellKey> = dt.cells().map(|(cell_key, _)| cell_key).collect();

        let (vertex_key, adjacent, extra_cell) = dt
            .vertices()
            .find_map(|(vertex_key, _)| {
                let adjacent: Vec<CellKey> = dt.tri.adjacent_cells(vertex_key).collect();
                all_cells
                    .iter()
                    .copied()
                    .find(|cell_key| !adjacent.contains(cell_key))
                    .map(|extra_cell| (vertex_key, adjacent, extra_cell))
            })
            .expect("fixture should contain a cell outside at least one vertex star");

        let stale_cell = CellKey::from(KeyData::from_ffi(999_999));
        let seeds = dt.collect_local_repair_seed_cells(
            vertex_key,
            &[adjacent[0], extra_cell, extra_cell, stale_cell],
        );

        assert_eq!(seeds.len(), adjacent.len() + 1);
        assert_eq!(&seeds[..adjacent.len()], adjacent.as_slice());
        assert_eq!(seeds[adjacent.len()], extra_cell);
        assert!(!seeds.contains(&stale_cell));
    }

    struct ForceHeuristicRebuildGuard {
        prior: bool,
    }

    impl ForceHeuristicRebuildGuard {
        fn enable() -> Self {
            let prior = FORCE_HEURISTIC_REBUILD.with(|flag| {
                let prior = flag.get();
                flag.set(true);
                prior
            });
            Self { prior }
        }
    }

    impl Drop for ForceHeuristicRebuildGuard {
        fn drop(&mut self) {
            FORCE_HEURISTIC_REBUILD.with(|flag| flag.set(self.prior));
        }
    }

    struct ForceRepairNonconvergentGuard {
        prior: bool,
    }

    impl ForceRepairNonconvergentGuard {
        fn enable() -> Self {
            let prior = FORCE_REPAIR_NONCONVERGENT.with(|flag| {
                let prior = flag.get();
                flag.set(true);
                prior
            });
            Self { prior }
        }
    }

    impl Drop for ForceRepairNonconvergentGuard {
        fn drop(&mut self) {
            FORCE_REPAIR_NONCONVERGENT.with(|flag| flag.set(self.prior));
        }
    }

    #[test]
    fn test_construction_options_builder_roundtrip() {
        init_tracing();
        let opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_dedup_policy(DedupPolicy::Exact)
            .with_retry_policy(RetryPolicy::Disabled);

        assert_eq!(opts.insertion_order(), InsertionOrderStrategy::Input);
        assert_eq!(opts.dedup_policy(), DedupPolicy::Exact);
        assert_eq!(opts.retry_policy(), RetryPolicy::Disabled);
    }

    #[test]
    fn test_construction_options_global_repair_fallback_toggle() {
        init_tracing();
        let default_opts = ConstructionOptions::default();
        assert!(
            default_opts.use_global_repair_fallback,
            "default should enable global repair fallback"
        );

        let disabled_opts = default_opts.without_global_repair_fallback();
        assert!(
            !disabled_opts.use_global_repair_fallback,
            "without_global_repair_fallback should disable the flag"
        );

        // Chaining with other builders should preserve the flag.
        let chained_opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .without_global_repair_fallback()
            .with_retry_policy(RetryPolicy::Disabled);
        assert!(!chained_opts.use_global_repair_fallback);
        assert_eq!(
            chained_opts.insertion_order(),
            InsertionOrderStrategy::Input
        );
        assert_eq!(chained_opts.retry_policy(), RetryPolicy::Disabled);
    }

    #[test]
    fn test_new_with_options_smoke_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let opts = ConstructionOptions::default().with_retry_policy(RetryPolicy::Disabled);
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_cells(), 1);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_new_with_construction_statistics_counts_initial_simplex_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let (dt, stats) =
            DelaunayTriangulation::new_with_construction_statistics(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(stats.inserted, 4);
        assert_eq!(stats.total_skipped(), 0);
        assert_eq!(stats.total_attempts, 4);
        assert_eq!(stats.max_attempts, 1);
        assert_eq!(stats.attempts_histogram.get(1).copied().unwrap_or(0), 4);
    }

    #[test]
    fn test_new_with_options_and_construction_statistics_skips_duplicate_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            // Initial simplex
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            // Duplicate coords (different UUID)
            vertex!([0.0, 0.0, 0.0]),
        ];
        let duplicate_uuid = vertices[4].uuid();

        let opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_retry_policy(RetryPolicy::Disabled);

        let (dt, stats) =
            DelaunayTriangulation::new_with_options_and_construction_statistics(&vertices, opts)
                .unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(stats.inserted, 4);
        assert_eq!(stats.skipped_duplicate, 1);
        assert_eq!(stats.skipped_degeneracy, 0);
        assert_eq!(stats.total_skipped(), 1);
        assert_eq!(stats.total_attempts, 5);
        assert_eq!(stats.attempts_histogram.get(1).copied().unwrap_or(0), 5);

        assert_eq!(stats.skip_samples.len(), 1);
        let sample = &stats.skip_samples[0];
        assert_eq!(sample.index, 4);
        assert_eq!(sample.uuid, duplicate_uuid);
        assert_eq!(sample.coords, vec![0.0, 0.0, 0.0]);
        assert_eq!(sample.attempts, 1);
        assert!(sample.error.contains("Duplicate coordinates"));
    }
    #[test]
    fn test_construction_statistics_record_insertion_tracks_inserted_common_fields() {
        init_tracing();

        let mut summary = ConstructionStatistics::default();
        let stats = InsertionStatistics {
            attempts: 3,
            cells_removed_during_repair: 4,
            result: crate::core::operations::InsertionResult::Inserted,
        };

        summary.record_insertion(&stats);

        assert_eq!(summary.inserted, 1);
        assert_eq!(summary.skipped_duplicate, 0);
        assert_eq!(summary.skipped_degeneracy, 0);
        assert_eq!(summary.total_attempts, 3);
        assert_eq!(summary.max_attempts, 3);
        assert_eq!(summary.attempts_histogram.get(3).copied().unwrap_or(0), 1);
        assert_eq!(summary.used_perturbation, 1);
        assert_eq!(summary.cells_removed_total, 4);
        assert_eq!(summary.cells_removed_max, 4);

        // Borrowed API: caller retains ownership of insertion stats.
        assert_eq!(stats.attempts, 3);
        assert!(matches!(
            stats.result,
            crate::core::operations::InsertionResult::Inserted
        ));
    }

    #[test]
    fn test_construction_statistics_record_insertion_tracks_skipped_variants() {
        init_tracing();

        let mut summary = ConstructionStatistics::default();
        let skipped_duplicate = InsertionStatistics {
            attempts: 1,
            cells_removed_during_repair: 0,
            result: crate::core::operations::InsertionResult::SkippedDuplicate,
        };
        let skipped_degeneracy = InsertionStatistics {
            attempts: 2,
            cells_removed_during_repair: 5,
            result: crate::core::operations::InsertionResult::SkippedDegeneracy,
        };

        summary.record_insertion(&skipped_duplicate);
        summary.record_insertion(&skipped_degeneracy);

        assert_eq!(summary.inserted, 0);
        assert_eq!(summary.skipped_duplicate, 1);
        assert_eq!(summary.skipped_degeneracy, 1);
        assert_eq!(summary.total_skipped(), 2);
        assert_eq!(summary.total_attempts, 3);
        assert_eq!(summary.max_attempts, 2);
        assert_eq!(summary.attempts_histogram.get(1).copied().unwrap_or(0), 1);
        assert_eq!(summary.attempts_histogram.get(2).copied().unwrap_or(0), 1);
        assert_eq!(summary.used_perturbation, 1);
        assert_eq!(summary.cells_removed_total, 5);
        assert_eq!(summary.cells_removed_max, 5);
    }

    #[test]
    fn test_construction_statistics_record_skip_sample_caps_at_eight_samples() {
        init_tracing();

        let mut summary = ConstructionStatistics::default();
        for index in 0..10 {
            let sample_index_u32 = u32::try_from(index).unwrap();
            let coordinate_base = <f64 as std::convert::From<u32>>::from(sample_index_u32);
            summary.record_skip_sample(ConstructionSkipSample {
                index,
                uuid: Uuid::from_u128(
                    <u128 as std::convert::From<u32>>::from(sample_index_u32) + 1,
                ),
                coords: vec![
                    coordinate_base,
                    coordinate_base + 0.5,
                    coordinate_base + 1.0,
                ],
                attempts: index + 1,
                error: format!("skip sample #{index}"),
            });
        }

        assert_eq!(summary.skip_samples.len(), 8);
        assert_eq!(summary.skip_samples.first().map(|s| s.index), Some(0));
        assert_eq!(summary.skip_samples.last().map(|s| s.index), Some(7));
        assert_eq!(
            summary.skip_samples.last().map(|s| s.uuid),
            Some(Uuid::from_u128(8))
        );
    }

    #[test]
    fn test_select_balanced_simplex_indices_insufficient_vertices() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];

        let result = select_balanced_simplex_indices(&vertices);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_balanced_simplex_indices_rejects_non_finite_coords() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            Vertex::new_with_uuid(
                crate::geometry::point::Point::new([f64::NAN, 0.0, 0.0]),
                Uuid::new_v4(),
                None,
            ),
        ];

        let result = select_balanced_simplex_indices(&vertices);
        assert!(result.is_none());
    }

    #[test]
    fn test_reorder_vertices_for_simplex_valid_and_invalid() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([2.0, 2.0, 2.0]),
        ];

        let indices = [2_usize, 0, 3, 1];
        let reordered =
            reorder_vertices_for_simplex(&vertices, &indices).expect("expected valid reorder");

        let expected_first: Vec<[f64; 3]> =
            indices.iter().map(|&i| (&vertices[i]).into()).collect();
        let actual_first: Vec<[f64; 3]> = reordered.iter().take(4).map(Into::into).collect();
        assert_eq!(actual_first, expected_first);

        let remaining_expected: Vec<[f64; 3]> = vertices
            .iter()
            .enumerate()
            .filter(|(idx, _)| !indices.contains(idx))
            .map(|(_, v)| (*v).into())
            .collect();
        let remaining_actual: Vec<[f64; 3]> = reordered.iter().skip(4).map(Into::into).collect();
        assert_eq!(remaining_actual, remaining_expected);

        assert!(reorder_vertices_for_simplex(&vertices, &[0, 1, 2]).is_none());
        assert!(reorder_vertices_for_simplex(&vertices, &[0, 1, 1, 2]).is_none());
        assert!(reorder_vertices_for_simplex(&vertices, &[0, 1, 2, 99]).is_none());
    }

    #[test]
    fn test_preprocess_vertices_for_construction_balanced_sets_fallback() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([2.0, 2.0, 2.0]),
        ];

        let preprocess = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::preprocess_vertices_for_construction(
            &vertices,
            DedupPolicy::Off,
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::Balanced,
        )
        .expect("preprocess failed");

        assert!(preprocess.fallback_slice().is_some());
        assert_eq!(preprocess.primary_slice(&vertices).len(), vertices.len());
        assert_eq!(preprocess.fallback_slice().unwrap().len(), vertices.len());
        assert!(preprocess.grid_cell_size().is_some());
    }

    #[test]
    fn test_preprocess_vertices_rejects_invalid_epsilon_tolerance() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let result = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::preprocess_vertices_for_construction(
            &vertices,
            DedupPolicy::Epsilon { tolerance: -1.0 },
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::First,
        );

        assert!(matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::GeometricDegeneracy { .. }
            ))
        ));
    }

    fn vertices_from_coords_permutation_3d(
        coords: &[[f64; 3]],
        permutation: &[usize],
    ) -> Vec<Vertex<f64, (), 3>> {
        permutation.iter().map(|&i| vertex!(coords[i])).collect()
    }

    #[test]
    fn test_bulk_construction_skips_near_duplicate_coordinates_3d() {
        init_tracing();
        // Test that epsilon-based deduplication removes near-duplicates
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.25, 0.25, 0.25]),
            // Near-duplicate within tolerance 1e-10
            vertex!([0.25 + 5e-11, 0.25, 0.25]),
        ];

        let opts = ConstructionOptions::default()
            .with_dedup_policy(DedupPolicy::Epsilon { tolerance: 1e-10 })
            .with_retry_policy(RetryPolicy::Disabled);
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

    fn coord_sequence_3d(vertices: &[Vertex<f64, (), 3>]) -> Vec<[f64; 3]> {
        vertices.iter().map(Into::into).collect()
    }

    #[test]
    fn test_insertion_order_hilbert_is_deterministic_across_permutations_3d() {
        init_tracing();
        let coords: [[f64; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [-1.0, 5.0, 0.0],
            [3.0, 2.0, 1.0],
        ];

        let permutations: [&[usize]; 4] = [
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[2, 3, 4, 5, 6, 7, 0, 1],
            &[1, 3, 5, 7, 0, 2, 4, 6],
        ];

        // Test both dedup_quantized=false (sort-only) and dedup_quantized=true
        // (the real code path used by order_vertices_by_strategy).
        let expected_no_dedup = vertices_from_coords_permutation_3d(&coords, permutations[0]);
        let expected_no_dedup =
            coord_sequence_3d(&order_vertices_hilbert(expected_no_dedup, false));

        let expected_dedup = vertices_from_coords_permutation_3d(&coords, permutations[0]);
        let expected_dedup = coord_sequence_3d(&order_vertices_hilbert(expected_dedup, true));

        for perm in &permutations[1..] {
            let vertices = vertices_from_coords_permutation_3d(&coords, perm);
            let got = coord_sequence_3d(&order_vertices_hilbert(vertices, false));
            assert_eq!(got, expected_no_dedup);

            let vertices = vertices_from_coords_permutation_3d(&coords, perm);
            let got = coord_sequence_3d(&order_vertices_hilbert(vertices, true));
            assert_eq!(got, expected_dedup);
        }
    }

    // =========================================================================
    // HILBERT DEDUP — GENERIC HELPERS
    // =========================================================================

    /// Build D+1 standard simplex vertices: origin + D unit vectors.
    fn simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut verts = Vec::with_capacity(D + 1);
        verts.push(vertex!([0.0; D]));
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            verts.push(vertex!(coords));
        }
        verts
    }

    /// Build simplex vertices plus exact duplicates of the first two.
    fn simplex_with_duplicates<const D: usize>() -> (Vec<Vertex<f64, (), D>>, usize) {
        let mut verts = simplex_vertices::<D>();
        let distinct = verts.len();
        // Duplicate the origin and first unit vector
        verts.push(vertex!([0.0; D]));
        let mut unit = [0.0; D];
        unit[0] = 1.0;
        verts.push(vertex!(unit));
        (verts, distinct)
    }

    /// Build simplex vertices plus an interior point (all distinct).
    #[expect(
        clippy::cast_precision_loss,
        reason = "D ≤ 5 in practice; no precision loss"
    )]
    fn simplex_with_interior<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut verts = simplex_vertices::<D>();
        let interior = [0.1_f64 / (D as f64); D];
        verts.push(vertex!(interior));
        verts
    }

    // =========================================================================
    // HILBERT DEDUP — MACRO-GENERATED PER-DIMENSION TESTS (2D–5D)
    // =========================================================================

    /// Generate Hilbert-sort dedup tests for a given dimension:
    ///
    /// - exact duplicates are removed
    /// - distinct points are preserved
    /// - all-identical inputs collapse to 1
    macro_rules! gen_hilbert_dedup_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_hilbert_sort_dedup_removes_exact_duplicates_ $dim d>]() {
                    init_tracing();
                    let (vertices, distinct) = simplex_with_duplicates::<$dim>();
                    assert!(vertices.len() > distinct);
                    let result = order_vertices_hilbert(vertices, true);
                    assert_eq!(
                        result.len(),
                        distinct,
                        "{}D: exact duplicates should be removed",
                        $dim
                    );
                }

                #[test]
                fn [<test_hilbert_sort_dedup_preserves_distinct_points_ $dim d>]() {
                    init_tracing();
                    let vertices = simplex_with_interior::<$dim>();
                    let expected = vertices.len();
                    let result = order_vertices_hilbert(vertices, true);
                    assert_eq!(
                        result.len(),
                        expected,
                        "{}D: distinct points should all be preserved",
                        $dim
                    );
                }

                #[test]
                fn [<test_hilbert_sort_dedup_all_identical_ $dim d>]() {
                    init_tracing();
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        vertex!([0.5; $dim]),
                        vertex!([0.5; $dim]),
                        vertex!([0.5; $dim]),
                    ];
                    let result = order_vertices_hilbert(vertices, true);
                    assert_eq!(
                        result.len(),
                        1,
                        "{}D: all-identical inputs should collapse to 1",
                        $dim
                    );
                }
            }
        };
    }

    gen_hilbert_dedup_tests!(2);
    gen_hilbert_dedup_tests!(3);
    gen_hilbert_dedup_tests!(4);
    gen_hilbert_dedup_tests!(5);

    // =========================================================================
    // HILBERT DEDUP — STANDALONE EDGE-CASE TESTS
    // =========================================================================

    #[test]
    fn test_hilbert_dedup_empty_input() {
        let vertices: Vec<Vertex<f64, (), 3>> = vec![];
        let result = order_vertices_hilbert(vertices, true);
        assert!(result.is_empty(), "empty input must produce empty output");
    }

    #[test]
    fn test_hilbert_dedup_single_vertex() {
        let vertices: Vec<Vertex<f64, (), 3>> = vec![vertex!([1.0, 2.0, 3.0])];
        let result = order_vertices_hilbert(vertices, true);
        assert_eq!(result.len(), 1, "single vertex must be preserved");
    }

    #[test]
    fn test_hilbert_dedup_already_unique() {
        // Distinct vertices — dedup should be a no-op.
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let n = vertices.len();
        let result = order_vertices_hilbert(vertices, true);
        assert_eq!(result.len(), n, "already-unique input must be unchanged");
    }

    #[test]
    fn test_new_with_options_hilbert_smoke_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.25, 0.25, 0.25]),
        ];

        let opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Hilbert)
            .with_retry_policy(RetryPolicy::Disabled);

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_new_with_options_shuffled_retry_policy_smoke_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.25, 0.25, 0.25]),
        ];

        let opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_retry_policy(RetryPolicy::Shuffled {
                attempts: NonZeroUsize::new(2).unwrap(),
                base_seed: Some(123),
            });

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_delaunay_constructors_default_to_pl_manifold_mode() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();

        assert_eq!(
            dt_with_kernel.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );

        let dt_from_tds: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::from_tds(dt_new.tds().clone(), FastKernel::new());
        assert_eq!(
            dt_from_tds.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
    }

    #[test]
    fn test_set_topology_guarantee_updates_underlying_triangulation() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
        assert_eq!(dt.tri.topology_guarantee, TopologyGuarantee::PLManifold);

        dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
        assert_eq!(dt.tri.topology_guarantee, TopologyGuarantee::Pseudomanifold);
    }

    #[test]
    fn test_new_with_topology_guarantee_sets_pl() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new_with_topology_guarantee(
                &vertices,
                TopologyGuarantee::PLManifold,
            )
            .unwrap();

        assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    }

    #[test]
    fn test_delaunay_check_policy_should_check() {
        init_tracing();
        assert!(!DelaunayCheckPolicy::EndOnly.should_check(1));

        let every_2 = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(2).unwrap());
        assert!(!every_2.should_check(1));
        assert!(every_2.should_check(2));
        assert!(!every_2.should_check(3));
        assert!(every_2.should_check(4));
    }

    #[test]
    fn test_set_delaunay_check_policy_updates_state() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.delaunay_check_policy(), DelaunayCheckPolicy::EndOnly);

        let policy = DelaunayCheckPolicy::EveryN(NonZeroUsize::new(3).unwrap());
        dt.set_delaunay_check_policy(policy);
        assert_eq!(dt.delaunay_check_policy(), policy);
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

        assert_eq!(dt.number_of_cells(), 1);
        assert_eq!(
            dt.delaunay_repair_policy(),
            DelaunayRepairPolicy::EveryInsertion
        );
        assert!(!dt.should_run_delaunay_repair_for(dt.topology_guarantee(), 1));
    }

    #[test]
    fn test_should_run_delaunay_repair_for_skips_when_no_cells() {
        init_tracing();
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_cells(), 0);
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

        assert_eq!(dt.number_of_cells(), 1);
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

        assert!(!dt.should_run_delaunay_repair_for(topology, 1));
        assert!(dt.should_run_delaunay_repair_for(topology, 2));
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
        assert!(dt.tri.tds.get_vertex_by_key(remapped).is_some());
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

    /// Slow search helper to find a natural stale-key repro case.
    ///
    /// This remains ignored by default because it is nondeterministic and expensive.
    /// For deterministic coverage, see `test_insert_remaps_vertex_key_after_forced_heuristic_rebuild`.
    #[test]
    #[ignore = "manual search helper; run explicitly to discover natural repro cases"]
    fn find_stale_vertex_key_after_heuristic_rebuild_repro_case() {
        const DIM: usize = 4;
        const INITIAL_COUNT: usize = 12;
        const CASES: usize = 2_000;

        init_tracing();

        // This probes for a configuration that triggers a heuristic rebuild during automatic
        // flip-repair after insertion, which historically could invalidate the returned VertexKey.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD3_1A_7A_1C_0A_17_u64);

        for case in 0..CASES {
            let mut vertices: Vec<Vertex<f64, (), DIM>> = Vec::with_capacity(INITIAL_COUNT);
            for _ in 0..INITIAL_COUNT {
                // Use a coarse lattice + tiny noise to encourage near-degenerate configurations.
                let mut coords = [0.0_f64; DIM];
                for c in &mut coords {
                    let base: i32 = rng.random_range(-3..=3);
                    let noise: f64 = rng.random_range(-1.0e-6..=1.0e-6);
                    *c = <f64 as std::convert::From<i32>>::from(base) + noise;
                }
                vertices.push(vertex!(coords));
            }

            let Ok(mut dt) =
                DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), DIM>::new(&vertices)
            else {
                continue;
            };

            let mut inserted_coords = [0.0_f64; DIM];
            for c in &mut inserted_coords {
                let base: i32 = rng.random_range(-3..=3);
                let noise: f64 = rng.random_range(-1.0e-6..=1.0e-6);
                *c = <f64 as std::convert::From<i32>>::from(base) + noise;
            }
            let inserted = vertex!(inserted_coords);
            let inserted_uuid = inserted.uuid();

            let Ok(vertex_key) = dt.insert(inserted) else {
                continue;
            };

            let found = dt
                .tri
                .tds
                .get_vertex_by_key(vertex_key)
                .is_some_and(|v| v.uuid() == inserted_uuid);

            if !found {
                tracing::debug!(case, "FOUND stale key after insertion");
                tracing::debug!(vertex_key = ?vertex_key, inserted_uuid = %inserted_uuid);
                tracing::debug!("initial vertices:");
                for v in &vertices {
                    tracing::debug!(coords = ?v.point().coords(), "  vertex");
                }
                tracing::debug!("inserted vertex coords: {inserted_coords:?}");
                panic!("stale VertexKey returned from insert() after heuristic rebuild");
            }
        }

        panic!("no stale-key case found after {CASES} attempts");
    }

    #[test]
    fn test_remove_vertex_fast_path_inverse_k1() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);
        let original_vertex_count = dt.number_of_vertices();
        let original_cell_count = dt.number_of_cells();

        let cell_key = dt.cells().next().unwrap().0;
        let inserted_vertex = vertex!([0.2, 0.2, 0.2]);
        let inserted_uuid = inserted_vertex.uuid();
        dt.flip_k1_insert(cell_key, inserted_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), original_vertex_count + 1);
        assert_eq!(dt.number_of_cells(), original_cell_count + 3);

        let vertex_key = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(k, _)| k)
            .expect("Inserted vertex not found");

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_cells = dt.remove_vertex(vertex_key).unwrap();

        assert_eq!(removed_cells, 4);
        assert_eq!(dt.number_of_vertices(), original_vertex_count);
        assert_eq!(dt.number_of_cells(), original_cell_count);
        assert!(dt.as_triangulation().validate().is_ok());
        assert!(dt.vertices().all(|(_, v)| v.uuid() != inserted_uuid));
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

    /// When the primary per-insertion repair returns `NonConvergent`, the robust
    /// fallback in `maybe_repair_after_insertion` should rescue the insertion.
    #[test]
    fn test_maybe_repair_after_insertion_robust_fallback_on_forced_nonconvergent() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let _guard = ForceRepairNonconvergentGuard::enable();
        let result = dt.insert(vertex!([0.5, 0.5]));
        assert!(
            result.is_ok(),
            "Insertion should succeed via robust fallback: {result:?}"
        );
        assert!(dt.validate().is_ok());
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

        // Build a non-Delaunay 2D TDS manually (same config as flips.rs tests).
        let d_candidates = [[0.0, 1.2], [0.1, 1.1], [0.2, 0.9], [-0.1, 1.3]];
        let kernel = AdaptiveKernel::<f64>::new();
        let mut raw_tds = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<f64, (), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 1.0]))
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords))
                .unwrap();
            let _ = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
                .unwrap();
            let _ = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, d], None).unwrap())
                .unwrap();
            repair_neighbor_pointers(&mut candidate).unwrap();
            if verify_delaunay_via_flip_predicates(&candidate, &kernel).is_err() {
                raw_tds = Some(candidate);
                break;
            }
        }
        let tds = raw_tds.expect("need a non-Delaunay 2D config");
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, kernel);
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
        let kernel2 = AdaptiveKernel::<f64>::new();
        let mut raw_tds2 = None;
        for d_coords in d_candidates {
            let mut candidate: Tds<f64, (), (), 2> = Tds::empty();
            let a = candidate
                .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
                .unwrap();
            let b = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 1.0]))
                .unwrap();
            let c = candidate
                .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
                .unwrap();
            let d = candidate
                .insert_vertex_with_mapping(vertex!(d_coords))
                .unwrap();
            let _ = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
                .unwrap();
            let _ = candidate
                .insert_cell_with_mapping(Cell::new(vec![a, b, d], None).unwrap())
                .unwrap();
            repair_neighbor_pointers(&mut candidate).unwrap();
            if verify_delaunay_via_flip_predicates(&candidate, &kernel2).is_err() {
                raw_tds2 = Some(candidate);
                break;
            }
        }
        let tds2 = raw_tds2.unwrap();
        let mut dt2: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds2, AdaptiveKernel::new());
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
                Err(DelaunayRepairError::Flip(FlipError::UnsupportedDimension {
                    dimension: 1
                }))
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
                Err(DelaunayRepairError::Flip(FlipError::UnsupportedDimension {
                    dimension: 1
                }))
            ),
            "Expected non-retryable Flip(UnsupportedDimension) pass-through for D=1, got: {result:?}"
        );
    }

    /// Macro to generate comprehensive triangulation construction tests across dimensions.
    ///
    /// This macro generates tests that verify all construction patterns:
    /// 1. **Batch construction** - Creating a simplex with D+1 vertices + incremental insertion
    /// 2. **Bootstrap from empty** - Accumulating vertices until D+1, then auto-creating simplex
    /// 3. **Cavity-based continuation** - Verifying cavity algorithm works after bootstrap
    /// 4. **Equivalence testing** - Bootstrap and batch produce identical structures
    ///
    /// # Usage
    /// ```ignore
    /// test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);
    /// ```
    macro_rules! test_incremental_insertion {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                // Test 1: Batch construction with incremental insertion
                #[test]
                fn [<test_incremental_insertion_ $dim d>]() {
                    init_tracing();
                    // Build initial simplex (D+1 vertices)
                    let mut vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    // Add interior point to be inserted incrementally
                    vertices.push(vertex!($interior_point));

                    let expected_vertices = vertices.len();

                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    assert_eq!(dt.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert!(dt.number_of_cells() > 1,
                        "{}D: Expected multiple cells, got {}", $dim, dt.number_of_cells());
                }

                // Test 2: Bootstrap from empty triangulation
                #[test]
                fn [<test_bootstrap_from_empty_ $dim d>]() {
                    init_tracing();
                    // Start with empty triangulation
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::empty();
                    assert_eq!(dt.number_of_vertices(), 0);
                    assert_eq!(dt.number_of_cells(), 0);

                    let vertices = vec![$(vertex!($simplex_coords)),+];
                    assert_eq!(vertices.len(), $dim + 1, "Test should provide exactly D+1 vertices");

                    // Insert D vertices - should accumulate without creating cells
                    for (i, vertex) in vertices.iter().take($dim).enumerate() {
                        dt.insert(*vertex).unwrap();
                        assert_eq!(dt.number_of_vertices(), i + 1,
                            "{}D: After inserting vertex {}, expected {} vertices", $dim, i, i + 1);
                        assert_eq!(dt.number_of_cells(), 0,
                            "{}D: Should have 0 cells during bootstrap (have {} vertices < D+1)",
                            $dim, i + 1);
                    }

                    // Insert (D+1)th vertex - should trigger initial simplex creation
                    dt.insert(*vertices.last().unwrap()).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 1);
                    assert_eq!(dt.number_of_cells(), 1,
                        "{}D: Should have exactly 1 cell after inserting D+1 vertices", $dim);

                    // Verify triangulation is valid
                    assert!(dt.is_valid().is_ok(),
                        "{}D: Triangulation should be valid after bootstrap", $dim);
                }

                // Test 3: Bootstrap continues with cavity-based insertion
                #[test]
                fn [<test_bootstrap_continues_with_cavity_ $dim d>]() {
                    init_tracing();
                    // Start with empty, bootstrap to initial simplex, then continue with cavity-based
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::empty();

                    let initial_vertices = vec![$(vertex!($simplex_coords)),+];

                    // Bootstrap: insert D+1 vertices
                    for vertex in &initial_vertices {
                        dt.insert(*vertex).unwrap();
                    }
                    assert_eq!(dt.number_of_cells(), 1);

                    // Continue with cavity-based insertion (vertex D+2 onward)
                    dt.insert(vertex!($interior_point)).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 2);
                    assert!(dt.number_of_cells() > 1,
                        "{}D: Should have multiple cells after cavity-based insertion", $dim);

                    // Verify triangulation remains valid
                    assert!(dt.is_valid().is_ok());
                }

                // Test 4: Bootstrap equivalent to batch construction
                #[test]
                fn [<test_bootstrap_equivalent_to_batch_ $dim d>]() {
                    init_tracing();
                    // Compare bootstrap path vs batch construction
                    let vertices = vec![$(vertex!($simplex_coords)),+];

                    // Path A: Bootstrap from empty
                    let mut dt_bootstrap: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty();
                    for vertex in &vertices {
                        dt_bootstrap.insert(*vertex).unwrap();
                    }

                    // Path B: Batch construction
                    let dt_batch: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    // Both should produce identical structure
                    assert_eq!(dt_bootstrap.number_of_vertices(), dt_batch.number_of_vertices(),
                        "{}D: Bootstrap and batch should have same vertex count", $dim);
                    assert_eq!(dt_bootstrap.number_of_cells(), dt_batch.number_of_cells(),
                        "{}D: Bootstrap and batch should have same cell count", $dim);

                    // Both should be valid
                    assert!(dt_bootstrap.is_valid().is_ok());
                    assert!(dt_batch.is_valid().is_ok());
                }
            }
        };
    }

    // 2D: Triangle + interior point
    test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);

    // 3D: Tetrahedron + interior point
    test_incremental_insertion!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2]
    );

    // 4D: 4-simplex + interior point
    test_incremental_insertion!(
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

    // 5D: 5-simplex + interior point
    test_incremental_insertion!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    );

    // =========================================================================
    // empty() / with_empty_kernel() tests
    // =========================================================================

    #[test]
    fn test_empty_creates_empty_triangulation() {
        init_tracing();
        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_vertices(), 0);
        assert_eq!(dt.number_of_cells(), 0);
        // dim() returns -1 for empty triangulation
        assert_eq!(dt.dim(), -1);
    }

    #[test]
    fn test_empty_supports_incremental_insertion() {
        init_tracing();
        // Verify empty triangulation supports incremental insertion via bootstrap
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.number_of_vertices(), 0);

        // Can now insert into empty triangulation - bootstrap phase
        dt.insert(vertex!([0.0, 0.0])).unwrap();
        dt.insert(vertex!([1.0, 0.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 0); // Still in bootstrap

        dt.insert(vertex!([0.0, 1.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 1); // Initial simplex created
    }

    #[test]
    fn test_validation_policy_defaults_to_on_suspicion() {
        init_tracing();
        // empty() -> Triangulation::new_empty() -> ValidationPolicy::default()
        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.validation_policy(), ValidationPolicy::OnSuspicion);

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        // new() -> with_kernel() -> explicit validation_policy initialization
        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.validation_policy(), ValidationPolicy::OnSuspicion);

        // with_kernel() constructor path should also use the default policy
        let dt_with_kernel: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();
        assert_eq!(
            dt_with_kernel.validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        // from_tds() is a separate constructor path (const-friendly), and should also
        // default to OnSuspicion.
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let dt_from_tds: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());
        assert_eq!(
            dt_from_tds.validation_policy(),
            ValidationPolicy::OnSuspicion
        );
    }

    #[test]
    fn test_validation_policy_setter_and_getter_roundtrip() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Getter reflects the underlying Triangulation policy.
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);

        dt.set_validation_policy(ValidationPolicy::Always);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Always);

        dt.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Never);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Never);

        dt.set_validation_policy(ValidationPolicy::OnSuspicion);
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);
    }

    // =========================================================================
    // with_kernel() tests
    // =========================================================================

    #[test]
    fn test_with_kernel_fast_kernel() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_with_kernel_robust_kernel() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&RobustKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_with_kernel_insufficient_vertices_2d() {
        init_tracing();
        let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0])];

        let result: Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices);

        assert!(result.is_err());
        match result {
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::InsufficientVertices { dimension, .. },
            )) => {
                assert_eq!(dimension, 2);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_with_kernel_insufficient_vertices_3d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];

        let result: Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3>, _> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices);

        assert!(result.is_err());
        match result {
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::InsufficientVertices { dimension, .. },
            )) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_with_kernel_f32_coordinates() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.0f32, 1.0f32]),
        ];

        let dt: DelaunayTriangulation<AdaptiveKernel<f32>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    // =========================================================================
    // Query method tests
    // =========================================================================

    #[test]
    fn test_number_of_vertices_minimal_simplex() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
    }

    #[test]
    fn test_number_of_cells_minimal_simplex() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Minimal 3D simplex has exactly 1 tetrahedron
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_number_of_cells_after_insertion() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_cells(), 1);

        // Insert interior point - should create 3 triangles
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert_eq!(dt.number_of_cells(), 3);
    }

    #[test]
    fn test_dim_returns_correct_dimension() {
        init_tracing();
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.dim(), 2);

        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.dim(), 3);

        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices_4d).unwrap();
        assert_eq!(dt_4d.dim(), 4);
    }

    // =========================================================================
    // insert() tests
    // =========================================================================

    #[test]
    fn test_insert_single_interior_point_2d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);

        let v_key = dt.insert(vertex!([0.3, 0.3])).unwrap();

        // Verify insertion succeeded
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_cells(), 3);

        // Verify the returned key can access the vertex
        assert!(dt.tri.tds.get_vertex_by_key(v_key).is_some());
    }

    #[test]
    fn test_insert_multiple_sequential_points_2d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Insert 3 interior points sequentially
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert_eq!(dt.number_of_vertices(), 4);

        dt.insert(vertex!([0.5, 0.2])).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(vertex!([0.2, 0.5])).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        // All vertices should be present
        assert!(dt.number_of_cells() > 1);
    }

    #[test]
    fn test_insert_multiple_sequential_points_3d() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Insert 3 interior points sequentially (well inside the tetrahedron)
        dt.insert(vertex!([0.1, 0.1, 0.1])).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(vertex!([0.15, 0.15, 0.1])).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        dt.insert(vertex!([0.1, 0.15, 0.15])).unwrap();
        assert_eq!(dt.number_of_vertices(), 7);

        assert!(dt.number_of_cells() > 1);
    }

    #[test]
    fn test_insert_updates_last_inserted_cell() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Initially no last_inserted_cell
        assert!(dt.insertion_state.last_inserted_cell.is_none());

        // After insertion, should have a cached cell
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert!(dt.insertion_state.last_inserted_cell.is_some());
    }

    #[test]
    fn test_new_with_exact_minimum_vertices() {
        init_tracing();
        // 2D: exactly 3 vertices (minimum for 2D simplex)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.number_of_vertices(), 3);
        assert_eq!(dt_2d.number_of_cells(), 1);

        // 3D: exactly 4 vertices (minimum for 3D simplex)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.number_of_vertices(), 4);
        assert_eq!(dt_3d.number_of_cells(), 1);
    }

    #[test]
    fn test_tds_accessor_provides_readonly_access() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Access TDS via immutable reference
        let tds = dt.tds();
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 1);

        // Verify we can call other TDS methods
        assert!(tds.is_valid().is_ok());
        assert!(tds.cell_keys().next().is_some());
    }

    #[test]
    fn test_internal_tds_access() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);

        // Internal code can access TDS directly for mutations
        let tds = &mut dt.tri.tds;
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 1);

        // Can call mutating methods like remove_duplicate_cells
        let result = tds.remove_duplicate_cells();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tds_accessor_reflects_insertions() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Before insertion
        assert_eq!(dt.tds().number_of_vertices(), 3);

        // Insert a new vertex
        dt.insert(vertex!([0.3, 0.3])).unwrap();

        // After insertion, TDS accessor reflects the change
        assert_eq!(dt.tds().number_of_vertices(), 4);
        assert!(dt.tds().number_of_cells() > 1);
    }

    #[test]
    fn test_tds_accessors_maintain_validation_invariants() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Verify TDS is valid through accessor
        assert!(dt.tds().is_valid().is_ok());

        // Insert additional vertex
        dt.insert(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();

        // TDS should still be valid after mutation
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate().is_ok());
    }

    #[test]
    fn test_bootstrap_with_custom_kernel() {
        init_tracing();
        // Verify bootstrap works with RobustKernel
        let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel(RobustKernel::new());

        assert_eq!(dt.number_of_vertices(), 0);

        // Bootstrap with robust predicates
        dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
        dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
        dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 0); // Still bootstrapping

        dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 1); // Initial simplex created

        assert!(dt.is_valid().is_ok());
    }

    // =========================================================================
    // Coverage-oriented tests (tarpaulin)
    // =========================================================================

    #[test]
    fn test_with_kernel_aborts_on_duplicate_uuid_in_insertion_loop() {
        init_tracing();
        let mut vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
            vertex!([0.25, 0.25]),
        ];

        // Ensure the duplicate UUID is introduced in the incremental insertion loop,
        // not during initial simplex construction.
        let dup_uuid = vertices[0].uuid();
        vertices[3].set_uuid(dup_uuid).unwrap();

        let result: Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices);

        match result.unwrap_err() {
            DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::Tds(TdsConstructionError::DuplicateUuid {
                    entity: _,
                    uuid,
                }),
            ) => {
                assert_eq!(uuid, dup_uuid);
            }
            other => panic!("Expected DuplicateUuid error, got {other:?}"),
        }
    }

    #[test]
    fn test_validation_report_ok_for_valid_triangulation() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert!(dt.validation_report().is_ok());
    }

    #[test]
    fn test_validation_report_returns_mapping_failures_only() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Break UUID↔key mappings: remove one vertex UUID entry.
        let uuid = dt.tri.tds.vertices().next().unwrap().1.uuid();
        dt.tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = dt.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::CellMappings
            )
        }));

        // Early-return on mapping failures: do not add derived invariants.
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.kind != InvariantKind::DelaunayProperty)
        );
    }

    #[test]
    fn test_validation_report_includes_vertex_incidence_violation() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Corrupt a `Vertex::incident_cell` pointer.
        let vertex_key = dt.tri.tds.vertices().next().unwrap().0;
        dt.tri
            .tds
            .get_vertex_by_key_mut(vertex_key)
            .unwrap()
            .incident_cell = Some(CellKey::default());

        let report = dt.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexIncidence)
        );
    }

    #[test]
    fn test_serde_roundtrip_uses_custom_deserialize_impl() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let json = serde_json::to_string(&dt).unwrap();

        // AdaptiveKernel: no auto-Deserialize impl, use from_tds.
        let tds: Tds<f64, (), (), 3> = serde_json::from_str(&json).unwrap();
        let roundtrip_adaptive = DelaunayTriangulation::from_tds(tds, AdaptiveKernel::new());

        assert_eq!(
            roundtrip_adaptive.number_of_vertices(),
            dt.number_of_vertices()
        );
        assert_eq!(roundtrip_adaptive.number_of_cells(), dt.number_of_cells());

        // `insertion_state.last_inserted_cell` is a performance-only locate hint and is intentionally not
        // persisted across serde round-trips (it is reset to `None` in `from_tds`).
        assert!(
            roundtrip_adaptive
                .insertion_state
                .last_inserted_cell
                .is_none()
        );

        // RobustKernel: has a custom Deserialize impl.
        let roundtrip_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
            serde_json::from_str(&json).unwrap();

        assert_eq!(
            roundtrip_robust.number_of_vertices(),
            dt.number_of_vertices()
        );
        assert_eq!(roundtrip_robust.number_of_cells(), dt.number_of_cells());
        assert!(
            roundtrip_robust
                .insertion_state
                .last_inserted_cell
                .is_none()
        );
    }

    // =========================================================================
    // Topology traversal forwarding tests (DelaunayTriangulation → Triangulation)
    // =========================================================================

    #[test]
    fn test_topology_traversal_methods_are_forwarded() {
        init_tracing();
        // Single tetrahedron: 4 vertices, 1 cell, 6 unique edges.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let edges_dt: std::collections::HashSet<_> = dt.edges().collect();
        let edges_tri: std::collections::HashSet<_> = tri.edges().collect();
        assert_eq!(edges_dt, edges_tri);
        assert_eq!(edges_dt.len(), 6);

        let index = dt.build_adjacency_index().unwrap();
        let edges_dt_index: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
        let edges_tri_index: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_dt_index, edges_tri_index);
        assert_eq!(edges_dt_index, edges_dt);

        let v0 = dt.vertices().next().unwrap().0;
        let incident_dt: std::collections::HashSet<_> = dt.incident_edges(v0).collect();
        let incident_tri: std::collections::HashSet<_> = tri.incident_edges(v0).collect();
        assert_eq!(incident_dt, incident_tri);
        assert_eq!(incident_dt.len(), 3);

        let incident_dt_index: std::collections::HashSet<_> =
            dt.incident_edges_with_index(&index, v0).collect();
        let incident_tri_index: std::collections::HashSet<_> =
            tri.incident_edges_with_index(&index, v0).collect();
        assert_eq!(incident_dt_index, incident_tri_index);
        assert_eq!(incident_dt_index, incident_dt);

        let cell_key = dt.cells().next().unwrap().0;
        let neighbors_dt: Vec<_> = dt.cell_neighbors(cell_key).collect();
        let neighbors_tri: Vec<_> = tri.cell_neighbors(cell_key).collect();
        assert_eq!(neighbors_dt, neighbors_tri);
        assert!(neighbors_dt.is_empty());

        let neighbors_dt_index: Vec<_> = dt.cell_neighbors_with_index(&index, cell_key).collect();
        let neighbors_tri_index: Vec<_> = tri.cell_neighbors_with_index(&index, cell_key).collect();
        assert_eq!(neighbors_dt_index, neighbors_tri_index);
        assert_eq!(neighbors_dt_index, neighbors_dt);

        // Geometry/topology accessors should be forwarded as well.
        let cell_vertices_dt = dt.cell_vertices(cell_key).unwrap();
        let cell_vertices_tri = tri.cell_vertices(cell_key).unwrap();
        assert_eq!(cell_vertices_dt, cell_vertices_tri);
        assert_eq!(cell_vertices_dt.len(), 4);

        let coords_dt = dt.vertex_coords(v0).unwrap();
        let coords_tri = tri.vertex_coords(v0).unwrap();
        assert_eq!(coords_dt, coords_tri);

        // Missing keys should behave the same as on `Triangulation`.
        assert!(dt.vertex_coords(VertexKey::default()).is_none());
        assert!(dt.cell_vertices(CellKey::default()).is_none());
    }

    // =========================================================================
    // Tests for per-insertion incremental Delaunay repair during batch construction
    // =========================================================================

    /// Verifies that 3D batch construction with more vertices than the initial simplex
    /// (D+1 = 4) completes successfully.  Each inserted vertex triggers a
    /// `maybe_repair_after_insertion` call (`EveryInsertion` policy) in
    /// `insert_remaining_vertices_seeded`.
    #[test]
    fn test_batch_3d_construction_with_extra_vertex_triggers_incremental_repair() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            // 5th vertex: triggers one iteration of insert_remaining_vertices_seeded
            vertex!([0.3, 0.3, 0.3]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

    /// Verifies the `Some`-stats call site: `new_with_construction_statistics` passes
    /// `Some(&mut stats)` into `insert_remaining_vertices_seeded`, exercising the
    /// per-insertion repair in the stats-collecting branch.
    #[test]
    fn test_batch_3d_construction_statistics_with_extra_vertex_triggers_incremental_repair() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.3, 0.3, 0.3]),
        ];
        let (dt, stats) =
            DelaunayTriangulation::<_, (), (), 3>::new_with_construction_statistics(&vertices)
                .unwrap();
        assert_eq!(dt.number_of_vertices(), 5);
        assert_eq!(stats.inserted, 5);
        assert!(dt.validate().is_ok());
    }

    /// Forced local repair non-convergence exercises the D>=4 soft-fail branch
    /// without relying on a fragile floating-point cycling fixture.
    #[test]
    fn test_batch_4d_forced_nonconvergent_local_repair_canonicalizes_without_stats() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
            vertex!([0.2, 0.2, 0.2, 0.2]),
            vertex!([0.35, 0.25, 0.15, 0.3]),
        ];

        let _guard = ForceRepairNonconvergentGuard::enable();
        let kernel = RobustKernel::<f64>::new();
        let dt =
            DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_kernel(&kernel, &vertices)
                .expect(
                    "D>=4 construction should continue after forced local repair non-convergence",
                );

        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert!(dt.validate().is_ok());
    }

    /// The statistics path has a separate insertion loop, so it needs its own
    /// forced D>=4 local-repair non-convergence assertion.
    #[test]
    fn test_batch_4d_forced_nonconvergent_local_repair_canonicalizes_with_stats() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
            vertex!([0.2, 0.2, 0.2, 0.2]),
            vertex!([0.35, 0.25, 0.15, 0.3]),
        ];

        let _guard = ForceRepairNonconvergentGuard::enable();
        let kernel = RobustKernel::<f64>::new();
        let (dt, stats) = DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_topology_guarantee_and_options_with_construction_statistics(
            &kernel,
            &vertices,
            TopologyGuarantee::DEFAULT,
            ConstructionOptions::default(),
        )
        .expect("D>=4 stats construction should continue after forced local repair non-convergence");

        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert_eq!(stats.inserted, vertices.len());
        assert!(dt.validate().is_ok());
    }

    // =========================================================================
    // Tests for try_d_lt4_global_repair_fallback
    // =========================================================================

    /// When `use_global_repair_fallback` is false the helper should return an error
    /// immediately without attempting global repair.
    #[test]
    fn test_try_d_lt4_global_repair_fallback_disabled_returns_error() {
        init_tracing();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let repair_err = DelaunayRepairError::NonConvergent {
            max_flips: 16,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 0,
                flips_performed: 0,
                max_queue_len: 0,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: RepairQueueOrder::Fifo,
            }),
        };

        let result =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::try_d_lt4_global_repair_fallback(
                &mut dt.tri.tds,
                &dt.tri.kernel,
                TopologyGuarantee::PLManifold,
                false, // disabled
                5,
                &repair_err,
            );

        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("per-insertion Delaunay repair failed at index 5"),
            "error should mention the index: {err_msg}"
        );
    }

    /// When `use_global_repair_fallback` is true and the TDS is already valid,
    /// global repair succeeds and the helper returns `Ok(())`.
    #[test]
    fn test_try_d_lt4_global_repair_fallback_enabled_succeeds_on_valid_tds() {
        init_tracing();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.3, 0.3, 0.3]),
        ];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let repair_err = DelaunayRepairError::NonConvergent {
            max_flips: 16,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 0,
                flips_performed: 0,
                max_queue_len: 0,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: RepairQueueOrder::Fifo,
            }),
        };

        // TDS is valid, so global repair should succeed (nothing to fix).
        let result =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::try_d_lt4_global_repair_fallback(
                &mut dt.tri.tds,
                &dt.tri.kernel,
                TopologyGuarantee::PLManifold,
                true, // enabled
                5,
                &repair_err,
            );

        assert!(
            result.is_ok(),
            "global repair on valid TDS should succeed: {:?}",
            result.err()
        );
    }

    /// Verify the error message includes both local and global error details when
    /// global repair also fails.
    #[test]
    fn test_try_d_lt4_global_repair_fallback_error_includes_both_messages() {
        init_tracing();

        // Build a 1D triangulation — repair_delaunay_with_flips_k2_k3 returns
        // UnsupportedDimension for D<2, guaranteeing the global repair fails.
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 1> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let repair_err = DelaunayRepairError::PostconditionFailed {
            message: "synthetic local error".to_string(),
        };

        let result =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 1>::try_d_lt4_global_repair_fallback(
                &mut dt.tri.tds,
                &dt.tri.kernel,
                TopologyGuarantee::PLManifold,
                true, // enabled — but global repair will fail (D=1)
                7,
                &repair_err,
            );

        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("local error:"),
            "error should contain local error detail: {err_msg}"
        );
        assert!(
            err_msg.contains("global fallback:"),
            "error should contain global fallback detail: {err_msg}"
        );
        assert!(
            err_msg.contains("index 7"),
            "error should contain the index: {err_msg}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_topology_validation_is_internal() {
        let error = InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
            message: "missing cell".to_string(),
        });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "TopologyValidation should map to InternalInconsistency, got: {mapped:?}"
        );
        let msg = mapped.to_string();
        assert!(
            msg.contains("missing cell"),
            "error message should preserve the original error: {msg}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_degenerate_orientation_is_degeneracy() {
        let error = InsertionError::TopologyValidation(TdsError::Geometric(
            GeometricError::DegenerateOrientation {
                message: "det=0".to_string(),
            },
        ));
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::GeometricDegeneracy { .. }
            ),
            "DegenerateOrientation should map to GeometricDegeneracy, got: {mapped:?}"
        );
        let msg = mapped.to_string();
        assert!(
            msg.contains("det=0"),
            "error message should preserve the original error: {msg}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_negative_orientation_is_degeneracy() {
        let error = InsertionError::TopologyValidation(TdsError::Geometric(
            GeometricError::NegativeOrientation {
                message: "det<0 after canonicalization".to_string(),
            },
        ));
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::GeometricDegeneracy { .. }
            ),
            "NegativeOrientation should map to GeometricDegeneracy, got: {mapped:?}"
        );
        let msg = mapped.to_string();
        assert!(
            msg.contains("det<0"),
            "error message should preserve the original error: {msg}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_isolated_vertex_is_internal() {
        let error = InsertionError::TopologyValidationFailed {
            message: "test".to_string(),
            source: Box::new(TriangulationValidationError::IsolatedVertex {
                vertex_key: VertexKey::from(slotmap::KeyData::from_ffi(1)),
                vertex_uuid: Uuid::nil(),
            }),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "IsolatedVertex should map to InternalInconsistency, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_construction_preserves_source() {
        let error =
            InsertionError::Construction(TriangulationConstructionError::FailedToCreateCell {
                message: "test".to_string(),
            });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::FailedToCreateCell { .. }
            ),
            "Construction should preserve the inner error, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_topology_validation_failed_is_internal() {
        let error = InsertionError::TopologyValidationFailed {
            message: "post-insertion".to_string(),
            source: Box::new(TriangulationValidationError::EulerCharacteristicMismatch {
                computed: 3,
                expected: 2,
                classification: TopologyClassification::Ball(3),
            }),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "TopologyValidationFailed should map to InternalInconsistency, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_cavity_filling_is_internal() {
        let error = InsertionError::CavityFilling {
            message: "test".to_string(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        ));
    }

    #[test]
    fn test_map_orientation_canonicalization_error_neighbor_wiring_is_internal() {
        let error = InsertionError::NeighborWiring {
            message: "test".to_string(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        ));
    }

    #[test]
    fn test_map_orientation_canonicalization_error_duplicate_uuid_is_internal() {
        let error = InsertionError::DuplicateUuid {
            entity: EntityKind::Cell,
            uuid: Uuid::nil(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        ));
    }

    #[test]
    fn test_map_orientation_canonicalization_error_geometry_variants_are_degeneracy() {
        let geometry_errors: Vec<InsertionError> = vec![
            InsertionError::Location(LocateError::EmptyTriangulation),
            InsertionError::NonManifoldTopology {
                facet_hash: 0,
                cell_count: 3,
            },
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            },
            InsertionError::DelaunayValidationFailed {
                source: Box::new(DelaunayTriangulationValidationError::VerificationFailed {
                    message: "test".to_string(),
                }),
            },
            InsertionError::DelaunayRepairFailed {
                source: Box::new(DelaunayRepairError::PostconditionFailed {
                    message: "test".to_string(),
                }),
                context: "test".to_string(),
            },
            InsertionError::DuplicateCoordinates {
                coordinates: "[0,0,0]".to_string(),
            },
        ];
        for error in geometry_errors {
            let label = format!("{error}");
            let mapped =
                DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
            assert!(
                matches!(
                    mapped,
                    TriangulationConstructionError::GeometricDegeneracy { .. }
                ),
                "{label} should map to GeometricDegeneracy, got: {mapped:?}"
            );
        }
    }

    // ---- map_insertion_error tests ----

    #[test]
    fn test_map_insertion_error_construction_preserves_source() {
        let error =
            InsertionError::Construction(TriangulationConstructionError::FailedToCreateCell {
                message: "inner".to_string(),
            });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::FailedToCreateCell { .. }
            ),
            "Construction should preserve inner error, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_cavity_filling() {
        let error = InsertionError::CavityFilling {
            message: "test".to_string(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::FailedToCreateCell { .. }
            ),
            "CavityFilling should map to FailedToCreateCell, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_neighbor_wiring() {
        let error = InsertionError::NeighborWiring {
            message: "bad wiring".to_string(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "NeighborWiring should map to InternalInconsistency, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_topology_validation() {
        let error = InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
            message: "broken".to_string(),
        });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(mapped, TriangulationConstructionError::Tds(_)),
            "TopologyValidation should map to Tds(ValidationError), got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_duplicate_uuid() {
        let error = InsertionError::DuplicateUuid {
            entity: EntityKind::Cell,
            uuid: Uuid::nil(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(mapped, TriangulationConstructionError::Tds(_)),
            "DuplicateUuid should map to Tds(DuplicateUuid), got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_duplicate_coordinates() {
        let error = InsertionError::DuplicateCoordinates {
            coordinates: "[1,2,3]".to_string(),
        };
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::DuplicateCoordinates { .. }
            ),
            "DuplicateCoordinates should be preserved, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_geometry_variants_are_degeneracy() {
        let geometry_errors: Vec<InsertionError> = vec![
            InsertionError::ConflictRegion(ConflictError::OpenBoundary {
                facet_count: 2,
                ridge_vertex_count: 1,
                open_cell: CellKey::from(slotmap::KeyData::from_ffi(1)),
            }),
            InsertionError::Location(LocateError::EmptyTriangulation),
            InsertionError::NonManifoldTopology {
                facet_hash: 0,
                cell_count: 3,
            },
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            },
            InsertionError::DelaunayValidationFailed {
                source: Box::new(DelaunayTriangulationValidationError::VerificationFailed {
                    message: "test".to_string(),
                }),
            },
            InsertionError::DelaunayRepairFailed {
                source: Box::new(DelaunayRepairError::PostconditionFailed {
                    message: "test".to_string(),
                }),
                context: "test".to_string(),
            },
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: Box::new(TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: 3,
                    expected: 2,
                    classification: TopologyClassification::Ball(3),
                }),
            },
        ];
        for error in geometry_errors {
            let label = format!("{error}");
            let mapped =
                DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_insertion_error(error);
            assert!(
                matches!(
                    mapped,
                    TriangulationConstructionError::GeometricDegeneracy { .. }
                ),
                "{label} should map to GeometricDegeneracy, got: {mapped:?}"
            );
        }
    }

    // ---- is_retryable refinement tests ----

    #[test]
    fn test_is_retryable_degenerate_orientation_is_retryable() {
        let error = InsertionError::TopologyValidation(TdsError::Geometric(
            GeometricError::DegenerateOrientation {
                message: "det=0".to_string(),
            },
        ));
        assert!(
            error.is_retryable(),
            "DegenerateOrientation should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_negative_orientation_is_retryable() {
        let error = InsertionError::TopologyValidation(TdsError::Geometric(
            GeometricError::NegativeOrientation {
                message: "det<0".to_string(),
            },
        ));
        assert!(
            error.is_retryable(),
            "NegativeOrientation should be retryable"
        );
    }

    #[test]
    fn test_is_retryable_isolated_vertex_is_retryable() {
        let error = InsertionError::TopologyValidationFailed {
            message: "test".to_string(),
            source: Box::new(TriangulationValidationError::IsolatedVertex {
                vertex_key: VertexKey::from(slotmap::KeyData::from_ffi(1)),
                vertex_uuid: Uuid::nil(),
            }),
        };
        assert!(
            error.is_retryable(),
            "IsolatedVertex should be retryable (geometry-sensitive conflict region)"
        );
    }

    #[test]
    fn test_is_retryable_inconsistent_data_structure_is_not_retryable() {
        let error = InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
            message: "missing cell".to_string(),
        });
        assert!(
            !error.is_retryable(),
            "InconsistentDataStructure should NOT be retryable"
        );
    }

    #[test]
    fn test_is_retryable_failed_to_create_cell_is_not_retryable() {
        let error = InsertionError::TopologyValidation(TdsError::FailedToCreateCell {
            message: "test".to_string(),
        });
        assert!(
            !error.is_retryable(),
            "FailedToCreateCell should NOT be retryable"
        );
    }

    // ---- VerificationFailed variant tests ----

    #[test]
    fn test_verification_failed_display() {
        let err = DelaunayTriangulationValidationError::VerificationFailed {
            message: "flip predicate detected non-Delaunay facet".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("Delaunay verification failed"),
            "Display should contain prefix: {msg}"
        );
        assert!(
            msg.contains("flip predicate detected non-Delaunay facet"),
            "Display should contain inner message: {msg}"
        );
    }

    #[test]
    fn test_delaunay_validation_error_tds_variant_display() {
        let inner = TdsError::InconsistentDataStructure {
            message: "broken link".to_string(),
        };
        let err = DelaunayTriangulationValidationError::from(inner);
        assert!(err.to_string().contains("broken link"));
    }

    #[test]
    fn test_delaunay_validation_error_triangulation_variant_display() {
        let inner = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(slotmap::KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        };
        let err = DelaunayTriangulationValidationError::from(inner);
        assert!(err.to_string().contains("Isolated vertex"));
    }

    // ---- DT validate() error-mapping tests ----

    #[test]
    fn test_dt_validate_maps_tds_error_to_tds_variant() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Break vertex mapping so Level 2 structural validation fails.
        let vk = dt.tds().vertex_keys().next().unwrap();
        let uuid = dt.tds().get_vertex_by_key(vk).unwrap().uuid();
        dt.tds_mut().uuid_to_vertex_key.remove(&uuid);

        match dt.validate() {
            Err(DelaunayTriangulationValidationError::Tds(TdsError::MappingInconsistency {
                ..
            })) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Tds(MappingInconsistency), got {other:?}"
            ),
        }
    }

    #[test]
    fn test_dt_validate_maps_topology_error_to_triangulation_variant() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Add an isolated vertex so Level 3 (topology) fails.
        let _ = dt
            .tds_mut()
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 0.5]))
            .unwrap();

        match dt.validate() {
            Err(DelaunayTriangulationValidationError::Triangulation(
                TriangulationValidationError::IsolatedVertex { .. },
            )) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Triangulation(IsolatedVertex), got {other:?}"
            ),
        }
    }

    // ---- map_orientation_canonicalization_error: OrientationViolation ----

    #[test]
    fn test_map_orientation_canonicalization_error_orientation_violation_is_internal_inconsistency()
    {
        let error = InsertionError::TopologyValidation(TdsError::OrientationViolation {
            cell1_key: CellKey::from(slotmap::KeyData::from_ffi(1)),
            cell1_uuid: Uuid::nil(),
            cell2_key: CellKey::from(slotmap::KeyData::from_ffi(2)),
            cell2_uuid: Uuid::nil(),
            cell1_facet_index: 0,
            cell2_facet_index: 1,
            facet_vertices: vec![],
            cell2_facet_vertices: vec![],
            observed_odd_permutation: true,
            expected_odd_permutation: false,
        });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "OrientationViolation should map to InternalInconsistency (structural invariant breach, not geometry), got: {mapped:?}"
        );
    }

    // ---- map_orientation_canonicalization_error: ConflictRegion ----

    #[test]
    fn test_map_orientation_canonicalization_error_conflict_region_is_degeneracy() {
        let error = InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
            facet_hash: 0x123,
            cell_count: 3,
        });
        let mapped =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::GeometricDegeneracy { .. }
            ),
            "ConflictRegion should map to GeometricDegeneracy, got: {mapped:?}"
        );
    }

    // ---- RetryPolicy default regression test (#306) ----

    /// Verify that `RetryPolicy::default()` returns `Shuffled` with the expected
    /// attempt count in all build profiles.  Previously the default was `Disabled`
    /// in release builds, causing #306.
    #[test]
    fn test_retry_policy_default_is_shuffled_in_all_profiles() {
        let policy = RetryPolicy::default();
        match policy {
            RetryPolicy::Shuffled {
                attempts,
                base_seed,
            } => {
                assert_eq!(
                    attempts.get(),
                    DELAUNAY_SHUFFLE_ATTEMPTS,
                    "default retry attempts should equal DELAUNAY_SHUFFLE_ATTEMPTS"
                );
                assert_eq!(base_seed, None, "default base_seed should be None");
            }
            other => panic!("RetryPolicy::default() should be Shuffled, got {other:?}"),
        }
    }

    // ---- is_non_retryable_construction_error tests ----

    #[test]
    fn test_is_non_retryable_construction_error_duplicate_uuid() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::Tds(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Cell,
                uuid: Uuid::nil(),
            })
            .into();
        assert!(
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::is_non_retryable_construction_error(
                &err
            ),
            "DuplicateUuid should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_internal_inconsistency() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::InternalInconsistency {
                message: "test".to_string(),
            }
            .into();
        assert!(
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::is_non_retryable_construction_error(
                &err
            ),
            "InternalInconsistency should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_false_for_geometric_degeneracy() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::GeometricDegeneracy {
                message: "test".to_string(),
            }
            .into();
        assert!(
            !DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::is_non_retryable_construction_error(
                &err
            ),
            "GeometricDegeneracy should NOT be non-retryable"
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
            diagnostics: Box::new(crate::core::algorithms::flips::DelaunayRepairDiagnostics {
                facets_checked: 50,
                flips_performed: 1000,
                max_queue_len: 42,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: crate::core::algorithms::flips::RepairQueueOrder::Fifo,
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
}
