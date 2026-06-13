//! Batch construction options, errors, statistics, and policy helpers.
//!
//! This module contains the configuration surface used by
//! [`DelaunayTriangulationBuilder`](crate::builder::DelaunayTriangulationBuilder)
//! and the batch constructors on [`DelaunayTriangulation`](crate::DelaunayTriangulation).
//! Use it when you need deterministic insertion ordering, duplicate handling,
//! initial-simplex selection, retry behavior, or construction telemetry without
//! importing flip editing or validation-only APIs.
//!
//! Most examples should import these items through
//! [`delaunay::prelude::construction`](crate::prelude::construction), which
//! bundles the builder, construction options, construction errors, and the
//! [`vertex!`](crate::vertex) macro.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::construction::{
//!     ConstructionOptions, DelaunayTriangulationBuilder,
//!     DelaunayTriangulationConstructionError, InitialSimplexStrategy,
//!     InsertionOrderStrategy, vertex,
//! };
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.0, 1.0]),
//! ];
//!
//! let options = ConstructionOptions::default()
//!     .with_insertion_order(InsertionOrderStrategy::Input)
//!     .with_initial_simplex_strategy(InitialSimplexStrategy::First);
//!
//! let dt = DelaunayTriangulationBuilder::new(&vertices)
//!     .construction_options(options)
//!     .build::<()>()?;
//!
//! assert_eq!(dt.number_of_vertices(), 3);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::builder::DelaunayTriangulationBuilder;
use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairStats, FlipError, LocalRepairPhaseTiming,
    repair_delaunay_local_single_pass, repair_delaunay_local_single_pass_timed,
    repair_delaunay_with_flips_k2_k3,
};
use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, HullExtensionReason, InsertionError, SpatialIndexConstructionFailure,
    TdsConstructionFailure,
};
use crate::core::algorithms::locate::{ConflictError, LocateError};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{
    FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SecureHashMap, SmallBuffer,
};
use crate::core::construction::TriangulationConstructionError;
use crate::core::insertion::record_duplicate_detection_metrics;
use crate::core::operations::{
    DelaunayInsertionState, InsertionOutcome, InsertionResult, InsertionStatistics,
    InsertionTelemetryMode, RepairDecision, TopologicalOperation,
};
use crate::core::simplex::SimplexValidationError;
use crate::core::tds::SimplexKey;
use crate::core::tds::{TdsConstructionError, TdsError};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::util::{
    HilbertBitDepth, coords_equal_exact, coords_within_epsilon, hilbert_quantize_batch_in_range,
    stable_hash_u64_slice,
};
use crate::core::validation::{TopologyGuarantee, TriangulationValidationError, ValidationPolicy};
use crate::core::vertex::Vertex;
use crate::diagnostics::{BatchLocalRepairTrigger, ConstructionTelemetry, LocalRepairSample};
use crate::geometry::coordinate_range::CoordinateRange;
use crate::geometry::kernel::{AdaptiveKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar, CoordinateValues};
use crate::geometry::util::{
    RandomPointGenerationError, safe_coords_to_f64, safe_usize_to_scalar, simplex_volume,
};
use crate::locality::{
    accumulate_live_simplex_seeds, clear_simplex_seed_set, retain_live_simplex_seeds,
};
use crate::repair::DelaunayRepairPolicy;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::DelaunayTriangulation;
use crate::validation::DelaunayTriangulationValidationError;
use core::{cmp::Ordering, fmt};
use num_traits::{NumCast, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::{
    env,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    time::{Duration, Instant},
};
use thiserror::Error;
use uuid::Uuid;

/// Number of deterministic shuffled reconstruction attempts used by the
/// default construction retry policy.
const DELAUNAY_SHUFFLE_ATTEMPTS: usize = 6;

const DELAUNAY_SHUFFLE_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

fn batch_repair_trace_enabled() -> bool {
    env::var_os("DELAUNAY_BATCH_REPAIR_TRACE").is_some()
}

#[cfg(test)]
pub(crate) mod test_hooks {
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairError, RepairQueueOrder,
    };
    use std::cell::Cell;

    thread_local! {
        static FORCE_HEURISTIC_REBUILD: Cell<bool> = const { Cell::new(false) };
        static FORCE_REPAIR_NONCONVERGENT: Cell<bool> = const { Cell::new(false) };
        static BATCH_LOCAL_REPAIR_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    pub fn force_heuristic_rebuild_enabled() -> bool {
        FORCE_HEURISTIC_REBUILD.with(Cell::get)
    }

    #[must_use]
    pub fn set_force_heuristic_rebuild(enabled: bool) -> bool {
        FORCE_HEURISTIC_REBUILD.with(|flag| {
            let prior = flag.get();
            flag.set(enabled);
            prior
        })
    }

    pub fn restore_force_heuristic_rebuild(prior: bool) {
        FORCE_HEURISTIC_REBUILD.with(|flag| flag.set(prior));
    }

    pub fn force_repair_nonconvergent_enabled() -> bool {
        FORCE_REPAIR_NONCONVERGENT.with(Cell::get)
    }

    #[must_use]
    pub fn set_force_repair_nonconvergent(enabled: bool) -> bool {
        FORCE_REPAIR_NONCONVERGENT.with(|flag| {
            let prior = flag.get();
            flag.set(enabled);
            prior
        })
    }

    pub fn restore_force_repair_nonconvergent(prior: bool) {
        FORCE_REPAIR_NONCONVERGENT.with(|flag| flag.set(prior));
    }

    pub fn reset_batch_local_repair_calls() {
        BATCH_LOCAL_REPAIR_CALLS.with(|calls| calls.set(0));
    }

    pub fn batch_local_repair_calls() -> usize {
        BATCH_LOCAL_REPAIR_CALLS.with(Cell::get)
    }

    pub fn record_batch_local_repair_call() {
        BATCH_LOCAL_REPAIR_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    }

    #[must_use]
    pub fn synthetic_nonconvergent_error() -> DelaunayRepairError {
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
}

/// Errors that can occur during Delaunay triangulation construction.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayTriangulationConstructionError {
    /// Lower-layer construction failure summarized for Delaunay construction.
    #[error(transparent)]
    Triangulation(DelaunayConstructionFailure),

    /// Input validation error from explicit combinatorial construction.
    #[error(transparent)]
    ExplicitConstruction(#[from] crate::builder::ExplicitConstructionError),
}

impl From<TriangulationConstructionError> for DelaunayTriangulationConstructionError {
    fn from(source: TriangulationConstructionError) -> Self {
        Self::Triangulation(source.into())
    }
}

/// Construction phase that invoked flip-based Delaunay repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayConstructionRepairPhase {
    /// Local repair during the bulk insertion loop.
    BatchLocal {
        /// Zero-based input index whose insertion triggered the repair.
        index: usize,
    },
    /// Seeded or fallback repair during construction finalization.
    Completion,
}

impl fmt::Display for DelaunayConstructionRepairPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchLocal { index } => write!(f, "batch local repair at input index {index}"),
            Self::Completion => f.write_str("completion repair"),
        }
    }
}

/// Pattern-matchable summary of a lower-layer construction failure.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayConstructionFailure {
    /// Lower-layer TDS construction failed.
    #[error("TDS construction failed: {reason}")]
    Tds {
        /// Structured TDS construction failure.
        #[source]
        reason: TdsConstructionFailure,
    },

    /// Failed to create a simplex during construction.
    #[error("failed to create simplex during construction: {message}")]
    FailedToCreateSimplex {
        /// Simplex creation failure detail.
        message: String,
    },

    /// Cavity filling failed during insertion.
    #[error("cavity filling failed during insertion: {source}")]
    InsertionCavityFilling {
        /// Structured cavity-filling failure.
        #[source]
        source: CavityFillingError,
    },

    /// Insufficient vertices were provided.
    #[error("insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// Attempted dimension.
        dimension: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Random point generation failed before construction could begin.
    #[error("random point generation failed: {source}")]
    RandomPointGeneration {
        /// Structured random point generation failure.
        #[source]
        source: RandomPointGenerationError,
    },

    /// Geometric degeneracy prevented construction.
    #[error("geometric degeneracy encountered during construction: {message}")]
    GeometricDegeneracy {
        /// Degeneracy detail.
        message: String,
    },

    /// Periodic quotient construction is not release-validated for this dimension.
    #[error(
        "periodic image-point construction is release-validated only up to {max_validated_dimension}D; {dimension}D scalable quotient construction is tracked by issue #{tracking_issue}"
    )]
    UnsupportedPeriodicDimension {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Highest dimension with release-validated periodic quotient construction.
        max_validated_dimension: usize,
        /// Tracking issue for extending periodic quotient support.
        tracking_issue: u32,
    },

    /// Internal construction invariant failed.
    #[error("internal inconsistency during construction: {message}")]
    InternalInconsistency {
        /// Inconsistency detail.
        message: String,
    },

    /// Flip-based Delaunay repair failed during construction.
    #[error("Delaunay repair failed during {phase}: {source}")]
    DelaunayRepair {
        /// Construction phase that invoked repair.
        phase: DelaunayConstructionRepairPhase,
        /// Underlying typed repair failure.
        #[source]
        source: Box<DelaunayRepairError>,
    },

    /// Duplicate coordinates were detected.
    #[error("duplicate coordinates detected: {coordinates}")]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },

    /// Spatial index construction failed during construction.
    #[error("spatial index construction failed during construction: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },

    /// Conflict-region extraction failed during insertion.
    #[error("conflict region failed during insertion: {source}")]
    InsertionConflictRegion {
        /// Structured conflict-region failure.
        #[source]
        source: ConflictError,
    },

    /// Point location failed during insertion.
    #[error("point location failed during insertion: {source}")]
    InsertionLocation {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },

    /// Incremental insertion detected non-manifold topology.
    #[error(
        "non-manifold topology during insertion: facet {facet_hash:#x} shared by {simplex_count} simplices"
    )]
    InsertionNonManifoldTopology {
        /// Hash of the over-shared facet.
        facet_hash: u64,
        /// Number of simplices sharing the facet.
        simplex_count: usize,
    },

    /// Hull extension failed during insertion.
    #[error("hull extension failed during insertion: {reason}")]
    InsertionHullExtension {
        /// Structured hull-extension failure reason.
        reason: HullExtensionReason,
    },

    /// Level 4 Delaunay validation failed during insertion.
    #[error("Delaunay validation failed during insertion: {source}")]
    InsertionDelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Level 3 topology validation failed during insertion.
    #[error("topology validation failed during insertion: {message}: {source}")]
    InsertionTopologyValidation {
        /// High-level insertion context.
        message: String,
        /// Underlying topology validation error.
        #[source]
        source: TriangulationValidationError,
    },

    /// Local facet repair would remove more simplices than the active budget allowed.
    #[error(
        "local facet repair removal budget exceeded during construction: attempted {attempted}, max {max_simplices_removed}"
    )]
    LocalRepairBudgetExceeded {
        /// Maximum simplices the repair budget allowed for removal.
        max_simplices_removed: usize,
        /// Number of simplices selected for removal.
        attempted: usize,
    },

    /// Final topology validation failed after construction.
    #[error("final topology validation failed after construction: {message}: {source}")]
    FinalTopologyValidation {
        /// Validation failure detail.
        message: String,
        /// Underlying validation error.
        #[source]
        source: crate::core::tds::InvariantErrorSummary,
    },

    /// Final Delaunay-property validation failed after construction.
    #[error("final Delaunay validation failed after construction: {source}")]
    FinalDelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
}

impl From<TriangulationConstructionError> for DelaunayConstructionFailure {
    fn from(source: TriangulationConstructionError) -> Self {
        match source {
            TriangulationConstructionError::Tds(source) => Self::Tds {
                reason: source.into(),
            },
            TriangulationConstructionError::FailedToCreateSimplex { message } => {
                Self::FailedToCreateSimplex { message }
            }
            TriangulationConstructionError::InsertionCavityFilling { source } => {
                Self::InsertionCavityFilling { source }
            }
            TriangulationConstructionError::InsufficientVertices { dimension, source } => {
                Self::InsufficientVertices { dimension, source }
            }
            TriangulationConstructionError::GeometricDegeneracy { message } => {
                Self::GeometricDegeneracy { message }
            }
            TriangulationConstructionError::UnsupportedPeriodicDimension {
                dimension,
                max_validated_dimension,
                tracking_issue,
            } => Self::UnsupportedPeriodicDimension {
                dimension,
                max_validated_dimension,
                tracking_issue,
            },
            TriangulationConstructionError::InternalInconsistency { message } => {
                Self::InternalInconsistency { message }
            }
            TriangulationConstructionError::DuplicateCoordinates { coordinates } => {
                Self::DuplicateCoordinates { coordinates }
            }
            TriangulationConstructionError::SpatialIndexConstruction { reason } => {
                Self::SpatialIndexConstruction { reason }
            }
            TriangulationConstructionError::InsertionConflictRegion { source } => {
                Self::InsertionConflictRegion { source }
            }
            TriangulationConstructionError::InsertionLocation { source } => {
                Self::InsertionLocation { source }
            }
            TriangulationConstructionError::InsertionNonManifoldTopology {
                facet_hash,
                simplex_count,
            } => Self::InsertionNonManifoldTopology {
                facet_hash,
                simplex_count,
            },
            TriangulationConstructionError::InsertionHullExtension { reason } => {
                Self::InsertionHullExtension { reason }
            }
            TriangulationConstructionError::InsertionDelaunayValidation { source } => {
                Self::InsertionDelaunayValidation { source }
            }
            TriangulationConstructionError::InsertionTopologyValidation { message, source } => {
                Self::InsertionTopologyValidation { message, source }
            }
            TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            } => Self::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            },
            TriangulationConstructionError::FinalTopologyValidation { message, source } => {
                Self::FinalTopologyValidation { message, source }
            }
        }
    }
}

/// Returns true when a repair error represents input geometry or predicate
/// instability that shuffled construction may be able to resolve.
fn is_geometric_repair_error(repair_err: &DelaunayRepairError) -> bool {
    match repair_err {
        DelaunayRepairError::NonConvergent { .. }
        | DelaunayRepairError::PostconditionFailed { .. } => true,
        DelaunayRepairError::VerificationFailed { source, .. } => {
            is_geometric_flip_error(source.as_ref())
        }
        DelaunayRepairError::Flip { source } => is_geometric_flip_error(source.as_ref()),
        DelaunayRepairError::OrientationCanonicalizationFailed { .. }
        | DelaunayRepairError::InvalidTopology { .. }
        | DelaunayRepairError::HeuristicRebuildFailed { .. } => false,
    }
}

/// Returns true for flip errors caused by geometric predicates or degenerate
/// replacement simplices rather than deterministic topology/simplex-key failures.
fn is_geometric_flip_error(error: &FlipError) -> bool {
    match error {
        FlipError::PredicateFailure { .. }
        | FlipError::DegenerateSimplex
        | FlipError::NegativeOrientation { .. } => true,
        FlipError::SimplexCreation(source) => matches!(
            source.as_ref(),
            SimplexValidationError::DegenerateSimplex
                | SimplexValidationError::CoordinateConversion { .. }
        ),
        _ => false,
    }
}

/// Strategy used to order input vertices before batch construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InsertionOrderStrategy {
    /// Preserve the caller-provided input order.
    Input,
    /// Sort vertices by Hilbert curve with deterministic tie-breaking.
    #[default]
    Hilbert,
}

/// Policy controlling optional preprocessing to remove duplicate vertices.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum DedupPolicy {
    /// Do not apply explicit preprocessing dedup.
    ///
    /// Batch construction still keeps its built-in Hilbert and per-insertion
    /// duplicate defenses; exact duplicates can still be skipped and reported
    /// through [`ConstructionStatistics`].
    #[default]
    Off,
    /// Remove exact coordinate duplicates before construction.
    Exact,
    /// Remove near-duplicates within the given Euclidean tolerance.
    Epsilon {
        /// Non-negative Euclidean tolerance.
        tolerance: f64,
    },
}

/// Strategy controlling how the initial D+1 simplex vertices are selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InitialSimplexStrategy {
    /// Use the first D+1 vertices after preprocessing.
    First,
    /// Choose a better-conditioned simplex using a deterministic farthest-point heuristic.
    Balanced,
    /// Choose the largest-volume simplex from a bounded real-vertex candidate pool.
    #[default]
    MaxVolume,
}

/// Policy controlling deterministic retries with alternative insertion orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryPolicy {
    /// Do not attempt shuffled reconstruction retries.
    Disabled,
    /// Retry construction with deterministic shuffles.
    Shuffled {
        /// Number of shuffled reconstruction attempts.
        attempts: NonZeroUsize,
        /// Optional base seed.
        base_seed: Option<u64>,
    },
    /// Retry with deterministic shuffles only in debug/test builds.
    DebugOnlyShuffled {
        /// Number of shuffled reconstruction attempts.
        attempts: NonZeroUsize,
        /// Optional base seed.
        base_seed: Option<u64>,
    },
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::Shuffled {
            attempts: NonZeroUsize::new(DELAUNAY_SHUFFLE_ATTEMPTS)
                .expect("DELAUNAY_SHUFFLE_ATTEMPTS must be non-zero"),
            base_seed: None,
        }
    }
}

/// Default local-repair cadence for batch construction.
const fn default_batch_repair_policy() -> DelaunayRepairPolicy {
    DelaunayRepairPolicy::EveryInsertion
}

/// Options controlling batch construction behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ConstructionOptions {
    pub(crate) insertion_order: InsertionOrderStrategy,
    pub(crate) dedup_policy: DedupPolicy,
    pub(crate) initial_simplex: InitialSimplexStrategy,
    pub(crate) retry_policy: RetryPolicy,
    pub(crate) batch_repair_policy: DelaunayRepairPolicy,
    /// Whether final bulk repair can fall back to a global repair pass.
    pub(crate) use_global_repair_fallback: bool,
}

impl Default for ConstructionOptions {
    fn default() -> Self {
        Self {
            insertion_order: InsertionOrderStrategy::default(),
            dedup_policy: DedupPolicy::default(),
            initial_simplex: InitialSimplexStrategy::default(),
            retry_policy: RetryPolicy::default(),
            batch_repair_policy: default_batch_repair_policy(),
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

    /// Returns the automatic local Delaunay repair policy used during batch construction.
    #[must_use]
    pub const fn batch_repair_policy(&self) -> DelaunayRepairPolicy {
        self.batch_repair_policy
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

    /// Sets the automatic local Delaunay repair policy used during batch construction.
    #[must_use]
    pub const fn with_batch_repair_policy(
        mut self,
        batch_repair_policy: DelaunayRepairPolicy,
    ) -> Self {
        self.batch_repair_policy = batch_repair_policy;
        self
    }

    /// Disables the D<4 global repair fallback.
    #[must_use]
    pub(crate) const fn without_global_repair_fallback(mut self) -> Self {
        self.use_global_repair_fallback = false;
        self
    }
}

/// Aggregate statistics collected during batch construction.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct ConstructionStatistics {
    /// Number of vertices successfully inserted.
    pub inserted: usize,
    /// Number of vertices skipped due to duplicate coordinates.
    pub skipped_duplicate: usize,
    /// Number of vertices skipped due to geometric degeneracy.
    pub skipped_degeneracy: usize,
    /// Total number of insertion attempts across all vertices.
    pub total_attempts: usize,
    /// Maximum attempts for any single vertex.
    pub max_attempts: usize,
    /// Histogram of attempts.
    pub attempts_histogram: Vec<usize>,
    /// Number of vertices that required perturbation.
    pub used_perturbation: usize,
    /// Total number of simplices removed during insertion repair bookkeeping.
    pub simplices_removed_total: usize,
    /// Maximum number of simplices removed during repair for one insertion.
    pub simplices_removed_max: usize,
    /// Aggregate batch-construction telemetry.
    pub telemetry: ConstructionTelemetry,
    /// Slowest transactional insertions observed during batch construction.
    pub slow_insertions: Vec<ConstructionSlowInsertionSample>,
    /// Representative skipped vertices recorded during batch construction.
    pub skip_samples: Vec<ConstructionSkipSample>,
}

/// A single skipped-vertex sample captured during batch construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConstructionSkipSample {
    /// Index in the construction insertion order.
    pub index: usize,
    /// UUID of the skipped vertex.
    pub uuid: Uuid,
    /// Coordinates converted to `f64` for logging/debugging.
    pub coords: Vec<f64>,
    /// Whether [`coords`](Self::coords) contains converted coordinates.
    pub coords_available: bool,
    /// Number of insertion attempts for this vertex.
    pub attempts: usize,
    /// Human-readable error message.
    pub error: String,
}

/// A slow transactional insertion sample captured during batch construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConstructionSlowInsertionSample {
    /// Index in the construction insertion order.
    pub index: usize,
    /// UUID of the inserted or skipped vertex.
    pub uuid: Uuid,
    /// Number of insertion attempts for this vertex.
    pub attempts: usize,
    /// Final insertion result for this vertex.
    pub result: InsertionResult,
    /// Wall-clock nanoseconds spent in insertion.
    pub elapsed_nanos: u64,
    /// Simplex count immediately after the insertion attempt.
    pub simplices_after: usize,
    /// Point-location calls performed by this insertion.
    pub locate_calls: usize,
    /// Facet-walk steps performed by this insertion.
    pub locate_walk_steps_total: usize,
    /// Conflict-region calls performed by this insertion.
    pub conflict_region_calls: usize,
    /// Conflict-region simplices observed by this insertion.
    pub conflict_region_simplices_total: usize,
    /// Cavity insertion calls performed by this insertion.
    pub cavity_insertion_calls: usize,
    /// Global exterior conflict scans performed by this insertion.
    pub global_conflict_scans: usize,
    /// Hull extension calls performed by this insertion.
    pub hull_extension_calls: usize,
    /// Post-insertion topology validations performed by this insertion.
    pub topology_validation_calls: usize,
}

/// Construction error that also carries aggregate statistics collected up to failure.
#[derive(Debug, Clone, Error)]
#[error("{error}")]
#[non_exhaustive]
#[must_use]
pub struct DelaunayTriangulationConstructionErrorWithStatistics {
    /// Underlying construction error.
    #[source]
    pub error: DelaunayTriangulationConstructionError,
    /// Aggregate construction statistics collected before the error.
    pub statistics: ConstructionStatistics,
}

impl ConstructionStatistics {
    /// Aggregates attempt counters shared by inserted, skipped, and duplicate outcomes.
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

        self.simplices_removed_total = self
            .simplices_removed_total
            .saturating_add(stats.simplices_removed_during_repair);
        self.simplices_removed_max = self
            .simplices_removed_max
            .max(stats.simplices_removed_during_repair);
    }

    const MAX_SKIP_SAMPLES: usize = 8;
    const MAX_SLOW_INSERTION_SAMPLES: usize = 8;

    /// Record a single insertion attempt.
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

    /// Record a slow insertion sample, preserving the top samples by elapsed time.
    pub fn record_slow_insertion_sample(&mut self, sample: ConstructionSlowInsertionSample) {
        let insert_at = self
            .slow_insertions
            .iter()
            .position(|existing| sample.elapsed_nanos > existing.elapsed_nanos)
            .unwrap_or(self.slow_insertions.len());
        if insert_at >= Self::MAX_SLOW_INSERTION_SAMPLES {
            return;
        }

        self.slow_insertions.insert(insert_at, sample);
        self.slow_insertions
            .truncate(Self::MAX_SLOW_INSERTION_SAMPLES);
    }

    /// Merges attempt-level statistics from another construction pass.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.inserted = self.inserted.saturating_add(other.inserted);
        self.skipped_duplicate = self
            .skipped_duplicate
            .saturating_add(other.skipped_duplicate);
        self.skipped_degeneracy = self
            .skipped_degeneracy
            .saturating_add(other.skipped_degeneracy);
        self.total_attempts = self.total_attempts.saturating_add(other.total_attempts);
        self.max_attempts = self.max_attempts.max(other.max_attempts);

        if self.attempts_histogram.len() < other.attempts_histogram.len() {
            self.attempts_histogram
                .resize(other.attempts_histogram.len(), 0);
        }
        for (idx, count) in other.attempts_histogram.iter().enumerate() {
            self.attempts_histogram[idx] = self.attempts_histogram[idx].saturating_add(*count);
        }

        self.used_perturbation = self
            .used_perturbation
            .saturating_add(other.used_perturbation);
        self.simplices_removed_total = self
            .simplices_removed_total
            .saturating_add(other.simplices_removed_total);
        self.simplices_removed_max = self.simplices_removed_max.max(other.simplices_removed_max);
        self.telemetry.merge_from(&other.telemetry);

        for sample in &other.skip_samples {
            if self.skip_samples.len() >= Self::MAX_SKIP_SAMPLES {
                break;
            }
            self.skip_samples.push(sample.clone());
        }

        for sample in &other.slow_insertions {
            self.record_slow_insertion_sample(sample.clone());
        }
    }

    /// Total number of skipped vertices.
    #[must_use]
    pub const fn total_skipped(&self) -> usize {
        self.skipped_duplicate + self.skipped_degeneracy
    }
}

// Per-insertion local-repair flip-budget tunables.
//
// Budget formula: `seed_simplices.len() * (D + 1) * FACTOR` with a minimum of
// `FLOOR`. Two regimes so that D>=4's higher queue demand does not force a
// global budget increase.
//
// The D>=4 constants are sized from the measured `max_queue` distribution on the
// 500-point 4D seeded repro (seed `0xD225B8A07E274AE6`, ball radius 100)
// captured in `docs/archive/issue_204_investigation.md`:
//
//   max_queue samples  min=91 p50=207 p90=281 p95=312 p99=409 max=416
//
// `FACTOR = 12` with `FLOOR = 96` yields a typical 300-flip budget (5-simplex seed
// set), covering p50-p90 and brushing p95. The p95-p99 tail is deferred to the
// final completion repair rather than paid for during every cadenced repair.
const LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4: usize = 12;
const LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4: usize = 96;
const LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_LT_4: usize = 4;
const LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_LT_4: usize = 16;
const LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_GE_4: usize = 24;
const LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_LT_4: usize = 16;

/// Per-insertion local Delaunay repair flip budget.
///
/// Computes `seeds * (D + 1) * FACTOR` with a minimum of `FLOOR`, using the
/// dimension-aware constants above.
pub(crate) const fn local_repair_flip_budget<const D: usize>(seed_simplices_len: usize) -> usize {
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
    let raw = seed_simplices_len
        .saturating_mul(D + 1)
        .saturating_mul(factor);
    if raw > floor { raw } else { floor }
}

/// Pending local repair frontier size that triggers an early batch repair.
///
/// The threshold keeps sparse repair cadences from letting a large seed
/// frontier accumulate. 3D uses a lower threshold because the 3000-point sweep
/// in #341 showed that repair cost rises sharply once the pending frontier
/// grows beyond the small-batch regime.
const fn local_repair_seed_backlog_threshold<const D: usize>() -> usize {
    let factor = if D >= 4 {
        LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_GE_4
    } else {
        LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_LT_4
    };
    (D + 1).saturating_mul(factor)
}

/// Decides whether batch construction should run local Delaunay repair now.
pub(crate) fn batch_local_repair_trigger<const D: usize>(
    policy: DelaunayRepairPolicy,
    insertion_count: usize,
    topology: TopologyGuarantee,
    pending_seed_simplices_len: usize,
) -> Option<BatchLocalRepairTrigger> {
    if policy == DelaunayRepairPolicy::Never
        || pending_seed_simplices_len == 0
        || !TopologicalOperation::FacetFlip.is_admissible_under(topology)
    {
        return None;
    }

    if matches!(
        policy.decide(insertion_count, topology, TopologicalOperation::FacetFlip,),
        RepairDecision::Proceed
    ) {
        return Some(BatchLocalRepairTrigger::Cadence);
    }

    (pending_seed_simplices_len >= local_repair_seed_backlog_threshold::<D>())
        .then_some(BatchLocalRepairTrigger::SeedBacklog)
}

// =============================================================================
// BATCH CONSTRUCTION ORDERING HELPERS (INTERNAL)
// =============================================================================

type VertexBuffer<T, U, const D: usize> = Vec<Vertex<T, U, D>>;

/// Preprocessed vertex ordering state used by batch construction to preserve
/// deterministic insertion order, retry fallback order, and deduplication-grid
/// reuse across public construction entry points.
pub(crate) struct PreprocessVertices<T, U, const D: usize> {
    primary: Option<VertexBuffer<T, U, D>>,
    fallback: Option<VertexBuffer<T, U, D>>,
    grid_cell_size: Option<T>,
}

impl<T, U, const D: usize> fmt::Debug for PreprocessVertices<T, U, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreprocessVertices")
            .field("primary_len", &self.primary.as_ref().map(Vec::len))
            .field("fallback_len", &self.fallback.as_ref().map(Vec::len))
            .field("has_grid_cell_size", &self.grid_cell_size.is_some())
            .finish()
    }
}

impl<T, U, const D: usize> PreprocessVertices<T, U, D> {
    /// Borrows the preprocessed vertex order when one exists, avoiding a clone
    /// for policies that leave the input unchanged.
    pub(crate) fn primary_slice<'a>(
        &'a self,
        input: &'a [Vertex<T, U, D>],
    ) -> &'a [Vertex<T, U, D>] {
        self.primary.as_deref().unwrap_or(input)
    }

    /// Exposes the original order as a retry fallback for balanced-simplex
    /// preprocessing.
    pub(crate) fn fallback_slice(&self) -> Option<&[Vertex<T, U, D>]> {
        self.fallback.as_deref()
    }

    /// Carries the dedup grid size forward so incremental insertion can reuse a
    /// compatible spatial index.
    pub(crate) const fn grid_cell_size(&self) -> Option<T>
    where
        T: Copy,
    {
        self.grid_cell_size
    }
}

pub(crate) type PreprocessVerticesResult<T, U, const D: usize> =
    Result<PreprocessVertices<T, U, D>, DelaunayTriangulationConstructionError>;

/// Hashes coordinates as a deterministic tiebreaker for partial vertex ordering.
fn vertex_coordinate_hash<T, U, const D: usize>(vertex: &Vertex<T, U, D>) -> u64
where
    T: CoordinateScalar,
{
    let mut hasher = FastHasher::default();
    vertex.hash(&mut hasher);
    hasher.finish()
}

fn total_cmp_for_coordinate<T>(left: &T, right: &T) -> Ordering
where
    T: CoordinateScalar,
{
    left.ordered_partial_cmp(right)
        .expect("CoordinateScalar::ordered_partial_cmp must define a total order")
}

fn compare_vertices_by_coordinates<T, U, const D: usize>(
    left: &Vertex<T, U, D>,
    right: &Vertex<T, U, D>,
) -> Ordering
where
    T: CoordinateScalar,
{
    for (left_coord, right_coord) in left.point().coords().iter().zip(right.point().coords()) {
        let ordering = total_cmp_for_coordinate(left_coord, right_coord);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

/// Produces a stable construction order when Hilbert ordering is unavailable or
/// unsuitable.
fn order_vertices_lexicographic<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
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
        compare_vertices_by_coordinates(a, b)
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
    dedup_quantized: bool,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
{
    match insertion_order {
        InsertionOrderStrategy::Input => vertices,
        InsertionOrderStrategy::Hilbert => order_vertices_hilbert(vertices, dedup_quantized),
    }
}

/// Provides a scalar-aware tolerance for dedup paths that need a nonzero grid
/// size even under exact duplicate policy.
pub(crate) fn default_duplicate_tolerance<T: CoordinateScalar>() -> T {
    <T as NumCast>::from(1e-10_f64).unwrap_or_else(T::default_tolerance)
}

/// Builds a hash-grid index after construction has validated its tolerance.
///
/// This preserves the parse-don't-validate boundary between public
/// construction inputs and the internal duplicate index: a failure here means
/// an upstream construction invariant was broken after validation, so it is
/// surfaced as a typed spatial-index construction failure.
fn hash_grid_from_validated_cell_size<T, const D: usize, I>(
    cell_size: T,
) -> Result<HashGridIndex<T, D, I>, DelaunayTriangulationConstructionError>
where
    T: CoordinateScalar,
{
    HashGridIndex::try_new(cell_size).map_err(|error| {
        DelaunayTriangulationConstructionError::from(
            TriangulationConstructionError::SpatialIndexConstruction {
                reason: error.into(),
            },
        )
    })
}

/// Verifies the hash grid can represent every input coordinate before choosing
/// the O(n) duplicate path.
fn hash_grid_usable_for_vertices<T, U, const D: usize>(
    grid: &HashGridIndex<T, D, usize>,
    vertices: &[Vertex<T, U, D>],
) -> bool
where
    T: CoordinateScalar,
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
{
    if !hash_grid_usable_for_vertices(grid, &vertices) {
        return dedup_vertices_exact_sorted(vertices);
    }
    grid.clear();
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    for v in vertices {
        let coords = *v.point().coords();
        let mut duplicate = false;
        let mut candidate_count = 0usize;
        let used_index = grid.for_each_candidate_vertex_key(&coords, |idx| {
            candidate_count = candidate_count.saturating_add(1);
            let existing_coords = unique[idx].point().coords();
            if coords_equal_exact(&coords, existing_coords) {
                duplicate = true;
                return false;
            }
            true
        });

        record_duplicate_detection_metrics(used_index, candidate_count, !used_index);

        if !duplicate {
            let idx = unique.len();
            unique.push(v);
            grid.insert_vertex(idx, &coords);
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
    // Quantized keys are derived directly from public coordinate input, so use
    // randomized hashing instead of `FastHashMap`.
    let mut buckets: SecureHashMap<
        QuantizedKey<D>,
        SmallBuffer<usize, BATCH_DEDUP_BUCKET_INLINE_CAPACITY>,
    > = SecureHashMap::default();
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
{
    if !hash_grid_usable_for_vertices(grid, &vertices) {
        return dedup_vertices_epsilon_quantized(vertices, epsilon);
    }
    grid.clear();
    let mut unique: Vec<Vertex<T, U, D>> = Vec::with_capacity(vertices.len());

    let epsilon_sq = epsilon * epsilon;
    for v in vertices {
        let coords = *v.point().coords();
        let mut duplicate = false;
        let mut candidate_count = 0usize;
        let used_index = grid.for_each_candidate_vertex_key(&coords, |idx| {
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
            grid.insert_vertex(idx, &coords);
        }
    }

    unique
}

/// Converts candidate simplex vertices to f64 coordinates for deterministic
/// preprocessing heuristics without hiding non-finite inputs.
fn vertices_coords_f64<T, U, const D: usize>(vertices: &[Vertex<T, U, D>]) -> Option<Vec<[f64; D]>>
where
    T: CoordinateScalar,
{
    let mut coords_f64: Vec<[f64; D]> = Vec::with_capacity(vertices.len());
    for v in vertices {
        let coords = safe_coords_to_f64(v.point().coords()).ok()?;
        if coords.iter().any(|coord| !coord.is_finite()) {
            return None;
        }
        coords_f64.push(coords);
    }
    Some(coords_f64)
}

/// Computes squared Euclidean distance for initial-simplex selection
/// heuristics that only need deterministic ordering.
fn squared_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum::<f64>()
}

/// Appends an index once so candidate pools remain small and deterministic.
fn push_unique_index(indices: &mut Vec<usize>, idx: usize) {
    if !indices.contains(&idx) {
        indices.push(idx);
    }
}

/// Computes the bounded candidate-pool size for max-volume simplex search.
const INITIAL_SIMPLEX_MAX_VOLUME_CANDIDATE_CAP: usize = 18;

const fn initial_simplex_candidate_cap<const D: usize>(point_count: usize) -> usize {
    let minimum = D.saturating_add(1);
    let bounded_cap = if INITIAL_SIMPLEX_MAX_VOLUME_CANDIDATE_CAP > minimum {
        INITIAL_SIMPLEX_MAX_VOLUME_CANDIDATE_CAP
    } else {
        minimum
    };
    let requested = D.saturating_add(1).saturating_mul(2).saturating_add(4);
    let target = if requested < bounded_cap {
        requested
    } else {
        bounded_cap
    };
    if point_count < target {
        point_count
    } else {
        target
    }
}

/// Finds the deterministic lexicographic anchor for a candidate pool.
fn lexicographic_min_index<const D: usize>(coords_f64: &[[f64; D]]) -> Option<usize> {
    if coords_f64.is_empty() {
        return None;
    }
    let mut lexicographic_min = 0usize;
    for idx in 1..coords_f64.len() {
        if coords_f64[idx].partial_cmp(&coords_f64[lexicographic_min]) == Some(Ordering::Less) {
            lexicographic_min = idx;
        }
    }
    Some(lexicographic_min)
}

/// Adds per-axis coordinate extrema to the candidate pool.
fn append_axis_extrema<const D: usize>(coords_f64: &[[f64; D]], candidates: &mut Vec<usize>) {
    for axis in 0..D {
        let mut min_idx = 0usize;
        let mut max_idx = 0usize;
        for idx in 1..coords_f64.len() {
            let coord = coords_f64[idx][axis];
            let min_coord = coords_f64[min_idx][axis];
            let max_coord = coords_f64[max_idx][axis];

            match coord.partial_cmp(&min_coord) {
                Some(Ordering::Less) => min_idx = idx,
                Some(Ordering::Equal)
                    if coords_f64[idx].partial_cmp(&coords_f64[min_idx])
                        == Some(Ordering::Less) =>
                {
                    min_idx = idx;
                }
                _ => {}
            }
            match coord.partial_cmp(&max_coord) {
                Some(Ordering::Greater) => max_idx = idx,
                Some(Ordering::Equal)
                    if coords_f64[idx].partial_cmp(&coords_f64[max_idx])
                        == Some(Ordering::Less) =>
                {
                    max_idx = idx;
                }
                _ => {}
            }
        }
        push_unique_index(candidates, min_idx);
        push_unique_index(candidates, max_idx);
    }
}

/// Extends the candidate pool with farthest-point samples until it reaches the
/// configured cap or exhausts usable points.
fn extend_candidate_pool_by_farthest_points<const D: usize>(
    coords_f64: &[[f64; D]],
    candidates: &mut Vec<usize>,
    candidate_cap: usize,
) {
    let mut selected_mask = vec![false; coords_f64.len()];
    for &idx in candidates.iter() {
        selected_mask[idx] = true;
    }

    let mut min_dist_sq = vec![f64::INFINITY; coords_f64.len()];
    for idx in 0..coords_f64.len() {
        if selected_mask[idx] {
            min_dist_sq[idx] = 0.0;
            continue;
        }
        for &candidate_idx in candidates.iter() {
            let dist = squared_distance(&coords_f64[idx], &coords_f64[candidate_idx]);
            if dist < min_dist_sq[idx] {
                min_dist_sq[idx] = dist;
            }
        }
    }

    while candidates.len() < candidate_cap {
        let mut best_idx: Option<usize> = None;
        let mut best_dist = -1.0_f64;

        for idx in 0..coords_f64.len() {
            if selected_mask[idx] {
                continue;
            }
            let dist = min_dist_sq[idx];
            if !dist.is_finite() {
                continue;
            }
            let replace = best_idx.is_none_or(|best_idx_val| match dist.partial_cmp(&best_dist) {
                Some(Ordering::Greater) => true,
                Some(Ordering::Equal) => {
                    coords_f64[idx].partial_cmp(&coords_f64[best_idx_val]) == Some(Ordering::Less)
                }
                _ => false,
            });
            if replace {
                best_idx = Some(idx);
                best_dist = dist;
            }
        }

        let Some(best_idx) = best_idx else {
            break;
        };
        push_unique_index(candidates, best_idx);
        selected_mask[best_idx] = true;

        for idx in 0..coords_f64.len() {
            if selected_mask[idx] {
                continue;
            }
            let dist = squared_distance(&coords_f64[idx], &coords_f64[best_idx]);
            if dist < min_dist_sq[idx] {
                min_dist_sq[idx] = dist;
            }
        }
    }
}

/// Chooses a bounded pool of real extreme vertices for max-volume simplex
/// search.
fn initial_simplex_candidate_pool_indices<const D: usize>(coords_f64: &[[f64; D]]) -> Vec<usize> {
    let candidate_cap = initial_simplex_candidate_cap::<D>(coords_f64.len());
    if candidate_cap == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::with_capacity(candidate_cap);
    if let Some(lexicographic_min) = lexicographic_min_index(coords_f64) {
        push_unique_index(&mut candidates, lexicographic_min);
    }
    append_axis_extrema(coords_f64, &mut candidates);
    extend_candidate_pool_by_farthest_points(coords_f64, &mut candidates, candidate_cap);

    candidates
}

/// Chooses a well-spread initial simplex to reduce early degeneracy in
/// incremental construction.
fn select_balanced_simplex_indices<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
) -> Option<Vec<usize>>
where
    T: CoordinateScalar,
{
    if vertices.len() < D + 1 {
        return None;
    }

    let coords_f64 = vertices_coords_f64(vertices)?;

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
        min_dist_sq[i] = squared_distance(&coords_f64[i], &coords_f64[seed_idx]);
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
            let dist_sq = squared_distance(&coords_f64[i], &coords_f64[best_idx]);
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

/// Advances a lexicographic combination in place so max-volume search can
/// enumerate bounded candidate pools without recursion.
fn advance_combination(indices: &mut [usize], upper: usize) -> bool {
    let len = indices.len();
    if len > upper {
        return false;
    }
    for pos in (0..len).rev() {
        if indices[pos] < pos + upper - len {
            indices[pos] += 1;
            for next in pos + 1..len {
                indices[next] = indices[next - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Scores a candidate simplex by f64 volume and rejects degenerate choices.
fn simplex_volume_for_indices<const D: usize>(
    coords_f64: &[[f64; D]],
    simplex_indices: &[usize],
) -> Option<f64> {
    if simplex_indices.len() != D + 1 {
        return None;
    }

    let mut points: SmallBuffer<Point<f64, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(simplex_indices.len());
    for &idx in simplex_indices {
        points.push(Point::new(coords_f64[idx]));
    }
    simplex_volume(&points)
        .ok()
        .filter(|volume| volume.is_finite() && *volume > 0.0)
}

/// Chooses the largest-volume nondegenerate real simplex from a bounded
/// extreme-vertex candidate pool.
fn select_max_volume_simplex_indices<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
) -> Option<Vec<usize>>
where
    T: CoordinateScalar,
{
    if vertices.len() < D + 1 {
        return None;
    }

    let coords_f64 = vertices_coords_f64(vertices)?;
    let candidates = initial_simplex_candidate_pool_indices(&coords_f64);
    if candidates.len() < D + 1 {
        return None;
    }

    let simplex_len = D + 1;
    let mut combination: Vec<usize> = (0..simplex_len).collect();
    let mut best_volume = 0.0_f64;
    let mut best_indices: Option<Vec<usize>> = None;

    loop {
        let simplex_indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> = combination
            .iter()
            .map(|&candidate_idx| candidates[candidate_idx])
            .collect();
        if let Some(volume) = simplex_volume_for_indices(&coords_f64, &simplex_indices)
            && volume > best_volume
        {
            best_volume = volume;
            best_indices = Some(simplex_indices.iter().copied().collect());
        }

        if !advance_combination(&mut combination, candidates.len()) {
            break;
        }
    }

    best_indices
}

/// Places the selected simplex first while preserving every remaining input
/// vertex exactly once.
fn reorder_vertices_for_simplex<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    simplex_indices: &[usize],
) -> Option<Vec<Vertex<T, U, D>>>
where
    T: Copy,
    U: Copy,
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
fn hilbert_bits_per_coord<const D: usize>() -> Option<HilbertBitDepth> {
    if D == 0 {
        return None;
    }

    let Ok(d_u32) = u32::try_from(D) else {
        return None;
    };

    // Hilbert indexing encodes D coordinates with `bits` bits each into a `u128`.
    // Use as many bits as possible (up to the `hilbert` module's `bits <= 31` bound).
    let bits_per_coord = (128_u32 / d_u32).min(31);
    HilbertBitDepth::try_new(bits_per_coord).ok()
}

/// Converts vertex coordinates for diagnostics without synthesizing sentinel values.
///
/// Returns `None` if any coordinate cannot be represented as `f64`, allowing
/// callers to omit diagnostic coordinates instead of hiding conversion failure
/// behind `NaN` or infinity.
pub(crate) fn vertex_coords_f64<T, U, const D: usize>(vertex: &Vertex<T, U, D>) -> Option<Vec<f64>>
where
    T: CoordinateScalar,
{
    vertex
        .point()
        .coords()
        .iter()
        .map(|coord| coord.to_f64().filter(|value| value.is_finite()))
        .collect()
}

/// Sort key for Hilbert ordering: `(Hilbert index, quantized coords, vertex, input index)`.
type HilbertSortKey<T, U, const D: usize> = (u128, [u32; D], Vertex<T, U, D>, usize);

/// Orders vertices along a Hilbert curve to improve insertion locality while
/// retaining deterministic lexicographic fallbacks.
fn order_vertices_hilbert<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
    dedup_quantized: bool,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
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

    if min.total_cmp(&max).is_eq() {
        return if dedup_quantized {
            dedup_vertices_exact_sorted(vertices)
        } else {
            order_vertices_lexicographic(vertices)
        };
    }

    let (Some(min_t), Some(max_t)) = (NumCast::from(min), NumCast::from(max)) else {
        return order_vertices_lexicographic(vertices);
    };

    let Ok(bounds) = CoordinateRange::try_new(min_t, max_t) else {
        return order_vertices_lexicographic(vertices);
    };

    // Quantize all coordinates and compute Hilbert indices in one
    // proof-carrying pass. `hilbert_quantize_batch_in_range` validates the
    // index width and quantization scale once, clamps every coordinate into
    // the bit-depth grid, and returns a batch whose indices are infallible, so
    // no per-coordinate range rescan is repeated here.
    let Ok(batch) = hilbert_quantize_batch_in_range(&vertices, bounds, bits_per_coord, |vertex| {
        *vertex.point().coords()
    }) else {
        // On quantization error, fall back to true lexicographic ordering.
        return order_vertices_lexicographic(vertices);
    };

    let (indices, quantized) = batch.into_indices_and_coordinates();
    if vertices.len() != quantized.len() || vertices.len() != indices.len() {
        return order_vertices_lexicographic(vertices);
    }

    // Pair indices with vertices, quantized coords, and input indices.
    let mut keyed: Vec<HilbertSortKey<T, U, D>> = vertices
        .into_iter()
        .zip(quantized)
        .zip(indices)
        .enumerate()
        .map(|(input_index, ((vertex, q), idx))| (idx, q, vertex, input_index))
        .collect();

    keyed.sort_by(
        |(a_idx, a_q, a_vertex, a_in), (b_idx, b_q, b_vertex, b_in)| {
            a_idx
                .cmp(b_idx)
                .then_with(|| a_q.cmp(b_q))
                .then_with(|| compare_vertices_by_coordinates(a_vertex, b_vertex))
                .then_with(|| a_in.cmp(b_in))
        },
    );

    if dedup_quantized {
        // Deduplicate at quantization resolution in a single linear sweep.
        // Because vertices sharing the same quantized cell are now adjacent
        // after sorting, we can eliminate duplicates without re-quantizing.
        let input_len = keyed.len();
        let mut prev_q: Option<[u32; D]> = None;
        let mut deduped = Vec::with_capacity(input_len);

        for (_, q, vertex, _) in keyed {
            if prev_q == Some(q) {
                continue;
            }
            prev_q = Some(q);
            deduped.push(vertex);
        }

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

// Most common case: f64 with AdaptiveKernel, no vertex or simplex data.
impl<const D: usize> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
    /// Creates a Delaunay triangulation from `f64` vertices with no attached data.
    ///
    /// This convenience constructor uses [`AdaptiveKernel`] and the default
    /// construction options.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may contain fewer vertices
    /// than the input slice. Use
    /// [`new_with_construction_statistics`](Self::new_with_construction_statistics)
    /// when skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if the initial simplex cannot be constructed, if
    /// insertion fails, or if final validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        Self::with_kernel(&AdaptiveKernel::<f64>::new(), vertices)
    }

    /// Creates a default `f64` triangulation and returns aggregate construction
    /// statistics.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may have skipped some input
    /// vertices. Inspect [`ConstructionStatistics::total_skipped`] and
    /// [`ConstructionStatistics::skip_samples`] when the caller requires a
    /// strict all-inputs-inserted contract.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if
    /// construction fails. The error includes partial statistics collected
    /// before failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics for compatibility"
    )]
    pub fn new_with_construction_statistics(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_options_and_statistics(
            &kernel,
            vertices,
            TopologyGuarantee::DEFAULT,
            ConstructionOptions::default(),
        )
    }

    /// Creates a default `f64` triangulation with explicit construction options
    /// and returns aggregate construction statistics.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may have skipped some input
    /// vertices. Inspect [`ConstructionStatistics::total_skipped`] and
    /// [`ConstructionStatistics::skip_samples`] when the caller requires a
    /// strict all-inputs-inserted contract.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if
    /// construction fails. The error includes partial statistics collected
    /// before failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DelaunayTriangulationBuilder, RetryPolicy, vertex,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let Some(attempts) = NonZeroUsize::new(2) else {
    ///     return Ok(());
    /// };
    /// let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
    ///     attempts,
    ///     base_seed: Some(7),
    /// });
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .construction_options(options)
    ///     .build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
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
        Self::with_options_and_statistics(&kernel, vertices, TopologyGuarantee::DEFAULT, options)
    }

    /// Creates a default `f64` triangulation with explicit construction options.
    ///
    /// This batch constructor may successfully build a triangulation after
    /// skipping duplicate or retry-exhausted degenerate input vertices. Use
    /// [`new_with_options_and_construction_statistics`](Self::new_with_options_and_construction_statistics)
    /// when skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if construction fails, or if the selected options are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DedupPolicy, DelaunayTriangulationBuilder, InsertionOrderStrategy,
    ///     vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let options = ConstructionOptions::default()
    ///     .with_insertion_order(InsertionOrderStrategy::Hilbert)
    ///     .with_dedup_policy(DedupPolicy::Exact);
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .construction_options(options)
    ///     .build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
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

    /// Creates a default `f64` triangulation with an explicit topology guarantee.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may contain fewer vertices
    /// than the input slice. Use
    /// [`new_with_construction_statistics`](Self::new_with_construction_statistics)
    /// when skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if construction fails or if the requested topology
    /// guarantee cannot be satisfied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
    /// };
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build::<()>()?;
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_topology_guarantee(
        vertices: &[Vertex<f64, (), D>],
        topology_guarantee: TopologyGuarantee,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let kernel = AdaptiveKernel::<f64>::new();
        Self::with_topology_guarantee(&kernel, vertices, topology_guarantee)
    }

    /// Creates an empty default `f64` triangulation with no attached data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulation;
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_simplices(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self::with_empty_kernel(AdaptiveKernel::<f64>::new())
    }

    /// Creates an empty default `f64` triangulation with an explicit topology guarantee.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulation, TopologyGuarantee,
    /// };
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[must_use]
    pub fn empty_with_topology_guarantee(topology_guarantee: TopologyGuarantee) -> Self {
        Self::with_empty_kernel_and_topology_guarantee(
            AdaptiveKernel::<f64>::new(),
            topology_guarantee,
        )
    }

    /// Creates a fluent builder for default `f64` Delaunay triangulations.
    ///
    /// For non-`f64` coordinates, vertex data, or custom kernels, construct
    /// [`DelaunayTriangulationBuilder::new`] directly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
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
// hashing, and deduplication — the kernel already guarantees `CoordinateScalar`;
// these paths add `NumCast`.

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: NumCast,
    U: DataType,
    V: DataType,
{
    /// Retries batch construction with deterministic shuffles so retryable
    /// degeneracies can be escaped reproducibly.
    #[expect(
        clippy::too_many_lines,
        reason = "construction retry flow keeps seed selection, validation, and diagnostics together"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "private construction retry helper threads orthogonal batch knobs explicitly"
    )]
    pub(crate) fn build_with_shuffled_retries(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
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
            batch_repair_policy,
            use_global_repair_fallback,
        ) {
            Ok(candidate) => match candidate.is_delaunay_via_flips() {
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
                batch_repair_policy,
                use_global_repair_fallback,
            ) {
                Ok(candidate) => match candidate.is_delaunay_via_flips() {
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
                },
                Err(err) => {
                    let err_string = err.to_string();
                    if Self::is_non_retryable_construction_error(&err) {
                        log_construction_retry_result(
                            attempt,
                            Some(attempt_seed),
                            perturbation_seed,
                            "failed",
                            Some(&err_string),
                            None,
                        );
                        return Err(err);
                    }
                    last_error = err_string;
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
    #[expect(
        clippy::too_many_lines,
        reason = "statistics variant mirrors construction retry flow for comparable diagnostics"
    )]
    #[expect(
        clippy::result_large_err,
        reason = "Internal helper propagates public by-value construction-statistics error type"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "statistics retry helper mirrors the non-statistics construction path"
    )]
    pub(crate) fn build_with_shuffled_retries_with_construction_statistics(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
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
        let mut aggregate_stats = ConstructionStatistics::default();

        // Attempt 0: original order, no extra perturbation salt.
        log_construction_retry_result(0, None, 0_u64, "started", None, None);
        let mut last_error: String =
            match Self::build_with_kernel_inner_seeded_with_construction_statistics(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                0_u64,
                true,
                grid_cell_size,
                batch_repair_policy,
                use_global_repair_fallback,
            ) {
                Ok((candidate, mut stats)) => {
                    let delaunay_started = Instant::now();
                    let delaunay_result = candidate.is_delaunay_via_flips();
                    stats
                        .telemetry
                        .record_construction_final_delaunay_validation_timing(
                            duration_nanos_saturating(delaunay_started.elapsed()),
                        );
                    match delaunay_result {
                        Ok(()) => {
                            aggregate_stats.merge_from(&stats);
                            log_construction_retry_result(
                                0,
                                None,
                                0_u64,
                                "succeeded",
                                None,
                                Some(&stats),
                            );
                            return Ok((candidate, aggregate_stats));
                        }
                        Err(err) => {
                            aggregate_stats.merge_from(&stats);
                            last_stats.replace(stats);
                            format!("Delaunay property violated after construction: {err}")
                        }
                    }
                }
                Err(err) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics { error, statistics } =
                        err;
                    aggregate_stats.merge_from(&statistics);
                    if Self::is_non_retryable_construction_error(&error) {
                        let last_error = error.to_string();
                        log_construction_retry_result(
                            0,
                            None,
                            0_u64,
                            "failed",
                            Some(&last_error),
                            Some(&statistics),
                        );
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics: aggregate_stats,
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
                batch_repair_policy,
                use_global_repair_fallback,
            ) {
                Ok((candidate, mut stats)) => {
                    let delaunay_started = Instant::now();
                    let delaunay_result = candidate.is_delaunay_via_flips();
                    stats
                        .telemetry
                        .record_construction_final_delaunay_validation_timing(
                            duration_nanos_saturating(delaunay_started.elapsed()),
                        );
                    match delaunay_result {
                        Ok(()) => {
                            aggregate_stats.merge_from(&stats);
                            log_construction_retry_result(
                                attempt,
                                Some(attempt_seed),
                                perturbation_seed,
                                "succeeded",
                                None,
                                Some(&stats),
                            );
                            return Ok((candidate, aggregate_stats));
                        }
                        Err(err) => {
                            aggregate_stats.merge_from(&stats);
                            last_stats.replace(stats);
                            last_error =
                                format!("Delaunay property violated after construction: {err}");
                        }
                    }
                }
                Err(err) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics { error, statistics } =
                        err;
                    aggregate_stats.merge_from(&statistics);
                    if Self::is_non_retryable_construction_error(&error) {
                        let last_error = error.to_string();
                        log_construction_retry_result(
                            attempt,
                            Some(attempt_seed),
                            perturbation_seed,
                            "failed",
                            Some(&last_error),
                            Some(&statistics),
                        );
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics: aggregate_stats,
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
        Err(DelaunayTriangulationConstructionErrorWithStatistics {
            error: TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Delaunay construction failed after shuffled reconstruction attempts: {last_error}"
                ),
            }
            .into(),
            statistics: aggregate_stats,
        })
    }

    /// Runs batch construction without statistics while preserving the same
    /// final validation path as the statistics variant.
    pub(crate) fn build_with_kernel_inner(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
        use_global_repair_fallback: bool,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        let dt = Self::build_with_kernel_inner_seeded(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
            batch_repair_policy,
            use_global_repair_fallback,
        )?;

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
        delaunay_result.map_err(|source| {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::FinalDelaunayValidation { source },
            )
        })?;

        Ok(dt)
    }

    /// Runs batch construction with aggregate statistics without changing the
    /// construction algorithm itself.
    #[expect(
        clippy::result_large_err,
        reason = "Internal helper propagates public by-value construction-statistics error type"
    )]
    pub(crate) fn build_with_kernel_inner_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
        use_global_repair_fallback: bool,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let (dt, mut stats) = Self::build_with_kernel_inner_seeded_with_construction_statistics(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
            batch_repair_policy,
            use_global_repair_fallback,
        )?;

        // `DelaunayCheckPolicy::EndOnly`: always run a final global Delaunay validation pass after
        // batch construction.
        tracing::debug!("post-construction: starting Delaunay validation (build stats)");
        let delaunay_started = Instant::now();
        let delaunay_result = dt.is_valid();
        let delaunay_elapsed = delaunay_started.elapsed();
        stats
            .telemetry
            .record_construction_final_delaunay_validation_timing(duration_nanos_saturating(
                delaunay_elapsed,
            ));
        tracing::debug!(
            elapsed = ?delaunay_elapsed,
            success = delaunay_result.is_ok(),
            "post-construction: Delaunay validation (build stats) completed"
        );
        if let Err(err) = delaunay_result {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error: DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::FinalDelaunayValidation { source: err },
                ),
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
    #[expect(
        clippy::too_many_arguments,
        reason = "seeded construction helper carries retry, repair, and validation knobs"
    )]
    fn build_with_kernel_inner_seeded_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
        use_global_repair_fallback: bool,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        if vertices.len() < D + 1 {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error: TriangulationConstructionError::InsufficientVertices {
                    dimension: D,
                    source: SimplexValidationError::InsufficientVertices {
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
        // per-insertion validation.  Running a full O(simplices) topology check after
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
        // per insertion.  Instead, insert_remaining_vertices_seeded accumulates the local
        // frontier touched by successful insertions and calls repair_delaunay_local_single_pass
        // at the requested cadence (no topology check, no heuristic rebuild, soft-fail on
        // non-convergence for D≥4).  Soft-failed repair frontiers are retained for the final
        // seeded repair in finalize_bulk_construction.
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

        let mut soft_fail_seeds: Vec<SimplexKey> = Vec::new();
        let mut pending_repair_seeds: Vec<SimplexKey> = Vec::new();
        let insert_loop_started = Instant::now();
        let insert_result = dt.insert_remaining_vertices_seeded(
            vertices,
            perturbation_seed,
            grid_cell_size,
            batch_repair_policy,
            Some(&mut stats),
            &mut pending_repair_seeds,
            &mut soft_fail_seeds,
        );
        stats
            .telemetry
            .record_construction_insert_loop_timing(duration_nanos_saturating(
                insert_loop_started.elapsed(),
            ));
        if let Err(error) = insert_result {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error,
                statistics: stats,
            });
        }

        let finalize_started = Instant::now();
        let finalize_result = dt.finalize_bulk_construction(
            original_validation_policy,
            original_repair_policy,
            run_final_repair,
            batch_repair_policy,
            &pending_repair_seeds,
            &soft_fail_seeds,
            Some(&mut stats.telemetry),
        );
        stats
            .telemetry
            .record_construction_finalize_timing(duration_nanos_saturating(
                finalize_started.elapsed(),
            ));
        if let Err(error) = finalize_result {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error,
                statistics: stats,
            });
        }

        Ok((dt, stats))
    }

    /// Implements the non-statistics seeded construction core for callers that
    /// only need the triangulation.
    #[expect(
        clippy::too_many_arguments,
        reason = "seeded construction helper carries retry, repair, and validation knobs"
    )]
    fn build_with_kernel_inner_seeded(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
        use_global_repair_fallback: bool,
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        if vertices.len() < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: SimplexValidationError::InsufficientVertices {
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
        let mut soft_fail_seeds: Vec<SimplexKey> = Vec::new();
        let mut pending_repair_seeds: Vec<SimplexKey> = Vec::new();
        dt.insert_remaining_vertices_seeded(
            vertices,
            perturbation_seed,
            grid_cell_size,
            batch_repair_policy,
            None,
            &mut pending_repair_seeds,
            &mut soft_fail_seeds,
        )?;
        dt.finalize_bulk_construction(
            original_validation_policy,
            original_repair_policy,
            run_final_repair,
            batch_repair_policy,
            &pending_repair_seeds,
            &soft_fail_seeds,
            None,
        )?;

        Ok(dt)
    }

    /// Records successful local-repair telemetry in one place so the repair loop
    /// stays focused on control flow.
    fn record_successful_local_repair_telemetry(
        telemetry: &mut ConstructionTelemetry,
        index: usize,
        trigger: BatchLocalRepairTrigger,
        seed_simplices_len: usize,
        repair_elapsed: Duration,
        phase_timing: &LocalRepairPhaseTiming,
        stats: &DelaunayRepairStats,
    ) {
        telemetry.record_local_repair_work(
            stats.facets_checked,
            stats.flips_performed,
            stats.max_queue_len,
        );
        telemetry.record_local_repair_sample(LocalRepairSample {
            index,
            trigger,
            seed_simplices: seed_simplices_len,
            elapsed_nanos: duration_nanos_saturating(repair_elapsed),
            items_checked: stats.facets_checked,
            flips_performed: stats.flips_performed,
            max_queue_len: stats.max_queue_len,
            facet_nanos: phase_timing.attempt_facet_nanos,
            ridge_nanos: phase_timing.attempt_ridge_nanos,
            postcondition_nanos: phase_timing.postcondition_nanos,
        });
    }

    /// Records timing and frontier telemetry for one local-repair attempt.
    fn record_local_repair_attempt_telemetry(
        telemetry: &mut ConstructionTelemetry,
        repair_elapsed: Duration,
        phase_timing: &LocalRepairPhaseTiming,
        seed_simplices_len: usize,
        trigger: BatchLocalRepairTrigger,
    ) {
        telemetry.record_local_repair_timing(duration_nanos_saturating(repair_elapsed));
        telemetry.record_local_repair_phase_timing(phase_timing);
        telemetry.record_local_repair_frontier(seed_simplices_len, trigger);
    }

    /// Repairs the currently accumulated batch-local seed frontier.
    #[expect(
        clippy::too_many_lines,
        reason = "local repair control flow keeps telemetry, rollback, and soft-fail handling together"
    )]
    fn repair_pending_local_seed_simplices(
        &mut self,
        index: usize,
        trigger: BatchLocalRepairTrigger,
        pending_seed_simplices: &mut Vec<SimplexKey>,
        pending_seen: &mut FastHashSet<SimplexKey>,
        soft_fail_seeds: &mut Vec<SimplexKey>,
        mut construction_telemetry: Option<&mut ConstructionTelemetry>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        retain_live_simplex_seeds(&self.tri.tds, pending_seed_simplices, pending_seen);
        if pending_seed_simplices.is_empty() {
            return Ok(());
        }

        #[cfg(test)]
        test_hooks::record_batch_local_repair_call();

        let seed_simplices_len = pending_seed_simplices.len();
        let max_flips = local_repair_flip_budget::<D>(seed_simplices_len);
        let trace_repair = batch_repair_trace_enabled();
        let mut phase_timing = LocalRepairPhaseTiming::default();
        if trace_repair {
            tracing::debug!(
                idx = index,
                seed_simplices = seed_simplices_len,
                max_flips,
                trigger = ?trigger,
                "bulk batch repair: starting local repair"
            );
        }
        let collect_telemetry = construction_telemetry.is_some();
        let repair_started = (collect_telemetry || trace_repair).then(Instant::now);

        let repair_result = {
            self.invalidate_locate_hint_cache();
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            let timing = if collect_telemetry {
                Some(&mut phase_timing)
            } else {
                None
            };
            repair_delaunay_local_single_pass_timed(
                tds,
                kernel,
                pending_seed_simplices,
                max_flips,
                timing,
            )
        };
        #[cfg(test)]
        let repair_result = if test_hooks::force_repair_nonconvergent_enabled() {
            Err(test_hooks::synthetic_nonconvergent_error())
        } else {
            repair_result
        };
        let repair_elapsed = repair_started.map(|started| started.elapsed());
        if let Some(telemetry) = construction_telemetry.as_mut() {
            Self::record_local_repair_attempt_telemetry(
                telemetry,
                repair_elapsed.unwrap_or_default(),
                &phase_timing,
                seed_simplices_len,
                trigger,
            );
        }

        match repair_result {
            Ok(stats) => {
                if let Some(telemetry) = construction_telemetry.as_mut() {
                    Self::record_successful_local_repair_telemetry(
                        telemetry,
                        index,
                        trigger,
                        seed_simplices_len,
                        repair_elapsed.unwrap_or_default(),
                        &phase_timing,
                        &stats,
                    );
                }
                if trace_repair {
                    tracing::debug!(
                        idx = index,
                        seed_simplices = seed_simplices_len,
                        flips = stats.flips_performed,
                        checked = stats.facets_checked,
                        max_queue = stats.max_queue_len,
                        elapsed = ?repair_elapsed.unwrap_or_default(),
                        "bulk batch repair: local repair succeeded"
                    );
                }
                clear_simplex_seed_set(pending_seed_simplices, pending_seen);
            }
            Err(repair_err) => {
                if trace_repair {
                    tracing::debug!(
                        idx = index,
                        seed_simplices = seed_simplices_len,
                        error = %repair_err,
                        elapsed = ?repair_elapsed.unwrap_or_default(),
                        "bulk batch repair: local repair failed"
                    );
                }
                if !Self::can_soft_fail(&repair_err) {
                    return Err(Self::map_hard_repair_error(index, repair_err));
                }
                tracing::debug!(
                    idx = index,
                    error = %repair_err,
                    seed_simplices = seed_simplices_len,
                    "bulk batch repair: local repair soft-failed; deferring seeds to final repair"
                );
                soft_fail_seeds.extend(pending_seed_simplices.iter().copied());
                clear_simplex_seed_set(pending_seed_simplices, pending_seen);
            }
        }
        Ok(())
    }

    /// Inserts the non-simplex vertices under a fixed perturbation seed so bulk
    /// construction retries are reproducible.
    #[expect(
        clippy::too_many_lines,
        reason = "seeded insertion loop keeps cache repair and retry diagnostics in one flow"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "seeded insertion loop needs batch repair and construction-statistics state"
    )]
    fn insert_remaining_vertices_seeded(
        &mut self,
        vertices: &[Vertex<K::Scalar, U, D>],
        perturbation_seed: u64,
        grid_cell_size: Option<K::Scalar>,
        batch_repair_policy: DelaunayRepairPolicy,
        construction_stats: Option<&mut ConstructionStatistics>,
        pending_repair_seeds: &mut Vec<SimplexKey>,
        soft_fail_seeds: &mut Vec<SimplexKey>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        let mut grid_index: Option<HashGridIndex<K::Scalar, D>> = match grid_cell_size {
            Some(cell_size) => Some(hash_grid_from_validated_cell_size(cell_size)?),
            None => None,
        };
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
            let initial_simplex_vertices = vertices.len().min(D + 1);
            let started = Instant::now();
            BatchProgressState {
                // The initial simplex is already present when this loop starts, so progress
                // and throughput only count the remaining bulk vertices — the counters live
                // in a "bulk-only" frame, 0…(input_len - (D+1)).
                input_vertices: vertices.len(),
                initial_simplex_vertices,
                bulk_vertices: vertices.len().saturating_sub(initial_simplex_vertices),
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
        let mut pending_repair_seen: FastHashSet<SimplexKey> =
            pending_repair_seeds.iter().copied().collect();

        match construction_stats {
            None => {
                // Insert remaining vertices incrementally.
                // Retryable geometric degeneracies are retried with perturbation and ultimately skipped
                // (transactional rollback) to keep the triangulation manifold. Duplicate/near-duplicate
                // coordinates are skipped immediately.
                for (offset, vertex) in vertices.iter().skip(D + 1).enumerate() {
                    let index = (D + 1).saturating_add(offset);
                    let uuid = vertex.uuid();
                    let coords = trace_insertion.then(|| vertex_coords_f64(vertex)).flatten();

                    if trace_insertion && let Some(coords) = coords.as_ref() {
                        tracing::debug!(index, %uuid, coords = ?coords, "[bulk] start");
                    }

                    let started = trace_insertion.then(Instant::now);
                    let mut insert = || {
                        // Pass the batch index through to transactional insertion so the
                        // lower-layer retryable-skip trace can point back to this exact
                        // bulk-construction position.
                        self.tri.insert_with_statistics_seeded_indexed_detailed(
                            *vertex,
                            None,
                            self.insertion_state.last_inserted_simplex,
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
                        let repair_seed_simplices = detail.repair_seed_simplices;
                        (
                            detail.outcome,
                            detail.stats,
                            repair_seed_simplices,
                            detail.delaunay_repair_required,
                        )
                    });
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: _,
                                hint,
                            },
                            _stats,
                            repair_seed_simplices,
                            delaunay_repair_required,
                        )) => {
                            inserted_vertices = inserted_vertices.saturating_add(1);
                            if trace_insertion && let Some(elapsed) = elapsed {
                                tracing::debug!(index, %uuid, elapsed = ?elapsed, "[bulk] inserted");
                            }
                            // Cache hint for faster subsequent insertions.
                            self.insertion_state.last_inserted_simplex = hint;
                            self.insertion_state.delaunay_repair_insertion_count = self
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);
                            // Batch local Delaunay repair: accumulate the local frontier
                            // touched by each successful insertion, then repair the whole
                            // frontier when the policy fires or the frontier grows too large.
                            // This keeps EveryN semantics local to the recent insertion window
                            // rather than repairing only the final insertion in the batch.
                            let topology = self.tri.topology_guarantee();
                            if delaunay_repair_required
                                && batch_repair_policy != DelaunayRepairPolicy::Never
                                && TopologicalOperation::FacetFlip.is_admissible_under(topology)
                                && self.tri.tds.number_of_simplices() > 0
                            {
                                accumulate_live_simplex_seeds(
                                    &self.tri.tds,
                                    &repair_seed_simplices,
                                    pending_repair_seeds,
                                    &mut pending_repair_seen,
                                );
                                if let Some(trigger) = batch_local_repair_trigger::<D>(
                                    batch_repair_policy,
                                    inserted_vertices,
                                    topology,
                                    pending_repair_seeds.len(),
                                ) {
                                    self.repair_pending_local_seed_simplices(
                                        index,
                                        trigger,
                                        pending_repair_seeds,
                                        &mut pending_repair_seen,
                                        soft_fail_seeds,
                                        None,
                                    )?;
                                }
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    bulk_processed: offset + 1,
                                    bulk_inserted: inserted_vertices,
                                    bulk_skipped: skipped_vertices,
                                    simplex_count: self.tri.tds.number_of_simplices(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Ok((
                            InsertionOutcome::Skipped { error },
                            stats,
                            _repair_seed_simplices,
                            _delaunay_repair_required,
                        )) => {
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
                                    bulk_processed: offset + 1,
                                    bulk_inserted: inserted_vertices,
                                    bulk_skipped: skipped_vertices,
                                    simplex_count: self.tri.tds.number_of_simplices(),
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
                    let coords = trace_insertion.then(|| vertex_coords_f64(vertex)).flatten();

                    if trace_insertion && let Some(coords) = coords.as_ref() {
                        tracing::debug!(index, %uuid, coords = ?coords, "[bulk] start");
                    }

                    let started = Instant::now();
                    let mut insert = || {
                        // Keep the stats and non-stats branches aligned so bulk-index-based
                        // tracing behaves the same regardless of whether the caller records
                        // construction statistics.
                        self.tri
                            .insert_with_statistics_seeded_indexed_detailed_with_telemetry(
                                *vertex,
                                None,
                                self.insertion_state.last_inserted_simplex,
                                perturbation_seed,
                                grid_index.as_mut(),
                                Some(index),
                                InsertionTelemetryMode::CountsAndTimings,
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
                    let elapsed = started.elapsed();
                    let elapsed_nanos = duration_nanos_saturating(elapsed);
                    let insert_result = insert_result.map(|detail| {
                        let repair_seed_simplices = detail.repair_seed_simplices;
                        (
                            detail.outcome,
                            detail.stats,
                            repair_seed_simplices,
                            detail.delaunay_repair_required,
                            detail.telemetry,
                        )
                    });
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: _,
                                hint,
                            },
                            stats,
                            repair_seed_simplices,
                            delaunay_repair_required,
                            telemetry,
                        )) => {
                            inserted_vertices = inserted_vertices.saturating_add(1);
                            if trace_insertion {
                                tracing::debug!(
                                    index,
                                    %uuid,
                                    attempts = stats.attempts,
                                    elapsed = ?elapsed,
                                    "[bulk] inserted"
                                );
                            }
                            construction_stats.record_insertion(&stats);
                            construction_stats.telemetry.record_insertion(&telemetry);
                            construction_stats
                                .telemetry
                                .record_insertion_timing(elapsed_nanos);
                            construction_stats.record_slow_insertion_sample(
                                ConstructionSlowInsertionSample {
                                    index,
                                    uuid,
                                    attempts: stats.attempts,
                                    result: stats.result,
                                    elapsed_nanos,
                                    simplices_after: self.tri.tds.number_of_simplices(),
                                    locate_calls: telemetry.locate_calls,
                                    locate_walk_steps_total: telemetry.locate_walk_steps_total,
                                    conflict_region_calls: telemetry.conflict_region_calls,
                                    conflict_region_simplices_total: telemetry
                                        .conflict_region_simplices_total,
                                    cavity_insertion_calls: telemetry.cavity_insertion_calls,
                                    global_conflict_scans: telemetry.global_conflict_scans,
                                    hull_extension_calls: telemetry.hull_extension_calls,
                                    topology_validation_calls: telemetry.topology_validation_calls,
                                },
                            );

                            // Cache hint for faster subsequent insertions.
                            self.insertion_state.last_inserted_simplex = hint;
                            self.insertion_state.delaunay_repair_insertion_count = self
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);
                            // Batch local repair: see the non-stats branch
                            // comment for full details.
                            let topology = self.tri.topology_guarantee();
                            if delaunay_repair_required
                                && batch_repair_policy != DelaunayRepairPolicy::Never
                                && TopologicalOperation::FacetFlip.is_admissible_under(topology)
                                && self.tri.tds.number_of_simplices() > 0
                            {
                                let seed_started = Instant::now();
                                let seed_simplices_added = accumulate_live_simplex_seeds(
                                    &self.tri.tds,
                                    &repair_seed_simplices,
                                    pending_repair_seeds,
                                    &mut pending_repair_seen,
                                );
                                construction_stats
                                    .telemetry
                                    .record_repair_seed_accumulation(
                                        duration_nanos_saturating(seed_started.elapsed()),
                                        seed_simplices_added,
                                    );
                                if let Some(trigger) = batch_local_repair_trigger::<D>(
                                    batch_repair_policy,
                                    inserted_vertices,
                                    topology,
                                    pending_repair_seeds.len(),
                                ) {
                                    self.repair_pending_local_seed_simplices(
                                        index,
                                        trigger,
                                        pending_repair_seeds,
                                        &mut pending_repair_seen,
                                        soft_fail_seeds,
                                        Some(&mut construction_stats.telemetry),
                                    )?;
                                }
                            }
                            log_bulk_progress_if_due(
                                BatchProgressSample {
                                    bulk_processed: offset + 1,
                                    bulk_inserted: inserted_vertices,
                                    bulk_skipped: skipped_vertices,
                                    simplex_count: self.tri.tds.number_of_simplices(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Ok((
                            InsertionOutcome::Skipped { error },
                            stats,
                            _repair_seed_simplices,
                            _delaunay_repair_required,
                            telemetry,
                        )) => {
                            skipped_vertices = skipped_vertices.saturating_add(1);
                            if trace_insertion {
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
                            construction_stats.telemetry.record_insertion(&telemetry);
                            construction_stats
                                .telemetry
                                .record_insertion_timing(elapsed_nanos);
                            construction_stats.record_slow_insertion_sample(
                                ConstructionSlowInsertionSample {
                                    index,
                                    uuid,
                                    attempts: stats.attempts,
                                    result: stats.result,
                                    elapsed_nanos,
                                    simplices_after: self.tri.tds.number_of_simplices(),
                                    locate_calls: telemetry.locate_calls,
                                    locate_walk_steps_total: telemetry.locate_walk_steps_total,
                                    conflict_region_calls: telemetry.conflict_region_calls,
                                    conflict_region_simplices_total: telemetry
                                        .conflict_region_simplices_total,
                                    cavity_insertion_calls: telemetry.cavity_insertion_calls,
                                    global_conflict_scans: telemetry.global_conflict_scans,
                                    hull_extension_calls: telemetry.hull_extension_calls,
                                    topology_validation_calls: telemetry.topology_validation_calls,
                                },
                            );

                            // Keep the first few skip samples so we have concrete reproduction anchors.
                            let (coords, coords_available) = vertex_coords_f64(vertex)
                                .map_or_else(|| (Vec::new(), false), |coords| (coords, true));
                            construction_stats.record_skip_sample(ConstructionSkipSample {
                                index,
                                uuid: vertex.uuid(),
                                coords,
                                coords_available,
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
                                    bulk_processed: offset + 1,
                                    bulk_inserted: inserted_vertices,
                                    bulk_skipped: skipped_vertices,
                                    simplex_count: self.tri.tds.number_of_simplices(),
                                    perturbation_seed,
                                },
                                &mut batch_progress,
                            );
                        }
                        Err(e) => {
                            if trace_insertion {
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
    #[expect(
        clippy::too_many_arguments,
        reason = "bulk finalization restores policies, repair state, and optional statistics telemetry"
    )]
    pub(crate) fn finalize_bulk_construction(
        &mut self,
        original_validation_policy: ValidationPolicy,
        original_repair_policy: DelaunayRepairPolicy,
        run_final_repair: bool,
        batch_repair_policy: DelaunayRepairPolicy,
        pending_repair_seeds: &[SimplexKey],
        soft_fail_seeds: &[SimplexKey],
        mut construction_telemetry: Option<&mut ConstructionTelemetry>,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        // Restore policies after batch construction.
        self.tri.validation_policy = original_validation_policy;
        self.insertion_state.delaunay_repair_policy = original_repair_policy;

        let has_simplices = self.tri.tds.number_of_simplices() > 0;
        let mut completion_seed_simplices = Vec::new();
        let mut completion_seen = FastHashSet::default();
        for &simplex_key in pending_repair_seeds.iter().chain(soft_fail_seeds.iter()) {
            if self.tri.tds.contains_simplex(simplex_key) && completion_seen.insert(simplex_key) {
                completion_seed_simplices.push(simplex_key);
            }
        }
        if run_final_repair
            && has_simplices
            && batch_repair_policy != DelaunayRepairPolicy::Never
            && !completion_seed_simplices.is_empty()
        {
            let repair_started = Instant::now();
            let repair_result = self.run_seeded_completion_repair(&completion_seed_simplices);
            if let Some(telemetry) = construction_telemetry.as_mut() {
                telemetry.record_construction_completion_repair_timing(duration_nanos_saturating(
                    repair_started.elapsed(),
                ));
            }
            repair_result?;
        }

        // Flip-based repair calls normalize_coherent_orientation() which makes all simplices
        // combinatorially coherent but can leave the global sign negative.  Re-canonicalize
        // geometric orientation to positive before validation (#258).
        let orientation_started = Instant::now();
        let orientation_result = self
            .tri
            .normalize_and_promote_positive_orientation()
            .map_err(Self::map_orientation_canonicalization_error);
        if let Some(telemetry) = construction_telemetry.as_mut() {
            telemetry.record_construction_orientation_timing(duration_nanos_saturating(
                orientation_started.elapsed(),
            ));
        }
        orientation_result?;

        tracing::debug!("post-construction: starting topology validation (finalize)");
        let validation_started = Instant::now();
        let validation_result = self.tri.validate();
        let validation_elapsed = validation_started.elapsed();
        if let Some(telemetry) = construction_telemetry.as_mut() {
            telemetry.record_construction_topology_validation_timing(duration_nanos_saturating(
                validation_elapsed,
            ));
        }
        tracing::debug!(
            elapsed = ?validation_elapsed,
            success = validation_result.is_ok(),
            "post-construction: topology validation (finalize) completed"
        );
        if let Err(err) = validation_result {
            return Err(TriangulationConstructionError::FinalTopologyValidation {
                message: "topology validation failed after construction".to_string(),
                source: err.into(),
            }
            .into());
        }

        Ok(())
    }

    fn run_seeded_completion_repair(
        &mut self,
        completion_seed_simplices: &[SimplexKey],
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        let seed_count = completion_seed_simplices.len();
        let max_flips = local_repair_flip_budget::<D>(seed_count);
        tracing::debug!(
            seed_count,
            max_flips,
            "post-construction: starting seeded completion Delaunay repair"
        );
        let repair_started = Instant::now();
        let repair_result = {
            self.invalidate_locate_hint_cache();
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_local_single_pass(tds, kernel, completion_seed_simplices, max_flips)
        };
        let repair_outcome = match repair_result {
            Ok(_) => Ok(()),
            Err(error) => self.try_final_global_repair_after_seeded_failure(&error),
        };
        tracing::debug!(
            elapsed = ?repair_started.elapsed(),
            success = repair_outcome.is_ok(),
            "post-construction: seeded completion Delaunay repair finished"
        );
        repair_outcome
    }

    fn try_final_global_repair_after_seeded_failure(
        &mut self,
        seeded_error: &DelaunayRepairError,
    ) -> Result<(), DelaunayTriangulationConstructionError> {
        if !self.insertion_state.use_global_repair_fallback || !Self::can_soft_fail(seeded_error) {
            let message = format!("Delaunay repair failed after construction: {seeded_error}");
            return Err(Self::map_completion_repair_error(
                message,
                seeded_error.clone(),
            ));
        }

        tracing::debug!(
            error = %seeded_error,
            "post-construction: seeded completion repair soft-failed; trying final global repair"
        );
        self.invalidate_locate_hint_cache();
        let topology = self.tri.topology_guarantee();
        let global_result = {
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology, None)
        };
        match global_result {
            Ok(_) => Ok(()),
            Err(global_error) => {
                let message = format!(
                    "Delaunay repair failed after construction: seeded local error: \
                     {seeded_error}; global fallback: {global_error}"
                );
                Err(Self::map_completion_repair_error(message, global_error))
            }
        }
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: NumCast,
    U: DataType,
    V: DataType,
{
    /// Creates an empty Delaunay triangulation with the given kernel.
    ///
    /// Use this when a caller needs a custom kernel but wants to insert vertices
    /// incrementally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayTriangulation;
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    ///     DelaunayTriangulation::with_empty_kernel(RobustKernel::new());
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// ```
    #[must_use]
    pub fn with_empty_kernel(kernel: K) -> Self {
        let duplicate_tolerance = default_duplicate_tolerance::<K::Scalar>();

        Self {
            tri: Triangulation::new_empty(kernel),
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: HashGridIndex::try_new(duplicate_tolerance).ok(),
        }
    }

    /// Creates an empty Delaunay triangulation with a topology guarantee.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    ///     DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
    ///         RobustKernel::new(),
    ///         TopologyGuarantee::PLManifold,
    ///     );
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    /// ```
    #[must_use]
    pub fn with_empty_kernel_and_topology_guarantee(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        let duplicate_tolerance = default_duplicate_tolerance::<K::Scalar>();

        let mut tri = Triangulation::new_empty(kernel);
        tri.topology_guarantee = topology_guarantee;
        tri.validation_policy = topology_guarantee.default_validation_policy();
        Self {
            tri,
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: HashGridIndex::try_new(duplicate_tolerance).ok(),
        }
    }

    /// Creates a Delaunay triangulation from vertices with an explicit kernel.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may contain fewer vertices
    /// than the input slice. Use
    /// [`with_options_and_statistics`](Self::with_options_and_statistics) when
    /// skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if construction fails or final validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let kernel = RobustKernel::<f64>::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_with_kernel::<_, ()>(&kernel)?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_kernel(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
        Self::with_topology_guarantee(kernel, vertices, TopologyGuarantee::DEFAULT)
    }

    /// Creates a Delaunay triangulation with an explicit topology guarantee.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may contain fewer vertices
    /// than the input slice. Use
    /// [`with_options_and_statistics`](Self::with_options_and_statistics) when
    /// skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if construction fails or if the requested topology
    /// guarantee cannot be satisfied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let kernel = RobustKernel::<f64>::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build_with_kernel::<_, ()>(&kernel)?;
    /// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    /// # Ok(())
    /// # }
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

    /// Creates a Delaunay triangulation with topology and construction options.
    ///
    /// This batch constructor may successfully build a triangulation after
    /// skipping duplicate or retry-exhausted degenerate input vertices. Use
    /// [`with_options_and_statistics`](Self::with_options_and_statistics) when
    /// skipped-input observability is required.
    ///
    /// # Errors
    /// Returns an error if construction fails, if validation fails, or if the
    /// selected preprocessing options are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionError> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let kernel = RobustKernel::<f64>::new();
    /// let dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .construction_options(ConstructionOptions::default())
    ///     .build_with_kernel::<_, ()>(&kernel)?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
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
            batch_repair_policy,
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
                            batch_repair_policy,
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
                            batch_repair_policy,
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
                batch_repair_policy,
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

    /// Creates a Delaunay triangulation with construction statistics.
    ///
    /// Batch construction is a best-effort ingestion path for duplicate or
    /// degenerate inputs: a successful construction may have skipped some input
    /// vertices. Inspect [`ConstructionStatistics::total_skipped`] and
    /// [`ConstructionStatistics::skip_samples`] when the caller requires a
    /// strict all-inputs-inserted contract.
    ///
    /// # Errors
    /// Returns [`DelaunayTriangulationConstructionErrorWithStatistics`] if
    /// construction fails. The error includes the partial statistics collected
    /// before failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     ConstructionOptions, DelaunayTriangulation, TopologyGuarantee, vertex,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// # fn main() -> Result<(), delaunay::DelaunayTriangulationConstructionErrorWithStatistics> {
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let kernel = RobustKernel::<f64>::new();
    /// let (dt, stats) =
    ///     DelaunayTriangulation::<RobustKernel<f64>, (), (), 3>::with_options_and_statistics(
    ///         &kernel,
    ///         &vertices,
    ///         TopologyGuarantee::PLManifold,
    ///         ConstructionOptions::default(),
    ///     )?;
    /// assert_eq!(dt.number_of_vertices(), stats.inserted);
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::result_large_err,
        reason = "Public API intentionally returns by-value construction statistics for compatibility"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "Statistics constructor handles preprocessing, retry, and fallback aggregation"
    )]
    pub fn with_options_and_statistics(
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
            batch_repair_policy,
            use_global_repair_fallback,
        } = options;

        let preprocessing_started = Instant::now();
        let preprocessed = match Self::preprocess_vertices_for_construction(
            vertices,
            dedup_policy,
            insertion_order,
            initial_simplex,
        ) {
            Ok(preprocessed) => preprocessed,
            Err(error) => {
                let mut statistics = ConstructionStatistics::default();
                statistics
                    .telemetry
                    .record_construction_preprocessing_timing(duration_nanos_saturating(
                        preprocessing_started.elapsed(),
                    ));
                return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                    error,
                    statistics,
                });
            }
        };
        let preprocessing_nanos = duration_nanos_saturating(preprocessing_started.elapsed());
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
                            batch_repair_policy,
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
                            batch_repair_policy,
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
                batch_repair_policy,
                use_global_repair_fallback,
            )
        };

        match build_with_vertices(primary_vertices) {
            Ok((dt, mut stats)) => {
                stats
                    .telemetry
                    .record_construction_preprocessing_timing(preprocessing_nanos);
                Ok((dt, stats))
            }
            Err(mut primary_err) => {
                let Some(fallback) = fallback_vertices else {
                    primary_err
                        .statistics
                        .telemetry
                        .record_construction_preprocessing_timing(preprocessing_nanos);
                    return Err(primary_err);
                };

                match build_with_vertices(fallback) {
                    Ok((dt, stats)) => {
                        let mut aggregate = primary_err.statistics;
                        aggregate.merge_from(&stats);
                        aggregate
                            .telemetry
                            .record_construction_preprocessing_timing(preprocessing_nanos);
                        Ok((dt, aggregate))
                    }
                    Err(fallback_err) => {
                        let mut aggregate = primary_err.statistics;
                        aggregate.merge_from(&fallback_err.statistics);
                        aggregate
                            .telemetry
                            .record_construction_preprocessing_timing(preprocessing_nanos);
                        Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error: fallback_err.error,
                            statistics: aggregate,
                        })
                    }
                }
            }
        }
    }

    /// Applies deduplication, insertion ordering, and initial-simplex selection
    /// before any topology is created.
    #[expect(
        clippy::too_many_lines,
        reason = "preprocessing keeps dedup, ordering, and initial-simplex selection in one boundary"
    )]
    pub(crate) fn preprocess_vertices_for_construction(
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
        let mut grid =
            hash_grid_from_validated_cell_size::<K::Scalar, D, usize>(grid_cell_size_value)?;

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
                dedup_policy != DedupPolicy::Off,
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
            InitialSimplexStrategy::MaxVolume => {
                let base = owned_vertices.unwrap_or_else(|| vertices.to_vec());
                if let Some(indices) = select_max_volume_simplex_indices(&base) {
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
    pub(crate) const fn is_non_retryable_construction_error(
        err: &DelaunayTriangulationConstructionError,
    ) -> bool {
        matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::Tds {
                    reason: TdsConstructionFailure::DuplicateUuid { .. }
                        | TdsConstructionFailure::Validation { .. },
                } | DelaunayConstructionFailure::InternalInconsistency { .. }
                    | DelaunayConstructionFailure::UnsupportedPeriodicDimension { .. }
                    | DelaunayConstructionFailure::SpatialIndexConstruction { .. }
                    | DelaunayConstructionFailure::DelaunayRepair { .. }
                    | DelaunayConstructionFailure::InsertionTopologyValidation { .. }
                    | DelaunayConstructionFailure::LocalRepairBudgetExceeded { .. }
                    | DelaunayConstructionFailure::FinalTopologyValidation { .. }
                    | DelaunayConstructionFailure::FinalDelaunayValidation { .. },
            )
        )
    }

    /// Identifies D≥4 local-repair failures that can safely try escalation and
    /// then enter the bounded soft-fail path.
    pub(crate) const fn can_soft_fail(repair_err: &DelaunayRepairError) -> bool {
        matches!(
            repair_err,
            DelaunayRepairError::NonConvergent { .. }
                | DelaunayRepairError::PostconditionFailed { .. }
        )
    }

    /// Converts non-soft-fail local-repair errors into construction failures so
    /// the bulk loop does not canonicalize or keep mutating after unexpected
    /// topology/flip failures.
    pub(crate) fn map_hard_repair_error(
        index: usize,
        repair_err: DelaunayRepairError,
    ) -> DelaunayTriangulationConstructionError {
        let message =
            format!("per-insertion Delaunay repair failed at index {index}: {repair_err}");
        if is_geometric_repair_error(&repair_err) {
            TriangulationConstructionError::GeometricDegeneracy { message }.into()
        } else {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::DelaunayRepair {
                    phase: DelaunayConstructionRepairPhase::BatchLocal { index },
                    source: Box::new(repair_err),
                },
            )
        }
    }

    pub(crate) fn map_completion_repair_error(
        message: String,
        repair_error: DelaunayRepairError,
    ) -> DelaunayTriangulationConstructionError {
        if is_geometric_repair_error(&repair_error) {
            TriangulationConstructionError::GeometricDegeneracy { message }.into()
        } else {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::DelaunayRepair {
                    phase: DelaunayConstructionRepairPhase::Completion,
                    source: Box::new(repair_error),
                },
            )
        }
    }

    /// Map an [`InsertionError`] from post-construction orientation canonicalization
    /// into a [`TriangulationConstructionError`].
    ///
    /// Structural / data-structure errors (missing simplices, broken invariants) become
    /// [`InternalInconsistency`](TriangulationConstructionError::InternalInconsistency)
    /// because they indicate algorithmic bugs rather than bad input geometry.
    /// Geometry-related failures (degenerate predicates, conflict regions, etc.) become
    /// [`GeometricDegeneracy`](TriangulationConstructionError::GeometricDegeneracy).
    ///
    /// NOTE: This match is intentionally exhaustive over `InsertionError`.
    /// When adding new variants, decide whether the failure mode is an internal
    /// bug or an input-geometry problem.
    pub(crate) fn map_orientation_canonicalization_error(
        error: InsertionError,
    ) -> TriangulationConstructionError {
        match error {
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
            // adjacent simplices) lands here rather than in the geometry arm above. After
            // normalize_coherent_orientation() BFS, a surviving violation would mean the
            // normalization algorithm failed its post-condition — an internal bug, not
            // bad input geometry. DegenerateOrientation / NegativeOrientation capture
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
            InsertionError::SpatialIndexConstruction { reason } => {
                TriangulationConstructionError::SpatialIndexConstruction { reason }
            }
            InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            } => TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            },
            InsertionError::DelaunayRepairFailed { source, context } => {
                let message = format!(
                    "Failed to canonicalize orientation after post-construction repair: \
                     Delaunay repair failed ({context}): {source}"
                );
                if is_geometric_repair_error(&source) {
                    TriangulationConstructionError::GeometricDegeneracy { message }
                } else {
                    TriangulationConstructionError::InternalInconsistency { message }
                }
            }
            // Geometry-related failures (degenerate input, predicate issues).
            error @ (InsertionError::ConflictRegion(_)
            | InsertionError::Location(_)
            | InsertionError::NonManifoldTopology { .. }
            | InsertionError::HullExtension { .. }
            | InsertionError::DelaunayValidationFailed { .. }
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
    pub(crate) fn map_insertion_error(error: InsertionError) -> TriangulationConstructionError {
        match error {
            InsertionError::CavityFilling { reason } => {
                TriangulationConstructionError::InsertionCavityFilling { source: reason }
            }
            InsertionError::NeighborWiring { reason } => {
                TriangulationConstructionError::InternalInconsistency {
                    message: format!("Neighbor wiring failed: {reason}"),
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
            InsertionError::DelaunayRepairFailed { source, context } => {
                let message = format!("Delaunay repair failed ({context}): {source}");
                if is_geometric_repair_error(&source) {
                    TriangulationConstructionError::GeometricDegeneracy { message }
                } else {
                    TriangulationConstructionError::InternalInconsistency { message }
                }
            }

            InsertionError::ConflictRegion(source) => {
                TriangulationConstructionError::InsertionConflictRegion { source }
            }
            InsertionError::Location(source) => {
                TriangulationConstructionError::InsertionLocation { source }
            }
            InsertionError::NonManifoldTopology {
                facet_hash,
                simplex_count,
            } => TriangulationConstructionError::InsertionNonManifoldTopology {
                facet_hash,
                simplex_count,
            },
            InsertionError::HullExtension { reason } => {
                TriangulationConstructionError::InsertionHullExtension { reason }
            }
            InsertionError::DelaunayValidationFailed { source } => {
                TriangulationConstructionError::InsertionDelaunayValidation { source }
            }
            InsertionError::TopologyValidationFailed { message, source } => {
                TriangulationConstructionError::InsertionTopologyValidation { message, source }
            }
            InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            } => TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            },
            InsertionError::SpatialIndexConstruction { reason } => {
                TriangulationConstructionError::SpatialIndexConstruction { reason }
            }
        }
    }

    /// Avoids retry work when construction has no incremental phase to reorder.
    pub(crate) const fn should_retry_construction(vertices: &[Vertex<K::Scalar, U, D>]) -> bool {
        D >= 2 && vertices.len() > D + 1
    }

    /// Derives an input-order-independent seed so shuffled retries are
    /// reproducible for the same vertex set.
    pub(crate) fn construction_shuffle_seed(vertices: &[Vertex<K::Scalar, U, D>]) -> u64 {
        let mut vertex_hashes = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let mut hasher = FastHasher::default();
            vertex.hash(&mut hasher);
            vertex_hashes.push(hasher.finish());
        }
        vertex_hashes.sort_unstable();
        stable_hash_u64_slice(&vertex_hashes)
    }

    /// Keeps construction retry shuffling deterministic for diagnostics and tests.
    pub(crate) fn shuffle_vertices(vertices: &mut [Vertex<K::Scalar, U, D>], seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        vertices.shuffle(&mut rng);
    }
}

/// Reads the optional batch-construction progress cadence from the environment.
///
/// `DELAUNAY_BULK_PROGRESS_EVERY` is the canonical knob. The large-scale debug
/// harness also reuses `DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY` so manual runs can
/// request periodic progress without additional wiring.
pub(crate) fn bulk_progress_every_from_env() -> Option<usize> {
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

/// Converts a measured duration to nanoseconds while saturating pathological
/// values that exceed the public telemetry counter width.
pub(crate) fn duration_nanos_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Snapshot of one batch-construction progress sample.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchProgressSample {
    pub(crate) bulk_processed: usize,
    pub(crate) bulk_inserted: usize,
    pub(crate) bulk_skipped: usize,
    pub(crate) simplex_count: usize,
    pub(crate) perturbation_seed: u64,
}

/// Rolling state used to compute periodic batch throughput summaries.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchProgressState {
    pub(crate) input_vertices: usize,
    pub(crate) initial_simplex_vertices: usize,
    pub(crate) bulk_vertices: usize,
    pub(crate) progress_every: usize,
    pub(crate) started: Instant,
    pub(crate) last_progress: Instant,
    pub(crate) last_processed: usize,
}

/// Emits periodic batch-construction progress for long-running release-mode
/// investigations such as the 4D large-scale debug harness.
///
/// Progress is emitted via `tracing::debug!`; enable with `RUST_LOG=debug` (the
/// large-scale debug harness wires this up automatically when
/// `DELAUNAY_BULK_PROGRESS_EVERY` is set).
pub(crate) fn log_bulk_progress_if_due(
    sample: BatchProgressSample,
    state: &mut Option<BatchProgressState>,
) {
    let Some(state) = state.as_mut() else {
        return;
    };
    if sample.bulk_processed == 0 {
        return;
    }

    // Always log the final sample, even when the total is not an exact multiple of the
    // requested cadence, so interrupted runs still end with a complete progress line.
    let should_log = sample.bulk_processed == state.bulk_vertices
        || sample.bulk_processed.is_multiple_of(state.progress_every);
    if !should_log {
        return;
    }

    let elapsed = state.started.elapsed();
    let chunk_elapsed = state.last_progress.elapsed();
    let chunk_processed = sample.bulk_processed.saturating_sub(state.last_processed);

    let overall_rate = safe_usize_to_scalar::<f64>(sample.bulk_processed)
        .ok()
        .map(|processed| processed / elapsed.as_secs_f64().max(1e-9));
    let chunk_rate = safe_usize_to_scalar::<f64>(chunk_processed)
        .ok()
        .map(|processed| processed / chunk_elapsed.as_secs_f64().max(1e-9));

    tracing::debug!(
        target: "delaunay::bulk_progress",
        perturbation_seed = format_args!("0x{:X}", sample.perturbation_seed),
        input_vertices = state.input_vertices,
        initial_simplex_vertices = state.initial_simplex_vertices,
        bulk_processed = sample.bulk_processed,
        bulk_vertices = state.bulk_vertices,
        bulk_inserted = sample.bulk_inserted,
        bulk_skipped = sample.bulk_skipped,
        simplices = sample.simplex_count,
        elapsed = ?elapsed,
        total_rate_pts_per_s = ?overall_rate,
        recent_rate_pts_per_s = ?chunk_rate,
        "bulk-construction progress"
    );

    state.last_progress = Instant::now();
    state.last_processed = sample.bulk_processed;
}

/// Emits retry-boundary events for release-mode large-scale construction runs.
pub(crate) fn log_construction_retry_start(
    attempt: usize,
    attempt_seed: u64,
    perturbation_seed: u64,
) {
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
pub(crate) fn log_construction_retry_result(
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
            simplices_removed_total = stats.simplices_removed_total,
            simplices_removed_max = stats.simplices_removed_max,
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

/// Enables release-visible retry-boundary tracing for bulk construction.
fn construction_retry_trace_enabled() -> bool {
    bulk_progress_every_from_env().is_some()
        || env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some()
        || env::var_os("DELAUNAY_INSERT_TRACE").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairVerificationContext, FlipContextError,
        FlipPredicateError, FlipPredicateOperation, RepairQueueOrder,
    };
    use crate::core::algorithms::incremental_insertion::{
        DelaunayRepairFailureContext, HullExtensionReason, NeighborWiringError,
        SpatialIndexConstructionFailure,
    };
    use crate::core::algorithms::locate::{ConflictError, LocateError};
    use crate::core::tds::{EntityKind, GeometricError, InvariantError, SimplexKey, VertexKey};
    use crate::core::validation::{
        TopologyGuarantee, TriangulationValidationError, ValidationPolicy,
    };
    use crate::core::vertex::VertexBuilder;
    use crate::diagnostics::BatchLocalRepairTrigger;
    use crate::geometry::coordinate_range::{CoordinateRangeError, CoordinateRangeOrdering};
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel, RobustKernel};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{
        CoordinateConversionError, CoordinateConversionValue,
    };
    use crate::geometry::util::RandomPointGenerationError;
    use crate::repair::DelaunayRepairPolicy;
    use crate::topology::characteristics::euler::TopologyClassification;
    use crate::validation::DelaunayTriangulationValidationError;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::num::NonZeroUsize;
    use std::sync::Once;
    use std::time::Instant;
    use uuid::Uuid;

    type TestDelaunay<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

    #[test]
    fn test_random_point_generation_error_variant_preserved() {
        let source = RandomPointGenerationError::InvalidCoordinateRange {
            source: CoordinateRangeError::NonIncreasing {
                ordering: CoordinateRangeOrdering::Decreasing,
                min: 5.0,
                max: 1.0,
            },
        };
        let err = DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::RandomPointGeneration {
                source: source.clone(),
            },
        );

        assert!(!matches!(
            &err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { .. }
            )
        ));

        let mut current_source = std::error::Error::source(&err);
        let mut preserved_source = None;
        while let Some(source_error) = current_source {
            if let Some(random_source) = source_error.downcast_ref::<RandomPointGenerationError>() {
                preserved_source = Some(random_source);
                break;
            }
            current_source = source_error.source();
        }
        let preserved_source =
            preserved_source.expect("RandomPointGeneration should preserve its typed source");
        assert_eq!(preserved_source, &source);
    }

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

    #[test]
    fn test_finalize_bulk_construction_validates_pseudomanifold_topology() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
            vertex!([2.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::Pseudomanifold,
        )
        .unwrap();
        assert!(
            dt.tri.tds.number_of_simplices() > 1,
            "regression setup needs an interior facet"
        );

        dt.tri.tds.clear_all_neighbors();

        let err = dt
            .finalize_bulk_construction(
                ValidationPolicy::OnSuspicion,
                DelaunayRepairPolicy::Never,
                false,
                DelaunayRepairPolicy::Never,
                &[],
                &[],
                None,
            )
            .unwrap_err();

        assert_matches!(
            err,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::FinalTopologyValidation { .. }
            )
        );
    }

    macro_rules! test_incremental_insertion {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_incremental_insertion_ $dim d>]() {
                    init_tracing();
                    let mut vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];
                    vertices.push(vertex!($interior_point));

                    let expected_vertices = vertices.len();
                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    assert_eq!(dt.number_of_vertices(), expected_vertices);
                    assert!(dt.number_of_simplices() > 1);
                }

                #[test]
                fn [<test_bootstrap_from_empty_ $dim d>]() {
                    init_tracing();
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty();
                    assert_eq!(dt.number_of_vertices(), 0);
                    assert_eq!(dt.number_of_simplices(), 0);

                    let vertices = vec![$(vertex!($simplex_coords)),+];
                    assert_eq!(vertices.len(), $dim + 1);

                    for (i, vertex) in vertices.iter().take($dim).enumerate() {
                        dt.insert(*vertex).unwrap();
                        assert_eq!(dt.number_of_vertices(), i + 1);
                        assert_eq!(dt.number_of_simplices(), 0);
                    }

                    dt.insert(*vertices.last().unwrap()).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 1);
                    assert_eq!(dt.number_of_simplices(), 1);
                    assert!(dt.is_valid().is_ok());
                }

                #[test]
                fn [<test_bootstrap_continues_with_cavity_ $dim d>]() {
                    init_tracing();
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty();
                    let initial_vertices = vec![$(vertex!($simplex_coords)),+];

                    for vertex in &initial_vertices {
                        dt.insert(*vertex).unwrap();
                    }
                    assert_eq!(dt.number_of_simplices(), 1);

                    dt.insert(vertex!($interior_point)).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 2);
                    assert!(dt.number_of_simplices() > 1);
                    assert!(dt.is_valid().is_ok());
                }

                #[test]
                fn [<test_bootstrap_equivalent_to_batch_ $dim d>]() {
                    init_tracing();
                    let vertices = vec![$(vertex!($simplex_coords)),+];

                    let mut dt_bootstrap: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty();
                    for vertex in &vertices {
                        dt_bootstrap.insert(*vertex).unwrap();
                    }

                    let dt_batch: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    assert_eq!(dt_bootstrap.number_of_vertices(), dt_batch.number_of_vertices());
                    assert_eq!(
                        dt_bootstrap.number_of_simplices(),
                        dt_batch.number_of_simplices()
                    );
                    assert!(dt_bootstrap.is_valid().is_ok());
                    assert!(dt_batch.is_valid().is_ok());
                }
            }
        };
    }

    test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);

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
        assert_eq!(dt.number_of_simplices(), 1);
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
        assert_eq!(dt.number_of_simplices(), 1);
    }

    #[test]
    fn test_with_kernel_insufficient_vertices_2d() {
        init_tracing();
        let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0])];

        let result: Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices);

        match result.unwrap_err() {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::InsufficientVertices { dimension, .. },
            ) => assert_eq!(dimension, 2),
            other => panic!("Expected InsufficientVertices error, got {other:?}"),
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

        match result.unwrap_err() {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::InsufficientVertices { dimension, .. },
            ) => assert_eq!(dimension, 3),
            other => panic!("Expected InsufficientVertices error, got {other:?}"),
        }
    }

    #[test]
    fn test_with_kernel_aborts_on_duplicate_uuid_in_insertion_loop() {
        init_tracing();
        let mut vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
            vertex!([0.25, 0.25]),
        ];

        let dup_uuid = vertices[0].uuid();
        vertices[3].set_uuid(dup_uuid).unwrap();

        let result: Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&AdaptiveKernel::new(), &vertices);

        match result.unwrap_err() {
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::Tds {
                    reason: TdsConstructionFailure::DuplicateUuid { entity: _, uuid },
                },
            ) => assert_eq!(uuid, dup_uuid),
            other => panic!("Expected DuplicateUuid error, got {other:?}"),
        }
    }

    #[test]
    fn test_batch_3d_construction_with_extra_vertex_triggers_incremental_repair() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.3, 0.3, 0.3]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

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
        let (dt, stats) =
            DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_options_and_statistics(
                &kernel,
                &vertices,
                TopologyGuarantee::DEFAULT,
                ConstructionOptions::default(),
            )
            .expect(
                "D>=4 stats construction should continue after forced local repair non-convergence",
            );

        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert_eq!(stats.inserted, vertices.len());
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_batch_4d_every_n_repair_cadence_runs_with_pending_seeds() {
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

        test_hooks::reset_batch_local_repair_calls();
        let _guard = ForceRepairNonconvergentGuard::enable();
        let kernel = RobustKernel::<f64>::new();
        let options = ConstructionOptions::default()
            .with_batch_repair_policy(DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap()));
        let (dt, stats) =
            DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_options_and_statistics(
                &kernel,
                &vertices,
                TopologyGuarantee::DEFAULT,
                options,
            )
            .expect("EveryN batch repair should soft-fail forced local non-convergence and finish");

        assert_eq!(dt.number_of_vertices(), vertices.len());
        assert_eq!(stats.inserted, vertices.len());
        assert_eq!(test_hooks::batch_local_repair_calls(), 1);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn construction_options_default_uses_batch_repair_cadence() {
        assert_eq!(
            ConstructionOptions::default().initial_simplex_strategy(),
            InitialSimplexStrategy::MaxVolume
        );
        assert_eq!(
            ConstructionOptions::default().batch_repair_policy(),
            DelaunayRepairPolicy::EveryInsertion
        );
        assert_eq!(
            DelaunayRepairPolicy::default(),
            DelaunayRepairPolicy::EveryInsertion
        );
    }

    #[test]
    fn construction_options_builder_roundtrip() {
        let opts = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_dedup_policy(DedupPolicy::Exact)
            .with_batch_repair_policy(DelaunayRepairPolicy::EveryN(NonZeroUsize::new(4).unwrap()))
            .with_retry_policy(RetryPolicy::Disabled);

        assert_eq!(opts.insertion_order(), InsertionOrderStrategy::Input);
        assert_eq!(opts.dedup_policy(), DedupPolicy::Exact);
        assert_eq!(
            opts.batch_repair_policy(),
            DelaunayRepairPolicy::EveryN(NonZeroUsize::new(4).unwrap())
        );
        assert_eq!(opts.retry_policy(), RetryPolicy::Disabled);
    }

    #[test]
    fn construction_options_global_repair_fallback_toggle() {
        let default_opts = ConstructionOptions::default();
        assert!(default_opts.use_global_repair_fallback);

        let disabled_opts = default_opts.without_global_repair_fallback();
        assert!(!disabled_opts.use_global_repair_fallback);

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
    fn construction_statistics_record_insertion_tracks_inserted_common_fields() {
        let mut summary = ConstructionStatistics::default();
        let stats = InsertionStatistics {
            attempts: 3,
            simplices_removed_during_repair: 4,
            result: InsertionResult::Inserted,
        };

        summary.record_insertion(&stats);

        assert_eq!(summary.inserted, 1);
        assert_eq!(summary.skipped_duplicate, 0);
        assert_eq!(summary.skipped_degeneracy, 0);
        assert_eq!(summary.total_attempts, 3);
        assert_eq!(summary.max_attempts, 3);
        assert_eq!(summary.attempts_histogram.get(3).copied().unwrap_or(0), 1);
        assert_eq!(summary.used_perturbation, 1);
        assert_eq!(summary.simplices_removed_total, 4);
        assert_eq!(summary.simplices_removed_max, 4);
        assert_eq!(stats.attempts, 3);
        assert_matches!(stats.result, InsertionResult::Inserted);
    }

    #[test]
    fn construction_statistics_record_insertion_tracks_skipped_variants() {
        let mut summary = ConstructionStatistics::default();
        let skipped_duplicate = InsertionStatistics {
            attempts: 1,
            simplices_removed_during_repair: 0,
            result: InsertionResult::SkippedDuplicate,
        };
        let skipped_degeneracy = InsertionStatistics {
            attempts: 2,
            simplices_removed_during_repair: 5,
            result: InsertionResult::SkippedDegeneracy,
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
        assert_eq!(summary.simplices_removed_total, 5);
        assert_eq!(summary.simplices_removed_max, 5);
    }

    #[test]
    fn construction_statistics_record_skip_sample_caps_at_eight_samples() {
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
                coords_available: true,
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
    fn construction_statistics_records_slowest_insertion_samples() {
        let mut summary = ConstructionStatistics::default();
        for index in 0..10 {
            let sample_index_u32 = u32::try_from(index).unwrap();
            summary.record_slow_insertion_sample(ConstructionSlowInsertionSample {
                index,
                uuid: Uuid::from_u128(
                    <u128 as std::convert::From<u32>>::from(sample_index_u32) + 1,
                ),
                attempts: 1,
                result: InsertionResult::Inserted,
                elapsed_nanos: <u64 as std::convert::From<u32>>::from(sample_index_u32) * 1_000,
                simplices_after: index,
                locate_calls: 1,
                locate_walk_steps_total: index,
                conflict_region_calls: 1,
                conflict_region_simplices_total: index,
                cavity_insertion_calls: 1,
                global_conflict_scans: 0,
                hull_extension_calls: 0,
                topology_validation_calls: 1,
            });
        }

        assert_eq!(summary.slow_insertions.len(), 8);
        assert_eq!(summary.slow_insertions.first().map(|s| s.index), Some(9));
        assert_eq!(summary.slow_insertions.last().map(|s| s.index), Some(2));
        assert!(
            summary
                .slow_insertions
                .windows(2)
                .all(|pair| pair[0].elapsed_nanos >= pair[1].elapsed_nanos)
        );
    }

    #[test]
    fn log_bulk_progress_if_due_updates_progress_state_only_when_due() {
        let sample = BatchProgressSample {
            bulk_processed: 5,
            bulk_inserted: 4,
            bulk_skipped: 1,
            simplex_count: 7,
            perturbation_seed: 0xCAFE,
        };

        let mut disabled = None;
        log_bulk_progress_if_due(sample, &mut disabled);
        assert!(disabled.is_none());

        let mut state = Some(BatchProgressState {
            input_vertices: 13,
            initial_simplex_vertices: 3,
            bulk_vertices: 10,
            progress_every: 5,
            started: Instant::now(),
            last_progress: Instant::now(),
            last_processed: 0,
        });

        log_bulk_progress_if_due(
            BatchProgressSample {
                bulk_processed: 0,
                ..sample
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 0);

        log_bulk_progress_if_due(
            BatchProgressSample {
                bulk_processed: 3,
                ..sample
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 0);

        log_bulk_progress_if_due(sample, &mut state);
        assert_eq!(state.as_ref().unwrap().last_processed, 5);

        log_bulk_progress_if_due(
            BatchProgressSample {
                bulk_processed: 10,
                bulk_inserted: 8,
                bulk_skipped: 2,
                simplex_count: 11,
                perturbation_seed: 0xBEEF,
            },
            &mut state,
        );
        assert_eq!(state.as_ref().unwrap().last_processed, 10);
    }

    #[test]
    fn test_vertex_coords_f64_converts_f64_vertex_coords() {
        init_tracing();
        let vertex: Vertex<f64, (), 3> = vertex!([1.25, -2.5, 3.75]);

        assert_eq!(vertex_coords_f64(&vertex), Some(vec![1.25, -2.5, 3.75]));
    }

    #[test]
    fn test_vertex_coords_f64_rejects_non_finite_coords() {
        init_tracing();
        let nan_vertex: Vertex<f64, (), 3> = VertexBuilder::default()
            .point(Point::new([1.0, f64::NAN, 3.0]))
            .build()
            .unwrap();
        let infinite_vertex: Vertex<f64, (), 3> = VertexBuilder::default()
            .point(Point::new([1.0, f64::INFINITY, 3.0]))
            .build()
            .unwrap();

        assert_eq!(vertex_coords_f64(&nan_vertex), None);
        assert_eq!(vertex_coords_f64(&infinite_vertex), None);
    }

    fn coord_sequence_2d(vertices: &[Vertex<f64, (), 2>]) -> Vec<[f64; 2]> {
        vertices.iter().map(|v| *v.point().coords()).collect()
    }

    #[test]
    fn order_vertices_input_preserves_order() {
        init_tracing();
        let vertices = vec![
            vertex!([2.0, 0.0]),
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
        ];
        let expected = coord_sequence_2d(&vertices);

        let ordered = order_vertices_by_strategy(vertices, InsertionOrderStrategy::Input, false);

        assert_eq!(coord_sequence_2d(&ordered), expected);
    }

    #[test]
    fn preprocess_hilbert_with_dedup_off_preserves_duplicate_vertices() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 0.0]),
        ];

        let preprocess = TestDelaunay::<2>::preprocess_vertices_for_construction(
            &vertices,
            DedupPolicy::Off,
            InsertionOrderStrategy::Hilbert,
            InitialSimplexStrategy::First,
        )
        .expect("preprocessing with dedup off should preserve duplicate vertices");

        assert_eq!(preprocess.primary_slice(&vertices).len(), vertices.len());
    }

    #[test]
    fn dedup_exact_sorted_without_grid() {
        init_tracing();
        let vertices = vec![
            vertex!([1.0, 0.0]),
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let unique = dedup_vertices_exact_sorted(vertices);

        assert_eq!(
            coord_sequence_2d(&unique),
            vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        );
    }

    #[test]
    fn dedup_exact_grid_fallback() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 0.0]),
        ];
        let mut grid = HashGridIndex::<f64, 2, usize>::try_new(1.0e-10).unwrap();

        let unique = dedup_vertices_exact_hash_grid(vertices, &mut grid);

        assert_eq!(coord_sequence_2d(&unique), vec![[0.0, 0.0], [1.0, 0.0]]);

        let vertices_6d = vec![
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let mut unusable_grid = HashGridIndex::<f64, 6, usize>::try_new(1.0e-10).unwrap();

        let fallback_unique = dedup_vertices_exact_hash_grid(vertices_6d, &mut unusable_grid);

        assert_eq!(fallback_unique.len(), 1);
    }

    #[test]
    fn epsilon_dedup_quantized_paths() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.09, 0.0]),
            vertex!([0.25, 0.0]),
        ];

        let unique = dedup_vertices_epsilon_quantized(vertices, 0.1);

        assert_eq!(coord_sequence_2d(&unique), vec![[0.0, 0.0], [0.25, 0.0]]);

        let zero_epsilon_vertices = vec![vertex!([0.0, 0.0]), vertex!([0.0, 0.0])];
        let zero_epsilon_unique = dedup_vertices_epsilon_quantized(zero_epsilon_vertices, 0.0);
        assert_eq!(zero_epsilon_unique.len(), 2);

        let nonfinite_vertices = vec![
            vertex!([0.0, 0.0]),
            Vertex::new_with_uuid(Point::new([f64::NAN, 0.0]), Uuid::new_v4(), None),
        ];
        let nonfinite_unique = dedup_vertices_epsilon_quantized(nonfinite_vertices, 0.1);
        assert_eq!(nonfinite_unique.len(), 2);

        let vertices_6d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let fallback_unique = dedup_vertices_epsilon_quantized(vertices_6d, 0.1);
        assert_eq!(fallback_unique.len(), 1);
    }

    #[test]
    fn dedup_epsilon_grid_fallback() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.05, 0.0]),
            vertex!([0.25, 0.0]),
        ];
        let mut grid = HashGridIndex::<f64, 2, usize>::try_new(0.1).unwrap();

        let unique = dedup_vertices_epsilon_hash_grid(vertices, 0.1, &mut grid);

        assert_eq!(coord_sequence_2d(&unique), vec![[0.0, 0.0], [0.25, 0.0]]);

        let fallback_vertices = vec![vertex!([0.0, 0.0]), vertex!([0.05, 0.0])];
        let mut unusable_grid = HashGridIndex::<f64, 2, usize>::try_new(0.1).unwrap();
        unusable_grid.remove_vertex(&0, &[f64::NAN, 0.0]);

        let fallback_unique =
            dedup_vertices_epsilon_hash_grid(fallback_vertices, 0.1, &mut unusable_grid);

        assert_eq!(fallback_unique.len(), 1);
    }

    #[test]
    fn preprocess_falls_back_when_grid_unusable() {
        init_tracing();
        let exact_vertices = vec![
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        let exact = TestDelaunay::<6>::preprocess_vertices_for_construction(
            &exact_vertices,
            DedupPolicy::Exact,
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::First,
        )
        .unwrap();

        assert_eq!(exact.primary_slice(&exact_vertices).len(), 2);
        assert!(exact.grid_cell_size().is_none());

        let epsilon_vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        let epsilon = TestDelaunay::<6>::preprocess_vertices_for_construction(
            &epsilon_vertices,
            DedupPolicy::Epsilon { tolerance: 0.1 },
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::First,
        )
        .unwrap();

        assert_eq!(epsilon.primary_slice(&epsilon_vertices).len(), 2);
        assert!(epsilon.grid_cell_size().is_none());
    }

    #[test]
    fn preprocess_zero_epsilon_keeps_base() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
        ];

        let preprocess = TestDelaunay::<3>::preprocess_vertices_for_construction(
            &vertices,
            DedupPolicy::Epsilon { tolerance: 0.0 },
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::Balanced,
        )
        .unwrap();

        assert!(preprocess.grid_cell_size().is_some());
        assert_eq!(preprocess.primary_slice(&vertices).len(), vertices.len());
        assert!(preprocess.fallback_slice().is_none());
    }

    #[test]
    fn hash_grid_from_validated_cell_size_rejects_invalid_internal_tolerance() {
        init_tracing();
        let error = hash_grid_from_validated_cell_size::<f64, 3, usize>(0.0)
            .expect_err("invalid internal hash-grid tolerance should fail");

        assert_matches!(
            error,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::SpatialIndexConstruction {
                    reason: SpatialIndexConstructionFailure::NonPositiveCellSize { value }
                }
            ) if value == CoordinateConversionValue::from_f64(0.0)
        );
    }

    #[test]
    fn quantize_and_neighbor_edges() {
        init_tracing();
        assert_eq!(quantize_coords(&[0.25, -0.25], 10.0), Some([2, -3]));
        assert_eq!(quantize_coords(&[f64::NAN, 0.0], 10.0), None);
        assert_eq!(quantize_coords(&[1.0e308, 0.0], 1.0e308), None);

        let mut visited = Vec::new();
        let mut current = [0_i64, 0_i64];
        let completed = visit_quantized_neighbors(0, &[4, 7], &mut current, &mut |neighbor| {
            visited.push(neighbor);
            visited.len() < 4
        });

        assert!(!completed);
        assert_eq!(visited.len(), 4);
    }

    #[test]
    fn hilbert_fallback_for_nonfinite_coords() {
        init_tracing();
        let vertices = vec![
            vertex!([1.0, 0.0]),
            Vertex::new_with_uuid(Point::new([f64::NAN, 0.0]), Uuid::new_v4(), None),
            vertex!([0.0, 0.0]),
        ];

        let ordered = order_vertices_hilbert(vertices, true);

        assert_eq!(ordered.len(), 3);
        assert!(
            ordered.iter().any(|v| v.point().coords()[0].is_nan()),
            "fallback ordering should preserve the non-finite vertex"
        );
        assert!(
            ordered
                .iter()
                .any(|v| coords_equal_exact(v.point().coords(), &[0.0, 0.0]))
        );
        assert!(
            ordered
                .iter()
                .any(|v| coords_equal_exact(v.point().coords(), &[1.0, 0.0]))
        );
    }

    #[test]
    fn hilbert_fallback_for_unsupported_dim() {
        init_tracing();
        let vertices = vec![vertex!([1.0; 17]), vertex!([0.0; 17])];

        let ordered = order_vertices_hilbert(vertices, true);

        assert!(coords_equal_exact(ordered[0].point().coords(), &[0.0; 17]));
        assert!(coords_equal_exact(ordered[1].point().coords(), &[1.0; 17]));
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
            Vertex::new_with_uuid(Point::new([f64::NAN, 0.0, 0.0]), Uuid::new_v4(), None),
        ];

        let result = select_balanced_simplex_indices(&vertices);
        assert!(result.is_none());
    }

    macro_rules! max_volume_axis_simplex_test {
        ($test_name:ident, $dimension:literal, [$($coords:expr),+ $(,)?], [$($expected_idx:expr),+ $(,)?]) => {
            #[test]
            fn $test_name() {
                init_tracing();
                let vertices: Vec<Vertex<f64, (), $dimension>> = vec![$(vertex!($coords)),+];

                let result = select_max_volume_simplex_indices(&vertices)
                    .expect("max-volume simplex selection failed");
                let expected_indices = [$($expected_idx),+];

                assert_eq!(result.len(), expected_indices.len());
                for expected_idx in expected_indices {
                    assert!(
                        result.contains(&expected_idx),
                        "expected selected simplex {result:?} to contain vertex index {expected_idx}"
                    );
                }
            }
        };
    }

    max_volume_axis_simplex_test!(
        test_select_max_volume_simplex_indices_prefers_largest_triangle_2d,
        2,
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [1.0, 1.0],
        ],
        [0, 3, 4]
    );

    max_volume_axis_simplex_test!(
        test_select_max_volume_simplex_indices_prefers_largest_tetrahedron,
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        [0, 4, 5, 6]
    );

    max_volume_axis_simplex_test!(
        test_select_max_volume_simplex_indices_prefers_largest_simplex_4d,
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ],
        [0, 5, 6, 7, 8]
    );

    max_volume_axis_simplex_test!(
        test_select_max_volume_simplex_indices_prefers_largest_simplex_5d,
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
        ],
        [0, 6, 7, 8, 9, 10]
    );

    #[test]
    fn test_select_max_volume_simplex_indices_rejects_degenerate_pool() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([3.0, 0.0, 0.0]),
        ];

        let result = select_max_volume_simplex_indices(&vertices);
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
    fn test_preprocess_vertices_for_construction_max_volume_sets_largest_simplex_first() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([10.0, 0.0, 0.0]),
            vertex!([0.0, 10.0, 0.0]),
            vertex!([0.0, 0.0, 10.0]),
        ];

        let preprocess = DelaunayTriangulation::<
            AdaptiveKernel<f64>,
            (),
            (),
            3,
        >::preprocess_vertices_for_construction(
            &vertices,
            DedupPolicy::Off,
            InsertionOrderStrategy::Input,
            InitialSimplexStrategy::MaxVolume,
        )
        .expect("preprocess failed");

        let primary = preprocess.primary_slice(&vertices);
        assert!(primary.len() >= 4);
        let first_simplex = &primary[..4];
        let first_simplex_contains = |expected_coords: [f64; 3]| {
            first_simplex.iter().any(|vertex| {
                vertex
                    .point()
                    .coords()
                    .iter()
                    .zip(expected_coords)
                    .all(|(actual, expected)| (*actual - expected).abs() <= f64::EPSILON)
            })
        };

        assert!(preprocess.fallback_slice().is_some());
        assert!(first_simplex_contains([0.0, 0.0, 0.0]));
        assert!(first_simplex_contains([10.0, 0.0, 0.0]));
        assert!(first_simplex_contains([0.0, 10.0, 0.0]));
        assert!(first_simplex_contains([0.0, 0.0, 10.0]));
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

        assert_matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { .. }
            ))
        );
    }

    #[test]
    fn stats_preprocess_error_defaults() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let options = ConstructionOptions::default().with_dedup_policy(DedupPolicy::Epsilon {
            tolerance: f64::NAN,
        });

        let error =
            DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), 3>::with_options_and_statistics(
                &AdaptiveKernel::new(),
                &vertices,
                TopologyGuarantee::PLManifold,
                options,
            )
            .expect_err("NaN epsilon should fail during preprocessing");

        assert_eq!(error.statistics.inserted, 0);
        assert_eq!(error.statistics.total_skipped(), 0);
        assert_eq!(error.statistics.total_attempts, 0);
        assert!(error.statistics.skip_samples.is_empty());
        assert_matches!(
            error.error,
            DelaunayTriangulationConstructionError::Triangulation(_)
        );
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
    fn simplex_with_interior<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut verts = simplex_vertices::<D>();
        let dimension = safe_usize_to_scalar::<f64>(D).expect("test dimensions fit in f64");
        let interior = [0.1_f64 / dimension; D];
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
        assert_eq!(dt.number_of_simplices(), 1);
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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
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
        assert!(sample.coords_available);
        assert_eq!(sample.attempts, 1);
        assert!(sample.error.contains("Duplicate coordinates"));
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
    fn test_empty_topology_guarantee_derives_validation_policy() {
        init_tracing();

        let strict: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(
                TopologyGuarantee::PLManifoldStrict,
            );
        assert_eq!(
            strict.topology_guarantee(),
            TopologyGuarantee::PLManifoldStrict
        );
        assert_eq!(strict.validation_policy(), ValidationPolicy::Always);

        let pseudomanifold: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                FastKernel::<f64>::new(),
                TopologyGuarantee::Pseudomanifold,
            );
        assert_eq!(
            pseudomanifold.topology_guarantee(),
            TopologyGuarantee::Pseudomanifold
        );
        assert_eq!(
            pseudomanifold.validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        let pl_manifold: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                FastKernel::<f64>::new(),
                TopologyGuarantee::PLManifold,
            );
        assert_eq!(
            pl_manifold.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
        assert_eq!(
            pl_manifold.validation_policy(),
            ValidationPolicy::ExplicitOnly
        );
    }

    #[test]
    fn test_empty_creates_empty_triangulation() {
        init_tracing();
        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_vertices(), 0);
        assert_eq!(dt.number_of_simplices(), 0);
        assert_eq!(dt.dim(), -1);
    }

    #[test]
    fn test_empty_supports_incremental_insertion() {
        init_tracing();
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.number_of_vertices(), 0);

        dt.insert(vertex!([0.0, 0.0])).unwrap();
        dt.insert(vertex!([1.0, 0.0])).unwrap();
        assert_eq!(dt.number_of_simplices(), 0);

        dt.insert(vertex!([0.0, 1.0])).unwrap();
        assert_eq!(dt.number_of_simplices(), 1);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "single classification table keeps soft-fail and hard-error mapping cases together"
    )]
    fn test_repair_soft_fail_classification() {
        let nonconvergent = DelaunayRepairError::NonConvergent {
            max_flips: 1000,
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
        assert!(TestDelaunay::<4>::can_soft_fail(&nonconvergent));

        let postcondition = DelaunayRepairError::PostconditionFailed {
            message: "unresolved facet".to_string(),
        };
        assert!(TestDelaunay::<4>::can_soft_fail(&postcondition));

        let flip_error =
            DelaunayRepairError::from(FlipError::UnsupportedDimension { dimension: 1 });
        assert!(!TestDelaunay::<4>::can_soft_fail(&flip_error));

        let topology_error = DelaunayRepairError::InvalidTopology {
            required: TopologyGuarantee::PLManifold,
            found: TopologyGuarantee::Pseudomanifold,
            message: "local repair requires manifold topology",
        };
        assert!(!TestDelaunay::<4>::can_soft_fail(&topology_error));

        let verification_error = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
            source: Box::new(FlipError::InvalidFlipContext {
                reason: Box::new(FlipContextError::MissingRemovedSimplexFrame),
            }),
        };
        assert!(!TestDelaunay::<4>::can_soft_fail(&verification_error));

        let canonicalization_error = DelaunayRepairError::OrientationCanonicalizationFailed {
            message: "after flip repair: broken orientation".to_string(),
        };
        assert!(!TestDelaunay::<4>::can_soft_fail(&canonicalization_error));

        let mapped_hard = TestDelaunay::<4>::map_hard_repair_error(23, flip_error);
        assert!(
            matches!(
                mapped_hard,
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::DelaunayRepair {
                        phase: DelaunayConstructionRepairPhase::BatchLocal { index: 23 },
                        ref source,
                    }
                ) if matches!(
                    source.as_ref(),
                    DelaunayRepairError::Flip {
                        source: flip_source
                    }
                        if matches!(
                            flip_source.as_ref(),
                            FlipError::UnsupportedDimension { dimension: 1 }
                        )
                )
            ),
            "deterministic hard D>=4 repair failures should stop shuffled retries: {mapped_hard:?}"
        );

        let geometric_error = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let mapped_geometric = TestDelaunay::<4>::map_hard_repair_error(24, geometric_error);
        assert!(
            matches!(
                mapped_geometric,
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::GeometricDegeneracy { ref message }
                ) if message.contains("per-insertion Delaunay repair failed at index 24")
                    && message.contains("degenerate simplex")
            ),
            "geometric hard D>=4 repair failures should remain retryable degeneracies: {mapped_geometric:?}"
        );

        let simplex_creation =
            DelaunayRepairError::from(FlipError::from(SimplexValidationError::DegenerateSimplex));
        let mapped_simplex_creation =
            TestDelaunay::<4>::map_hard_repair_error(25, simplex_creation);
        assert!(
            matches!(
                mapped_simplex_creation,
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::GeometricDegeneracy { ref message }
                ) if message.contains("per-insertion Delaunay repair failed at index 25")
                    && message.contains("Degenerate simplex")
            ),
            "geometric simplex creation failures should remain retryable degeneracies: {mapped_simplex_creation:?}"
        );

        let duplicate_simplex_creation =
            DelaunayRepairError::from(FlipError::from(SimplexValidationError::DuplicateVertices));
        assert!(
            !TestDelaunay::<4>::can_soft_fail(&duplicate_simplex_creation),
            "non-geometric simplex creation failures should remain hard repair errors"
        );

        let mapped_verification = TestDelaunay::<4>::map_hard_repair_error(26, verification_error);
        assert!(
            matches!(
                mapped_verification,
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::DelaunayRepair {
                        phase: DelaunayConstructionRepairPhase::BatchLocal { index: 26 },
                        ref source,
                    }
                ) if matches!(**source, DelaunayRepairError::VerificationFailed { .. })
            ),
            "verification context failures should stop shuffled retries: {mapped_verification:?}"
        );

        let predicate_verification = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::PredicateFailure {
                reason: Box::new(FlipPredicateError::CoordinateConversion {
                    operation: FlipPredicateOperation::K2SimplexAInSphere,
                    source: CoordinateConversionError::ConversionFailed {
                        coordinate_index: 0,
                        coordinate_value: CoordinateConversionValue::Other(
                            "in_sphere failed".to_string(),
                        ),
                        from_type: "f64",
                        to_type: "f64",
                    },
                }),
            }),
        };
        let mapped_predicate = TestDelaunay::<4>::map_hard_repair_error(27, predicate_verification);
        assert!(
            matches!(
                mapped_predicate,
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::GeometricDegeneracy { ref message }
                ) if message.contains("per-insertion Delaunay repair failed at index 27")
                    && message.contains("in_sphere failed")
            ),
            "verification predicate failures should remain geometric: {mapped_predicate:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_topology_validation_is_internal() {
        let error = InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
            message: "missing simplex".to_string(),
        });
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "TopologyValidation should map to InternalInconsistency, got: {mapped:?}"
        );
        let msg = mapped.to_string();
        assert!(
            msg.contains("missing simplex"),
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
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
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
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
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
            source: TriangulationValidationError::IsolatedVertex {
                vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                vertex_uuid: Uuid::nil(),
            },
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "IsolatedVertex should map to InternalInconsistency, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_topology_validation_failed_is_internal() {
        let error = InsertionError::TopologyValidationFailed {
            message: "post-insertion".to_string(),
            source: TriangulationValidationError::EulerCharacteristicMismatch {
                computed: 3,
                expected: 2,
                classification: TopologyClassification::Ball(3),
            },
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
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
            reason: CavityFillingError::EmptyFanTriangulation,
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_neighbor_wiring_is_internal() {
        let error = InsertionError::NeighborWiring {
            reason: NeighborWiringError::MissingSimplex {
                simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
            },
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_preserves_repair_budget() {
        let error = InsertionError::MaxSimplicesRemovedExceeded {
            max_simplices_removed: 2,
            attempted: 3,
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_duplicate_uuid_is_internal() {
        let error = InsertionError::DuplicateUuid {
            entity: EntityKind::Simplex,
            uuid: Uuid::nil(),
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InternalInconsistency { .. }
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_geometry_variants_are_degeneracy() {
        let geometry_errors: Vec<InsertionError> = vec![
            InsertionError::Location(LocateError::EmptyTriangulation),
            InsertionError::NonManifoldTopology {
                facet_hash: 0,
                simplex_count: 3,
            },
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            },
            InsertionError::DelaunayValidationFailed {
                source: DelaunayTriangulationValidationError::VerificationFailed {
                    message: "test".to_string(),
                },
            },
            InsertionError::DelaunayRepairFailed {
                source: Box::new(DelaunayRepairError::PostconditionFailed {
                    message: "test".to_string(),
                }),
                context: DelaunayRepairFailureContext::LocalRepair,
            },
            InsertionError::DuplicateCoordinates {
                coordinates: CoordinateValues::from([0.0, 0.0, 0.0]),
            },
        ];
        for error in geometry_errors {
            let label = format!("{error}");
            let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
            assert!(
                matches!(
                    mapped,
                    TriangulationConstructionError::GeometricDegeneracy { .. }
                ),
                "{label} should map to GeometricDegeneracy, got: {mapped:?}"
            );
        }
    }

    #[test]
    fn test_map_orientation_canonicalization_error_hard_repair_is_internal() {
        let error = InsertionError::DelaunayRepairFailed {
            source: Box::new(DelaunayRepairError::VerificationFailed {
                context: DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
                source: Box::new(FlipError::InvalidFlipContext {
                    reason: Box::new(FlipContextError::MissingRemovedSimplexFrame),
                }),
            }),
            context: DelaunayRepairFailureContext::OrientationCanonicalization,
        };
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { ref message }
                    if message.contains("orientation canonicalization")
                        && message.contains("removed simplex frame")
            ),
            "hard repair errors during orientation canonicalization should be internal: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_cavity_filling() {
        let error = InsertionError::CavityFilling {
            reason: CavityFillingError::EmptyFanTriangulation,
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InsertionCavityFilling {
                    source: CavityFillingError::EmptyFanTriangulation
                }
            ),
            "CavityFilling should preserve its typed construction source, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_neighbor_wiring() {
        let error = InsertionError::NeighborWiring {
            reason: NeighborWiringError::MissingSimplex {
                simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
            },
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
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
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert!(
            matches!(mapped, TriangulationConstructionError::Tds(_)),
            "TopologyValidation should map to Tds(ValidationError), got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_duplicate_uuid() {
        let error = InsertionError::DuplicateUuid {
            entity: EntityKind::Simplex,
            uuid: Uuid::nil(),
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert!(
            matches!(mapped, TriangulationConstructionError::Tds(_)),
            "DuplicateUuid should map to Tds(DuplicateUuid), got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_duplicate_coordinates() {
        let error = InsertionError::DuplicateCoordinates {
            coordinates: CoordinateValues::from([1.0, 2.0, 3.0]),
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::DuplicateCoordinates { .. }
            ),
            "DuplicateCoordinates should be preserved, got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_preserves_typed_insertion_sources() {
        let conflict = InsertionError::ConflictRegion(ConflictError::OpenBoundary {
            facet_count: 2,
            ridge_vertex_count: 1,
            open_simplex: SimplexKey::from(KeyData::from_ffi(1)),
        });
        let mapped = TestDelaunay::<3>::map_insertion_error(conflict);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionConflictRegion {
                source: ConflictError::OpenBoundary { .. },
            }
        );

        let location = InsertionError::Location(LocateError::EmptyTriangulation);
        let mapped = TestDelaunay::<3>::map_insertion_error(location);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionLocation {
                source: LocateError::EmptyTriangulation,
            }
        );

        let non_manifold = InsertionError::NonManifoldTopology {
            facet_hash: 0xab,
            simplex_count: 3,
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(non_manifold);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionNonManifoldTopology {
                facet_hash: 0xab,
                simplex_count: 3,
            }
        );

        let hull = InsertionError::HullExtension {
            reason: HullExtensionReason::NoVisibleFacets,
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(hull);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionHullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            }
        );

        let delaunay = InsertionError::DelaunayValidationFailed {
            source: DelaunayTriangulationValidationError::VerificationFailed {
                message: "test".to_string(),
            },
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(delaunay);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionDelaunayValidation { .. }
        );

        let topology = InsertionError::TopologyValidationFailed {
            message: "test".to_string(),
            source: TriangulationValidationError::EulerCharacteristicMismatch {
                computed: 3,
                expected: 2,
                classification: TopologyClassification::Ball(3),
            },
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(topology);
        assert_matches!(
            mapped,
            TriangulationConstructionError::InsertionTopologyValidation { .. }
        );
    }

    #[test]
    fn test_delaunay_construction_failure_preserves_typed_generic_sources() {
        let conflict_source = ConflictError::OpenBoundary {
            facet_count: 2,
            ridge_vertex_count: 1,
            open_simplex: SimplexKey::from(KeyData::from_ffi(1)),
        };
        let failure = DelaunayConstructionFailure::from(
            TriangulationConstructionError::InsertionConflictRegion {
                source: conflict_source,
            },
        );
        assert_matches!(
            failure,
            DelaunayConstructionFailure::InsertionConflictRegion {
                source: ConflictError::OpenBoundary { .. },
            }
        );

        let failure = DelaunayConstructionFailure::from(
            TriangulationConstructionError::InsertionHullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            },
        );
        assert_matches!(
            failure,
            DelaunayConstructionFailure::InsertionHullExtension {
                reason: HullExtensionReason::NoVisibleFacets,
            }
        );

        let failure = DelaunayConstructionFailure::from(
            TriangulationConstructionError::InsertionDelaunayValidation {
                source: DelaunayTriangulationValidationError::VerificationFailed {
                    message: "delaunay check".to_string(),
                },
            },
        );
        assert_matches!(
            failure,
            DelaunayConstructionFailure::InsertionDelaunayValidation {
                source: DelaunayTriangulationValidationError::VerificationFailed { .. },
            }
        );

        let vertex_key = VertexKey::from(KeyData::from_ffi(2));
        let failure = DelaunayConstructionFailure::from(
            TriangulationConstructionError::InsertionTopologyValidation {
                message: "post-insertion".to_string(),
                source: TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    vertex_uuid: Uuid::nil(),
                },
            },
        );
        assert_matches!(
            failure,
            DelaunayConstructionFailure::InsertionTopologyValidation {
                message,
                source: TriangulationValidationError::IsolatedVertex {
                    vertex_key: preserved_key,
                    ..
                },
            } if message == "post-insertion" && preserved_key == vertex_key
        );
    }

    #[test]
    fn test_map_insertion_error_hard_repair_is_internal() {
        let error = InsertionError::DelaunayRepairFailed {
            source: Box::new(DelaunayRepairError::from(FlipError::UnsupportedDimension {
                dimension: 1,
            })),
            context: DelaunayRepairFailureContext::LocalRepair,
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { ref message }
                    if message.contains("local repair")
                        && message.contains("Bistellar flip not supported for D=1")
            ),
            "hard repair errors during insertion should be internal: {mapped:?}"
        );
    }

    #[test]
    fn test_map_insertion_error_preserves_repair_budget() {
        let error = InsertionError::MaxSimplicesRemovedExceeded {
            max_simplices_removed: 2,
            attempted: 3,
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
        );

        let public_error: DelaunayTriangulationConstructionError = mapped.into();
        assert_matches!(
            public_error,
            DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::LocalRepairBudgetExceeded {
                    max_simplices_removed: 2,
                    attempted: 3,
                }
            )
        );
    }

    #[test]
    fn test_map_insertion_error_spatial_index_construction_is_typed() {
        let error = InsertionError::SpatialIndexConstruction {
            reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                value: CoordinateConversionValue::from_f64(0.0),
            },
        };
        let mapped = TestDelaunay::<3>::map_insertion_error(error);
        assert_matches!(
            mapped,
            TriangulationConstructionError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize { value }
            } if value == CoordinateConversionValue::from_f64(0.0)
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_orientation_violation_is_internal_inconsistency()
    {
        let error = InsertionError::TopologyValidation(TdsError::OrientationViolation {
            simplex1_key: SimplexKey::from(KeyData::from_ffi(1)),
            simplex1_uuid: Uuid::nil(),
            simplex2_key: SimplexKey::from(KeyData::from_ffi(2)),
            simplex2_uuid: Uuid::nil(),
            simplex1_facet_index: 0,
            simplex2_facet_index: 1,
            facet_vertices: vec![],
            simplex2_facet_vertices: vec![],
            observed_odd_permutation: true,
            expected_odd_permutation: false,
        });
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::InternalInconsistency { .. }
            ),
            "OrientationViolation should map to InternalInconsistency (structural invariant breach, not geometry), got: {mapped:?}"
        );
    }

    #[test]
    fn test_map_orientation_canonicalization_error_conflict_region_is_degeneracy() {
        let error = InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
            facet_hash: 0x123,
            simplex_count: 3,
        });
        let mapped = TestDelaunay::<3>::map_orientation_canonicalization_error(error);
        assert!(
            matches!(
                mapped,
                TriangulationConstructionError::GeometricDegeneracy { .. }
            ),
            "ConflictRegion should map to GeometricDegeneracy, got: {mapped:?}"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_duplicate_uuid() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::Tds(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Simplex,
                uuid: Uuid::nil(),
            })
            .into();
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&err),
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
            TestDelaunay::<3>::is_non_retryable_construction_error(&err),
            "InternalInconsistency should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_repair_budget() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
            .into();
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&err),
            "Local repair budget exhaustion should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_spatial_index_construction() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::from_f64(0.0),
                },
            }
            .into();
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&err),
            "Spatial index construction failures should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_tds_validation() {
        let err: DelaunayTriangulationConstructionError = TriangulationConstructionError::Tds(
            TdsConstructionError::ValidationError(TdsError::InconsistentDataStructure {
                message: "test".to_string(),
            }),
        )
        .into();
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&err),
            "TDS validation failures should be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_topology_validation_buckets() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(1));
        let insertion_err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::InsertionTopologyValidation {
                message: "test".to_string(),
                source: TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    vertex_uuid: Uuid::nil(),
                },
            }
            .into();
        let final_err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::FinalTopologyValidation {
                message: "test".to_string(),
                source: InvariantError::Triangulation(
                    TriangulationValidationError::IsolatedVertex {
                        vertex_key,
                        vertex_uuid: Uuid::nil(),
                    },
                )
                .into(),
            }
            .into();
        let final_delaunay_err = DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::FinalDelaunayValidation {
                source: DelaunayTriangulationValidationError::VerificationFailed {
                    message: "final Level 4 check failed".to_string(),
                },
            },
        );

        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&insertion_err),
            "InsertionTopologyValidation should be non-retryable"
        );
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&final_err),
            "FinalTopologyValidation should be non-retryable"
        );
        assert!(
            TestDelaunay::<3>::is_non_retryable_construction_error(&final_delaunay_err),
            "FinalDelaunayValidation should be non-retryable"
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
            !TestDelaunay::<3>::is_non_retryable_construction_error(&err),
            "GeometricDegeneracy should NOT be non-retryable"
        );
    }

    #[test]
    fn test_is_non_retryable_construction_error_true_for_unsupported_periodic_dimension() {
        let err: DelaunayTriangulationConstructionError =
            TriangulationConstructionError::UnsupportedPeriodicDimension {
                dimension: 4,
                max_validated_dimension: 3,
                tracking_issue: 416,
            }
            .into();
        assert!(
            TestDelaunay::<4>::is_non_retryable_construction_error(&err),
            "UnsupportedPeriodicDimension should be non-retryable"
        );
    }

    #[test]
    fn retry_policy_default_is_shuffled_in_all_profiles() {
        let policy = RetryPolicy::default();
        match policy {
            RetryPolicy::Shuffled {
                attempts,
                base_seed,
            } => {
                assert_eq!(attempts.get(), DELAUNAY_SHUFFLE_ATTEMPTS);
                assert_eq!(base_seed, None);
            }
            other => panic!("RetryPolicy::default() should be Shuffled, got {other:?}"),
        }
    }

    macro_rules! gen_local_repair_flip_budget_tests {
        ($dim:literal, $floor:ident, $factor:ident) => {
            pastey::paste! {
                #[test]
                fn [<local_repair_flip_budget_uses_dimension_specific_floor_and_factor_ $dim d>]() {
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
    fn local_repair_seed_backlog_threshold_uses_dimension_regimes() {
        assert_eq!(
            local_repair_seed_backlog_threshold::<3>(),
            4 * LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_LT_4
        );
        assert_eq!(
            local_repair_seed_backlog_threshold::<4>(),
            5 * LOCAL_REPAIR_SEED_BACKLOG_FACTOR_D_GE_4
        );
    }

    #[test]
    fn batch_local_repair_trigger_prefers_cadence_over_backlog() {
        let policy = DelaunayRepairPolicy::EveryN(NonZeroUsize::new(4).unwrap());
        let threshold = local_repair_seed_backlog_threshold::<3>();

        assert_eq!(
            batch_local_repair_trigger::<3>(policy, 4, TopologyGuarantee::PLManifold, threshold),
            Some(BatchLocalRepairTrigger::Cadence)
        );
    }

    #[test]
    fn batch_local_repair_trigger_runs_every_insertion_below_backlog() {
        assert_eq!(
            batch_local_repair_trigger::<3>(
                DelaunayRepairPolicy::EveryInsertion,
                1,
                TopologyGuarantee::PLManifold,
                1,
            ),
            Some(BatchLocalRepairTrigger::Cadence)
        );
        assert_eq!(
            batch_local_repair_trigger::<3>(
                DelaunayRepairPolicy::EveryInsertion,
                0,
                TopologyGuarantee::PLManifold,
                1,
            ),
            None
        );
    }

    #[test]
    fn batch_local_repair_trigger_repairs_early_on_seed_backlog() {
        let policy = DelaunayRepairPolicy::EveryN(NonZeroUsize::new(128).unwrap());
        let threshold = local_repair_seed_backlog_threshold::<3>();

        assert_eq!(
            batch_local_repair_trigger::<3>(policy, 7, TopologyGuarantee::PLManifold, threshold),
            Some(BatchLocalRepairTrigger::SeedBacklog)
        );
        assert_eq!(
            batch_local_repair_trigger::<3>(
                policy,
                7,
                TopologyGuarantee::PLManifold,
                threshold - 1
            ),
            None
        );
    }

    #[test]
    fn batch_local_repair_trigger_respects_policy_and_topology() {
        let threshold = local_repair_seed_backlog_threshold::<3>();

        assert_eq!(
            batch_local_repair_trigger::<3>(
                DelaunayRepairPolicy::Never,
                7,
                TopologyGuarantee::PLManifold,
                threshold
            ),
            None
        );
        assert_eq!(
            batch_local_repair_trigger::<3>(
                DelaunayRepairPolicy::EveryN(NonZeroUsize::new(128).unwrap()),
                7,
                TopologyGuarantee::PLManifold,
                0
            ),
            None
        );
        assert_eq!(
            batch_local_repair_trigger::<3>(
                DelaunayRepairPolicy::EveryN(NonZeroUsize::new(128).unwrap()),
                7,
                TopologyGuarantee::Pseudomanifold,
                threshold
            ),
            Some(BatchLocalRepairTrigger::SeedBacklog)
        );
    }
}
