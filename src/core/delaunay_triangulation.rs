//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

#![forbid(unsafe_code)]

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::algorithms::flips::{
    DelaunayRepairError, DelaunayRepairStats, FlipError, apply_bistellar_flip_k1_inverse,
    repair_delaunay_local_single_pass, repair_delaunay_with_flips_k2_k3,
};
use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::cell::Cell;
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{CellKeyBuffer, FastHashMap, FastHasher, SmallBuffer};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::operations::{
    DelaunayInsertionState, InsertionOutcome, InsertionStatistics, RepairDecision,
    TopologicalOperation,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{
    TopologyGuarantee, Triangulation, TriangulationConstructionError, TriangulationValidationError,
    ValidationPolicy, record_duplicate_detection_metrics,
};
use crate::core::triangulation_data_structure::{
    CellKey, InvariantKind, InvariantViolation, Tds, TdsConstructionError, TdsValidationError,
    TriangulationValidationReport, VertexKey,
};
use crate::core::util::{
    coords_equal_exact, coords_within_epsilon, hilbert_index, stable_hash_u64_slice,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{FastKernel, Kernel, RobustKernel};
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateScalar, ScalarAccumulative, ScalarSummable,
};
use crate::topology::manifold::validate_ridge_links_for_cells;
use core::cmp::Ordering;
use num_traits::{NumCast, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

#[cfg(any(test, debug_assertions))]
const DELAUNAY_SHUFFLE_ATTEMPTS: usize = 6;
const DELAUNAY_SHUFFLE_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

#[cfg(any(test, debug_assertions))]
const HEURISTIC_REBUILD_ATTEMPTS: usize = 6;
#[cfg(not(any(test, debug_assertions)))]
const HEURISTIC_REBUILD_ATTEMPTS: usize = 2;

thread_local! {
    static HEURISTIC_REBUILD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    static FORCE_HEURISTIC_REBUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct HeuristicRebuildRecursionGuard {
    prior_depth: usize,
}

impl HeuristicRebuildRecursionGuard {
    fn enter() -> Self {
        let prior_depth = HEURISTIC_REBUILD_DEPTH.with(|depth| {
            let prior = depth.get();
            depth.set(prior.saturating_add(1));
            prior
        });
        Self { prior_depth }
    }

    fn in_progress() -> bool {
        HEURISTIC_REBUILD_DEPTH.with(|depth| depth.get() > 0)
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
/// use delaunay::core::delaunay_triangulation::DelaunayTriangulationConstructionError;
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
}

/// Errors that can occur during Delaunay triangulation validation (Level 4).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::delaunay_triangulation::DelaunayTriangulationValidationError;
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
    /// Lower-layer validation error (Levels 1–3).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// A cell violates the empty circumsphere property.
    #[error(
        "Delaunay property violated: Cell {cell_uuid} (key: {cell_key:?}) violates empty circumsphere invariant"
    )]
    DelaunayViolation {
        /// Key of the violating cell.
        cell_key: CellKey,
        /// UUID of the violating cell (or nil if the UUID mapping is unavailable).
        cell_uuid: Uuid,
    },

    /// Numeric predicate failure during Delaunay validation.
    #[error(
        "Numeric predicate failure while validating Delaunay property for cell {cell_uuid} (key: {cell_key:?}), vertex {vertex_key:?}: {source}"
    )]
    NumericPredicateError {
        /// The key of the cell whose circumsphere was being tested.
        cell_key: CellKey,
        /// UUID of the cell whose predicate evaluation failed (or nil if unavailable).
        cell_uuid: Uuid,
        /// The key of the vertex being classified relative to the circumsphere.
        vertex_key: VertexKey,
        /// Underlying robust predicate error (e.g., conversion failure).
        #[source]
        source: CoordinateConversionError,
    },
}

// =============================================================================
// BATCH CONSTRUCTION OPTIONS
// =============================================================================

/// Strategy used to order input vertices before batch construction.
///
/// The default is [`InsertionOrderStrategy::Hilbert`], which improves spatial locality during
/// bulk insertion.
///
/// If you need to preserve the caller-provided order (for example to control the initial simplex
/// vertices), use [`InsertionOrderStrategy::Input`].
///
/// Note: Morton ordering can improve spatial locality, but it may cause flip cycle issues during
/// Delaunay repair with certain point distributions.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::delaunay_triangulation::{ConstructionOptions, InsertionOrderStrategy};
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
    /// Sort vertices by their coordinates (lexicographic order, `OrderedFloat` semantics).
    ///
    /// This ordering is deterministic and does not depend on vertex UUIDs (which are random by
    /// default).
    Lexicographic,
    /// Sort vertices by Morton / Z-order curve (quantized, normalized coordinates).
    ///
    /// This ordering can improve spatial locality during bulk insertion, reducing point location
    /// cost. However, it may cause flip cycle issues during Delaunay repair with certain point
    /// distributions.
    ///
    /// Ties are broken lexicographically by coordinates, then by original input index.
    Morton,
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
/// This is intended as an *explicit* opt-in for callers who want a predictable preprocessing step.
/// The default is [`DedupPolicy::Off`].
///
/// # Examples
///
/// ```rust
/// use delaunay::core::delaunay_triangulation::{ConstructionOptions, DedupPolicy};
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
    /// Do not preprocess input vertices (legacy behavior).
    #[default]
    Off,
    /// Remove exact coordinate duplicates (NaN-aware, +0.0 == -0.0).
    Exact,
    /// Remove near-duplicates within the given Euclidean tolerance.
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
/// If enabled, the constructor deterministically retries construction with alternative insertion
/// orders (shuffles) when the final Delaunay property check fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::delaunay_triangulation::{ConstructionOptions, RetryPolicy};
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
    /// In debug/test builds, retry construction with a small number of deterministic shuffles if the
    /// final Delaunay property check fails.
    ///
    /// In release builds, this is treated as [`RetryPolicy::Disabled`]. Prefer
    /// [`RetryPolicy::Shuffled`] if you want retries in all build modes.
    DebugOnlyShuffled {
        /// Number of shuffled reconstruction attempts (excluding the original-order attempt).
        attempts: NonZeroUsize,
        /// Optional base seed. If `None`, a deterministic seed is derived from the vertex set.
        base_seed: Option<u64>,
    },
}

#[cfg(any(test, debug_assertions))]
impl Default for RetryPolicy {
    fn default() -> Self {
        Self::DebugOnlyShuffled {
            attempts: NonZeroUsize::new(DELAUNAY_SHUFFLE_ATTEMPTS)
                .expect("DELAUNAY_SHUFFLE_ATTEMPTS must be non-zero"),
            base_seed: None,
        }
    }
}

#[cfg(not(any(test, debug_assertions)))]
impl Default for RetryPolicy {
    fn default() -> Self {
        Self::Disabled
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
/// use delaunay::core::delaunay_triangulation::{
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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct ConstructionOptions {
    insertion_order: InsertionOrderStrategy,
    dedup_policy: DedupPolicy,
    initial_simplex: InitialSimplexStrategy,
    retry_policy: RetryPolicy,
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
    pub statistics: Box<ConstructionStatistics>,
}

impl ConstructionStatistics {
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
struct PreprocessVertices<T, U, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
{
    primary: Option<VertexBuffer<T, U, D>>,
    fallback: Option<VertexBuffer<T, U, D>>,
    grid_cell_size: Option<T>,
}

impl<T, U, const D: usize> PreprocessVertices<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    fn primary_slice<'a>(&'a self, input: &'a [Vertex<T, U, D>]) -> &'a [Vertex<T, U, D>] {
        self.primary.as_deref().unwrap_or(input)
    }

    fn fallback_slice(&self) -> Option<&[Vertex<T, U, D>]> {
        self.fallback.as_deref()
    }

    const fn grid_cell_size(&self) -> Option<T>
    where
        T: Copy,
    {
        self.grid_cell_size
    }
}

type PreprocessVerticesResult<T, U, const D: usize> =
    Result<PreprocessVertices<T, U, D>, DelaunayTriangulationConstructionError>;

fn vertex_coordinate_hash<T, U, const D: usize>(vertex: &Vertex<T, U, D>) -> u64
where
    T: CoordinateScalar,
    U: DataType,
{
    let mut hasher = FastHasher::default();
    vertex.hash(&mut hasher);
    hasher.finish()
}

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

fn morton_bits_per_coord<const D: usize>() -> Option<u32> {
    if !(2..=5).contains(&D) {
        return None;
    }

    let Ok(d_u32) = u32::try_from(D) else {
        return None;
    };

    let bits_per_coord = u64::BITS / d_u32;
    if bits_per_coord == 0 || bits_per_coord >= u64::BITS {
        return None;
    }

    Some(bits_per_coord)
}

fn morton_code<const D: usize>(quantized: [u64; D], bits_per_coord: u32) -> u64 {
    let mut code = 0_u64;

    for bit in (0..bits_per_coord).rev() {
        for &q in &quantized {
            let b = (q >> bit) & 1;
            code = (code << 1) | b;
        }
    }

    code
}

fn order_vertices_morton<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    let Some(bits_per_coord) = morton_bits_per_coord::<D>() else {
        return order_vertices_lexicographic(vertices);
    };

    // Compute bounding box in f64 for normalization. If any coordinate is non-finite,
    // fall back to lexicographic ordering (Morton normalization assumes finite values).
    let mut min = [f64::INFINITY; D];
    let mut max = [f64::NEG_INFINITY; D];

    for v in &vertices {
        let coords = v.point().coords();
        for axis in 0..D {
            let Some(c) = coords[axis].to_f64() else {
                return order_vertices_lexicographic(vertices);
            };
            if !c.is_finite() {
                return order_vertices_lexicographic(vertices);
            }
            min[axis] = min[axis].min(c);
            max[axis] = max[axis].max(c);
        }
    }

    let mut inv_range = [0.0_f64; D];
    for axis in 0..D {
        let range = max[axis] - min[axis];
        inv_range[axis] = if range > 0.0 { 1.0 / range } else { 0.0 };
    }

    let max_quant = (1_u64 << bits_per_coord) - 1;
    let scale = <f64 as From<u32>>::from(u32::try_from(max_quant).unwrap_or(u32::MAX));

    let mut keyed: Vec<(u64, Vertex<T, U, D>, usize)> = vertices
        .into_iter()
        .enumerate()
        .map(|(input_index, vertex)| {
            let coords = vertex.point().coords();
            let mut q = [0_u64; D];

            for axis in 0..D {
                let c = coords[axis].to_f64().unwrap_or(0.0);
                let norm = if inv_range[axis] == 0.0 {
                    0.0
                } else {
                    (c - min[axis]) * inv_range[axis]
                };
                let clamped = norm.clamp(0.0, 1.0);
                q[axis] = (clamped * scale).floor().to_u64().unwrap_or(0);
            }

            let code = morton_code::<D>(q, bits_per_coord);
            (code, vertex, input_index)
        })
        .collect();

    keyed.sort_by(|(a_code, a_vertex, a_idx), (b_code, b_vertex, b_idx)| {
        a_code
            .cmp(b_code)
            .then_with(|| a_vertex.partial_cmp(b_vertex).unwrap_or(Ordering::Equal))
            .then_with(|| a_idx.cmp(b_idx))
    });

    keyed.into_iter().map(|(_, v, _)| v).collect()
}

const BATCH_DEDUP_BUCKET_INLINE_CAPACITY: usize = 8;
const BATCH_DEDUP_MAX_DIMENSION: usize = 5;

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
        InsertionOrderStrategy::Lexicographic => order_vertices_lexicographic(vertices),
        InsertionOrderStrategy::Morton => order_vertices_morton(vertices),
        InsertionOrderStrategy::Hilbert => order_vertices_hilbert(vertices),
    }
}

fn default_duplicate_tolerance<T: CoordinateScalar>() -> T {
    <T as NumCast>::from(1e-10_f64).unwrap_or_else(T::default_tolerance)
}

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
                dist_sq = dist_sq + diff * diff;
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

fn order_vertices_hilbert<T, U, const D: usize>(
    vertices: Vec<Vertex<T, U, D>>,
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

    let mut keyed: Vec<(u128, Vertex<T, U, D>, usize)> = vertices
        .into_iter()
        .enumerate()
        .map(|(input_index, vertex)| {
            let idx = hilbert_index(vertex.point().coords(), bounds, bits_per_coord);
            (idx, vertex, input_index)
        })
        .collect();

    keyed.sort_by(|(a_idx, a_vertex, a_in), (b_idx, b_vertex, b_in)| {
        a_idx
            .cmp(b_idx)
            .then_with(|| a_vertex.partial_cmp(b_vertex).unwrap_or(Ordering::Equal))
            .then_with(|| a_in.cmp(b_in))
    });

    keyed.into_iter().map(|(_, v, _)| v).collect()
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
pub struct DelaunayTriangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
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

// Most common case: f64 with FastKernel, no vertex or cell data
impl<const D: usize> DelaunayTriangulation<FastKernel<f64>, (), (), D> {
    /// Create a Delaunay triangulation from vertices with no data (most common case).
    ///
    /// This is the simplest constructor for the most common use case:
    /// - f64 coordinates
    /// - Fast floating-point predicates  
    /// - No vertex data
    /// - No cell data
    ///
    /// No type annotations needed! The compiler can infer everything.
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
        Self::with_kernel(&FastKernel::<f64>::new(), vertices)
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
    pub fn new_with_construction_statistics(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let kernel = FastKernel::<f64>::new();
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
    pub fn new_with_options_and_construction_statistics(
        vertices: &[Vertex<f64, (), D>],
        options: ConstructionOptions,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    {
        let kernel = FastKernel::<f64>::new();
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
    /// use delaunay::core::delaunay_triangulation::{
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
        let kernel = FastKernel::<f64>::new();
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
        let kernel = FastKernel::<f64>::new();
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
        Self::with_empty_kernel(FastKernel::<f64>::new())
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
        Self::with_empty_kernel_and_topology_guarantee(FastKernel::<f64>::new(), topology_guarantee)
    }
}

// Generic implementation for all kernels
impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: ScalarAccumulative + NumCast,
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
    /// Most users should use [`DelaunayTriangulation::new()`] instead, which uses fast predicates
    /// by default. Use this method only if you need:
    /// - Custom coordinate precision (f32, custom types)
    /// - Explicit robust/exact arithmetic predicates
    /// - Specialized kernel implementations
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
        Self::with_topology_guarantee(kernel, vertices, TopologyGuarantee::DEFAULT)
    }

    /// Create a Delaunay triangulation with an explicit topology guarantee.
    ///
    /// Passing [`TopologyGuarantee::PLManifold`] enforces ridge-link validation during
    /// construction and validates vertex-links at completion. Use
    /// [`TopologyGuarantee::PLManifoldStrict`] for per-insertion vertex-link checks.
    ///
    /// # Debug/Test Behavior
    /// In debug/test builds (for `D >= 3` with more than `D + 1` vertices), the constructor may
    /// retry construction with a handful of shuffled insertion orders if the Delaunay property
    /// check fails. Release builds skip these shuffled reconstruction attempts.
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
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
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
    /// use delaunay::core::delaunay_triangulation::{
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
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
        let ConstructionOptions {
            insertion_order,
            dedup_policy,
            initial_simplex,
            retry_policy,
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
                        );
                    }
                }
            }

            Self::build_with_kernel_inner(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                grid_cell_size,
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
    pub fn with_topology_guarantee_and_options_with_construction_statistics(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        options: ConstructionOptions,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    where
        K::Scalar: ScalarSummable,
    {
        let ConstructionOptions {
            insertion_order,
            dedup_policy,
            initial_simplex,
            retry_policy,
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
                statistics: Box::new(ConstructionStatistics::default()),
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
                        );
                    }
                }
            }

            Self::build_with_kernel_inner_with_construction_statistics(
                <K as Clone>::clone(kernel),
                vertices,
                topology_guarantee,
                grid_cell_size,
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

    fn preprocess_vertices_for_construction(
        vertices: &[Vertex<K::Scalar, U, D>],
        dedup_policy: DedupPolicy,
        insertion_order: InsertionOrderStrategy,
        initial_simplex: InitialSimplexStrategy,
    ) -> PreprocessVerticesResult<K::Scalar, U, D>
    where
        K::Scalar: ScalarSummable,
    {
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

    #[allow(clippy::too_many_lines)]
    fn build_with_shuffled_retries(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
        let base_seed = base_seed.unwrap_or_else(|| Self::construction_shuffle_seed(vertices));

        #[cfg(debug_assertions)]
        let log_shuffle = std::env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some();

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
        let mut last_error: String = match Self::build_with_kernel_inner_seeded(
            <K as Clone>::clone(kernel),
            vertices,
            topology_guarantee,
            0_u64,
            true,
            grid_cell_size,
        ) {
            Ok(candidate) => match crate::core::util::is_delaunay_property_only(&candidate.tri.tds)
            {
                Ok(()) => return Ok(candidate),
                Err(err) => format!("Delaunay property violated after construction: {err}"),
            },
            Err(err) => {
                // Some construction errors are deterministic and should not be masked
                // by shuffled retry logic (e.g. duplicate UUIDs).
                if matches!(
                    &err,
                    DelaunayTriangulationConstructionError::Triangulation(
                        TriangulationConstructionError::Tds(
                            TdsConstructionError::DuplicateUuid { .. }
                        )
                    )
                ) {
                    return Err(err);
                }
                err.to_string()
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

            match Self::build_with_kernel_inner_seeded(
                <K as Clone>::clone(kernel),
                &shuffled,
                topology_guarantee,
                perturbation_seed,
                true,
                grid_cell_size,
            ) {
                Ok(candidate) => {
                    match crate::core::util::is_delaunay_property_only(&candidate.tri.tds) {
                        Ok(()) => return Ok(candidate),
                        Err(err) => {
                            last_error =
                                format!("Delaunay property violated after construction: {err}");
                        }
                    }
                }
                Err(err) => {
                    if matches!(
                        &err,
                        DelaunayTriangulationConstructionError::Triangulation(
                            TriangulationConstructionError::Tds(
                                TdsConstructionError::DuplicateUuid { .. }
                            )
                        )
                    ) {
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

    #[allow(clippy::too_many_lines)]
    fn build_with_shuffled_retries_with_construction_statistics(
        kernel: &K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        attempts: NonZeroUsize,
        base_seed: Option<u64>,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    where
        K::Scalar: ScalarSummable,
    {
        let base_seed = base_seed.unwrap_or_else(|| Self::construction_shuffle_seed(vertices));

        #[cfg(debug_assertions)]
        let log_shuffle = std::env::var_os("DELAUNAY_DEBUG_SHUFFLE").is_some();

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
                    // Some construction errors are deterministic and should not be masked
                    // by shuffled retry logic (e.g. duplicate UUIDs).
                    if matches!(
                        &error,
                        DelaunayTriangulationConstructionError::Triangulation(
                            TriangulationConstructionError::Tds(
                                TdsConstructionError::DuplicateUuid { .. }
                            )
                        )
                    ) {
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics,
                        });
                    }
                    last_stats.replace(*statistics);
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

            match Self::build_with_kernel_inner_seeded_with_construction_statistics(
                <K as Clone>::clone(kernel),
                &shuffled,
                topology_guarantee,
                perturbation_seed,
                true,
                grid_cell_size,
            ) {
                Ok((candidate, stats)) => {
                    match crate::core::util::is_delaunay_property_only(&candidate.tri.tds) {
                        Ok(()) => return Ok((candidate, stats)),
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
                    if matches!(
                        &error,
                        DelaunayTriangulationConstructionError::Triangulation(
                            TriangulationConstructionError::Tds(
                                TdsConstructionError::DuplicateUuid { .. }
                            )
                        )
                    ) {
                        return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                            error,
                            statistics,
                        });
                    }
                    last_stats.replace(*statistics);
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
        }

        // Treat persistent construction failures or Delaunay violations as hard construction
        // errors so callers can deterministically reject.
        let statistics = Box::new(last_stats.unwrap_or_default());
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

    const fn should_retry_construction(vertices: &[Vertex<K::Scalar, U, D>]) -> bool {
        D >= 2 && vertices.len() > D + 1
    }

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

    fn shuffle_vertices(vertices: &mut [Vertex<K::Scalar, U, D>], seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        vertices.shuffle(&mut rng);
    }

    fn build_with_kernel_inner(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
        let dt = Self::build_with_kernel_inner_seeded(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
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

    fn build_with_kernel_inner_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    where
        K::Scalar: ScalarSummable,
    {
        let (dt, stats) = Self::build_with_kernel_inner_seeded_with_construction_statistics(
            kernel,
            vertices,
            topology_guarantee,
            0,
            true,
            grid_cell_size,
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
                    statistics: Box::new(stats),
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
                statistics: Box::new(stats),
            });
        }

        Ok((dt, stats))
    }

    fn build_with_kernel_inner_seeded_with_construction_statistics(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<(Self, ConstructionStatistics), DelaunayTriangulationConstructionErrorWithStatistics>
    where
        K::Scalar: ScalarSummable,
    {
        if vertices.len() < D + 1 {
            return Err(DelaunayTriangulationConstructionErrorWithStatistics {
                error: TriangulationConstructionError::InsufficientVertices {
                    dimension: D,
                    source: crate::core::cell::CellValidationError::InsufficientVertices {
                        actual: vertices.len(),
                        expected: D + 1,
                        dimension: D,
                    },
                }
                .into(),
                statistics: Box::new(ConstructionStatistics::default()),
            });
        }

        // Build initial simplex directly (no Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Triangulation::<K, U, V, D>::build_initial_simplex(initial_vertices).map_err(
            |error| DelaunayTriangulationConstructionErrorWithStatistics {
                error: error.into(),
                statistics: Box::new(ConstructionStatistics::default()),
            },
        )?;

        let mut dt = Self {
            tri: Triangulation {
                kernel,
                tds,
                validation_policy: ValidationPolicy::default(),
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        };

        // During batch construction, enforce topology guarantees:
        // - PLManifoldStrict: always validate (vertex-link checks) on each insertion
        // - PLManifold: always validate (ridge-link checks) on each insertion
        // - Pseudomanifold: keep debug-only strictness for safety without release overhead
        let original_validation_policy = dt.tri.validation_policy;
        dt.tri.validation_policy = if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_during_insertion()
            || dt.tri.topology_guarantee.requires_ridge_links()
        {
            ValidationPolicy::Always
        } else {
            ValidationPolicy::DebugOnly
        };

        // Disable maybe_repair_after_insertion during bulk construction: its full pipeline
        // (multi-pass repair + topology validation + heuristic rebuild) is too expensive
        // per insertion.  Instead, insert_remaining_vertices_seeded runs a targeted
        // repair_delaunay_with_flips_k2_k3 call directly after each insertion (no topology
        // check, no heuristic rebuild, soft-fail on non-convergence).  Soft-failed
        // insertions record their adjacent cells in soft_fail_seeds, which is used as the
        // seed for the final global repair in finalize_bulk_construction.  If no soft-fails
        // occurred, the seed is empty and the global repair returns immediately with an
        // empty queue (no work needed).
        let original_repair_policy = dt.insertion_state.delaunay_repair_policy;
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;

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
                statistics: Box::new(stats),
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
                statistics: Box::new(stats),
            });
        }

        Ok((dt, stats))
    }

    fn build_with_kernel_inner_seeded(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
        topology_guarantee: TopologyGuarantee,
        perturbation_seed: u64,
        run_final_repair: bool,
        grid_cell_size: Option<K::Scalar>,
    ) -> Result<Self, DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
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
                validation_policy: ValidationPolicy::default(),
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        };

        // During batch construction, enforce topology guarantees:
        // - PLManifoldStrict: always validate (vertex-link checks) on each insertion
        // - PLManifold: always validate (ridge-link checks) on each insertion
        // - Pseudomanifold: keep debug-only strictness for safety without release overhead
        let original_validation_policy = dt.tri.validation_policy;
        dt.tri.validation_policy = if dt
            .tri
            .topology_guarantee
            .requires_vertex_links_during_insertion()
            || dt.tri.topology_guarantee.requires_ridge_links()
        {
            ValidationPolicy::Always
        } else {
            ValidationPolicy::DebugOnly
        };

        // See the _with_construction_statistics variant for the repair policy rationale.
        let original_repair_policy = dt.insertion_state.delaunay_repair_policy;
        dt.insertion_state.delaunay_repair_policy = DelaunayRepairPolicy::Never;
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

    #[allow(clippy::too_many_lines)]
    fn insert_remaining_vertices_seeded(
        &mut self,
        vertices: &[Vertex<K::Scalar, U, D>],
        perturbation_seed: u64,
        grid_cell_size: Option<K::Scalar>,
        construction_stats: Option<&mut ConstructionStatistics>,
        _soft_fail_seeds: &mut Vec<CellKey>,
    ) -> Result<(), DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
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

        let trace_insertion = std::env::var_os("DELAUNAY_INSERT_TRACE").is_some();

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
                        eprintln!("[bulk] start idx={index} uuid={uuid} coords={coords:?}");
                    }

                    let started = trace_insertion.then(std::time::Instant::now);
                    let mut insert = || {
                        self.tri.insert_with_statistics_seeded_indexed(
                            *vertex,
                            None,
                            self.insertion_state.last_inserted_cell,
                            perturbation_seed,
                            grid_index.as_mut(),
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
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: v_key,
                                hint,
                            },
                            _stats,
                        )) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] inserted idx={index} uuid={uuid} elapsed={elapsed:?}"
                                );
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
                            // D=2, Rajan 1991/Joe 1991 for D=3); hard-fail on non-convergence
                            // triggers a shuffle retry at the outer level.
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
                                let seed_cells: Vec<CellKey> =
                                    self.tri.adjacent_cells(v_key).collect();
                                if !seed_cells.is_empty() {
                                    let max_flips = (seed_cells.len() * (D + 1) * 4).max(16);
                                    let repair_result = {
                                        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                                        repair_delaunay_local_single_pass(
                                            tds,
                                            kernel,
                                            &seed_cells,
                                            max_flips,
                                        )
                                    };
                                    if let Err(repair_err) = repair_result {
                                        if D < 4 {
                                            // Hard-fail for D<4: proven convergent, so a
                                            // failure here is a real numerical problem.
                                            tracing::debug!(
                                                error = %repair_err,
                                                idx = index,
                                                "bulk: per-insertion repair failed; \
                                                 aborting this vertex ordering"
                                            );
                                            return Err(
                                                TriangulationConstructionError::GeometricDegeneracy {
                                                    message: format!(
                                                        "per-insertion Delaunay repair \
                                                         failed at index {index}: \
                                                         {repair_err}"
                                                    ),
                                                }
                                                .into(),
                                            );
                                        }
                                        // Soft-fail for D≥4: non-convergence is expected in
                                        // degenerate configurations; continue building.
                                        tracing::debug!(
                                            error = %repair_err,
                                            idx = index,
                                            "bulk D≥4: per-insertion repair non-convergent; \
                                             continuing (both_positive_artifact handled)"
                                        );
                                    }
                                }
                            }
                        }
                        Ok((InsertionOutcome::Skipped { error }, stats)) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] skipped idx={index} uuid={uuid} attempts={} elapsed={elapsed:?} err={error}",
                                    stats.attempts
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
                        }
                        Err(e) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] failed idx={index} uuid={uuid} elapsed={elapsed:?} err={e}"
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
                        eprintln!("[bulk] start idx={index} uuid={uuid} coords={coords:?}");
                    }

                    let started = trace_insertion.then(std::time::Instant::now);
                    let mut insert = || {
                        self.tri.insert_with_statistics_seeded_indexed(
                            *vertex,
                            None,
                            self.insertion_state.last_inserted_cell,
                            perturbation_seed,
                            grid_index.as_mut(),
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
                    match insert_result {
                        Ok((
                            InsertionOutcome::Inserted {
                                vertex_key: v_key,
                                hint,
                            },
                            stats,
                        )) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] inserted idx={index} uuid={uuid} attempts={} elapsed={elapsed:?}",
                                    stats.attempts
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
                                let seed_cells: Vec<CellKey> =
                                    self.tri.adjacent_cells(v_key).collect();
                                if !seed_cells.is_empty() {
                                    let max_flips = (seed_cells.len() * (D + 1) * 4).max(16);
                                    let repair_result = {
                                        let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                                        repair_delaunay_local_single_pass(
                                            tds,
                                            kernel,
                                            &seed_cells,
                                            max_flips,
                                        )
                                    };
                                    if let Err(repair_err) = repair_result {
                                        if D < 4 {
                                            tracing::debug!(
                                                error = %repair_err,
                                                idx = index,
                                                "bulk: per-insertion repair failed; \
                                                 aborting this vertex ordering"
                                            );
                                            return Err(
                                                TriangulationConstructionError::GeometricDegeneracy {
                                                    message: format!(
                                                        "per-insertion Delaunay repair \
                                                         failed at index {index}: \
                                                         {repair_err}"
                                                    ),
                                                }
                                                .into(),
                                            );
                                        }
                                        tracing::debug!(
                                            error = %repair_err,
                                            idx = index,
                                            "bulk D≥4: per-insertion repair non-convergent; \
                                             continuing (both_positive_artifact handled)"
                                        );
                                    }
                                }
                            }
                        }
                        Ok((InsertionOutcome::Skipped { error }, stats)) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] skipped idx={index} uuid={uuid} attempts={} elapsed={elapsed:?} err={error}",
                                    stats.attempts
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
                        }
                        Err(e) => {
                            if trace_insertion && let Some(elapsed) = elapsed {
                                eprintln!(
                                    "[bulk] failed idx={index} uuid={uuid} elapsed={elapsed:?} err={e}"
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

    fn finalize_bulk_construction(
        &mut self,
        original_validation_policy: ValidationPolicy,
        original_repair_policy: DelaunayRepairPolicy,
        run_final_repair: bool,
        soft_fail_seeds: &[CellKey],
    ) -> Result<(), DelaunayTriangulationConstructionError>
    where
        K::Scalar: ScalarSummable,
    {
        // Restore policies after batch construction.
        self.tri.validation_policy = original_validation_policy;
        self.insertion_state.delaunay_repair_policy = original_repair_policy;

        let topology = self.tri.topology_guarantee();
        if run_final_repair && self.should_run_delaunay_repair_for(topology, 0) {
            // Use a single-pass seeded repair bounded to the soft-fail neighbourhood.
            // An empty soft_fail_seeds slice means per-insertion repair had no soft-fails:
            // seed_repair_queues leaves the queue empty → O(1) exit.
            // A non-empty slice targets only the affected neighbourhoods.
            //
            // For D≥4 we intentionally do NOT fall back to all-cells seeding here:
            // the D≥4 flip graph is not guaranteed convergent (Edelsbrunner-Shah 1996)
            // and a full-triangulation repair pass is prohibitively slow in debug mode.
            // Correctness is ensured by the is_delaunay_property_only() check in
            // build_with_shuffled_retries (which handles both_positive_artifact cases).
            let finalize_seeds = soft_fail_seeds;
            tracing::debug!(
                seed_count = finalize_seeds.len(),
                "post-construction: starting global Delaunay repair (finalize)"
            );
            let repair_started = Instant::now();
            let global_repair_max_flips = (finalize_seeds.len() * (D + 1) * 16).max(512);
            let repair_result = {
                let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                repair_delaunay_local_single_pass(
                    tds,
                    kernel,
                    finalize_seeds,
                    global_repair_max_flips,
                )
                .map(|_| ())
            };
            let repair_outcome: Result<(), DelaunayTriangulationConstructionError> =
                match repair_result {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        // Single-pass did not converge or encountered an error.
                        // Return a construction error so that build_with_shuffled_retries
                        // can try another vertex ordering (the intended fallback for
                        // hard-to-repair configurations).  We intentionally do NOT run
                        // run_flip_repair_fallbacks here: that chain can internally trigger
                        // a full-triangulation None-seed repair (O(cells × queues/cell))
                        // which is prohibitively slow for D≥4 with many cells.
                        Err(TriangulationConstructionError::GeometricDegeneracy {
                            message: format!("Delaunay repair failed after construction: {e}"),
                        }
                        .into())
                    }
                };
            tracing::debug!(
                elapsed = ?repair_started.elapsed(),
                success = repair_outcome.is_ok(),
                "post-construction: global Delaunay repair (finalize) completed"
            );
            repair_outcome?;
        }

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

    fn map_insertion_error(error: InsertionError) -> TriangulationConstructionError {
        match error {
            // Preserve underlying construction errors (e.g. duplicate UUID).
            InsertionError::Construction(source) => source,
            InsertionError::CavityFilling { message } => {
                TriangulationConstructionError::FailedToCreateCell { message }
            }
            InsertionError::NeighborWiring { message } => {
                TriangulationConstructionError::from(TdsConstructionError::ValidationError(
                    TdsValidationError::InvalidNeighbors { message },
                ))
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
            | InsertionError::TopologyValidationFailed { .. }) => {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: insertion_error.to_string(),
                }
            }
        }
    }

    // TODO: Implement after bistellar flips + robust insertion (v0.7.0+)
    // /// Create a Delaunay triangulation with a specified topological space.
    // ///
    // /// This will allow constructing Delaunay triangulations on different topologies
    // /// (Euclidean, spherical, toroidal) with appropriate boundary conditions
    // /// and topology validation. This method should delegate to
    // /// [`Triangulation::with_topology`] after constructing the TDS.
    // ///
    // /// Requires:
    // /// - Bistellar flips for topology-preserving operations
    // /// - Robust Delaunay insertion that respects topology constraints
    // ///
    // /// # Examples (future)
    // ///
    // /// ```rust,ignore
    // /// use delaunay::prelude::query::*;
    // /// use delaunay::topology::spaces::ToroidalSpace;
    // ///
    // /// let space = ToroidalSpace::new([1.0, 1.0, 1.0]);
    // /// let dt = DelaunayTriangulation::with_topology(
    // ///     FastKernel::new(),
    // ///     space,
    // ///     &vertices
    // /// ).unwrap();
    // /// ```
    // pub fn with_topology<T>(
    //     kernel: K,
    //     topology: T,
    //     vertices: &[Vertex<K::Scalar, U, D>],
    // ) -> Result<Self, TriangulationConstructionError>
    // where
    //     K::Scalar: CoordinateScalar,
    //     T: TopologicalSpace,
    // {
    //     // Build TDS with Delaunay property, then delegate to Triangulation layer
    //     unimplemented!("Requires bistellar flips + robust insertion")
    // }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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

    /// Returns a reference to the underlying triangulation data structure.
    ///
    /// This provides access to the purely combinatorial Tds layer for
    /// advanced operations and performance testing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    /// budget is exhausted.
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayRepairError`] if the repair fails to converge or an underlying
    /// flip operation fails.
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
        K::Scalar: ScalarSummable,
    {
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
        repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology)
    }

    fn repair_delaunay_with_flips_robust(
        &mut self,
        seed_cells: Option<&[CellKey]>,
    ) -> Result<DelaunayRepairStats, DelaunayRepairError>
    where
        K::Scalar: ScalarSummable,
    {
        let topology = self.tri.topology_guarantee();
        let kernel = RobustKernel::<K::Scalar>::new();
        let (tds, kernel) = (&mut self.tri.tds, &kernel);
        repair_delaunay_with_flips_k2_k3(tds, kernel, seed_cells, topology)
    }

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
    fn remap_vertex_key_by_uuid(&self, vertex_uuid: Uuid) -> Result<VertexKey, InsertionError> {
        self.tri
            .tds
            .vertex_key_from_uuid(&vertex_uuid)
            .ok_or_else(|| InsertionError::CavityFilling {
                message: format!(
                    "Inserted vertex with uuid {vertex_uuid} missing after heuristic rebuild"
                ),
            })
    }
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

    fn run_flip_repair_fallbacks(
        &mut self,
        seed_cells: Option<&[CellKey]>,
    ) -> Result<bool, DelaunayRepairError>
    where
        K::Scalar: ScalarSummable,
    {
        // Avoid unbounded recursion (and stack overflows) when heuristic rebuild itself triggers
        // local-repair fallbacks during incremental insertion.
        let nested_rebuild = HeuristicRebuildRecursionGuard::in_progress();
        let forced = Self::force_heuristic_rebuild_enabled();

        // During a heuristic rebuild attempt, never allow a *nested* heuristic rebuild.
        // We still allow a robust flip-repair pass as a best-effort fallback.
        if nested_rebuild {
            match self.repair_delaunay_with_flips_robust(seed_cells) {
                Ok(_) => return Ok(false),
                Err(robust_err) => {
                    return Err(DelaunayRepairError::HeuristicRebuildFailed {
                        message: format!(
                            "Nested heuristic rebuild disabled during heuristic rebuild (robust repair failed: {robust_err})"
                        ),
                    });
                }
            }
        }

        // Outside of heuristic rebuild, first try a fully robust repair pass unless the test-only
        // force flag requests a heuristic rebuild.
        if !forced && self.repair_delaunay_with_flips_robust(seed_cells).is_ok() {
            return Ok(false);
        }

        let outcome =
            self.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())?;
        Ok(outcome.used_heuristic())
    }

    /// Runs flip-based Delaunay repair with an optional heuristic rebuild fallback.
    ///
    /// This first attempts the standard two-pass flip repair. If it fails to converge (or if
    /// the result cannot be verified as Delaunay), it rebuilds the triangulation from the
    /// current vertex set using a shuffled insertion order and a perturbation seed, then runs
    /// a final flip-repair pass.
    ///
    /// The returned outcome marks whether the heuristic fallback was used and records
    /// the seeds needed to reproduce it (if desired).
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayRepairError`] if the flip-based repair fails or if the heuristic
    /// rebuild fallback cannot construct a valid triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayRepairHeuristicConfig;
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
        K::Scalar: ScalarSummable,
    {
        if Self::force_heuristic_rebuild_enabled() {
            let base_seed = self.heuristic_rebuild_base_seed();
            let seeds = config.resolve_seeds(base_seed);
            let (candidate, stats, used_seeds) = self.rebuild_with_heuristic(seeds)?;
            *self = candidate;
            return Ok(DelaunayRepairOutcome {
                stats,
                heuristic: Some(used_seeds),
            });
        }
        match self.repair_delaunay_with_flips() {
            Ok(stats) => Ok(DelaunayRepairOutcome {
                stats,
                heuristic: None,
            }),
            Err(
                DelaunayRepairError::NonConvergent { .. }
                | DelaunayRepairError::PostconditionFailed { .. },
            ) => {
                if let Ok(stats) = self.repair_delaunay_with_flips_robust(None) {
                    return Ok(DelaunayRepairOutcome {
                        stats,
                        heuristic: None,
                    });
                }
                let base_seed = self.heuristic_rebuild_base_seed();
                let seeds = config.resolve_seeds(base_seed);
                let (candidate, stats, used_seeds) = self.rebuild_with_heuristic(seeds)?;
                *self = candidate;
                Ok(DelaunayRepairOutcome {
                    stats,
                    heuristic: Some(used_seeds),
                })
            }
            Err(err) => Err(err),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn rebuild_with_heuristic(
        &self,
        base_seeds: DelaunayRepairHeuristicSeeds,
    ) -> Result<(Self, DelaunayRepairStats, DelaunayRepairHeuristicSeeds), DelaunayRepairError>
    where
        K::Scalar: ScalarSummable,
    {
        use rand::{SeedableRng, seq::SliceRandom};

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
                let topology = self.tri.topology_guarantee();
                let mut candidate = Self::with_empty_kernel_and_topology_guarantee(
                    self.tri.kernel.clone(),
                    topology,
                );

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
                    let (outcome, _stats) = {
                        let (tri, spatial_index) =
                            (&mut candidate.tri, &mut candidate.spatial_index);
                        tri.insert_with_statistics_seeded_indexed(
                            vertex,
                            None,
                            hint,
                            seeds.perturbation_seed,
                            spatial_index.as_mut(),
                        )
                        .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                            message: format!(
                                "heuristic rebuild insertion failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                            ),
                        })?
                    };

                    match outcome {
                        InsertionOutcome::Inserted { vertex_key, hint } => {
                            let mut hint = hint;
                            candidate.insertion_state.last_inserted_cell = hint;
                            candidate.insertion_state.delaunay_repair_insertion_count = candidate
                                .insertion_state
                                .delaunay_repair_insertion_count
                                .saturating_add(1);

                            let (_vertex_key, used_heuristic) = candidate
                                .maybe_repair_after_insertion(vertex_key, hint)
                                .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                                    message: format!(
                                        "heuristic rebuild repair failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                                    ),
                                })?;

                            if used_heuristic {
                                candidate.insertion_state.last_inserted_cell = None;
                                hint = None;
                            }

                            candidate
                                .maybe_check_after_insertion()
                                .map_err(|e| DelaunayRepairError::HeuristicRebuildFailed {
                                    message: format!(
                                        "heuristic rebuild Delaunay check failed at idx={idx} uuid={uuid} coords={coords:?}: {e}"
                                    ),
                                })?;

                            // Keep the parameter "used".
                            let _ = hint;
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
                let stats = repair_delaunay_with_flips_k2_k3(tds, kernel, None, topology)?;

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

    fn collect_vertices_for_rebuild(&self) -> Vec<Vertex<K::Scalar, U, D>> {
        self.tri
            .tds
            .vertices()
            .map(|(_, vertex)| Vertex::new_with_uuid(*vertex.point(), vertex.uuid(), vertex.data))
            .collect()
    }

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

    /// Returns the topology guarantee used for Level 3 topology validation.
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.tri.topology_guarantee()
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    pub fn cell_vertices(&self, c: CellKey) -> Option<&[VertexKey]>
    where
        K::Scalar: CoordinateScalar,
    {
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
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]>
    where
        K::Scalar: CoordinateScalar,
    {
        self.as_triangulation().vertex_coords(v)
    }

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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
    pub fn insert(&mut self, vertex: Vertex<K::Scalar, U, D>) -> Result<VertexKey, InsertionError>
    where
        K::Scalar: ScalarSummable,
    {
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
            let (outcome, _stats) = {
                let (tri, spatial_index) = (&mut self.tri, &mut self.spatial_index);
                tri.insert_with_statistics_seeded_indexed(
                    vertex,
                    None,
                    hint,
                    0,
                    spatial_index.as_mut(),
                )?
            };

            match outcome {
                InsertionOutcome::Inserted {
                    vertex_key: v_key,
                    hint,
                } => {
                    self.insertion_state.last_inserted_cell = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    let (v_key, used_heuristic) = self.maybe_repair_after_insertion(v_key, hint)?;
                    if used_heuristic {
                        self.insertion_state.last_inserted_cell = None;
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
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K::Scalar: ScalarSummable,
    {
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
            let (outcome, stats) = {
                let (tri, spatial_index) = (&mut self.tri, &mut self.spatial_index);
                tri.insert_with_statistics_seeded_indexed(
                    vertex,
                    None,
                    hint,
                    0,
                    spatial_index.as_mut(),
                )?
            };

            let outcome = match outcome {
                InsertionOutcome::Inserted { vertex_key, hint } => {
                    let mut hint = hint;
                    self.insertion_state.last_inserted_cell = hint;
                    self.insertion_state.delaunay_repair_insertion_count = self
                        .insertion_state
                        .delaunay_repair_insertion_count
                        .saturating_add(1);
                    let (vertex_key, used_heuristic) =
                        self.maybe_repair_after_insertion(vertex_key, hint)?;
                    if used_heuristic {
                        self.insertion_state.last_inserted_cell = None;
                        hint = None;
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
                if let Some((tds, insertion_state, spatial_index)) = snapshot {
                    self.tri.tds = tds;
                    self.insertion_state = insertion_state;
                    self.spatial_index = spatial_index;
                }
                Err(err)
            }
        }
    }

    fn maybe_repair_after_insertion(
        &mut self,
        mut vertex_key: VertexKey,
        hint: Option<CellKey>,
    ) -> Result<(VertexKey, bool), InsertionError>
    where
        K::Scalar: ScalarSummable,
    {
        let topology = self.tri.topology_guarantee();
        if !self.should_run_delaunay_repair_for(
            topology,
            self.insertion_state.delaunay_repair_insertion_count,
        ) {
            return Ok((vertex_key, false));
        }

        let vertex_uuid = self
            .tri
            .tds
            .get_vertex_by_key(vertex_key)
            .map(Vertex::uuid)
            .ok_or_else(|| InsertionError::CavityFilling {
                message: format!("Inserted vertex {vertex_key:?} missing before Delaunay repair"),
            })?;
        let seed_cells: Vec<CellKey> = self.tri.adjacent_cells(vertex_key).collect();
        let hint_seed = hint.and_then(|ck| {
            if !self.tri.tds.contains_cell(ck) {
                if std::env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
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
            if !contains_vertex && std::env::var_os("DELAUNAY_REPAIR_TRACE").is_some() {
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

        let mut used_heuristic = false;
        if Self::force_heuristic_rebuild_enabled() {
            used_heuristic = self
                .run_flip_repair_fallbacks(seed_ref)
                .map_err(|fallback_err| InsertionError::CavityFilling {
                    message: format!(
                        "Delaunay repair failed (forced heuristic rebuild); robust repair fallback also failed; heuristic rebuild fallback also failed ({fallback_err})"
                    ),
                })?;
            if used_heuristic {
                vertex_key = self.remap_vertex_key_by_uuid(vertex_uuid)?;
            }
        } else {
            let repair_result = {
                let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
                repair_delaunay_with_flips_k2_k3(tds, kernel, seed_ref, topology).map(|_| ())
            };

            match repair_result {
                Ok(()) => {}
                Err(
                    e @ (DelaunayRepairError::NonConvergent { .. }
                    | DelaunayRepairError::PostconditionFailed { .. }),
                ) => {
                    // Deterministic rebuild fallback to avoid committing a non-Delaunay triangulation.
                    //
                    // NOTE: This is intentionally expensive, but is only triggered when local repair
                    // fails to converge or leaves a detectable Delaunay violation.
                    used_heuristic = self
                        .run_flip_repair_fallbacks(seed_ref)
                        .map_err(|fallback_err| InsertionError::CavityFilling {
                            message: format!(
                                "Delaunay repair failed ({e}); robust repair fallback also failed; heuristic rebuild fallback also failed ({fallback_err})"
                            ),
                        })?;
                    if used_heuristic {
                        vertex_key = self.remap_vertex_key_by_uuid(vertex_uuid)?;
                    }
                }
                Err(e) => {
                    return Err(InsertionError::CavityFilling {
                        message: format!("Delaunay repair failed: {e}"),
                    });
                }
            }
        }

        // Topology safety-net: flip-based repair is a topological operation and must not
        // violate the requested topology guarantee.
        //
        // In practice, higher-dimensional flip sequences can transiently (or permanently)
        // introduce PL-manifold violations (e.g., disconnected ridge links). Catch those
        // locally and surface an insertion error so the outer transactional guard can roll
        // back the insertion.
        if topology.requires_ridge_links() {
            let local_cells: Vec<CellKey> = self.tri.adjacent_cells(vertex_key).collect();
            if !local_cells.is_empty()
                && let Err(err) = validate_ridge_links_for_cells(&self.tri.tds, &local_cells)
            {
                return Err(InsertionError::TopologyValidationFailed {
                    message: "Topology invalid after Delaunay repair".to_string(),
                    source: TriangulationValidationError::from(err),
                });
            }
        }
        Ok((vertex_key, used_heuristic))
    }

    fn maybe_check_after_insertion(&self) -> Result<(), InsertionError>
    where
        K::Scalar: ScalarSummable,
    {
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
                message: e.to_string(),
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
    /// * `vertex` - Reference to the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of cells that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns error if the vertex-cell incidence cannot be rebuilt, indicating data structure corruption.
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
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Get a vertex to remove
    /// let vertex_to_remove = dt.vertices().next().unwrap().1.clone();
    /// let cells_before = dt.number_of_cells();
    ///
    /// // Remove the vertex and all cells containing it
    /// let cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();
    /// println!("Removed {} cells along with the vertex", cells_removed);
    ///
    /// // Vertex removal preserves Levels 1–3 but may not preserve the Delaunay property.
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// ```
    pub fn remove_vertex(
        &mut self,
        vertex: &Vertex<K::Scalar, U, D>,
    ) -> Result<usize, TriangulationValidationError>
    where
        K::Scalar: ScalarSummable,
    {
        let Some(vertex_key) = self.tri.tds.vertex_key_from_uuid(&vertex.uuid()) else {
            return Ok(0);
        };

        // Fast path: inverse k=1 flip when the vertex star is a simplex.
        let mut seed_cells: Option<CellKeyBuffer> = None;
        let cells_removed = match apply_bistellar_flip_k1_inverse(
            &mut self.tri.tds,
            &self.tri.kernel,
            vertex_key,
        ) {
            Ok(info) => {
                seed_cells = Some(info.new_cells);
                info.removed_cells.len()
            }
            Err(FlipError::NeighborWiring { message }) => {
                return Err(TdsValidationError::InconsistentDataStructure {
                    message: format!("inverse k=1 flip failed during remove_vertex: {message}"),
                }
                .into());
            }
            Err(_) => self
                .tri
                .remove_vertex(vertex)
                .map_err(TriangulationValidationError::from)?,
        };

        let topology = self.tri.topology_guarantee();
        if self.should_run_delaunay_repair_for(topology, 0) {
            let seed_ref = seed_cells.as_deref();
            let (tds, kernel) = (&mut self.tri.tds, &self.tri.kernel);
            repair_delaunay_with_flips_k2_k3(tds, kernel, seed_ref, topology).map_err(|e| {
                TdsValidationError::InconsistentDataStructure {
                    message: format!("Delaunay repair failed after vertex removal: {e}"),
                }
            })?;
        }

        Ok(cells_removed)
    }

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
    pub fn is_valid(&self) -> Result<(), DelaunayTriangulationValidationError>
    where
        K::Scalar: ScalarSummable,
    {
        // Use fast flip-based verification (O(cells) instead of O(cells × vertices))
        self.is_delaunay_via_flips().map_err(|err| {
            // Convert DelaunayRepairError to DelaunayTriangulationValidationError
            TriangulationValidationError::from(TdsValidationError::InconsistentDataStructure {
                message: format!("Delaunay property violation detected: {err}"),
            })
            .into()
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
    pub fn is_delaunay_via_flips(&self) -> Result<(), DelaunayRepairError>
    where
        K::Scalar: ScalarSummable,
    {
        crate::core::algorithms::flips::verify_delaunay_via_flip_predicates(
            &self.tri.tds,
            &self.tri.kernel,
        )
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
    pub fn validate(&self) -> Result<(), DelaunayTriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.validate()?;
        self.is_valid()
    }

    /// Create a `DelaunayTriangulation` from a deserialized `Tds` with a default kernel.
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::core::triangulation_data_structure::Tds;
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
        Self {
            tri: Triangulation {
                kernel,
                tds,
                validation_policy: ValidationPolicy::OnSuspicion,
                topology_guarantee: TopologyGuarantee::DEFAULT,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        }
    }

    /// Create a `DelaunayTriangulation` from a deserialized `Tds` with an explicit topology guarantee.
    ///
    /// This is identical to [`from_tds`](Self::from_tds), but allows callers to set the desired
    /// [`TopologyGuarantee`] at construction time.
    #[must_use]
    pub const fn from_tds_with_topology_guarantee(
        tds: Tds<K::Scalar, U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Self {
        Self {
            tri: Triangulation {
                kernel,
                tds,
                validation_policy: ValidationPolicy::OnSuspicion,
                topology_guarantee,
            },
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
        }
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
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport>
    where
        K::Scalar: CoordinateScalar,
    {
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
                if let Err(e) = self.is_valid() {
                    report.violations.push(InvariantViolation {
                        kind: InvariantKind::DelaunayProperty,
                        error: e.into(),
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

/// Custom `Deserialize` implementation for the common case: `FastKernel<f64>` with no custom data.
///
/// This specialization provides convenient deserialization for the most common use case:
/// triangulations with `f64` coordinates, `FastKernel`, and no custom vertex/cell data.
///
/// # Why This Specialization?
///
/// Kernels are stateless and can be reconstructed on deserialization. We only serialize
/// the `Tds` (which contains all the geometric and topological data), then reconstruct
/// the kernel wrapper on deserialization.
///
/// This specialization is limited to `FastKernel<f64>` because:
/// - It's the most common configuration (matches `DelaunayTriangulation::new()` default)
/// - Rust doesn't allow overlapping `impl` blocks for generic types
/// - Custom kernels are rare and can deserialize manually
///
/// # Note on Locate Hint Persistence
/// The internal `insertion_state.last_inserted_cell` "locate hint" is intentionally **not** serialized.
/// Deserialization reconstructs a fresh triangulation via [`from_tds()`](Self::from_tds),
/// which resets the hint to `None`. This only affects performance for the first few
/// insertions after loading.
///
/// # Usage with Custom Kernels
///
/// If you're using a custom kernel (e.g., `RobustKernel`) or custom data types,
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
/// let dt = DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::new(&vertices)?;
/// let json = serde_json::to_string(&dt)?;
///
/// // Deserialize with custom kernel
/// let tds: Tds<f64, (), (), 3> = serde_json::from_str(&json)?;
/// let dt_robust = DelaunayTriangulation::from_tds(tds, RobustKernel::new());
/// # Ok(())
/// # }
/// ```
impl<'de, const D: usize> Deserialize<'de> for DelaunayTriangulation<FastKernel<f64>, (), (), D>
where
    Tds<f64, (), (), D>: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let tds = Tds::deserialize(deserializer)?;
        Ok(Self::from_tds(tds, FastKernel::new()))
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
/// use delaunay::core::delaunay_triangulation::DelaunayRepairPolicy;
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
/// use delaunay::core::delaunay_triangulation::DelaunayRepairHeuristicConfig;
///
/// let config = DelaunayRepairHeuristicConfig {
///     shuffle_seed: Some(7),
///     perturbation_seed: Some(11),
/// };
/// assert_eq!(config.shuffle_seed, Some(7));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DelaunayRepairHeuristicConfig {
    /// Optional RNG seed used to shuffle vertex insertion order.
    pub shuffle_seed: Option<u64>,
    /// Optional seed used to vary the deterministic perturbation pattern.
    pub perturbation_seed: Option<u64>,
}

impl DelaunayRepairHeuristicConfig {
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
/// use delaunay::core::delaunay_triangulation::DelaunayRepairHeuristicSeeds;
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
/// use delaunay::core::delaunay_triangulation::DelaunayRepairOutcome;
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
/// use delaunay::core::delaunay_triangulation::DelaunayCheckPolicy;
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
    use crate::core::algorithms::flips::DelaunayRepairError;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::triangulation::flips::BistellarFlips;
    use crate::vertex;
    use rand::{RngExt, SeedableRng};
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
    fn test_new_with_options_smoke_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let opts = ConstructionOptions::default().with_retry_policy(RetryPolicy::Disabled);
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
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

        let preprocess = DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::preprocess_vertices_for_construction(
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

        let result = DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::preprocess_vertices_for_construction(
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
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(dt.validate().is_ok());
    }

    fn coord_sequence_3d(vertices: &[Vertex<f64, (), 3>]) -> Vec<[f64; 3]> {
        vertices.iter().map(Into::into).collect()
    }

    #[test]
    fn test_insertion_order_lexicographic_is_deterministic_across_permutations_3d() {
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

        let expected_vertices = vertices_from_coords_permutation_3d(&coords, permutations[0]);
        let expected = coord_sequence_3d(&order_vertices_lexicographic(expected_vertices));

        for perm in &permutations[1..] {
            let vertices = vertices_from_coords_permutation_3d(&coords, perm);
            let got = coord_sequence_3d(&order_vertices_lexicographic(vertices));
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_insertion_order_morton_is_deterministic_across_permutations_3d() {
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

        let expected_vertices = vertices_from_coords_permutation_3d(&coords, permutations[0]);
        let expected = coord_sequence_3d(&order_vertices_morton(expected_vertices));

        for perm in &permutations[1..] {
            let vertices = vertices_from_coords_permutation_3d(&coords, perm);
            let got = coord_sequence_3d(&order_vertices_morton(vertices));
            assert_eq!(got, expected);
        }
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

        let expected_vertices = vertices_from_coords_permutation_3d(&coords, permutations[0]);
        let expected = coord_sequence_3d(&order_vertices_hilbert(expected_vertices));

        for perm in &permutations[1..] {
            let vertices = vertices_from_coords_permutation_3d(&coords, perm);
            let got = coord_sequence_3d(&order_vertices_hilbert(vertices));
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_new_with_options_lexicographic_and_morton_smoke_3d() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.25, 0.25, 0.25]),
        ];

        for insertion_order in [
            InsertionOrderStrategy::Lexicographic,
            InsertionOrderStrategy::Morton,
            InsertionOrderStrategy::Hilbert,
        ] {
            let opts = ConstructionOptions::default()
                .with_insertion_order(insertion_order)
                .with_retry_policy(RetryPolicy::Disabled);

            let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
                DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

            assert_eq!(dt.number_of_vertices(), 5);
            assert!(dt.validate().is_ok());
        }
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

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
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

        let dt_new: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_empty: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::empty();
        assert_eq!(dt_empty.topology_guarantee(), TopologyGuarantee::PLManifold);

        let dt_with_kernel: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices).unwrap();

        assert_eq!(
            dt_with_kernel.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );

        let dt_from_tds: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(dt_new.tds().clone(), FastKernel::new());
        assert_eq!(
            dt_from_tds.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
    }

    #[test]
    fn test_set_topology_guarantee_updates_underlying_triangulation() {
        init_tracing();
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::empty();

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

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
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
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::empty();
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
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 1> =
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
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> = DelaunayTriangulation::empty();

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
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
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
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::EveryN(NonZeroUsize::new(2).unwrap()));
        let topology = dt.topology_guarantee();

        assert!(!dt.should_run_delaunay_repair_for(topology, 1));
        assert!(dt.should_run_delaunay_repair_for(topology, 2));
    }

    #[test]
    fn test_run_flip_repair_fallbacks_smoke_ok_with_local_seed() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 0.2]),
        ];
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let before_vertices = dt.number_of_vertices();
        let before_cells = dt.number_of_cells();

        let seed_cell = dt.cells().next().unwrap().0;
        let seeds = [seed_cell];

        let _ = dt.run_flip_repair_fallbacks(Some(&seeds)).unwrap();

        assert_eq!(dt.number_of_vertices(), before_vertices);
        assert_eq!(dt.number_of_cells(), before_cells);
        assert!(dt.validate().is_ok());
    }

    #[test]
    fn test_insert_remaps_vertex_key_after_forced_heuristic_rebuild() {
        init_tracing();
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let inserted = vertex!([0.25, 0.25]);
        let inserted_uuid = inserted.uuid();
        let _guard = ForceHeuristicRebuildGuard::enable();

        let (outcome, _stats) = dt.insert_with_statistics(inserted).unwrap();
        let InsertionOutcome::Inserted { vertex_key, .. } = outcome else {
            panic!("Expected successful insertion outcome");
        };

        let remapped = dt
            .tri
            .tds
            .vertex_key_from_uuid(&inserted_uuid)
            .expect("Inserted vertex UUID missing after forced heuristic rebuild");

        assert_eq!(vertex_key, remapped);
        assert!(
            dt.insertion_state.last_inserted_cell.is_none(),
            "Heuristic rebuild should clear locate hint"
        );
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

            let Ok(mut dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), DIM>::new(&vertices)
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

        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
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

        let vertex_to_remove = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(_, v)| *v)
            .expect("Inserted vertex not found");

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_cells = dt.remove_vertex(&vertex_to_remove).unwrap();

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

        let mut dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let result = dt.repair_delaunay_with_flips();
        assert!(
            !matches!(result, Err(DelaunayRepairError::InvalidTopology { .. })),
            "Flip-based repair should be admissible under PLManifold topology"
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
        let dt_with_kernel: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices).unwrap();
        assert_eq!(
            dt_with_kernel.validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        // from_tds() is a separate constructor path (const-friendly), and should also
        // default to OnSuspicion.
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let dt_from_tds: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
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

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices);

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

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 3>, _> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices);

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

        let dt: DelaunayTriangulation<FastKernel<f32>, (), (), 2> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices).unwrap();

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

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(&FastKernel::new(), &vertices);

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

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let json = serde_json::to_string(&dt).unwrap();
        let roundtrip: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            serde_json::from_str(&json).unwrap();

        assert_eq!(roundtrip.number_of_vertices(), dt.number_of_vertices());
        assert_eq!(roundtrip.number_of_cells(), dt.number_of_cells());

        // `insertion_state.last_inserted_cell` is a performance-only locate hint and is intentionally not
        // persisted across serde round-trips (it is reset to `None` in `from_tds`).
        assert!(roundtrip.insertion_state.last_inserted_cell.is_none());
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
        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
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
            DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::new_with_construction_statistics(
                &vertices,
            )
            .unwrap();
        assert_eq!(dt.number_of_vertices(), 5);
        assert_eq!(stats.inserted, 5);
        assert!(dt.validate().is_ok());
    }
}
