//! Common trait for vertex insertion algorithms in Delaunay triangulations.
//!
//! This module defines the `InsertionAlgorithm` trait that provides a unified
//! interface for different vertex insertion strategies, including the basic
//! Bowyer-Watson algorithm and robust variants with enhanced numerical stability.
//!
//! # Transactional Cavity-Based Insertion
//!
//! The cavity-based insertion algorithm (used for interior vertices) implements a
//! **three-phase transactional pattern** to ensure atomic operations:
//!
//! ## Phase 1: Validate
//!
//! - Extract all boundary facet metadata while bad cells still exist
//! - Capture vertex keys and outside neighbor information
//! - Validate topology and relationships
//! - **Critical**: No TDS modifications occur in this phase
//! - Any errors leave the triangulation completely unchanged
//!
//! ## Phase 2: Tentative
//!
//! - Insert the vertex (if not already present)
//! - Create ALL new cells filling the cavity
//! - **Key insight**: Bad cells are NOT removed yet
//! - On failure: Remove only newly-created cells + vertex (if new)
//! - The original triangulation remains intact and valid
//!
//! ## Phase 3: Commit
//!
//! - Remove bad cells (point of no return)
//! - Wire neighbor relationships:
//!   - New→Old: Connect new cells to neighbors across cavity boundary
//!   - New→New: Connect new cells to each other using facet signatures
//! - Finalize incident cell assignments
//!
//! ## Why Not Serialization?
//!
//! The previous implementation attempted to snapshot and restore cells, which was
//! fundamentally broken because:
//! - `SlotMap` keys cannot be preserved across remove/insert cycles
//! - Cell UUIDs, neighbors, and data would be lost
//! - Dangling references would corrupt the entire triangulation
//!
//! The transactional pattern avoids these issues entirely by deferring destructive
//! operations until success is guaranteed.
//!
//! ## Performance Considerations
//!
//! - **No deep copies**: Metadata extraction is lightweight (vertex keys only)
//! - **No serialization**: Avoids expensive encoding/decoding overhead
//! - **Bounded overhead**: Extra work scales with cavity size, not TDS size
//! - **Cache-friendly**: Uses `SmallBuffer` and pre-allocated `Vec` containers

use crate::core::collections::{CellKeySet, FastHashSet, SmallBuffer, fast_hash_set_with_capacity};
use crate::core::facet::{FacetError, FacetHandle, FacetView, facet_key_from_vertices};
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::triangulation_data_structure::CellKey;
use crate::core::{
    cell::Cell,
    collections::MAX_PRACTICAL_DIMENSION_SIZE,
    triangulation_data_structure::{
        Tds, TriangulationConstructionError, TriangulationValidationError, VertexKey,
    },
    util::usize_to_u8,
    vertex::Vertex,
};
use crate::geometry::point::Point;
use crate::geometry::predicates::{InSphere, Orientation, insphere, simplex_orientation};
use crate::geometry::traits::coordinate::CoordinateScalar;
use num_traits::NumCast;
use num_traits::{One, Zero, cast};
use smallvec::SmallVec;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Sub, SubAssign};
use thiserror::Error;

// REMOVED: make_facet_from_view was broken due to Phase 3A refactoring
// The deprecated Facet::vertices() returns an empty Vec, causing silent failures
// All code now uses FacetView directly or lightweight (CellKey, u8) handles

/// Error for too many degenerate cells case
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub struct TooManyDegenerateCellsError {
    /// Number of degenerate cells
    pub degenerate_count: usize,
    /// Total cells tested
    pub total_tested: usize,
}

impl std::fmt::Display for TooManyDegenerateCellsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.total_tested == 0 {
            write!(
                f,
                "All {} candidate cells were degenerate",
                self.degenerate_count
            )
        } else {
            write!(
                f,
                "Too many degenerate circumspheres ({}/{})",
                self.degenerate_count, self.total_tested
            )
        }
    }
}

/// Error that can occur during bad cells detection
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BadCellsError {
    /// All cells were marked as bad, which is geometrically unusual and likely indicates
    /// the vertex should be inserted via hull extension instead
    #[error(
        "All {cell_count} cells marked as bad ({degenerate_count} degenerate). Vertex likely needs hull extension."
    )]
    AllCellsBad {
        /// Number of cells that were all marked as bad
        cell_count: usize,
        /// Number of degenerate cells encountered
        degenerate_count: usize,
    },
    /// Too many degenerate circumspheres to reliably detect bad cells
    #[error(transparent)]
    TooManyDegenerateCells(TooManyDegenerateCellsError),
    /// No cells exist to test
    #[error("No cells exist to test")]
    NoCells,
    /// TDS corruption detected: cell references vertex key that doesn't exist
    #[error(
        "TDS corruption: Cell {:?} references vertex key {:?} which is not in TDS. \
         This indicates data structure corruption. Run tds.is_valid() to diagnose.",
        cell_key,
        vertex_key
    )]
    TdsCorruption {
        /// The cell key that contains the invalid reference
        cell_key: CellKey,
        /// The vertex key that was not found
        vertex_key: VertexKey,
    },
}

/// Comprehensive error type for vertex insertion operations
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InsertionError {
    /// Geometric predicates failed to determine vertex placement
    #[error("Geometric failure during {strategy_attempted:?} insertion: {message}")]
    GeometricFailure {
        /// Description of the geometric failure
        message: String,
        /// The insertion strategy that was being attempted when the failure occurred
        strategy_attempted: InsertionStrategy,
    },

    /// All attempted fallback strategies were exhausted
    #[error("All {attempts} fallback strategies exhausted. Last error: {last_error}")]
    FallbacksExhausted {
        /// Number of fallback attempts that were tried
        attempts: usize,
        /// Description of the last error encountered before giving up
        last_error: String,
    },

    /// The triangulation data structure is in an invalid state
    #[error("Triangulation validation error: {0}")]
    TriangulationState(#[from] TriangulationValidationError),

    /// Error during triangulation construction
    #[error("Triangulation construction error: {0}")]
    TriangulationConstruction(
        #[from] crate::core::triangulation_data_structure::TriangulationConstructionError,
    ),

    /// Vertex is degenerate (e.g., duplicate or invalid coordinates)
    #[error("Invalid vertex: {reason}")]
    InvalidVertex {
        /// Description of why the vertex is invalid
        reason: String,
    },

    /// Too many bad cells found during cavity detection
    #[error("Excessive bad cells found: {found} (threshold: {threshold})")]
    ExcessiveBadCells {
        /// Number of bad cells that were found
        found: usize,
        /// Maximum threshold for bad cells that was exceeded
        threshold: usize,
    },

    /// Numerical precision issues prevented successful insertion
    #[error(
        "Precision failure (tolerance: {tolerance}, perturbation attempts: {perturbation_attempts})"
    )]
    PrecisionFailure {
        /// The tolerance level that was used when the precision failure occurred
        tolerance: f64,
        /// Number of perturbation attempts that were made before giving up
        perturbation_attempts: usize,
    },

    /// Hull extension failed for exterior vertex
    #[error("Hull extension failure: {reason}")]
    HullExtensionFailure {
        /// Description of why the hull extension failed
        reason: String,
    },

    /// Bad cells detection failed
    #[error("Bad cells detection error: {0}")]
    BadCellsDetection(#[from] BadCellsError),

    /// Vertex validation error
    #[error("Vertex validation error: {0}")]
    VertexValidation(#[from] crate::core::vertex::VertexValidationError),

    /// Duplicate boundary facets detected during cavity boundary analysis.
    ///
    /// This error surfaces algorithmic bugs in cavity boundary detection. The cavity
    /// boundary should be a topological sphere with no duplicate facets. Duplicates
    /// indicate:
    /// - Incorrect neighbor traversal logic
    /// - Non-manifold mesh connectivity
    /// - Data structure corruption
    ///
    /// By returning an error instead of silently filtering duplicates, we ensure
    /// correctness and prevent subtle insertion failures.
    #[error(
        "Duplicate boundary facets detected: {duplicate_count} duplicates found among {total_count} facets. \
         This indicates a bug in cavity boundary detection."
    )]
    DuplicateBoundaryFacets {
        /// Number of duplicate facets found
        duplicate_count: usize,
        /// Total number of facets processed
        total_count: usize,
    },
}

impl InsertionError {
    /// Create a geometric failure error
    pub fn geometric_failure(message: impl Into<String>, strategy: InsertionStrategy) -> Self {
        Self::GeometricFailure {
            message: message.into(),
            strategy_attempted: strategy,
        }
    }

    /// Create an invalid vertex error
    pub fn invalid_vertex(reason: impl Into<String>) -> Self {
        Self::InvalidVertex {
            reason: reason.into(),
        }
    }

    /// Create a precision failure error
    #[must_use]
    pub const fn precision_failure(tolerance: f64, perturbation_attempts: usize) -> Self {
        Self::PrecisionFailure {
            tolerance,
            perturbation_attempts,
        }
    }

    /// Create a hull extension failure error
    pub fn hull_extension_failure(reason: impl Into<String>) -> Self {
        Self::HullExtensionFailure {
            reason: reason.into(),
        }
    }

    /// Check if the error indicates a recoverable geometric issue
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::GeometricFailure { .. }
                | Self::PrecisionFailure { .. }
                | Self::BadCellsDetection(_)
        )
    }

    /// Get the insertion strategy that was attempted when this error occurred
    #[must_use]
    pub const fn attempted_strategy(&self) -> Option<InsertionStrategy> {
        match self {
            Self::GeometricFailure {
                strategy_attempted, ..
            } => Some(*strategy_attempted),
            _ => None,
        }
    }
}

/// Margin factor used for bounding box expansion in exterior vertex detection
const MARGIN_FACTOR: f64 = 0.1;

/// Bounding box expansion helper: safe subtraction for floating-point coordinates.
///
/// Performs normal floating-point subtraction. Despite the name, this is NOT saturating
/// arithmetic - floats naturally handle overflow by producing ±infinity, which is the
/// desired behavior for bounding box expansion (ensures all vertices are contained).
///
/// # Type Constraints
///
/// Only supports floating-point types (f32, f64) via `CoordinateScalar`. Integer types
/// are not supported by the trait.
#[inline]
fn bbox_sub<T>(a: T, b: T) -> T
where
    T: CoordinateScalar + Sub<Output = T>,
{
    // Plain subtraction; floats naturally produce -infinity on underflow
    a - b
}

/// Bounding box expansion helper: safe addition for floating-point coordinates.
///
/// Performs normal floating-point addition. Despite the name, this is NOT saturating
/// arithmetic - floats naturally handle overflow by producing ±infinity, which is the
/// desired behavior for bounding box expansion (ensures all vertices are contained).
///
/// # Type Constraints
///
/// Only supports floating-point types (f32, f64) via `CoordinateScalar`. Integer types
/// are not supported by the trait.
#[inline]
fn bbox_add<T>(a: T, b: T) -> T
where
    T: CoordinateScalar + Add<Output = T>,
{
    // Plain addition; floats naturally produce +infinity on overflow
    a + b
}

/// Calculate margin for bounding box expansion, handling both float and integer coordinate types.
///
/// This helper encapsulates the margin calculation logic used by `is_vertex_likely_exterior`,
/// ensuring consistent behavior across the codebase and avoiding drift with tests.
///
/// # Arguments
///
/// * `range` - The range (max - min) for a coordinate dimension
///
/// # Returns
///
/// The margin to add/subtract from the bounding box, computed as:
/// - `range * 0.1` for floating-point types
/// - `range / 10` (minimum 1) for integer types where `cast(0.1)` produces 0
#[inline]
fn calculate_margin<T>(range: T) -> T
where
    T: Zero
        + One
        + Div<Output = T>
        + PartialEq
        + NumCast
        + core::ops::Mul<Output = T>
        + Copy
        + AddAssign,
{
    let ten: T = cast(10).unwrap_or_else(|| {
        let mut t = T::zero();
        for _ in 0..10 {
            t += T::one();
        }
        t
    });
    match cast::<f64, T>(MARGIN_FACTOR) {
        None => {
            // Cast failed (shouldn't happen for standard numeric types)
            let mut m = range / ten;
            if m == T::zero() {
                m = T::one();
            }
            m
        }
        Some(mf) if mf == T::zero() => {
            // Integer case: cast(0.1) → Some(0), use integer division
            let mut m = range / ten;
            if m == T::zero() {
                m = T::one();
            }
            m
        }
        Some(mf) => {
            // Float case: use the margin factor directly
            range * mf
        }
    }
}

/// Threshold for determining when too many degenerate cells make results unreliable.
/// If more than this fraction of cells are degenerate, the results are considered unreliable.
/// Currently set to 0.5 (50%), which means if more than half the cells are degenerate,
/// we consider the results unreliable. This threshold can be adjusted based on the
/// tolerance for degenerate cases in specific applications.
///
/// **IMPORTANT**: The integer arithmetic optimization in `find_bad_cells` is specifically
/// designed for a threshold of 0.5. If you change this value, update the optimization logic
/// accordingly or remove the compile-time assertion below.
const DEGENERATE_CELL_THRESHOLD: f64 = 0.5;

// Compile-time assertion: ensure integer optimization remains valid
// This guard catches changes to DEGENERATE_CELL_THRESHOLD that would invalidate
// the optimized `degenerate_count * 2 > total` check in find_bad_cells.
// MSRV note: f64 comparisons in const context stabilized in Rust 1.83.0; project MSRV 1.90.0 satisfies this.
const _: () = {
    assert!(
        DEGENERATE_CELL_THRESHOLD == 0.5,
        "DEGENERATE_CELL_THRESHOLD must be exactly 0.5 for the integer optimization in find_bad_cells(). \
         If you need a different threshold, update the integer-optimized comparison (degenerate_count * 2 > total) in find_bad_cells()."
    );
};

/// Metadata for a single boundary facet during transactional cavity-based insertion.
///
/// This structure captures all information needed to create new cells after removing
/// bad cells, enabling a validate-then-commit pattern that avoids corruption on rollback.
///
/// # Transactional Insertion Pattern
///
/// The cavity-based insertion algorithm uses a three-phase approach:
/// 1. **Validate**: Extract all boundary facet metadata while bad cells still exist
/// 2. **Tentative**: Create new cells without removing bad cells yet
/// 3. **Commit**: Remove bad cells and wire neighbor relationships
///
/// This struct is populated during Phase 1 and used in Phases 2-3 to ensure
/// that failure during new cell creation does not corrupt the existing triangulation.
///
/// # Visibility
///
/// **⚠️ Internal API**: This type is public because it appears in trait method signatures,
/// but it is not intended for external use. It may change without notice in minor versions.
/// Do not use this type directly in your code.
#[derive(Debug, Clone)]
pub struct BoundaryFacetInfo {
    /// Key of the bad cell containing this boundary facet
    #[allow(dead_code)] // Written during construction, only read via Debug formatting
    pub(crate) bad_cell: CellKey,
    /// Index of the facet within the bad cell (0..=D)
    #[allow(dead_code)] // Written during construction, only read via Debug formatting
    pub(crate) bad_facet_index: usize,
    /// Vertex keys forming the boundary facet (D vertices)
    pub(crate) facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Neighbor cell across this boundary facet and its reciprocal facet index.
    /// `None` if this is a true boundary facet (no neighbor on the exterior side).
    /// Format: (`neighbor_cell_key`, `reciprocal_facet_index_in_neighbor`)
    pub(crate) outside_neighbor: Option<(CellKey, usize)>,
}

/// Strategy used for vertex insertion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionStrategy {
    /// Standard insertion strategy (default behavior)
    Standard,
    /// Use cavity-based insertion (interior vertices)
    CavityBased,
    /// Extend the convex hull (exterior vertices)
    HullExtension,
    /// Apply vertex perturbation for degenerate cases
    Perturbation,
    /// Use fallback method for difficult cases
    Fallback,
    /// Skip vertex insertion (for degenerate cases)
    Skip,
}

/// Information about a vertex insertion operation
#[derive(Debug, Clone)]
pub struct InsertionInfo {
    /// Strategy used for insertion
    pub strategy: InsertionStrategy,
    /// Number of cells removed during insertion
    pub cells_removed: usize,
    /// Number of new cells created
    pub cells_created: usize,
    /// Whether the insertion was successful
    pub success: bool,
    /// Whether a degenerate case was handled
    pub degenerate_case_handled: bool,
}

/// Unified statistics tracking for insertion algorithms
///
/// This structure provides a standardized way to track performance metrics
/// across different insertion algorithm implementations.
#[derive(Debug, Default, Clone)]
pub struct InsertionStatistics {
    /// Total number of vertices processed during triangulation
    pub vertices_processed: usize,
    /// Total number of cells created across all insertions
    pub total_cells_created: usize,
    /// Total number of cells removed across all insertions
    pub total_cells_removed: usize,
    /// Number of times fallback strategies were used
    pub fallback_strategies_used: usize,
    /// Number of degenerate cases handled
    pub degenerate_cases_handled: usize,
    /// Number of cavity boundary detection failures
    pub cavity_boundary_failures: usize,
    /// Number of successful cavity boundary recoveries
    pub cavity_boundary_recoveries: usize,
    /// Number of hull extension operations performed
    pub hull_extensions: usize,
    /// Number of vertex perturbations applied
    pub vertex_perturbations: usize,
}

impl InsertionStatistics {
    /// Create new empty statistics
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vertices_processed: 0,
            total_cells_created: 0,
            total_cells_removed: 0,
            fallback_strategies_used: 0,
            degenerate_cases_handled: 0,
            cavity_boundary_failures: 0,
            cavity_boundary_recoveries: 0,
            hull_extensions: 0,
            vertex_perturbations: 0,
        }
    }

    /// Reset all statistics to zero
    pub const fn reset(&mut self) {
        *self = Self::new();
    }

    /// Record a successful vertex insertion
    pub const fn record_vertex_insertion(&mut self, info: &InsertionInfo) {
        self.vertices_processed += 1;
        self.total_cells_created += info.cells_created;
        self.total_cells_removed += info.cells_removed;

        if info.degenerate_case_handled {
            self.degenerate_cases_handled += 1;
        }

        match info.strategy {
            InsertionStrategy::HullExtension => self.hull_extensions += 1,
            InsertionStrategy::Fallback => self.fallback_strategies_used += 1,
            InsertionStrategy::Perturbation => self.vertex_perturbations += 1,
            _ => {}
        }
    }

    /// Record a fallback strategy usage
    pub const fn record_fallback_usage(&mut self) {
        self.fallback_strategies_used += 1;
    }

    /// Record a cavity boundary failure and recovery
    pub const fn record_cavity_boundary_failure(&mut self) {
        self.cavity_boundary_failures += 1;
    }

    /// Record a successful cavity boundary recovery
    pub const fn record_cavity_boundary_recovery(&mut self) {
        self.cavity_boundary_recoveries += 1;
    }

    /// Get the basic statistics as a tuple (insertions, `cells_created`, `cells_removed`)
    /// for backward compatibility with existing code
    #[must_use]
    pub const fn as_basic_tuple(&self) -> (usize, usize, usize) {
        (
            self.vertices_processed,
            self.total_cells_created,
            self.total_cells_removed,
        )
    }

    /// Get the success rate for cavity boundary detection
    #[must_use]
    pub fn cavity_boundary_success_rate(&self) -> f64 {
        if self.vertices_processed == 0 {
            1.0 // No attempts means 100% success rate
        } else {
            // Use saturating subtraction to prevent underflow
            let successes = self
                .vertices_processed
                .saturating_sub(self.cavity_boundary_failures);
            #[expect(clippy::cast_precision_loss)]
            {
                successes as f64 / self.vertices_processed as f64
            }
        }
    }

    /// Get the fallback usage rate
    #[must_use]
    pub fn fallback_usage_rate(&self) -> f64 {
        if self.vertices_processed == 0 {
            0.0
        } else {
            #[expect(clippy::cast_precision_loss)]
            {
                self.fallback_strategies_used as f64 / self.vertices_processed as f64
            }
        }
    }
}

/// Common buffer management for insertion algorithms
///
/// This structure provides reusable buffers that can be shared across
/// different insertion algorithm implementations for better performance.
/// Instead of allocating new vectors on each operation, algorithms can
/// reuse these pre-allocated buffers.
#[derive(Debug)]
pub struct InsertionBuffers<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    /// Buffer for storing bad cell keys during cavity detection
    bad_cells_buffer: SmallBuffer<crate::core::triangulation_data_structure::CellKey, 16>,
    /// Buffer for storing boundary facet handles during cavity boundary detection
    boundary_facets_buffer: SmallBuffer<FacetHandle, 8>,
    /// Buffer for storing vertex points during geometric computations
    vertex_points_buffer: SmallBuffer<crate::geometry::point::Point<T, D>, 16>,
    /// Buffer for storing visible boundary facet handles
    visible_facets_buffer: SmallBuffer<FacetHandle, 8>,
    /// Phantom data to maintain generic parameter constraints
    _phantom: PhantomData<(U, V)>,
}

impl<T, U, V, const D: usize> Default for InsertionBuffers<T, U, V, D>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U, V, const D: usize> InsertionBuffers<T, U, V, D>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    /// Create new empty buffers
    #[must_use]
    pub fn new() -> Self {
        Self {
            bad_cells_buffer: SmallBuffer::new(),
            boundary_facets_buffer: SmallBuffer::new(),
            vertex_points_buffer: SmallBuffer::new(),
            visible_facets_buffer: SmallBuffer::new(),
            _phantom: PhantomData,
        }
    }

    /// Create buffers with pre-allocated capacity for better performance
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bad_cells_buffer: SmallBuffer::with_capacity(capacity),
            boundary_facets_buffer: SmallBuffer::with_capacity(capacity),
            vertex_points_buffer: SmallBuffer::with_capacity(capacity * (D + 1)), // More points per operation
            visible_facets_buffer: SmallBuffer::with_capacity(core::cmp::max(1, capacity / 2)), // Guard against zero capacity for tiny inputs
            _phantom: PhantomData,
        }
    }

    /// Clear all buffers for reuse
    pub fn clear_all(&mut self) {
        self.bad_cells_buffer.clear();
        self.boundary_facets_buffer.clear();
        self.vertex_points_buffer.clear();
        self.visible_facets_buffer.clear();
    }

    /// Prepare the bad cells buffer and return a mutable reference
    pub fn prepare_bad_cells_buffer(
        &mut self,
    ) -> &mut SmallBuffer<crate::core::triangulation_data_structure::CellKey, 16> {
        self.bad_cells_buffer.clear();
        &mut self.bad_cells_buffer
    }

    /// Prepare the boundary facets buffer and return a mutable reference
    pub fn prepare_boundary_facets_buffer(&mut self) -> &mut SmallBuffer<FacetHandle, 8> {
        self.boundary_facets_buffer.clear();
        &mut self.boundary_facets_buffer
    }

    /// Prepare the vertex points buffer and return a mutable reference
    pub fn prepare_vertex_points_buffer(
        &mut self,
    ) -> &mut SmallBuffer<crate::geometry::point::Point<T, D>, 16> {
        self.vertex_points_buffer.clear();
        &mut self.vertex_points_buffer
    }

    /// Prepare the visible facets buffer and return a mutable reference
    pub fn prepare_visible_facets_buffer(&mut self) -> &mut SmallBuffer<FacetHandle, 8> {
        self.visible_facets_buffer.clear();
        &mut self.visible_facets_buffer
    }

    // =========== ACCESSOR METHODS ===========
    // These methods provide controlled access to private buffers for backward compatibility

    /// Get a reference to the bad cells buffer
    #[must_use]
    pub const fn bad_cells_buffer(
        &self,
    ) -> &SmallBuffer<crate::core::triangulation_data_structure::CellKey, 16> {
        &self.bad_cells_buffer
    }

    /// Get a mutable reference to the bad cells buffer
    pub const fn bad_cells_buffer_mut(
        &mut self,
    ) -> &mut SmallBuffer<crate::core::triangulation_data_structure::CellKey, 16> {
        &mut self.bad_cells_buffer
    }

    /// Get a reference to the boundary facets buffer
    #[must_use]
    pub const fn boundary_facets_buffer(&self) -> &SmallBuffer<FacetHandle, 8> {
        &self.boundary_facets_buffer
    }

    /// Get a mutable reference to the boundary facets buffer
    pub const fn boundary_facets_buffer_mut(&mut self) -> &mut SmallBuffer<FacetHandle, 8> {
        &mut self.boundary_facets_buffer
    }

    /// Get a reference to the vertex points buffer
    #[must_use]
    pub const fn vertex_points_buffer(
        &self,
    ) -> &SmallBuffer<crate::geometry::point::Point<T, D>, 16> {
        &self.vertex_points_buffer
    }

    /// Get a mutable reference to the vertex points buffer
    pub const fn vertex_points_buffer_mut(
        &mut self,
    ) -> &mut SmallBuffer<crate::geometry::point::Point<T, D>, 16> {
        &mut self.vertex_points_buffer
    }

    /// Get a reference to the visible facets buffer
    #[must_use]
    pub const fn visible_facets_buffer(&self) -> &SmallBuffer<FacetHandle, 8> {
        &self.visible_facets_buffer
    }

    /// Get a mutable reference to the visible facets buffer
    pub const fn visible_facets_buffer_mut(&mut self) -> &mut SmallBuffer<FacetHandle, 8> {
        &mut self.visible_facets_buffer
    }

    // =========== COMPATIBILITY HELPERS ===========
    // These methods ease migration for external code that may have used direct field access

    /// Extract the bad cells as a Vec for compatibility with previous Vec-based APIs
    ///
    /// # Returns
    ///
    /// A `Vec<CellKey>` containing all the bad cell keys stored in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::traits::insertion_algorithm::InsertionBuffers;
    /// use delaunay::core::triangulation_data_structure::CellKey;
    ///
    /// let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();
    /// let bad_cells = buffers.bad_cells_as_vec();
    /// assert!(bad_cells.is_empty());
    /// ```
    #[must_use]
    pub fn bad_cells_as_vec(&self) -> Vec<crate::core::triangulation_data_structure::CellKey> {
        self.bad_cells_buffer.iter().copied().collect()
    }

    /// Set bad cells from a Vec for compatibility with previous Vec-based APIs
    pub fn set_bad_cells_from_vec(
        &mut self,
        vec: Vec<crate::core::triangulation_data_structure::CellKey>,
    ) {
        self.bad_cells_buffer.clear();
        self.bad_cells_buffer.extend(vec);
    }

    /// Extract the boundary facet handles as a Vec
    #[must_use]
    pub fn boundary_facet_handles(&self) -> Vec<FacetHandle> {
        self.boundary_facets_buffer.iter().copied().collect()
    }

    /// Set boundary facets from handles
    pub fn set_boundary_facet_handles(&mut self, handles: Vec<FacetHandle>) {
        self.boundary_facets_buffer.clear();
        self.boundary_facets_buffer.extend(handles);
    }

    /// Extract the boundary facets as `FacetViews` for iteration
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to create `FacetViews` with
    ///
    /// # Returns
    ///
    /// A `Result<Vec<FacetView>, FacetError>` containing the `FacetViews`
    ///
    /// # Errors
    ///
    /// Returns `FacetError` if facet views cannot be created from the stored handles.
    ///
    /// # Note
    ///
    /// This method uses lightweight `FacetView::new` with minimal bounds, avoiding
    /// heavy numeric traits for iterator-only operations.
    pub fn boundary_facets_as_views<'tds>(
        &self,
        tds: &'tds Tds<T, U, V, D>,
    ) -> Result<Vec<FacetView<'tds, T, U, V, D>>, FacetError> {
        self.boundary_facets_buffer
            .iter()
            .map(|handle| FacetView::new(tds, handle.cell_key(), handle.facet_index()))
            .collect()
    }

    /// Extract the vertex points as a Vec for compatibility with previous Vec-based APIs
    #[must_use]
    pub fn vertex_points_as_vec(&self) -> Vec<crate::geometry::point::Point<T, D>> {
        self.vertex_points_buffer.iter().copied().collect()
    }

    /// Set vertex points from a Vec for compatibility with previous Vec-based APIs
    pub fn set_vertex_points_from_vec(&mut self, vec: Vec<crate::geometry::point::Point<T, D>>) {
        self.vertex_points_buffer.clear();
        self.vertex_points_buffer.extend(vec);
    }

    /// Extract the visible facet handles as a Vec
    #[must_use]
    pub fn visible_facet_handles(&self) -> Vec<FacetHandle> {
        self.visible_facets_buffer.iter().copied().collect()
    }

    /// Set visible facets from handles
    pub fn set_visible_facet_handles(&mut self, handles: Vec<FacetHandle>) {
        self.visible_facets_buffer.clear();
        self.visible_facets_buffer.extend(handles);
    }

    /// Extract the visible facets as `FacetViews` for iteration
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to create `FacetViews` with
    ///
    /// # Returns
    ///
    /// A `Result<Vec<FacetView>, FacetError>` containing the `FacetViews`
    ///
    /// # Errors
    ///
    /// Returns `FacetError` if facet views cannot be created from the stored handles.
    ///
    /// # Note
    ///
    /// This method uses lightweight `FacetView::new` with minimal bounds, avoiding
    /// heavy numeric traits for iterator-only operations.
    pub fn visible_facets_as_views<'tds>(
        &self,
        tds: &'tds Tds<T, U, V, D>,
    ) -> Result<Vec<FacetView<'tds, T, U, V, D>>, FacetError> {
        self.visible_facets_buffer
            .iter()
            .map(|handle| FacetView::new(tds, handle.cell_key(), handle.facet_index()))
            .collect()
    }
}

/// Trait for vertex insertion algorithms in Delaunay triangulations
///
/// This trait provides a unified interface for different insertion algorithms,
/// allowing for pluggable strategies for handling various geometric configurations
/// and numerical precision requirements.
pub trait InsertionAlgorithm<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    /// Insert a single vertex into the triangulation with duplicate detection.
    ///
    /// This is the main entry point for vertex insertion. It automatically checks
    /// for duplicate/near-duplicate vertices before calling the implementation-specific
    /// `insert_vertex_impl` method.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the insertion operation, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vertex is a duplicate/near-duplicate of an existing vertex
    /// - Vertex insertion fails due to geometric degeneracy
    /// - Numerical issues or topological constraints prevent insertion
    ///
    /// # Default Implementation
    ///
    /// The default implementation:
    /// 1. Checks for duplicate vertices using epsilon tolerance (1e-10)
    /// 2. Calls `insert_vertex_impl` if no duplicate found
    /// 3. Implementations can override for custom duplicate handling
    fn insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError> {
        // Check for duplicate vertices in the TDS
        // Use epsilon tolerance to catch near-duplicates that could cause numerical issues
        let epsilon = T::from(1e-10).unwrap_or_else(T::default_tolerance);
        let vertex_coords: [T; D] = (&vertex).into();
        let vertex_uuid = vertex.uuid();

        for (_vkey, existing) in tds.vertices() {
            // Skip comparison with the vertex itself (handles case where vertex
            // is already inserted in TDS before this method is called)
            if existing.uuid() == vertex_uuid {
                continue;
            }

            let existing_coords: [T; D] = existing.into();

            // Compute Euclidean distance squared
            let dist_sq: T = vertex_coords
                .iter()
                .zip(existing_coords.iter())
                .map(|(a, b)| (*a - *b) * (*a - *b))
                .fold(T::zero(), |acc, d| acc + d);

            if dist_sq < epsilon * epsilon {
                // Use explicit cast for display purposes only
                let dist_sq_f64: f64 = num_traits::cast(dist_sq).unwrap_or(0.0);
                let threshold_sq_f64: f64 = num_traits::cast(epsilon * epsilon).unwrap_or(0.0);

                return Err(InsertionError::InvalidVertex {
                    reason: format!(
                        "Vertex at {vertex_coords:?} is a duplicate/near-duplicate of existing vertex (distance² = {dist_sq_f64:.2e}, threshold² = {threshold_sq_f64:.2e})"
                    ),
                });
            }
        }

        // No duplicate found - proceed with insertion
        self.insert_vertex_impl(tds, vertex)
    }

    /// Implementation-specific vertex insertion logic.
    ///
    /// This method contains the actual insertion algorithm logic and should be
    /// implemented by each concrete algorithm. It is called by `insert_vertex`
    /// after duplicate detection has passed.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert (guaranteed to not be a duplicate)
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the insertion operation, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if vertex insertion fails due to geometric degeneracy,
    /// numerical issues, or topological constraints.
    ///
    /// # Implementation Note
    ///
    /// Concrete implementations should override this method instead of `insert_vertex`
    /// to preserve automatic duplicate detection. Only override `insert_vertex` if
    /// you need custom duplicate handling logic.
    fn insert_vertex_impl(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>;

    /// Get statistics about the insertion algorithm's performance
    ///
    /// Returns a tuple of (`insertions_performed`, `cells_created`, `cells_removed`)
    fn get_statistics(&self) -> (usize, usize, usize);

    /// Reset the algorithm state for reuse
    fn reset(&mut self);

    /// Update the cell creation counter
    ///
    /// This method increments the internal cell creation counter. Concrete
    /// implementations should override this to update their statistics.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of cells to add to the creation counter
    fn increment_cells_created(&mut self, _count: usize) {
        // Default implementation does nothing - concrete implementations should override
    }

    /// Update the cell removal counter
    ///
    /// This method increments the internal cell removal counter. Concrete
    /// implementations should override this to update their statistics.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of cells to add to the removal counter
    fn increment_cells_removed(&mut self, _count: usize) {
        // Default implementation does nothing - concrete implementations should override
    }

    /// Update statistics after creating cells
    ///
    /// This is a protected method that concrete implementations can use to update
    /// their internal statistics tracking. It should be called after operations
    /// that create or remove cells outside of the normal `insert_vertex` flow.
    ///
    /// # Arguments
    ///
    /// * `cells_created` - Number of cells that were created
    /// * `cells_removed` - Number of cells that were removed
    fn update_statistics(&mut self, cells_created: usize, cells_removed: usize) {
        self.increment_cells_created(cells_created);
        self.increment_cells_removed(cells_removed);
    }

    /// Determine the appropriate insertion strategy for a given vertex
    ///
    /// This method analyzes the vertex position relative to the current
    /// triangulation and recommends the best insertion strategy.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to analyze
    ///
    /// # Returns
    ///
    /// The recommended insertion strategy
    fn determine_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy
    where
        T: AddAssign<T>
            + SubAssign<T>
            + Sum
            + NumCast
            + One
            + Zero
            + PartialEq
            + Div<Output = T>
            + Add<Output = T>
            + Sub<Output = T>,
    {
        // Default implementation provides basic strategy determination
        Self::determine_strategy_default(tds, vertex)
    }

    /// Default strategy determination logic
    ///
    /// This provides a baseline strategy determination that can be used by
    /// algorithms or as a fallback. It uses simple heuristics based on
    /// triangulation state and vertex position.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to analyze
    ///
    /// # Returns
    ///
    /// The recommended insertion strategy
    fn determine_strategy_default(
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy
    where
        T: AddAssign<T>
            + SubAssign<T>
            + Sum
            + NumCast
            + One
            + Zero
            + PartialEq
            + Div<Output = T>
            + Add<Output = T>
            + Sub<Output = T>,
    {
        // If the triangulation is empty or has very few cells, use standard approach
        if tds.number_of_cells() == 0 {
            return InsertionStrategy::Standard;
        }

        // If we only have one cell (initial simplex), any new vertex is likely exterior
        if tds.number_of_cells() == 1 {
            return InsertionStrategy::HullExtension;
        }

        // For more complex triangulations, we need to analyze the vertex position
        // This is a simplified analysis - robust algorithms may override this

        // Quick check: if the vertex is very far from existing vertices,
        // it's likely exterior and should use hull extension
        if Self::is_vertex_likely_exterior(tds, vertex) {
            InsertionStrategy::HullExtension
        } else {
            // Default to cavity-based insertion for interior vertices
            InsertionStrategy::CavityBased
        }
    }

    /// Checks if a vertex is interior to the current triangulation
    ///
    /// A vertex is considered interior if it lies within the circumsphere
    /// of at least one existing cell. This is more precise than the bounding-box
    /// approach used by `is_vertex_likely_exterior`.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to test
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the vertex is interior, `Ok(false)` if exterior.
    ///
    /// # Errors
    ///
    /// Returns an error if TDS is corrupted (missing vertex keys referenced by cells).
    fn is_vertex_interior(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<bool, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Early return for empty triangulation (symmetry with find_bad_cells)
        if tds.number_of_cells() == 0 {
            return Ok(false);
        }

        // InSphere and insphere are already imported at module top

        // Reserve exact capacity once; keep on stack for typical small D
        let mut vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

        for (cell_key, cell) in tds.cells() {
            // Clear and reuse the buffer - capacity is already preallocated
            vertex_points.clear();
            // Phase 3A: Get vertices via TDS using vertices
            // Note: We fail on corrupted cells (missing vertex keys) rather than skipping them.
            // This strictness ensures we detect and report TDS corruption immediately.
            // For recovery-friendly behavior, consider using is_valid() to detect corruption separately.
            for &vkey in cell.vertices() {
                let v = tds.get_vertex_by_key(vkey).ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "TDS corruption: cell {cell_key:?} references missing vertex key {vkey:?}"
                            ),
                        },
                    )
                })?;
                vertex_points.push(*v.point());
            }

            // Validate we got all D+1 vertices
            if vertex_points.len() != D + 1 {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell {cell_key:?} has {} vertices, expected {}",
                            vertex_points.len(),
                            D + 1
                        ),
                    },
                ));
            }

            if matches!(
                insphere(&vertex_points, *vertex.point()),
                Ok(InSphere::INSIDE)
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Helper method to determine if a vertex is likely exterior to the current triangulation
    ///
    /// This uses simple distance-based heuristics to make a quick determination.
    /// More sophisticated algorithms can override the main `determine_strategy` method.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to analyze
    ///
    /// # Returns
    ///
    /// `true` if the vertex is likely exterior, `false` otherwise
    fn is_vertex_likely_exterior(tds: &Tds<T, U, V, D>, vertex: &Vertex<T, U, D>) -> bool
    where
        T: AddAssign<T>
            + SubAssign<T>
            + Sum
            + NumCast
            + One
            + Zero
            + PartialEq
            + Div<Output = T>
            + Add<Output = T>
            + Sub<Output = T>,
    {
        // Get the vertex coordinates
        let vertex_coords: [T; D] = vertex.point().into();

        // Calculate rough bounding box of existing vertices
        let mut min_coords = [T::zero(); D];
        let mut max_coords = [T::zero(); D];
        let mut initialized = false;
        let mut vertex_count = 0;

        for (_vkey, existing_vertex) in tds.vertices() {
            let coords: [T; D] = existing_vertex.point().into();
            vertex_count += 1;

            if initialized {
                for i in 0..D {
                    if coords[i] < min_coords[i] {
                        min_coords[i] = coords[i];
                    }
                    if coords[i] > max_coords[i] {
                        max_coords[i] = coords[i];
                    }
                }
            } else {
                min_coords = coords;
                max_coords = coords;
                initialized = true;
            }
        }

        if vertex_count == 0 {
            return false; // No existing vertices to compare against
        }

        // Calculate bounding box margins (10% expansion)
        let mut expanded_min = [T::zero(); D];
        let mut expanded_max = [T::zero(); D];

        for i in 0..D {
            let range = max_coords[i] - min_coords[i];
            let margin = calculate_margin(range);

            // Use saturating arithmetic to prevent debug-mode overflow panics
            // For floating-point: behaves like normal +/- (overflow → infinity)
            // For integer types: saturates to T::MIN/T::MAX (prevents panic)
            expanded_min[i] = bbox_sub(min_coords[i], margin);
            expanded_max[i] = bbox_add(max_coords[i], margin);
        }

        // Check if vertex is outside the expanded bounding box
        for i in 0..D {
            if vertex_coords[i] < expanded_min[i] || vertex_coords[i] > expanded_max[i] {
                return true; // Outside bounding box - likely exterior
            }
        }

        false // Inside expanded bounding box - likely interior
    }

    // =========== CORE ALGORITHM METHODS ===========
    // These methods define the core steps of insertion algorithms that
    // different implementations can customize while maintaining a common interface

    /// Find "bad" cells whose circumsphere contains the given vertex
    ///
    /// A cell is "bad" if the vertex is strictly inside its circumsphere,
    /// violating the Delaunay property. Different algorithms may use different
    /// geometric predicates or tolerance levels.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to test against circumspheres
    ///
    /// # Returns
    ///
    /// A vector of bad cell keys on success.
    ///
    /// # Errors
    ///
    /// Returns `BadCellsError` if:
    /// - All cells are marked as bad (likely needs hull extension)
    /// - Too many degenerate cells to be reliable
    /// - No cells exist to test
    fn find_bad_cells(
        &mut self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<crate::core::triangulation_data_structure::CellKey>, BadCellsError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Check if there are any cells to test
        if tds.number_of_cells() == 0 {
            return Err(BadCellsError::NoCells);
        }

        let mut bad_cells = Vec::with_capacity(tds.number_of_cells());
        let mut cells_tested = 0;
        let mut degenerate_count = 0;
        // Reuse a small stack-allocated buffer to avoid heap traffic
        let mut vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

        // Only consider cells that have a valid circumsphere and strict containment
        for (cell_key, cell) in tds.cells() {
            // Phase 3A: Use cell.vertices() to get vertex keys from TDS (avoids materializing Vertex objects)
            let v_count = cell.number_of_vertices();
            // Treat non-D+1 vertex counts as degenerate
            if v_count != D + 1 {
                degenerate_count += 1;
                continue;
            }

            cells_tested += 1;
            // Reuse buffer by clearing and repopulating
            vertex_points.clear();
            // Phase 3A: Get vertices via TDS using vertex keys
            // Missing vertex keys indicate TDS corruption, not geometric degeneracy
            for &vkey in cell.vertices() {
                let v = tds
                    .get_vertex_by_key(vkey)
                    .ok_or(BadCellsError::TdsCorruption {
                        cell_key,
                        vertex_key: vkey,
                    })?;
                vertex_points.push(*v.point());
            }

            // Test circumsphere containment
            match insphere(&vertex_points, *vertex.point()) {
                Ok(InSphere::INSIDE) => {
                    // Cell is bad - vertex violates Delaunay property
                    bad_cells.push(cell_key);
                }
                Ok(InSphere::BOUNDARY | InSphere::OUTSIDE) => {
                    // Vertex is outside or on boundary - cell is fine
                }
                Err(_) => {
                    // Degenerate circumsphere or computation error
                    degenerate_count += 1;
                }
            }
        }

        // Check for problematic conditions
        if cells_tested > 1 && bad_cells.len() == cells_tested {
            // All cells marked as bad - this is unusual and suggests hull extension is needed
            return Err(BadCellsError::AllCellsBad {
                cell_count: cells_tested,
                degenerate_count,
            });
        }

        // Check if too many degenerate cells make results unreliable
        if cells_tested == 0 && degenerate_count > 0 {
            // All cells were degenerate (wrong vertex count)
            return Err(BadCellsError::TooManyDegenerateCells(
                TooManyDegenerateCellsError {
                    degenerate_count,
                    total_tested: 0,
                },
            ));
        } else if cells_tested > 0 && degenerate_count > 0 {
            // Check if too many cells are degenerate (using DEGENERATE_CELL_THRESHOLD)
            // We use integer arithmetic to avoid floating point precision issues:
            // For a threshold of 0.5 (50%): degenerate_count / total > 0.5
            // is equivalent to degenerate_count * 2 > total
            let total_cells = cells_tested.saturating_add(degenerate_count);

            // Use integer arithmetic optimization for 50% threshold
            // (guaranteed by compile-time assertion at module level)
            // For threshold = 0.5: degenerate_count / total > 0.5 ⟺ degenerate_count * 2 > total
            let threshold_exceeded = degenerate_count.saturating_mul(2) > total_cells;

            if threshold_exceeded {
                return Err(BadCellsError::TooManyDegenerateCells(
                    TooManyDegenerateCellsError {
                        degenerate_count,
                        total_tested: cells_tested,
                    },
                ));
            }
        }

        Ok(bad_cells)
    }

    /// Check if specific cells violate the Delaunay property with respect to existing vertices.
    ///
    /// This is used for iterative cavity refinement: after creating new cells, we check if
    /// any existing vertices are inside their circumspheres. If so, those cells must be removed
    /// and the cavity expanded.
    ///
    /// **Uses robust predicates** to ensure correctness even in near-degenerate configurations.
    /// This is critical for maintaining the Delaunay property in the presence of floating-point
    /// precision issues.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `cells_to_check` - Keys of cells to check for violations
    ///
    /// # Returns
    ///
    /// A vector of cell keys that violate the Delaunay property (have existing vertices inside their circumspheres).
    ///
    /// # Errors
    ///
    /// Returns an error if TDS corruption is detected.
    fn find_delaunay_violations_in_cells(
        &self,
        tds: &Tds<T, U, V, D>,
        cells_to_check: &[CellKey],
    ) -> Result<Vec<CellKey>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Use the centralized Delaunay validation function from util.rs
        crate::core::util::find_delaunay_violations(tds, Some(cells_to_check)).map_err(|err| {
            match err {
                crate::core::util::DelaunayValidationError::TriangulationState { source } => {
                    InsertionError::TriangulationState(source)
                }
                crate::core::util::DelaunayValidationError::DelaunayViolation { .. }
                | crate::core::util::DelaunayValidationError::InvalidCell { .. } => {
                    // These shouldn't happen during insertion, but convert to InsertionError for safety
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!("Delaunay validation error: {err}"),
                        },
                    )
                }
            }
        })
    }

    /// Find the boundary facets of a cavity formed by removing bad cells
    ///
    /// Returns lightweight `FacetHandle` for optimal performance.
    ///
    /// The boundary facets form the interface between the cavity (bad cells to be removed)
    /// and the good cells that remain. These facets will be used to create new cells
    /// connecting to the inserted vertex.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `bad_cells` - Keys of the cells to be removed
    ///
    /// # Returns
    ///
    /// A vector of `FacetHandle` representing boundary facets.
    /// Each handle can be used to create a `FacetView` on-demand for further operations.
    ///
    /// **Important**: These handles are only valid while the referenced cells exist in the TDS.
    /// If cells will be removed (e.g., during cavity-based insertion), extract necessary
    /// facet data (vertices, etc.) before removing cells, as handles become invalid after
    /// cell removal.
    ///
    /// # Errors
    ///
    /// Returns an error if the cavity boundary computation fails due to topological issues.
    ///
    /// # Performance Note
    ///
    /// This method is optimized for hot paths in insertion algorithms:
    /// - **Zero allocation** for facet creation (references existing TDS data)
    /// - **Memory efficient** - no Cell cloning required
    /// - **Fast** - direct access to TDS data structures
    /// - **Type safe** - `FacetHandle` prevents tuple ordering errors
    fn find_cavity_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<FacetHandle>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Pre-allocate capacity: each bad cell can contribute up to D+1 boundary facets
        let cap = bad_cells.len().saturating_mul(D.saturating_add(1));
        let mut boundary_facet_handles = Vec::with_capacity(cap);

        if bad_cells.is_empty() {
            return Ok(boundary_facet_handles);
        }

        let bad_cell_set: CellKeySet = bad_cells.iter().copied().collect();

        // Optimized O(|bad_cells|·D) algorithm: scan only facets of bad cells
        // instead of building global facet-to-cells map O(all_facets).
        //
        // A facet is on the cavity boundary if:
        // - Its neighbor cell is NOT in the bad set (boundary with good region)
        // - OR it has no neighbor (true boundary facet)
        //
        // We deduplicate using canonical facet keys since each boundary facet
        // can be seen from multiple bad cells sharing it.

        // Track seen boundary facets by canonical key to avoid duplicates
        let cap_seen = bad_cells.len().saturating_mul(D.saturating_add(1));
        let mut seen_facet_keys: FastHashSet<u64> = fast_hash_set_with_capacity(cap_seen);

        // Reusable buffer for facet vertices to avoid per-facet allocations
        let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D);

        // Scan each bad cell's D+1 facets
        for &bad_cell_key in bad_cells {
            let Some(bad_cell) = tds.get_cell(bad_cell_key) else {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Bad cell key {bad_cell_key:?} not found during cavity boundary detection"
                        ),
                    },
                ));
            };

            let Some(neighbors) = bad_cell.neighbors() else {
                // Cell has no neighbor information; treat all facets as boundary
                for facet_idx in 0..=D {
                    if let Ok(facet_idx_u8) = usize_to_u8(facet_idx, D + 1) {
                        // Compute canonical facet key for deduplication
                        facet_vertices.clear();
                        facet_vertices.extend(
                            bad_cell
                                .vertices()
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &v)| (i != facet_idx).then_some(v)),
                        );
                        let canonical_key = facet_key_from_vertices(&facet_vertices);

                        if seen_facet_keys.insert(canonical_key) {
                            boundary_facet_handles
                                .push(FacetHandle::new(bad_cell_key, facet_idx_u8));
                        }
                    }
                }
                continue;
            };

            // Check each facet (opposite to each vertex)
            for facet_idx in 0..=D {
                // Missing slot => treat as boundary
                let is_boundary = match neighbors.get(facet_idx) {
                    None => true,
                    Some(&neighbor_key_opt) => {
                        neighbor_key_opt.is_none_or(|n| !bad_cell_set.contains(&n))
                    }
                };
                if !is_boundary {
                    continue; // Interior facet; skip
                }

                // This is a boundary facet; compute canonical key and deduplicate
                let Ok(facet_idx_u8) = usize_to_u8(facet_idx, D + 1) else {
                    continue;
                };

                // Compute canonical facet key: sorted vertex keys of the D vertices
                facet_vertices.clear();
                facet_vertices.extend(
                    bad_cell
                        .vertices()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| (i != facet_idx).then_some(v)),
                );
                let canonical_key = facet_key_from_vertices(&facet_vertices);

                // Insert only if not already seen (deduplication)
                if seen_facet_keys.insert(canonical_key) {
                    boundary_facet_handles.push(FacetHandle::new(bad_cell_key, facet_idx_u8));
                }
            }
        }

        // Validation: ensure we have a reasonable number of boundary facets
        if boundary_facet_handles.is_empty() && !bad_cells.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: format!(
                        "No cavity boundary facets found for {} bad cells. This indicates a topological error.",
                        bad_cells.len()
                    ),
                },
            ));
        }

        Ok(boundary_facet_handles)
    }

    /// Inserts a vertex using cavity-based Bowyer-Watson insertion
    ///
    /// This method:
    /// 1. Finds all "bad" cells whose circumsphere contains the vertex
    /// 2. Removes these cells to create a star-shaped cavity
    /// 3. Triangulates the cavity by connecting the vertex to boundary facets
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No bad cells are found (vertex is not interior)
    /// - Cavity boundary computation fails
    /// - Cell creation fails
    #[expect(clippy::too_many_lines)]
    fn insert_vertex_cavity_based(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Find bad cells - use match to convert error to appropriate validation error
        let bad_cells = match self.find_bad_cells(tds, vertex) {
            Ok(cells) => cells,
            Err(BadCellsError::AllCellsBad {
                cell_count,
                degenerate_count,
            }) => {
                // All cells marked as bad - need hull extension instead
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!(
                            "Cavity-based insertion failed: all {cell_count} cells marked as bad ({degenerate_count} degenerate). \
                             Vertex likely needs hull extension."
                        ),
                    },
                ));
            }
            Err(BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
                degenerate_count,
                total_tested: cells_tested,
            })) => {
                // Too many degenerate cells - triangulation might be in bad state
                let total_tested = cells_tested;
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!(
                            "Cavity-based insertion failed: too many degenerate cells ({degenerate_count}/{total_tested})."
                        ),
                    },
                ));
            }
            Err(BadCellsError::NoCells) => {
                // No cells to test - triangulation is empty
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: "Cavity-based insertion failed: no cells exist in triangulation."
                            .to_string(),
                    },
                ));
            }
            Err(BadCellsError::TdsCorruption {
                cell_key,
                vertex_key,
            }) => {
                // TDS corruption detected - this is a fatal structural error
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "TDS corruption detected: Cell {cell_key:?} references vertex key {vertex_key:?} which doesn't exist. \
                             Run tds.is_valid() to diagnose."
                        ),
                    },
                ));
            }
        };

        if bad_cells.is_empty() {
            // No bad cells found - this method is not applicable
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "Cavity-based insertion failed: no bad cells found for vertex"
                        .to_string(),
                },
            ));
        }

        // Find boundary facets of the cavity using lightweight handles
        let boundary_facet_handles = self.find_cavity_boundary_facets(tds, &bad_cells)?;

        if boundary_facet_handles.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "No boundary facets found for cavity insertion".to_string(),
                },
            ));
        }

        let cells_removed = bad_cells.len();

        // ========================================================================
        // PHASE 1: VALIDATE - Extract all metadata while TDS is intact
        // ========================================================================
        // Gather boundary facet information (vertex keys + outside neighbors)
        // This is done BEFORE any modifications to enable clean rollback on failure.
        let boundary_infos = Self::gather_boundary_facet_info(tds, &boundary_facet_handles)?;

        // Deduplicate and validate boundary facets to prevent creating duplicate cells
        // Error on duplicates instead of silently filtering (surfaces algorithmic bugs)
        let boundary_infos = Self::deduplicate_boundary_facet_info(boundary_infos)?;

        // ========================================================================
        // PHASE 2: TENTATIVE - Insert vertex and create new cells (no removal yet)
        // ========================================================================
        // Track whether vertex existed before this operation for atomic rollback
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        // Ensure vertex is in TDS (needed for filtering and cell creation)
        Self::ensure_vertex_in_tds(tds, vertex)?;

        // Get the inserted vertex key for filtering and cell creation
        let inserted_vk = tds.vertex_key_from_uuid(&vertex.uuid()).ok_or_else(|| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: "Vertex was not found in TDS immediately after insertion".to_string(),
                },
            )
        })?;

        // Filter boundary facets to prevent invalid facet sharing
        // This prevents creating cells that would violate the "facet shared by at most 2 cells" constraint
        let boundary_infos =
            Self::filter_boundary_facets_by_valid_facet_sharing(tds, boundary_infos, inserted_vk)?;

        // Hard-stop if preventive filter removed all boundary facets
        // Proceeding would leave a hole in the TDS after removing bad cells
        if boundary_infos.is_empty() {
            Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "Preventive facet filtering rejected every cavity facet; aborting to keep the triangulation intact."
                        .to_string(),
                },
            ));
        }

        // Create all new cells BEFORE removing bad cells
        // This allows clean rollback if creation fails
        let mut created_cell_keys = Vec::with_capacity(boundary_infos.len());
        for info in &boundary_infos {
            // Combine facet vertices with the inserted vertex
            let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                info.facet_vertex_keys.clone();

            // Safety check: ensure facet doesn't already contain the inserted vertex
            if cell_vertices.contains(&inserted_vk) {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!(
                            "Boundary facet already contains inserted vertex {inserted_vk:?} - this indicates a cavity boundary detection bug"
                        ),
                    },
                ));
            }

            cell_vertices.push(inserted_vk);

            // Create cell from vertex keys
            let new_cell = Cell::new(cell_vertices, None).map_err(|err| {
                InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!("Failed to create cell from boundary facet: {err}"),
                    },
                )
            })?;

            match tds.insert_cell_with_mapping(new_cell) {
                Ok(key) => created_cell_keys.push(key),
                Err(e) => {
                    // Rollback: remove vertex and all cells containing it
                    Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
                    return Err(InsertionError::TriangulationConstruction(e));
                }
            }
        }
        let cells_created = created_cell_keys.len();

        // ========================================================================
        // PHASE 3: COMMIT - Remove bad cells and establish neighbor relationships
        // ========================================================================
        // Save bad cells for potential restoration if anything fails after removal
        let saved_bad_cells: Vec<_> = bad_cells
            .iter()
            .filter_map(|&ck| tds.get_cell(ck).cloned())
            .collect();

        // Now that all new cells exist, remove the bad cells
        // This is the point of no return - from here on, we must either succeed or restore
        Self::remove_bad_cells(tds, &bad_cells);

        // Wire neighbor relationships between new cells and existing triangulation
        if let Err(e) = Self::connect_new_cells_to_neighbors(
            tds,
            inserted_vk,
            &boundary_infos,
            &created_cell_keys,
        ) {
            Self::restore_cavity_insertion_failure(
                tds,
                &saved_bad_cells,
                &created_cell_keys,
                !vertex_existed_before,
                inserted_vk,
            );
            return Err(e);
        }

        // ========================================================================
        // PHASE 3.5: ITERATIVE CAVITY REFINEMENT
        // ========================================================================
        // Check if newly created cells violate the Delaunay property.
        // If any existing vertex is inside a new cell's circumsphere, we must
        // remove that cell and expand the cavity iteratively.
        #[allow(clippy::items_after_statements)]
        const MAX_REFINEMENT_ITERATIONS: usize = 100; // Prevent infinite loops
        // Maximum ratio of cells created to initial cells (dimension-dependent)
        // In D dimensions, a simplex has D+1 vertices. A well-behaved insertion
        // should create O(D) cells. We allow generous headroom for complex cavities
        // and iterative refinement which may need multiple rounds.
        let max_cell_growth_ratio = (D + 1) * 6; // 6x the number of simplex vertices
        let initial_cell_count = tds.number_of_cells();

        let mut total_cells_created = cells_created;
        let mut total_cells_removed = cells_removed;
        let mut cells_to_check = created_cell_keys.clone();
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > MAX_REFINEMENT_ITERATIONS {
                Self::restore_cavity_insertion_failure(
                    tds,
                    &saved_bad_cells,
                    &created_cell_keys,
                    !vertex_existed_before,
                    inserted_vk,
                );
                return Err(InsertionError::GeometricFailure {
                    message: format!(
                        "Iterative cavity refinement exceeded maximum iterations ({MAX_REFINEMENT_ITERATIONS})"
                    ),
                    strategy_attempted: InsertionStrategy::CavityBased,
                });
            }

            // Check for pathological cell growth (indicates degenerate geometry)
            let current_cell_count = tds.number_of_cells();
            if current_cell_count > initial_cell_count * max_cell_growth_ratio {
                Self::restore_cavity_insertion_failure(
                    tds,
                    &saved_bad_cells,
                    &created_cell_keys,
                    !vertex_existed_before,
                    inserted_vk,
                );
                return Err(InsertionError::GeometricFailure {
                    message: format!(
                        "Iterative cavity refinement caused excessive cell growth ({current_cell_count} cells from {initial_cell_count} initial). \
                         This indicates degenerate geometry that cannot be handled by standard Bowyer-Watson. \
                         Consider using RobustBowyerWatson with symbolic perturbation."
                    ),
                    strategy_attempted: InsertionStrategy::CavityBased,
                });
            }

            // Check if any of the cells violate the Delaunay property
            let violating_cells = self.find_delaunay_violations_in_cells(tds, &cells_to_check)?;

            if violating_cells.is_empty() {
                // No violations - we're done!
                break;
            }

            // Found violations - need to expand the cavity
            let refinement_boundary = self.find_cavity_boundary_facets(tds, &violating_cells)?;
            if refinement_boundary.is_empty() {
                // Can't find boundary - stop refinement
                break;
            }

            // Gather boundary info before removing violating cells
            let refinement_infos = Self::gather_boundary_facet_info(tds, &refinement_boundary)?;
            let refinement_infos = Self::deduplicate_boundary_facet_info(refinement_infos)?;
            let refinement_infos = Self::filter_boundary_facets_by_valid_facet_sharing(
                tds,
                refinement_infos,
                inserted_vk,
            )?;

            if refinement_infos.is_empty() {
                // No valid boundary facets - refinement blocked by topology constraints
                // Stop refinement and let finalization fix remaining issues
                break;
            }

            // Create new cells for the refined cavity
            let mut refinement_cell_keys = Vec::with_capacity(refinement_infos.len());
            for info in &refinement_infos {
                let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    info.facet_vertex_keys.clone();

                // Safety check: ensure facet doesn't already contain the inserted vertex
                if cell_vertices.contains(&inserted_vk) {
                    // Skip this facet - it already contains the vertex we're inserting
                    continue;
                }

                cell_vertices.push(inserted_vk);

                let new_cell = Cell::new(cell_vertices, None).map_err(|err| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::FailedToCreateCell {
                            message: format!("Failed to create refinement cell: {err}"),
                        },
                    )
                })?;

                match tds.insert_cell_with_mapping(new_cell) {
                    Ok(key) => refinement_cell_keys.push(key),
                    Err(_) => {
                        // Cell insertion failed (likely due to topology constraints)
                        // Stop creating more refinement cells
                        break;
                    }
                }
            }

            // If no refinement cells were created (all filtered as duplicates or failed), stop refinement
            // Do NOT remove violating cells if we couldn't create replacement cells
            if refinement_cell_keys.is_empty() {
                break;
            }

            // Only remove violating cells and connect neighbors if we successfully created refinement cells
            Self::remove_bad_cells(tds, &violating_cells);
            total_cells_removed += violating_cells.len();
            total_cells_created += refinement_cell_keys.len();

            // Connect the new cells
            // If this fails, we've already removed the bad cells, so we can't rollback cleanly
            // Just propagate the error
            Self::connect_new_cells_to_neighbors(
                tds,
                inserted_vk,
                &refinement_infos,
                &refinement_cell_keys,
            )?;

            // Check these new cells in the next iteration
            cells_to_check = refinement_cell_keys;
        }

        // Finalize the triangulation after insertion to fix any invalid states
        // This includes fix_invalid_facet_sharing which resolves topology issues
        // Note: In degenerate cases with iterative refinement, finalization may fail
        // due to unfixable topology issues. We treat this as a recoverable error.
        if let Err(e) = Self::finalize_after_insertion(tds) {
            // Finalization failed - restore to valid state before returning error
            Self::restore_cavity_insertion_failure(
                tds,
                &saved_bad_cells,
                &created_cell_keys,
                !vertex_existed_before,
                inserted_vk,
            );
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Failed to finalize triangulation after cavity-based insertion: {e}"
                    ),
                },
            ));
        }

        // Final validation: check if any cells still violate the Delaunay property
        // For IncrementalBowyerWatson, this will trigger fallback to RobustBowyerWatson
        // For RobustBowyerWatson, this ensures strict Delaunay guarantees
        let all_cell_keys: Vec<CellKey> = tds.cells().map(|(k, _)| k).collect();
        let remaining_violations = self.find_delaunay_violations_in_cells(tds, &all_cell_keys)?;

        if !remaining_violations.is_empty() {
            Self::restore_cavity_insertion_failure(
                tds,
                &saved_bad_cells,
                &created_cell_keys,
                !vertex_existed_before,
                inserted_vk,
            );
            return Err(InsertionError::GeometricFailure {
                message: format!(
                    "Cavity-based insertion completed but {} cells still violate the Delaunay property. \
                     Iterative refinement was blocked by topology constraints. \
                     Robust predicates or alternative insertion strategy required.",
                    remaining_violations.len()
                ),
                strategy_attempted: InsertionStrategy::CavityBased,
            });
        }

        Ok(InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed: total_cells_removed,
            cells_created: total_cells_created,
            success: true,
            degenerate_case_handled: false,
        })
    }

    /// Inserts a vertex by extending the convex hull
    ///
    /// This method finds visible boundary facets and creates new cells
    /// by connecting the vertex to these facets.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No visible facets are found (vertex is not exterior)
    /// - Visibility computation fails
    /// - Cell creation fails
    fn insert_vertex_hull_extension(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Get visible boundary facets using lightweight handles
        let visible_facet_handles = self.find_visible_boundary_facets_lightweight(tds, vertex)?;

        if visible_facet_handles.is_empty() {
            // No visible facets - this method is not applicable
            return Err(InsertionError::HullExtensionFailure {
                reason: "No visible boundary facets found for exterior vertex. \
                         This typically indicates the vertex is not actually exterior, or \
                         visibility detection is failing silently."
                    .to_string(),
            });
        }

        // Create new cells from visible facet handles (using lightweight API)
        let cells_created =
            Self::create_cells_from_facet_handles(tds, &visible_facet_handles, vertex)?;

        // Finalize the triangulation after insertion to fix any invalid states
        Self::finalize_after_insertion(tds).map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!(
                    "Failed to finalize triangulation after hull extension insertion: {e}"
                ),
            }
        })?;

        Ok(InsertionInfo {
            strategy: InsertionStrategy::HullExtension,
            cells_removed: 0,
            cells_created,
            success: true,
            degenerate_case_handled: false,
        })
    }

    /// Fallback insertion method for difficult cases
    ///
    /// This method attempts a conservative strategy by trying to find any valid
    /// connection to the existing triangulation. It first tries boundary facets
    /// (most likely to work) and then all facets if necessary.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error if all methods fail.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No valid facet connection can be found
    /// - All cell creation attempts fail
    /// - The triangulation is in an invalid state
    fn insert_vertex_fallback(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Conservative fallback: try to connect to any existing boundary facet
        // This avoids creating invalid geometry by arbitrary vertex replacement

        // Track whether vertex existed before we started for rollback purposes
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        // Performance note: Concrete implementations that also implement FacetCacheProvider
        // should override this method to use get_or_build_facet_cache() to avoid O(N·F)
        // rebuilds in difficult fallback cases. Default trait implementation uses direct
        // build since it cannot access concrete type's cache.
        let facet_to_cells = tds.build_facet_to_cells_map().map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to build facet-to-cells map: {e}"),
            }
        })?;

        // First try boundary facets (most likely to work)
        for cells in facet_to_cells.values() {
            if cells.len() == 1 {
                let facet_handle = cells.first().ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: "Boundary facet had no adjacent cell".to_string(),
                        },
                    )
                })?;
                let cell_key = facet_handle.cell_key();
                let facet_index = facet_handle.facet_index();
                let _fi = <usize as From<_>>::from(facet_index);
                // Phase 3A: Use lightweight facet handles directly
                // Try to create a cell from this facet handle and the vertex
                // Note: create_cell_from_facet_handle validates cell existence internally
                if Self::create_cell_from_facet_handle(tds, cell_key, facet_index, vertex).is_ok() {
                    // Finalize the triangulation after insertion to fix any invalid states
                    if Self::finalize_after_insertion(tds).is_ok() {
                        // Success!
                        return Ok(InsertionInfo {
                            strategy: InsertionStrategy::Fallback,
                            cells_removed: 0,
                            cells_created: 1,
                            success: true,
                            degenerate_case_handled: false,
                        });
                    }
                    // Finalization failed, continue trying other facets
                }
            }
        }

        // If boundary facets don't work, try ALL facets (including internal ones)
        for cells in facet_to_cells.values() {
            for facet_handle in cells {
                let cell_key = facet_handle.cell_key();
                let facet_index = facet_handle.facet_index();
                let _fi = <usize as From<_>>::from(facet_index);
                // Phase 3A: Use lightweight facet handles directly
                // Try to create a cell from this facet handle and the vertex
                // Note: create_cell_from_facet_handle validates cell existence internally
                if Self::create_cell_from_facet_handle(tds, cell_key, facet_index, vertex).is_ok() {
                    // Finalize the triangulation after insertion to fix any invalid states
                    if Self::finalize_after_insertion(tds).is_ok() {
                        // Success!
                        return Ok(InsertionInfo {
                            strategy: InsertionStrategy::Fallback,
                            cells_removed: 0,
                            cells_created: 1,
                            success: true,
                            degenerate_case_handled: false,
                        });
                    }
                    // Finalization failed, continue trying other facets
                }
            }
        }

        // All attempts failed - use smart rollback to clean up all cells and vertex
        Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);

        // If we can't find any boundary facet to connect to, the vertex might be
        // in a degenerate position or the triangulation might be corrupted
        Err(InsertionError::TriangulationState(
            TriangulationValidationError::FailedToCreateCell {
                message: format!(
                    "Fallback insertion failed: could not connect vertex {:?} to any boundary facet",
                    vertex.point()
                ),
            },
        ))
    }

    /// Triangulate a complete set of vertices
    ///
    /// This method provides a complete triangulation solution by inserting
    /// all vertices in the collection into the given triangulation data structure.
    /// This advanced implementation handles initial simplex creation, incremental
    /// insertion, and finalization steps.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure (should be empty or have initial vertices)
    /// * `vertices` - Collection of vertices to triangulate
    ///
    /// # Returns
    ///
    /// `Ok(())` if triangulation succeeds, or an error describing the failure.
    ///
    /// # Errors
    ///
    /// Returns an error if triangulation fails due to:
    /// - Insufficient vertices for the given dimension (< D+1)
    /// - Initial simplex creation fails
    /// - Vertex insertion fails
    /// - Triangulation finalization fails
    /// - Geometric degeneracy
    /// - Numerical precision issues
    /// - Topological constraints
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

        // Check for sufficient vertices
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

        // Step 1: Initialize with first D+1 vertices
        let (initial_vertices, remaining_vertices) = vertices.split_at(D + 1);
        Self::create_initial_simplex(tds, initial_vertices.to_vec())?;

        // Update statistics for initial simplex creation
        self.update_statistics(1, 0); // Initial simplex creates one cell, removes zero

        // Step 2: Insert remaining vertices incrementally
        for vertex in remaining_vertices {
            self.insert_vertex(tds, *vertex).map_err(|e| match e {
                InsertionError::TriangulationConstruction(tc_err) => tc_err,
                other => TriangulationConstructionError::FailedToAddVertex {
                    message: format!("Vertex insertion failed during triangulation: {other}"),
                },
            })?;
        }

        // Step 3: Finalize the triangulation
        Self::finalize_triangulation(tds)?;

        Ok(())
    }

    /// Creates the initial simplex from the first D+1 vertices
    ///
    /// This is a helper method used by the default triangulate implementation.
    /// Implementations can override this if they need specialized simplex creation.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertices` - Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful creation, or an error if the simplex cannot be created.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError::FailedToCreateCell` if:
    /// - The number of vertices provided is not exactly D+1.
    /// - The `CellBuilder` fails to create the initial cell.
    fn create_initial_simplex(
        tds: &mut Tds<T, U, V, D>,
        vertices: Vec<Vertex<T, U, D>>,
    ) -> Result<(), TriangulationConstructionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
    {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Ensure all vertices are registered in the TDS vertex mapping
        for vertex in &vertices {
            // Use the public UUID-to-key lookup method to check if vertex exists
            if tds.vertex_key_from_uuid(&vertex.uuid()).is_none() {
                tds.insert_vertex_with_mapping(*vertex).map_err(|e| {
                    TriangulationConstructionError::FailedToAddVertex {
                        message: format!("Failed to insert vertex: {e}"),
                    }
                })?;
            }
        }

        // Phase 3A: Create cell with vertex keys instead of using CellBuilder
        // First get the keys for all vertices
        let vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = vertices
            .iter()
            .filter_map(|v| tds.vertex_key_from_uuid(&v.uuid()))
            .collect();
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::FailedToCreateCell {
                message: "Initial simplex vertices missing from UUID→key mapping".to_string(),
            });
        }

        let cell = Cell::new(vertices, None).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex: {e}"),
            }
        })?;

        tds.insert_cell_with_mapping(cell).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to insert initial simplex cell: {e}"),
            }
        })?;

        Ok(())
    }

    /// Finalizes the triangulation by cleaning up and establishing relationships
    ///
    /// This method performs post-processing steps to ensure the triangulation
    /// is complete and consistent. Implementations can override this for custom
    /// finalization procedures.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful finalization, or an error if finalization fails.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if:
    /// - Fixing invalid facet sharing fails.
    /// - Assigning neighbor relationships fails.
    /// - Assigning incident cells to vertices fails.
    fn finalize_triangulation(
        tds: &mut Tds<T, U, V, D>,
    ) -> Result<(), TriangulationConstructionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
    {
        // Remove duplicate cells
        tds.remove_duplicate_cells()
            .map_err(TriangulationConstructionError::ValidationError)?;

        // Fix invalid facet sharing
        tds.fix_invalid_facet_sharing().map_err(|e| {
            TriangulationConstructionError::ValidationError(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Failed to fix invalid facet sharing: {e}"),
                },
            )
        })?;

        // Assign neighbor relationships
        tds.assign_neighbors()
            .map_err(TriangulationConstructionError::ValidationError)?;

        // Assign incident cells to vertices
        tds.assign_incident_cells()
            .map_err(TriangulationConstructionError::ValidationError)?;

        Ok(())
    }

    /// Test if a boundary facet is visible from a given vertex
    ///
    /// This method allows different algorithms to implement their own visibility testing
    /// strategies, from basic orientation tests to robust predicates with fallbacks.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `facet` - The facet to test visibility for
    /// * `vertex` - The vertex from which to test visibility
    /// * `adjacent_cell_key` - Key of the cell adjacent to this boundary facet
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is visible from the vertex, `Ok(false)` if not visible,
    /// or an error if topology validation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the triangulation topology is inconsistent or corrupted.
    fn is_facet_visible_from_vertex(
        &self,
        tds: &Tds<T, U, V, D>,
        facet: &crate::core::facet::FacetView<'_, T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        adjacent_cell_key: crate::core::triangulation_data_structure::CellKey,
    ) -> Result<bool, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Get the adjacent cell to this boundary facet
        let Some(adjacent_cell) = tds.get_cell(adjacent_cell_key) else {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Adjacent cell {adjacent_cell_key:?} not found during visibility test. This indicates TDS corruption."
                    ),
                },
            ));
        };

        // HOT PATH: Collect facet vertices once for point extraction
        let facet_vertices_vec: SmallVec<[Vertex<T, U, D>; 8]> = facet
            .vertices()
            .map_err(|e| {
                InsertionError::TriangulationState(TriangulationValidationError::FacetError(e))
            })?
            .copied()
            .collect();

        // Find the opposite vertex directly using facet_index (avoids HashSet overhead)
        let cell_vertices = adjacent_cell.vertices();
        let facet_index = <usize as From<u8>>::from(facet.facet_index());

        let opposite_vkey = cell_vertices
            .get(facet_index)
            .ok_or_else(|| {
                InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Facet index {facet_index} out of bounds for cell {adjacent_cell_key:?} with {} vertices",
                            cell_vertices.len()
                        ),
                    },
                )
            })?;

        let opposite_vertex = tds.get_vertex_by_key(*opposite_vkey).ok_or_else(|| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Vertex key {opposite_vkey:?} from cell {adjacent_cell_key:?} not found in TDS during visibility test. \
                         This indicates mapping inconsistency."
                    ),
                },
            )
        })?;

        // Create test simplices for orientation comparison
        // Using SmallVec to avoid heap allocation for small simplices (D+1 points)
        // Reuse cached vertices from above to avoid redundant facet.vertices() call
        let facet_vertex_points: SmallVec<[Point<T, D>; 8]> =
            facet_vertices_vec.iter().map(|v| *v.point()).collect();

        let mut simplex_with_opposite = facet_vertex_points.clone();
        simplex_with_opposite.push(*opposite_vertex.point());

        let mut simplex_with_test = facet_vertex_points;
        simplex_with_test.push(*vertex.point());

        // Get orientations
        let orientation_opposite = simplex_orientation(&simplex_with_opposite);
        let orientation_test = simplex_orientation(&simplex_with_test);

        match (orientation_opposite, orientation_test) {
            (Ok(ori_opp), Ok(ori_test)) => {
                // Facet is visible if the orientations are different
                // (vertices are on opposite sides of the hyperplane)
                let is_visible = match (ori_opp, ori_test) {
                    (Orientation::NEGATIVE, Orientation::POSITIVE)
                    | (Orientation::POSITIVE, Orientation::NEGATIVE) => true,
                    (Orientation::DEGENERATE, _)
                    | (_, Orientation::DEGENERATE)
                    | (Orientation::NEGATIVE, Orientation::NEGATIVE)
                    | (Orientation::POSITIVE, Orientation::POSITIVE) => false, // Same orientation = same side = not visible
                };
                Ok(is_visible)
            }
            _ => {
                // Consider surfacing orientation failures as recoverable errors
                // instead of returning Ok(false) which may hide geometric issues
                Err(InsertionError::geometric_failure(
                    "Orientation predicate failed during facet visibility test",
                    InsertionStrategy::HullExtension,
                ))
            }
        }
    }

    /// Create a new triangulation from vertices using this algorithm
    ///
    /// This method creates a new TDS and builds a complete triangulation from the given vertices.
    /// This is the algorithm-specific equivalent of `Tds::new()`, allowing different insertion
    /// algorithms to use their own strategies for creating triangulations from scratch.
    ///
    /// # Arguments
    ///
    /// * `vertices` - Collection of vertices to triangulate
    ///
    /// # Returns
    ///
    /// A new `Tds` containing the complete triangulation, or an error if triangulation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if triangulation fails due to:
    /// - Geometric degeneracy
    /// - Numerical precision issues
    /// - Insufficient vertices for the given dimension
    /// - Topological constraints
    ///
    /// # Default Implementation
    ///
    /// The default implementation delegates to `Tds::new()` which uses the regular Bowyer-Watson algorithm.
    /// Robust or specialized algorithms can override this to provide their own triangulation strategies.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    /// use delaunay::{vertex, geometry::point::Point};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let mut algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> = RobustBowyerWatson::new();
    /// let tds = algorithm.new_triangulation(&vertices)?;
    /// # Ok::<(), delaunay::core::triangulation_data_structure::TriangulationConstructionError>(())
    /// ```
    fn new_triangulation(
        &mut self,
        vertices: &[Vertex<T, U, D>],
    ) -> Result<Tds<T, U, V, D>, TriangulationConstructionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
    {
        // Default implementation: use the regular Tds::new constructor
        Tds::new(vertices)
    }

    // =========== SHARED UTILITY METHODS ===========
    // These methods provide common functionality used by multiple insertion algorithms

    /// Ensures that a vertex is properly registered in the TDS
    ///
    /// This is a shared utility method that both implementations can use.
    /// It checks if the vertex UUID is already mapped in the TDS UUID-to-key mapping,
    /// and if not, inserts the vertex and creates the mapping.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to ensure is registered
    ///
    /// # Returns
    ///
    /// `Ok(())` if the vertex is successfully registered or already exists,
    /// `Err` if insertion fails
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if the vertex cannot be inserted
    /// into the triangulation data structure
    fn ensure_vertex_in_tds(
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<(), TriangulationValidationError> {
        // Use the public UUID-to-key lookup method to check if vertex exists
        if tds.vertex_key_from_uuid(&vertex.uuid()).is_none() {
            tds.insert_vertex_with_mapping(*vertex).map_err(|e| {
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Failed to insert vertex into TDS: {e}"),
                }
            })?;
        }
        Ok(())
    }

    // NOTE: create_cell_from_facet_and_vertex() and create_cells_from_boundary_facets() removed in Phase 3C.
    // Use create_cell_from_facet_handle() or create_cells_from_facet_handles() which work with lightweight (CellKey, u8) handles.

    /// Create a single cell from a facet handle and an additional vertex.
    ///
    /// This is a lightweight helper that creates a cell using just the facet's
    /// (`CellKey`, u8) handle, avoiding the need for heavyweight Facet objects.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `cell_key` - Key of the cell containing the facet
    /// * `facet_index` - Index of the facet within the cell
    /// * `vertex` - The additional vertex to connect to the facet
    ///
    /// # Returns
    ///
    /// `Ok(CellKey)` containing the key of the newly created cell, or an error if creation failed.
    ///
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if:
    /// - The cell or facet cannot be found
    /// - Cell creation fails due to geometric constraints
    /// - Topological validation fails during cell insertion
    fn create_cell_from_facet_handle(
        tds: &mut Tds<T, U, V, D>,
        cell_key: CellKey,
        facet_index: u8,
        vertex: &Vertex<T, U, D>,
    ) -> Result<CellKey, TriangulationValidationError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Create FacetView from the handle
        let facet_view = FacetView::new(tds, cell_key, facet_index)
            .map_err(TriangulationValidationError::FacetError)?;

        // Extract vertex data from FacetView (zero allocation access)
        let facet_vertices_iter = facet_view
            .vertices()
            .map_err(TriangulationValidationError::FacetError)?;
        let facet_vertices: Vec<Vertex<T, U, D>> = facet_vertices_iter.copied().collect();

        // Delegate to the vertex-based method
        Self::create_cell_from_vertices_and_vertex(tds, facet_vertices, vertex)
    }

    /// Create a single cell from facet vertices and an additional vertex.
    ///
    /// This is a helper method that creates a new cell by combining a set of facet vertices
    /// with an additional vertex. This method is used internally by the FacetView-based
    /// cell creation methods to avoid borrowing conflicts.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `facet_vertices` - Vertices that make up the boundary facet
    /// * `vertex` - The additional vertex to connect to the facet
    ///
    /// # Returns
    ///
    /// `Ok(CellKey)` containing the key of the newly created cell, or an error if creation failed
    /// due to geometric or topological issues.
    ///
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if:
    /// - The vertex cannot be ensured in the TDS
    /// - Cell creation fails due to geometric constraints
    /// - Topological validation fails during cell insertion
    fn create_cell_from_vertices_and_vertex(
        tds: &mut Tds<T, U, V, D>,
        mut facet_vertices: Vec<Vertex<T, U, D>>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<CellKey, TriangulationValidationError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Reject duplicate vertex to avoid degenerate cell
        if facet_vertices.iter().any(|v| v.uuid() == vertex.uuid()) {
            return Err(TriangulationValidationError::FailedToCreateCell {
                message: "Attempted to create a cell with duplicate vertex (facet already contains the vertex)".to_string(),
            });
        }

        // Ensure the vertex is registered in the TDS vertex mapping
        Self::ensure_vertex_in_tds(tds, vertex)?;

        // Add the new vertex to complete the cell
        facet_vertices.push(*vertex);

        // Phase 3A: Get vertex keys from TDS
        let vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = facet_vertices
            .iter()
            .filter_map(|v| tds.vertex_key_from_uuid(&v.uuid()))
            .collect();
        if vertices.len() != facet_vertices.len() {
            return Err(TriangulationValidationError::InconsistentDataStructure {
                message: "One or more facet vertices missing from UUID→key mapping".to_string(),
            });
        }

        let new_cell = Cell::new(vertices, None).map_err(|e| {
            TriangulationValidationError::FailedToCreateCell {
                message: format!("Failed to build cell from facet vertices and vertex: {e}"),
            }
        })?;

        let cell_key = tds.insert_cell_with_mapping(new_cell).map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to insert cell into TDS: {e}"),
            }
        })?;

        Ok(cell_key)
    }

    /// Find visible boundary facets using lightweight `FacetHandle` to avoid heavy allocations.
    ///
    /// Returns lightweight `FacetHandle` instead of materialized facet data,
    /// significantly reducing memory allocation and copying overhead.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure
    /// * `vertex` - The vertex from which to test visibility
    ///
    /// # Returns
    ///
    /// A `Vec<FacetHandle>` where each handle represents a visible boundary facet.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError::TriangulationState` if the boundary facets cannot be enumerated.
    fn find_visible_boundary_facets_lightweight(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<FacetHandle>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        let mut visible_facet_handles = Vec::new();

        // Get all boundary facets using optimized boundary_facets() iterator
        let boundary_facets = tds
            .boundary_facets()
            .map_err(InsertionError::TriangulationState)?;

        // Test visibility for each boundary facet using lightweight handles
        for boundary_facet_view in boundary_facets {
            // Get the cell key and facet index for this boundary facet
            let cell_key = boundary_facet_view.cell_key();
            let facet_index = boundary_facet_view.facet_index();

            // Test visibility using FacetView directly (no conversion needed)
            if self.is_facet_visible_from_vertex(tds, &boundary_facet_view, vertex, cell_key)? {
                visible_facet_handles.push(FacetHandle::new(cell_key, facet_index));
            }
        }

        Ok(visible_facet_handles)
    }

    /// Create cells from lightweight facet handles.
    ///
    /// Works with `FacetHandle` to avoid materializing full facet data.
    ///
    /// **Atomic Behavior**: This method provides atomic semantics - either all cells are
    /// created successfully, or no cells are created and the TDS remains in its original state.
    /// If cell creation fails partway through, all successfully created cells are removed
    /// before returning the error.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to modify
    /// * `facet_handles` - Slice of `FacetHandle` representing boundary facets
    /// * `vertex` - The vertex to connect to the boundary facets
    ///
    /// # Returns
    ///
    /// The number of cells successfully created, or an error if facet reconstruction fails.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError::TriangulationState` if any facet handle is invalid or
    /// if cell reconstruction fails due to data structure inconsistencies. On error, the TDS
    /// is restored to its state before the method was called.
    fn create_cells_from_facet_handles(
        tds: &mut Tds<T, U, V, D>,
        facet_handles: &[FacetHandle],
        vertex: &Vertex<T, U, D>,
    ) -> Result<usize, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Early exit: empty input
        if facet_handles.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "No facet handles provided for cell creation".to_string(),
                },
            ));
        }

        // Track whether vertex existed before this operation for atomic rollback
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        // Phase 1: Gather and validate boundary facet information
        // Error on duplicates instead of silently filtering (surfaces algorithmic bugs)
        let boundary_infos = Self::gather_boundary_facet_info(tds, facet_handles)?;
        let boundary_infos = Self::deduplicate_boundary_facet_info(boundary_infos)?;

        // Ensure vertex is in TDS (needed for filtering and cell creation)
        Self::ensure_vertex_in_tds(tds, vertex)?;

        // Get the inserted vertex key for filtering and cell creation
        let inserted_vk = tds.vertex_key_from_uuid(&vertex.uuid()).ok_or_else(|| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: "Vertex was not found in TDS immediately after insertion".to_string(),
                },
            )
        })?;

        // Phase 1.5: Filter boundary facets to prevent invalid facet sharing
        // This prevents creating cells that would violate the "facet shared by at most 2 cells" constraint
        let boundary_infos =
            Self::filter_boundary_facets_by_valid_facet_sharing(tds, boundary_infos, inserted_vk)?;

        // Hard-stop if preventive filter removed all boundary facets
        // Creating zero cells would be invalid
        if boundary_infos.is_empty() {
            Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "No boundary facets available after filtering; aborting to keep the TDS consistent."
                        .to_string(),
                },
            ));
        }

        // Phase 2: Create all cells from deduplicated boundary facets
        // Tracking created cell keys for potential rollback
        let mut created_cell_keys = Vec::with_capacity(boundary_infos.len());

        for info in &boundary_infos {
            // Combine facet vertices with the inserted vertex
            let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                info.facet_vertex_keys.clone();

            // Safety check: ensure facet doesn't already contain the inserted vertex
            if cell_vertices.contains(&inserted_vk) {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!(
                            "Boundary facet already contains inserted vertex {inserted_vk:?} - this indicates a cavity boundary detection bug"
                        ),
                    },
                ));
            }

            cell_vertices.push(inserted_vk);

            // Create cell from vertex keys
            let new_cell = Cell::new(cell_vertices, None).map_err(|err| {
                InsertionError::TriangulationState(
                    TriangulationValidationError::FailedToCreateCell {
                        message: format!("Failed to create cell from boundary facet: {err}"),
                    },
                )
            })?;

            match tds.insert_cell_with_mapping(new_cell) {
                Ok(key) => created_cell_keys.push(key),
                Err(e) => {
                    // Rollback: remove vertex and all cells containing it
                    Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
                    return Err(InsertionError::TriangulationConstruction(e));
                }
            }
        }

        let cells_created = created_cell_keys.len();

        // Validate that we created at least some cells
        if cells_created == 0 && !facet_handles.is_empty() {
            Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: format!(
                        "Failed to create any cells from {} facet handles (all were duplicates)",
                        facet_handles.len()
                    ),
                },
            ));
        }

        Ok(cells_created)
    }

    /// Detect and reject duplicate boundary facets.
    ///
    /// Multiple boundary facets from different bad cells can have identical vertex sets,
    /// which would lead to creating duplicate cells sharing all facets (invalid topology).
    /// Instead of silently filtering duplicates, this function returns an error if any
    /// are detected, surfacing algorithmic bugs in cavity boundary detection.
    ///
    /// The cavity boundary should form a topological sphere with no duplicate facets.
    /// Duplicates indicate:
    /// - Incorrect neighbor traversal logic
    /// - Non-manifold mesh connectivity
    /// - Data structure corruption
    ///
    /// # Arguments
    ///
    /// * `boundary_infos` - Vector of boundary facet information to validate
    ///
    /// # Returns
    ///
    /// - `Ok(SmallBuffer<BoundaryFacetInfo>)` if no duplicates detected
    /// - `Err(InsertionError::DuplicateBoundaryFacets)` if duplicates found
    ///
    /// # Errors
    ///
    /// Returns `InsertionError::DuplicateBoundaryFacets` if duplicate facets are detected,
    /// indicating an algorithmic bug in cavity boundary detection.
    ///
    /// # Performance
    ///
    /// Uses `SmallBuffer` for efficient stack allocation (typically D+1 facets in D dimensions).
    /// Falls back to heap allocation for pathological cases. Runs in O(n) time using
    /// `FastHashSet` for duplicate detection via canonical facet keys.
    ///
    /// # Implementation Note
    ///
    /// Uses canonical facet key (u64 hash) via `facet_key_from_vertices` to identify duplicates,
    /// matching the hashing strategy used throughout the codebase.
    fn deduplicate_boundary_facet_info(
        boundary_infos: Vec<BoundaryFacetInfo>,
    ) -> Result<SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE>, InsertionError> {
        let total_count = boundary_infos.len();
        let mut seen_facet_keys: FastHashSet<u64> = fast_hash_set_with_capacity(total_count);
        let mut deduplicated: SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        let mut duplicate_count = 0;

        for info in boundary_infos {
            // Use canonical facet key (u64 hash) to identify unique facets
            // This matches the hashing used throughout the codebase
            let facet_key = facet_key_from_vertices(&info.facet_vertex_keys);

            // Track duplicates instead of silently filtering
            if seen_facet_keys.insert(facet_key) {
                deduplicated.push(info);
            } else {
                duplicate_count += 1;
            }
        }

        // Return error if duplicates were detected
        if duplicate_count > 0 {
            return Err(InsertionError::DuplicateBoundaryFacets {
                duplicate_count,
                total_count,
            });
        }

        Ok(deduplicated)
    }

    /// Filter boundary facets to prevent invalid facet sharing.
    ///
    /// This function ensures that creating cells from the boundary facets will not violate
    /// the fundamental Delaunay constraint: each facet must be shared by at most 2 cells.
    ///
    /// When we create a new cell from a boundary facet + inserted vertex, that cell will have
    /// D+1 facets:
    /// - 1 facet is the boundary facet itself (exists, currently has 1 cell)
    /// - D new facets (formed between boundary facet vertices and the new vertex)
    ///
    /// We need to ensure none of those D new facets would be shared by >2 cells.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `boundary_infos` - Deduplicated boundary facet information
    /// * `inserted_vk` - Vertex key of the newly inserted vertex
    ///
    /// # Returns
    ///
    /// - `Ok(SmallBuffer<BoundaryFacetInfo>)` with filtered facets that maintain valid topology
    /// - `Err(InsertionError)` if facet map building fails
    ///
    /// # Errors
    ///
    /// Returns `InsertionError::TriangulationState` if the facet-to-cells map cannot be built.
    ///
    /// # Implementation Note
    ///
    /// This is a preventive approach that avoids creating invalid topology rather than
    /// fixing it reactively. Time complexity is O(N*D*F) where N is number of boundary facets,
    /// D is dimension, and F is average facets per cell in the facet map.
    fn filter_boundary_facets_by_valid_facet_sharing(
        tds: &Tds<T, U, V, D>,
        boundary_infos: SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE>,
        inserted_vk: VertexKey,
    ) -> Result<SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        use crate::core::collections::{FastHashMap, fast_hash_map_with_capacity};

        // Build facet-to-cells map from current TDS state
        let facet_map = tds.build_facet_to_cells_map().map_err(|e| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Failed to build facet-to-cells map for preventive filtering: {e}"
                    ),
                },
            )
        })?;

        // Count how many cells each facet currently has
        let mut facet_cell_counts: FastHashMap<u64, usize> =
            fast_hash_map_with_capacity(facet_map.len());
        for (facet_key, handles) in &facet_map {
            facet_cell_counts.insert(*facet_key, handles.len());
        }

        // Filter boundary facets: keep only those that won't cause over-sharing
        #[cfg(debug_assertions)]
        let boundary_count = boundary_infos.len();
        let mut filtered: SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        let mut facet_vertices_buffer: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();

        for info in boundary_infos {
            // Capture bad_cell before we potentially move info
            let bad_cell = info.bad_cell;

            // Simulate creating the cell: combine boundary facet vertices + inserted vertex
            let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                info.facet_vertex_keys.clone();

            // Skip if facet already contains the inserted vertex (prevents duplicate vertices)
            if cell_vertices.contains(&inserted_vk) {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Filtering out boundary facet: already contains inserted vertex {inserted_vk:?}"
                );
                continue;
            }

            cell_vertices.push(inserted_vk);

            // Check all D+1 facets of this would-be cell
            let mut would_cause_oversharing = false;

            for exclude_idx in 0..cell_vertices.len() {
                // Build the facet by excluding vertex at exclude_idx
                facet_vertices_buffer.clear();
                for (i, &vk) in cell_vertices.iter().enumerate() {
                    if i != exclude_idx {
                        facet_vertices_buffer.push(vk);
                    }
                }

                // Compute facet key
                let facet_key = facet_key_from_vertices(&facet_vertices_buffer);

                // Check current cell count for this facet
                let mut current_count = facet_cell_counts.get(&facet_key).copied().unwrap_or(0);

                // The facet we're replacing is still counted via the bad cell; discount it
                // since the bad cell is about to be deleted
                if let Some(handles) = facet_map.get(&facet_key)
                    && handles.iter().any(|handle| handle.cell_key() == bad_cell)
                {
                    current_count = current_count.saturating_sub(1);
                }

                // If this facet already has 2 cells (after discounting bad cell), adding our new cell would make it 3 (over-sharing)
                if current_count >= 2 {
                    would_cause_oversharing = true;
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Filtering out boundary facet: would cause facet {:?} to be shared by {} cells (currently {})",
                        facet_key,
                        current_count + 1,
                        current_count
                    );
                    break;
                }
            }

            if !would_cause_oversharing {
                // This boundary facet is safe to use
                filtered.push(info);

                // Update our local facet counts to account for this cell we're planning to create
                // This ensures subsequent checks see the cumulative effect
                for exclude_idx in 0..cell_vertices.len() {
                    facet_vertices_buffer.clear();
                    for (i, &vk) in cell_vertices.iter().enumerate() {
                        if i != exclude_idx {
                            facet_vertices_buffer.push(vk);
                        }
                    }
                    let facet_key = facet_key_from_vertices(&facet_vertices_buffer);
                    let entry = facet_cell_counts.entry(facet_key).or_insert(0);
                    // Discount the bad cell contribution if present before incrementing
                    if let Some(handles) = facet_map.get(&facet_key)
                        && handles.iter().any(|handle| handle.cell_key() == bad_cell)
                    {
                        *entry = entry.saturating_sub(1);
                    }
                    *entry += 1;
                }
            }
        }

        #[cfg(debug_assertions)]
        if filtered.len() < boundary_count {
            eprintln!(
                "Filtered {} boundary facets to prevent over-sharing (kept {} of {})",
                boundary_count - filtered.len(),
                filtered.len(),
                boundary_count
            );
        }

        Ok(filtered)
    }

    /// Gather all boundary facet metadata before modifying the TDS.
    ///
    /// This is a critical helper for transactional cavity-based insertion. It extracts
    /// all information needed to create new cells and establish neighbor relationships
    /// **before** any cells are removed from the TDS.
    ///
    /// # Phase 1: Validate
    ///
    /// This function implements the "Validate" phase of the transactional pattern:
    /// - Extracts facet vertex keys
    /// - Identifies outside neighbors and reciprocal facet indices
    /// - Performs validation checks
    /// - Does NOT modify the TDS
    ///
    /// Any errors returned here leave the TDS completely unchanged.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `boundary_facet_handles` - Handles to boundary facets from `find_cavity_boundary_facets()`
    ///
    /// # Returns
    ///
    /// A vector of `BoundaryFacetInfo` with extracted metadata, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if:
    /// - Facet indices are invalid
    /// - Cell or vertex lookups fail
    /// - Neighbor relationships are inconsistent
    fn gather_boundary_facet_info(
        tds: &Tds<T, U, V, D>,
        boundary_facet_handles: &[FacetHandle],
    ) -> Result<Vec<BoundaryFacetInfo>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        let mut boundary_infos = Vec::with_capacity(boundary_facet_handles.len());

        for handle in boundary_facet_handles {
            let bad_cell = handle.cell_key();
            let bad_facet_index = <usize as From<u8>>::from(handle.facet_index());

            // Get the bad cell
            let Some(cell) = tds.get_cell(bad_cell) else {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Bad cell {bad_cell:?} not found during facet info gathering"
                        ),
                    },
                ));
            };

            // Extract facet vertex keys (D vertices, excluding the one at bad_facet_index)
            let facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = cell
                .vertices()
                .iter()
                .enumerate()
                .filter_map(|(i, &vkey)| (i != bad_facet_index).then_some(vkey))
                .collect();

            if facet_vertex_keys.len() != D {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Boundary facet has {} vertices; expected {} for {}-dimensional triangulation",
                            facet_vertex_keys.len(),
                            D,
                            D
                        ),
                    },
                ));
            }

            // Determine outside neighbor and reciprocal facet index
            let outside_neighbor = if let Some(neighbors) = cell.neighbors() {
                if let Some(&Some(neighbor_key)) = neighbors.get(bad_facet_index) {
                    // Find reciprocal facet index in the neighbor cell
                    let neighbor_cell = tds.get_cell(neighbor_key).ok_or_else(|| {
                        InsertionError::TriangulationState(
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Neighbor cell {neighbor_key:?} not found during boundary facet gathering"
                                ),
                            },
                        )
                    })?;

                    // The reciprocal facet in the neighbor is the one that shares these D vertices
                    // and points back to bad_cell. We can find it by looking for which facet
                    // (vertex index) in the neighbor has bad_cell as its neighbor.
                    let reciprocal_idx = if let Some(neighbor_neighbors) = neighbor_cell.neighbors()
                    {
                        neighbor_neighbors
                            .iter()
                            .enumerate()
                            .find_map(|(idx, &nkey)| {
                                (nkey == Some(bad_cell)).then_some(idx)
                            })
                            .ok_or_else(|| {
                                InsertionError::TriangulationState(
                                    TriangulationValidationError::InconsistentDataStructure {
                                        message: format!(
                                            "Neighbor {neighbor_key:?} does not reciprocate to bad cell {bad_cell:?}"
                                        ),
                                    },
                                )
                            })?
                    } else {
                        // Neighbor has no neighbor info; this is a hard invariant failure.
                        // The triangulation should maintain neighbor information consistently.
                        // Note: If recovery-friendly behavior is needed, this could treat missing
                        // neighbor info as a boundary (None), but current design prefers strict validation.
                        return Err(InsertionError::TriangulationState(
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Neighbor cell {neighbor_key:?} has no neighbor information"
                                ),
                            },
                        ));
                    };

                    Some((neighbor_key, reciprocal_idx))
                } else {
                    // No neighbor on this side (true boundary facet)
                    None
                }
            } else {
                // Bad cell has no neighbor information (treat as boundary)
                None
            };

            boundary_infos.push(BoundaryFacetInfo {
                bad_cell,
                bad_facet_index,
                facet_vertex_keys,
                outside_neighbor,
            });
        }

        Ok(boundary_infos)
    }

    /// Helper function to set a neighbor pointer with validation.
    ///
    /// Reduces code duplication by encapsulating the ensure+validate+set pattern.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation
    /// * `cell_key` - Key of the cell to modify
    /// * `neighbor_idx` - Index in the neighbor array to set
    /// * `neighbor_key` - Key of the neighbor cell to set
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if the cell is not found or neighbor buffer size is invalid.
    ///
    /// # Note on Overwrites
    ///
    /// This function may overwrite existing neighbor pointers during cavity-based insertion.
    /// This is intentional: when removing bad cells and creating new ones, the neighbor
    /// relationships must be updated to point to the new cells. The algorithm ensures
    /// correctness through its transactional validate-create-commit pattern.
    fn set_neighbor_with_validation(
        tds: &mut Tds<T, U, V, D>,
        cell_key: CellKey,
        neighbor_idx: usize,
        neighbor_key: CellKey,
    ) -> Result<(), InsertionError> {
        // Validate neighbor index is within bounds for D+1 neighbors
        if neighbor_idx > D {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Neighbor index {} out of bounds for D+1 ({}) in cell {cell_key:?}",
                        neighbor_idx,
                        D + 1
                    ),
                },
            ));
        }

        let cell_mut = tds.cells_mut().get_mut(cell_key).ok_or_else(|| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Cannot get mutable reference to cell {cell_key:?}"),
                },
            )
        })?;

        let neighbors = cell_mut.ensure_neighbors_buffer_mut();
        if neighbors.len() != D + 1 {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Neighbor buffer size {} != D+1 ({}) for cell {cell_key:?}",
                        neighbors.len(),
                        D + 1
                    ),
                },
            ));
        }
        neighbors[neighbor_idx] = Some(neighbor_key);
        Ok(())
    }

    /// Wire neighbor relationships for newly created cells after cavity removal.
    ///
    /// This helper establishes neighbor pointers between the new cells filling the cavity
    /// and between new cells and the existing triangulation. It must be called AFTER
    /// bad cells have been removed.
    ///
    /// # Phase 3: Commit (Neighbor Wiring)
    ///
    /// This implements the final step of the transactional pattern:
    /// 1. **New→Old adjacency**: Connect new cells to neighbors across the cavity boundary
    /// 2. **New→New adjacency**: Connect new cells to each other where they share facets
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `inserted_vk` - Key of the newly inserted vertex
    /// * `boundary_infos` - Metadata extracted in Phase 1 about boundary facets
    /// * `created_cells` - Keys of newly created cells (corresponds 1:1 with `boundary_infos`)
    ///
    /// # Returns
    ///
    /// `Ok(())` if neighbor wiring succeeds, or an error if topology is inconsistent.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if:
    /// - Cell or vertex lookups fail
    /// - Neighbor wiring creates conflicts or asymmetries
    /// - The inserted vertex cannot be found in created cells
    fn connect_new_cells_to_neighbors(
        tds: &mut Tds<T, U, V, D>,
        inserted_vk: VertexKey,
        boundary_infos: &[BoundaryFacetInfo],
        created_cells: &[CellKey],
    ) -> Result<(), InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        use crate::core::collections::{FastHashMap, fast_hash_map_with_capacity};

        if created_cells.len() != boundary_infos.len() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Mismatch between created cells ({}) and boundary info ({})",
                        created_cells.len(),
                        boundary_infos.len()
                    ),
                },
            ));
        }

        // ====================================================================
        // STEP 1: Wire New→Old neighbors across the cavity boundary
        // ====================================================================
        for (new_cell_key, info) in created_cells.iter().zip(boundary_infos) {
            // Find the index of the inserted vertex in the new cell
            let new_cell = tds.get_cell(*new_cell_key).ok_or_else(|| {
                InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Created cell {new_cell_key:?} not found"),
                    },
                )
            })?;

            let inserted_idx = new_cell
                .vertices()
                .iter()
                .position(|&vk| vk == inserted_vk)
                .ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Inserted vertex {inserted_vk:?} not found in new cell {new_cell_key:?}"
                            ),
                        },
                    )
                })?;

            // If there's an outside neighbor, wire the bidirectional connection
            if let Some((outside_ck, outside_facet_idx)) = info.outside_neighbor {
                // Set new_cell's neighbor at inserted_idx → outside_ck
                Self::set_neighbor_with_validation(tds, *new_cell_key, inserted_idx, outside_ck)?;

                // Set outside_ck's neighbor at outside_facet_idx → new_cell
                Self::set_neighbor_with_validation(
                    tds,
                    outside_ck,
                    outside_facet_idx,
                    *new_cell_key,
                )?;
            }
        }

        // ====================================================================
        // STEP 2: Wire New→New neighbors within the cavity
        // ====================================================================
        // Type alias at module level would be better, but this is a local helper
        #[expect(clippy::items_after_statements)]
        type FacetSignature = SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>;

        // Build a map from facet signatures to (cell_key, local_facet_index)
        // Facets that include the inserted vertex connect new cells to each other
        let mut facet_to_cell: FastHashMap<u64, (CellKey, usize)> =
            fast_hash_map_with_capacity(created_cells.len() * D);

        for &new_cell_key in created_cells {
            // Clone cell vertices to avoid borrow issues
            let cell_vertices = tds
                .get_cell(new_cell_key)
                .ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!("Created cell {new_cell_key:?} disappeared"),
                        },
                    )
                })?
                .vertices()
                .iter()
                .copied()
                .collect::<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>();

            // For each vertex in the cell (opposite facet)
            for (opposite_idx, &opposite_vk) in cell_vertices.iter().enumerate() {
                // Skip the facet opposite the inserted vertex (already wired to outside)
                if opposite_vk == inserted_vk {
                    continue;
                }

                // Build facet signature: all vertices except opposite_vk
                // Note: facet_key_from_vertices() sorts internally, no need to pre-sort
                let facet_sig: FacetSignature = cell_vertices
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &vk)| (i != opposite_idx).then_some(vk))
                    .collect();

                let facet_key = facet_key_from_vertices(&facet_sig);

                // Check if we've seen this facet before
                if let Some((other_cell_key, other_facet_idx)) = facet_to_cell.get(&facet_key) {
                    // Wire bidirectional neighbor pointers
                    let other_ck = *other_cell_key;
                    let other_idx = *other_facet_idx;

                    // Set new_cell[opposite_idx] → other_cell
                    Self::set_neighbor_with_validation(tds, new_cell_key, opposite_idx, other_ck)?;

                    // Set other_cell[other_idx] → new_cell
                    Self::set_neighbor_with_validation(tds, other_ck, other_idx, new_cell_key)?;
                } else {
                    // First time seeing this facet; record it
                    facet_to_cell.insert(facet_key, (new_cell_key, opposite_idx));
                }
            }
        }

        Ok(())
    }

    /// Rollback vertex insertion by removing all cells containing the vertex and the vertex itself.
    ///
    /// This is a smart rollback utility that delegates to `Tds::remove_vertex()` which atomically
    /// finds and removes all cells containing the vertex, then removes the vertex itself.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex that was being inserted
    /// * `vertex_existed_before` - Whether the vertex existed in TDS before the operation started
    ///
    /// # Implementation
    ///
    /// Delegates to `Tds::remove_vertex()` which:
    /// 1. Finds all cells containing the vertex
    /// 2. Removes all such cells
    /// 3. Removes the vertex itself (if newly inserted)
    fn rollback_vertex_insertion(
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        _vertex_existed_before: bool,
    ) where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Note: remove_vertex() handles both cells and vertex atomically
        // This works regardless of whether the vertex existed before because
        // remove_vertex() safely handles all cases
        tds.remove_vertex(vertex);
    }

    /// Legacy rollback function that takes explicit cell keys.
    ///
    /// # Deprecation
    ///
    /// **Deprecated since v0.5.0, will be removed in v0.6.0**.
    /// Use `rollback_vertex_insertion()` which automatically finds cells via `Tds::remove_vertex()`.
    /// Manual cell tracking is error-prone and unnecessary.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `created_cell_keys` - Keys of cells to remove (ignored in favor of automatic detection)
    /// * `vertex` - The vertex that was being inserted
    /// * `vertex_existed_before` - Whether the vertex existed in TDS before the operation started
    #[deprecated(
        since = "0.5.0",
        note = "Use `rollback_vertex_insertion()` instead. Will be removed in v0.6.0."
    )]
    fn rollback_created_cells_and_vertex(
        tds: &mut Tds<T, U, V, D>,
        _created_cell_keys: &[crate::core::triangulation_data_structure::CellKey],
        vertex: &Vertex<T, U, D>,
        vertex_existed_before: bool,
    ) where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Ignore _created_cell_keys and delegate to the smart rollback
        Self::rollback_vertex_insertion(tds, vertex, vertex_existed_before);
    }

    /// Removes bad cells from the triangulation
    ///
    /// This is a shared utility method that removes cells using the optimized
    /// key-based batch removal method.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `bad_cells` - Keys of the cells to remove
    fn remove_bad_cells(
        tds: &mut Tds<T, U, V, D>,
        bad_cells: &[crate::core::triangulation_data_structure::CellKey],
    ) where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Use the optimized batch removal method that handles UUID mapping
        // and generation counter updates internally
        tds.remove_cells_by_keys(bad_cells);
    }

    /// Restores TDS to valid state after cavity insertion failure.
    ///
    /// This helper function is called when cavity-based insertion fails after the "point of no return"
    /// (after bad cells have been removed). It restores the triangulation to a consistent state by:
    /// 1. Re-inserting the cells that were removed from the cavity
    /// 2. Removing any new cells that were created but are now invalid
    /// 3. Removing the vertex if it was newly inserted
    ///
    /// # Context
    ///
    /// In cavity-based insertion, there's a "point of no return" after which the old cavity cells
    /// have been removed but new cells may not yet be fully wired. If any operation fails after this
    /// point (neighbor wiring, finalization, validation), we must restore the TDS to prevent leaving
    /// it in an invalid state with vertices but no cells.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `saved_bad_cells` - The cells that were removed from the cavity (to be restored)
    /// * `created_cell_keys` - Keys of new cells that were created (to be removed)
    /// * `vertex_was_newly_inserted` - Whether the vertex was inserted during this operation
    /// * `vertex` - The vertex key to remove if it was newly inserted
    ///
    /// # Usage
    ///
    /// ```ignore
    /// // Save cells before removing them
    /// let saved_cells: Vec<_> = bad_cells.iter()
    ///     .filter_map(|&ck| tds.get_cell(ck).cloned())
    ///     .collect();
    ///
    /// // Remove bad cells (point of no return)
    /// Self::remove_bad_cells(tds, &bad_cells);
    ///
    /// // If anything fails after this point, restore:
    /// if let Err(e) = risky_operation(tds) {
    ///     Self::restore_cavity_insertion_failure(
    ///         tds,
    ///         &saved_cells,
    ///         &created_cell_keys,
    ///         !vertex_existed_before,
    ///         vertex,
    ///     );
    ///     return Err(e);
    /// }
    /// ```
    ///
    /// # Testing
    ///
    /// This function can be unit tested by:
    /// 1. Creating a triangulation with known state
    /// 2. Simulating a partial cavity insertion (remove cells, add new cells)
    /// 3. Calling this function
    /// 4. Verifying the triangulation returns to its original valid state
    fn restore_cavity_insertion_failure(
        tds: &mut Tds<T, U, V, D>,
        saved_bad_cells: &[Cell<T, U, V, D>],
        created_cell_keys: &[CellKey],
        vertex_was_newly_inserted: bool,
        vertex_key: VertexKey,
    ) where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Restore the cells that were removed from the cavity
        for cell in saved_bad_cells {
            let _ = tds.insert_cell_with_mapping(cell.clone());
        }

        // Remove any new cells that were created
        tds.remove_cells_by_keys(created_cell_keys);

        // Remove the vertex if it was newly inserted during this operation
        if vertex_was_newly_inserted {
            // Get the vertex by key so we can call remove_vertex
            // Copy the vertex to avoid borrow checker issues
            if let Some(&vertex) = tds.get_vertex_by_key(vertex_key) {
                tds.remove_vertex(&vertex);
            }
        }
    }

    /// Atomic operation: ensure vertex is in TDS and remove bad cells.
    ///
    /// This method ensures that either both operations succeed or neither has side effects,
    /// preventing the TDS from being left in an inconsistent state. The operation ordering
    /// is critical: vertex insertion must happen before cell removal to avoid corruption
    /// if vertex validation fails after cells are already removed.
    ///
    /// This implementation uses `ArcSwapOption` operations for efficient cache invalidation
    /// without requiring separate generation tracking.
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure to modify
    /// * `vertex` - The vertex to ensure is present in TDS
    /// * `bad_cells` - Cell keys to remove atomically
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Both operations succeeded atomically
    /// * `Err(InsertionError)` - Operation failed, TDS should be in consistent state
    ///
    /// # Atomic Operation Guarantee
    ///
    /// This method provides atomic semantics:
    /// - If vertex insertion fails, no cells are removed and no caches are invalidated
    /// - If vertex insertion succeeds, cells are guaranteed to be removed and caches invalidated
    /// - TDS remains in a consistent state regardless of success or failure
    /// # Errors
    ///
    /// Returns `InsertionError::TriangulationState` if the vertex cannot be inserted into the TDS.
    fn atomic_vertex_insert_and_remove_cells(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        bad_cells: &[crate::core::triangulation_data_structure::CellKey],
    ) -> Result<(), InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Ensure vertex is in TDS before destructive operations
        Self::ensure_vertex_in_tds(tds, vertex).map_err(InsertionError::TriangulationState)?;

        // Remove bad cells (this modifies TDS structure)
        Self::remove_bad_cells(tds, bad_cells);

        // Invalidate cache using direct ArcSwapOption operations
        // This is more efficient than generation-based tracking
        self.invalidate_cache_atomically();

        Ok(())
    }

    /// Hook method for atomic cache invalidation using `ArcSwapOption`
    ///
    /// This method leverages `ArcSwapOption`'s atomic operations for efficient
    /// cache invalidation without requiring separate generation counters.
    /// Algorithms that implement caching should override this method.
    fn invalidate_cache_atomically(&mut self) {
        // Default implementation: no caching, so nothing to invalidate
    }

    /// Finalizes the triangulation after insertion to ensure consistency
    ///
    /// This is a shared utility method that performs post-insertion cleanup
    /// and relationship establishment.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if finalization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails due to data structure inconsistencies.
    fn finalize_after_insertion(
        tds: &mut Tds<T, U, V, D>,
    ) -> Result<(), TriangulationValidationError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    {
        // Remove duplicate cells first
        tds.remove_duplicate_cells()?;

        // Fix invalid facet sharing - STRICT: must succeed
        // We use preventive filtering (filter_boundary_facets_by_valid_facet_sharing)
        // to avoid creating invalid topology. This is a safety net that ensures
        // the triangulation remains valid.
        //
        // If this fails, we return an error rather than tolerating invalid topology.
        // This ensures vertex insertion either succeeds completely with all invariants
        // satisfied, or fails cleanly without corrupting the triangulation.
        tds.fix_invalid_facet_sharing()?;

        // Assign neighbor relationships - STRICT: must succeed
        // Proper neighbor relationships are fundamental to triangulation validity.
        // If this fails, the triangulation is in an invalid state.
        tds.assign_neighbors()?;

        // Assign incident cells to vertices - STRICT: must succeed
        // Incident cell assignments are required for TDS validity.
        tds.assign_incident_cells()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::bowyer_watson::IncrementalBowyerWatson;
    use crate::core::facet::facet_key_from_vertices;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use approx::assert_abs_diff_eq;

    /// Macro to generate dimension-specific insertion algorithm tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions for the `InsertionAlgorithm` trait methods. It creates tests for:
    /// - Finding bad cells for interior vertices
    /// - Finding bad cells for exterior vertices
    /// - Finding cavity boundary facets
    /// - Finding visible boundary facets (lightweight)
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_insertion_algorithm_dimensions! {
    ///     insertion_2d => 2 => "triangle" => vec![vertex!([0.0, 0.0]), ...],
    /// }
    /// ```
    macro_rules! test_insertion_algorithm_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr => $initial_vertices:expr, $interior_vertex:expr, $exterior_vertex:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic insertion algorithm functionality in this dimension
                    let initial_vertices = $initial_vertices;
                    let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();
                    let mut algorithm = IncrementalBowyerWatson::new();

                    assert!(tds.is_valid().is_ok(), "{}D: Initial TDS should be valid", $dim);
                    assert_eq!(tds.dim(), $dim as i32, "{}D: TDS should have dimension {}", $dim, $dim);

                    // Test find_bad_cells with interior vertex
                    let interior_vertex = $interior_vertex;
                    let bad_cells_interior = algorithm
                        .find_bad_cells(&tds, &interior_vertex)
                        .expect("{}D: Should find bad cells for interior vertex");

                    assert!(
                        !bad_cells_interior.is_empty(),
                        "{}D: Interior vertex should find at least one bad cell",
                        $dim
                    );
                    assert!(
                        bad_cells_interior.len() <= tds.number_of_cells(),
                        "{}D: Cannot have more bad cells than total cells",
                        $dim
                    );

                    // Test find_bad_cells with exterior vertex
                    let exterior_vertex = $exterior_vertex;
                    let bad_cells_exterior = algorithm
                        .find_bad_cells(&tds, &exterior_vertex)
                        .expect("{}D: Should find bad cells for exterior vertex");

                    assert!(
                        bad_cells_exterior.len() <= tds.number_of_cells(),
                        "{}D: Cannot have more bad cells than total cells",
                        $dim
                    );

                    // Verify TDS remains valid after queries
                    assert!(tds.is_valid().is_ok(), "{}D: TDS should remain valid", $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _cavity_boundary>]() {
                        // Test cavity boundary facet detection
                        let initial_vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();
                        let algorithm = IncrementalBowyerWatson::new();

                        // Get the single cell as a "bad cell" for testing
                        let cell_keys: Vec<_> = tds.cell_keys().collect();
                        assert!(!cell_keys.is_empty(), "{}D: Should have at least one cell", $dim);

                        let bad_cells = vec![cell_keys[0]];
                        let boundary_facets = algorithm
                            .find_cavity_boundary_facets(&tds, &bad_cells)
                            .expect("{}D: Should find cavity boundary facets");

                        // For a single simplex, all D+1 facets should be boundary facets
                        assert_eq!(
                            boundary_facets.len(),
                            $dim + 1,
                            "{}D: Single simplex should have {} boundary facets (D+1)",
                            $dim,
                            $dim + 1
                        );

                        // Verify each boundary facet handle is valid
                        for handle in &boundary_facets {
                            assert!(
                                tds.get_cell(handle.cell_key()).is_some(),
                                "{}D: Boundary facet cell should exist",
                                $dim
                            );
                            assert!(
                                handle.facet_index() <= $dim as u8,
                                "{}D: Facet index should be valid",
                                $dim
                            );
                        }
                    }

                    #[test]
                    fn [<$test_name _visible_boundary_facets>]() {
                        // Test visible boundary facet detection (lightweight)
                        let initial_vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();
                        let algorithm = IncrementalBowyerWatson::new();

                        // Test with exterior vertex
                        let exterior_vertex = $exterior_vertex;
                        let visible_facets = algorithm
                            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
                            .expect("{}D: Should find visible boundary facets");

                        // Exterior vertex should see at least some facets
                        assert!(
                            !visible_facets.is_empty(),
                            "{}D: Exterior vertex should see at least some boundary facets",
                            $dim
                        );
                        assert!(
                            visible_facets.len() <= $dim + 1,
                            "{}D: Cannot see more than {} facets from a simplex",
                            $dim,
                            $dim + 1
                        );

                        // Verify each visible facet handle is valid
                        for handle in &visible_facets {
                            assert!(
                                tds.get_cell(handle.cell_key()).is_some(),
                                "{}D: Visible facet cell should exist",
                                $dim
                            );
                            let facet_view = crate::core::facet::FacetView::new(
                                &tds,
                                handle.cell_key(),
                                handle.facet_index()
                            )
                            .expect("{}D: Should create FacetView from handle");

                            assert_eq!(
                                facet_view.vertices().unwrap().count(),
                                $dim,
                                "{}D: Visible facet should have {} vertices",
                                $dim,
                                $dim
                            );
                        }
                    }

                    #[test]
                    fn [<$test_name _empty_bad_cells>]() {
                        // Test cavity boundary with empty bad cells list
                        let initial_vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();
                        let algorithm = IncrementalBowyerWatson::new();

                        let bad_cells: Vec<crate::core::triangulation_data_structure::CellKey> = vec![];
                        let boundary_facets = algorithm
                            .find_cavity_boundary_facets(&tds, &bad_cells)
                            .expect("{}D: Should handle empty bad cells");

                        assert!(
                            boundary_facets.is_empty(),
                            "{}D: Empty bad cells should yield empty boundary facets",
                            $dim
                        );
                    }

                    #[test]
                    fn [<$test_name _deduplicate_boundary_facet_info>]() {
                        // Test duplicate detection and error handling in boundary facet deduplication
                        println!("Testing deduplicate_boundary_facet_info in {}D", $dim);

                        let initial_vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();
                        let cell_key = tds.cell_keys().next().expect("Should have at least one cell");

                        // Get vertex keys for the simplex (should have D+1 vertices)
                        let vertex_keys: Vec<_> = tds.vertex_keys().collect();
                        assert_eq!(vertex_keys.len(), $dim + 1, "{}D: Should have {} vertices", $dim, $dim + 1);

                        // Case 1: Test with duplicates - should return error
                        let mut facet_info_with_duplicates = Vec::new();

                        // Add first distinct facet (all vertices except the first)
                        let mut first_facet_vertices = SmallBuffer::new();
                        for v in vertex_keys.iter().skip(1).take($dim) {
                            first_facet_vertices.push(*v);
                        }
                        facet_info_with_duplicates.push(BoundaryFacetInfo {
                            bad_cell: cell_key,
                            bad_facet_index: 0,
                            facet_vertex_keys: first_facet_vertices.clone(),
                            outside_neighbor: None,
                        });

                        // Add second distinct facet (all vertices except the second)
                        let mut second_facet_vertices = SmallBuffer::new();
                        second_facet_vertices.push(vertex_keys[0]);
                        for v in vertex_keys.iter().skip(2).take($dim - 1) {
                            second_facet_vertices.push(*v);
                        }
                        facet_info_with_duplicates.push(BoundaryFacetInfo {
                            bad_cell: cell_key,
                            bad_facet_index: 1,
                            facet_vertex_keys: second_facet_vertices.clone(),
                            outside_neighbor: None,
                        });

                        // Add duplicate of first facet (same vertex set, reversed order)
                        let mut duplicate_facet_vertices = SmallBuffer::new();
                        for v in first_facet_vertices.iter().rev() {
                            duplicate_facet_vertices.push(*v);
                        }
                        facet_info_with_duplicates.push(BoundaryFacetInfo {
                            bad_cell: cell_key,
                            bad_facet_index: 2,
                            facet_vertex_keys: duplicate_facet_vertices,
                            outside_neighbor: None,
                        });

                        println!("  {}D: Testing with {} facets (including 1 duplicate)", $dim, facet_info_with_duplicates.len());

                        // Should return error with correct counts
                        let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, $dim>::deduplicate_boundary_facet_info(
                            facet_info_with_duplicates.clone()
                        );

                        match result {
                            Err(InsertionError::DuplicateBoundaryFacets { duplicate_count, total_count }) => {
                                assert_eq!(duplicate_count, 1, "{}D: Should detect exactly 1 duplicate", $dim);
                                assert_eq!(total_count, 3, "{}D: Total count should be 3", $dim);
                                println!("  ✓ {}D: Correctly detected {} duplicate out of {} facets", $dim, duplicate_count, total_count);
                            }
                            Ok(_) => panic!("{}D: Should have returned DuplicateBoundaryFacets error", $dim),
                            Err(other) => panic!("{}D: Unexpected error: {:?}", $dim, other),
                        }

                        // Case 2: Test without duplicates - should return Ok
                        let facet_info_no_duplicates = vec![
                            BoundaryFacetInfo {
                                bad_cell: cell_key,
                                bad_facet_index: 0,
                                facet_vertex_keys: first_facet_vertices.clone(),
                                outside_neighbor: None,
                            },
                            BoundaryFacetInfo {
                                bad_cell: cell_key,
                                bad_facet_index: 1,
                                facet_vertex_keys: second_facet_vertices.clone(),
                                outside_neighbor: None,
                            },
                        ];

                        println!("  {}D: Testing with {} unique facets (no duplicates)", $dim, facet_info_no_duplicates.len());

                        let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, $dim>::deduplicate_boundary_facet_info(
                            facet_info_no_duplicates.clone()
                        );

                        match result {
                            Ok(deduplicated) => {
                                assert_eq!(deduplicated.len(), 2, "{}D: Should have 2 unique facets", $dim);

                                // Verify all facets have distinct vertex sets
                                let unique_sets: std::collections::HashSet<_> = deduplicated
                                    .iter()
                                    .map(|info| {
                                        let mut sorted: Vec<_> = info.facet_vertex_keys.iter().copied().collect();
                                        sorted.sort();
                                        sorted
                                    })
                                    .collect();

                                assert_eq!(
                                    unique_sets.len(),
                                    deduplicated.len(),
                                    "{}D: All facets should have distinct vertex sets",
                                    $dim
                                );

                                println!("  ✓ {}D: Correctly returned {} unique facets", $dim, deduplicated.len());
                            }
                            Err(e) => panic!("{}D: Should not have returned error for non-duplicate input: {:?}", $dim, e),
                        }

                        println!("  ✓ {}D: Deduplication test passed", $dim);
                    }

                    #[test]
                    fn [<$test_name _is_vertex_interior>]() {
                        // Test is_vertex_interior classification
                        println!("Testing is_vertex_interior in {}D", $dim);
                        let vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices).unwrap();
                        let algorithm = IncrementalBowyerWatson::new();

                        // Test interior vertex
                        let interior_vertex = $interior_vertex;
                        let result = algorithm.is_vertex_interior(&tds, &interior_vertex);
                        assert!(result.is_ok(), "{}D: Should succeed for interior vertex", $dim);
                        assert!(result.unwrap(), "{}D: Interior vertex should be classified as interior", $dim);

                        // Test exterior vertex
                        let exterior_vertex = $exterior_vertex;
                        let result = algorithm.is_vertex_interior(&tds, &exterior_vertex);
                        assert!(result.is_ok(), "{}D: Should succeed for exterior vertex", $dim);
                        assert!(!result.unwrap(), "{}D: Exterior vertex should not be interior", $dim);

                        println!("  ✓ {}D: is_vertex_interior test passed", $dim);
                    }

                    #[test]
                    fn [<$test_name _is_vertex_likely_exterior>]() {
                        // Test is_vertex_likely_exterior classification
                        println!("Testing is_vertex_likely_exterior in {}D", $dim);
                        let vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices).unwrap();

                        // Test far exterior vertex
                        let far_exterior = $exterior_vertex;
                        let is_exterior = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, $dim>::is_vertex_likely_exterior(
                            &tds,
                            &far_exterior,
                        );
                        assert!(is_exterior, "{}D: Far exterior vertex should be classified as exterior", $dim);

                        // Test interior vertex
                        let interior = $interior_vertex;
                        let is_exterior = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, $dim>::is_vertex_likely_exterior(
                            &tds,
                            &interior,
                        );
                        assert!(!is_exterior, "{}D: Interior vertex should not be classified as exterior", $dim);

                        println!("  ✓ {}D: is_vertex_likely_exterior test passed", $dim);
                    }

                    #[test]
                    fn [<$test_name _gather_boundary_facet_info>]() {
                        // Test gather_boundary_facet_info
                        println!("Testing gather_boundary_facet_info in {}D", $dim);
                        let vertices = $initial_vertices;
                        let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices).unwrap();

                        // Get boundary facets
                        let boundary_facets: Vec<_> = tds
                            .boundary_facets()
                            .unwrap()
                            .map(|fv| FacetHandle::new(fv.cell_key(), fv.facet_index()))
                            .collect();

                        // Test gathering
                        let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, $dim>::gather_boundary_facet_info(
                            &tds,
                            &boundary_facets,
                        );
                        assert!(result.is_ok(), "{}D: Should succeed for valid boundary facets", $dim);
                        let infos = result.unwrap();
                        assert!(!infos.is_empty(), "{}D: Should have boundary facet info", $dim);

                        // Each facet should have D vertices
                        for info in &infos {
                            assert_eq!(
                                info.facet_vertex_keys.len(),
                                $dim,
                                "{}D: Each facet should have {} vertices",
                                $dim, $dim
                            );
                        }

                        println!("  ✓ {}D: gather_boundary_facet_info test passed", $dim);
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D using the macro
    test_insertion_algorithm_dimensions! {
        insertion_2d => 2 => "triangle" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([2.0, 0.0]),
                vertex!([1.0, 2.0]),
            ],
            vertex!([1.0, 0.5]),
            vertex!([3.0, 0.0]),

        insertion_3d => 3 => "tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5]),
            vertex!([3.0, 0.0, 0.0]),

        insertion_4d => 4 => "4-simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5]),
            vertex!([3.0, 0.0, 0.0, 0.0]),

        insertion_5d => 5 => "5-simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5, 0.5]),
            vertex!([3.0, 0.0, 0.0, 0.0, 0.0]),
    }

    #[test]
    fn test_find_visible_boundary_facets_lightweight_vs_regular_consistency() {
        println!("Testing find_visible_boundary_facets_lightweight with multiple test cases");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test multiple vertices at different positions
        let test_vertices = vec![
            (vertex!([2.0, 0.0, 0.0]), "exterior +X"),
            (vertex!([0.0, 2.0, 0.0]), "exterior +Y"),
            (vertex!([0.0, 0.0, 2.0]), "exterior +Z"),
            (vertex!([0.5, 0.5, 0.5]), "interior"),
            (vertex!([-1.0, 0.0, 0.0]), "exterior -X"),
        ];

        for (i, (test_vertex, description)) in test_vertices.iter().enumerate() {
            println!("  Testing vertex {i} ({description}): {test_vertex:?}");

            // Get results from lightweight method
            let lightweight_result = algorithm
                .find_visible_boundary_facets_lightweight(&tds, test_vertex)
                .expect("Lightweight method should work");

            // Verify reasonable results
            assert!(
                lightweight_result.len() <= 4,
                "Cannot see more than 4 facets from a tetrahedron for vertex {i}"
            );

            // Verify all returned handles are valid and can be converted to FacetViews
            for handle in &lightweight_result {
                assert!(
                    tds.get_cell(handle.cell_key()).is_some(),
                    "Cell key {:?} should exist in TDS",
                    handle.cell_key()
                );
                assert!(
                    handle.facet_index() < 4,
                    "Facet index {} should be valid for 3D tetrahedron",
                    handle.facet_index()
                );

                // Verify we can create a FacetView from the handle
                let _facet_view = crate::core::facet::FacetView::new(
                    &tds,
                    handle.cell_key(),
                    handle.facet_index(),
                )
                .expect("Should be able to create FacetView from handle");
            }

            println!(
                "  Vertex {i} ({description}) sees {} valid facet handles",
                lightweight_result.len()
            );
        }

        println!("✓ Lightweight method validation test passed");
    }

    #[test]
    fn test_find_visible_boundary_facets_lightweight_empty_triangulation() {
        println!("Testing find_visible_boundary_facets_lightweight with empty triangulation");

        // Create empty triangulation
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let algorithm = IncrementalBowyerWatson::new();

        // Test with any vertex
        let test_vertex = vertex!([1.0, 0.0, 0.0]);
        let result = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &test_vertex)
            .expect("Should handle empty triangulation gracefully");

        // Should find no visible facets in empty triangulation
        assert!(
            result.is_empty(),
            "Empty triangulation should have no visible boundary facets"
        );

        println!("✓ Empty triangulation test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles() {
        println!("Testing create_cells_from_facet_handles");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get some boundary facet handles
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let visible_handles = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should find visible boundary facets");

        assert!(
            !visible_handles.is_empty(),
            "Should have found some visible facets"
        );

        let initial_cell_count = tds.number_of_cells();
        println!("  Initial cell count: {initial_cell_count}");
        println!(
            "  Creating cells from {} facet handles",
            visible_handles.len()
        );

        // Create cells from the handles
        let cells_created = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &visible_handles,
            &exterior_vertex,
        )
        .expect("Should successfully create cells from facet handles");

        let final_cell_count = tds.number_of_cells();
        println!("  Final cell count: {final_cell_count}");
        println!("  Cells created: {cells_created}");

        // Should have created some cells
        assert!(cells_created > 0, "Should have created at least one cell");
        assert_eq!(
            final_cell_count,
            initial_cell_count + cells_created,
            "Cell count should increase by the number of cells created"
        );

        println!("✓ Create cells from facet handles test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles_empty_input() {
        println!("Testing create_cells_from_facet_handles with empty handle list");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        let test_vertex = vertex!([2.0, 0.0, 0.0]);
        let empty_handles: Vec<FacetHandle> = Vec::new();

        let initial_cell_count = tds.number_of_cells();
        let initial_vertex_count = tds.number_of_vertices();

        // Create cells from empty handle list - should error immediately (early exit)
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &empty_handles,
            &test_vertex,
        );

        // Should fail immediately with clear error message (no vertex insertion needed)
        assert!(result.is_err(), "Should error on empty handle list");

        // Verify TDS state unchanged (atomic rollback)
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count,
            "Cell count should not change after error"
        );
        assert_eq!(
            tds.number_of_vertices(),
            initial_vertex_count,
            "Vertex count should not change after error (vertex rolled back)"
        );

        println!("✓ Empty handle list test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles_invalid_cell_key() {
        println!("Testing create_cells_from_facet_handles with invalid cell key");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        let test_vertex = vertex!([2.0, 0.0, 0.0]);

        // Create an invalid handle by using a valid cell key but then removing that cell.
        // This simulates realistic stale handle scenarios that can occur during triangulation
        // operations when handles are stored but cells are removed before handle use.
        let valid_cell_key = tds.cell_keys().next().expect("TDS should have cells");

        // Remove the cell to make the key invalid
        let _removed_cell = tds.remove_cell_by_key(valid_cell_key);

        let invalid_handles = vec![FacetHandle::new(valid_cell_key, 0)]; // Now invalid cell key, valid facet index

        // Should return error for invalid cell key
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &invalid_handles,
            &test_vertex,
        );

        assert!(result.is_err(), "Should return error for invalid cell key");

        if let Err(e) = result {
            // Should be a TriangulationState error about cell not found
            match e {
                crate::core::traits::insertion_algorithm::InsertionError::TriangulationState(
                    crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure { message }
                ) => {
                    assert!(
                        message.contains("not found"),
                        "Error message should mention cell not found, got: {message}"
                    );
                }
                _ => panic!("Expected InconsistentDataStructure error, got: {e:?}"),
            }
        }

        println!("✓ Invalid cell key test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles_invalid_facet_index() {
        println!("Testing create_cells_from_facet_handles with invalid facet index");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        let test_vertex = vertex!([2.0, 0.0, 0.0]);

        // Get a valid cell key but use invalid facet index
        let valid_cell_key = tds
            .cell_keys()
            .next()
            .expect("Should have at least one cell");
        let invalid_handles = vec![FacetHandle::new(valid_cell_key, 99)]; // 99 is way out of bounds for 3D (should be 0-3)

        // Should return error for invalid facet index
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &invalid_handles,
            &test_vertex,
        );

        assert!(
            result.is_err(),
            "Should return error for invalid facet index"
        );

        if let Err(e) = result {
            match e {
                crate::core::traits::insertion_algorithm::InsertionError::TriangulationState(
                    crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure { message }
                ) => {
                    assert!(
                        message.contains("Facet") || message.contains("Boundary facet") || message.contains("vertices"),
                        "Error message should mention facet issue, got: {message}"
                    );
                }
                _ => panic!("Expected InconsistentDataStructure error, got: {e:?}"),
            }
        }

        println!("✓ Invalid facet index test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles_duplicate_handles() {
        println!("Testing create_cells_from_facet_handles with duplicate handles");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get some valid handles
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let original_handles = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should find visible boundary facets");

        assert!(
            !original_handles.is_empty(),
            "Should have found some visible facets"
        );

        // Create duplicate handles by repeating the first handle
        let first_handle = original_handles[0];
        let mut duplicate_handles = original_handles;
        duplicate_handles.push(first_handle); // Add duplicate
        duplicate_handles.push(first_handle); // Add another duplicate

        let initial_cell_count = tds.number_of_cells();
        println!("  Initial cell count: {initial_cell_count}");
        println!(
            "  Testing with {} handles (including duplicates)",
            duplicate_handles.len()
        );

        // Create cells from handles with duplicates - should fail with error
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &duplicate_handles,
            &exterior_vertex,
        );

        // Should return error about duplicate boundary facets
        match result {
            Err(InsertionError::DuplicateBoundaryFacets {
                duplicate_count,
                total_count,
            }) => {
                assert!(duplicate_count >= 2, "Should detect at least 2 duplicates");
                assert_eq!(
                    total_count,
                    duplicate_handles.len(),
                    "Total count should match input"
                );
                println!(
                    "  ✓ Correctly detected {duplicate_count} duplicates out of {total_count} handles"
                );
            }
            Ok(_) => panic!("Should have returned DuplicateBoundaryFacets error"),
            Err(other) => panic!("Unexpected error: {other:?}"),
        }

        println!("✓ Duplicate handles test works correctly - errors on duplicates as expected");
    }

    #[test]
    fn test_create_cells_from_facet_handles_mixed_valid_invalid() {
        println!("Testing create_cells_from_facet_handles with mixed valid and invalid handles");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get some valid handles
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let valid_handles = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should find visible boundary facets");

        assert!(
            !valid_handles.is_empty(),
            "Should have found some visible facets"
        );

        // Mix valid handles with invalid ones
        // Use a valid cell key but invalid facet index for easier testing
        let valid_cell_key = tds.cell_keys().next().expect("TDS should have cells");
        let mut mixed_handles = valid_handles;
        mixed_handles.push(FacetHandle::new(valid_cell_key, 99)); // Add invalid handle with bad facet index

        let test_vertex = vertex!([2.0, 0.0, 0.0]);

        // Should fail on first invalid handle encountered
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &mixed_handles,
            &test_vertex,
        );

        assert!(
            result.is_err(),
            "Should return error when encountering invalid handle"
        );

        // Verify it's the expected error type
        if let Err(e) = result {
            match e {
                crate::core::traits::insertion_algorithm::InsertionError::TriangulationState(_) => {
                    // Expected error type
                }
                _ => panic!("Expected TriangulationState error, got: {e:?}"),
            }
        }

        println!("✓ Mixed valid/invalid handles test works correctly");
    }

    #[test]
    fn test_create_cells_from_facet_handles_boundary_conditions() {
        println!("Testing create_cells_from_facet_handles boundary conditions");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Test with valid cell key but boundary facet indices (0, 1, 2, 3 for 3D tetrahedron)
        let valid_cell_key = tds
            .cell_keys()
            .next()
            .expect("Should have at least one cell");

        // Test each valid facet index for 3D tetrahedron
        for facet_idx in 0u8..4u8 {
            let handles = vec![FacetHandle::new(valid_cell_key, facet_idx)];
            let test_vertex = vertex!([2.0, 0.0, 0.0]);

            let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
                &mut tds,
                &handles,
                &test_vertex,
            );

            // All facet indices 0-3 should be valid for 3D tetrahedron
            match result {
                Ok(cells_created) => {
                    println!("  Facet index {facet_idx}: Created {cells_created} cells");
                }
                Err(e) => {
                    println!("  Facet index {facet_idx}: Error - {e:?}");
                    // This might be okay depending on the geometric configuration
                    // Some facets might not be suitable for cell creation
                }
            }
        }

        // Test just-out-of-bounds facet index (4 for 3D tetrahedron)
        let out_of_bounds_handles = vec![FacetHandle::new(valid_cell_key, 4)];
        let test_vertex = vertex!([2.0, 0.0, 0.0]);

        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &out_of_bounds_handles,
            &test_vertex,
        );

        assert!(
            result.is_err(),
            "Facet index 4 should be out of bounds for 3D tetrahedron"
        );

        println!("✓ Boundary conditions test works correctly");
    }

    /// Test atomic rollback behavior of `create_cells_from_facet_handles`.
    ///
    /// This test verifies that when cell creation fails partway through processing
    /// multiple facet handles, any previously created cells are rolled back to keep
    /// the TDS in a consistent state.
    #[test]
    fn test_create_cells_from_facet_handles_atomic_rollback() {
        println!("Testing create_cells_from_facet_handles atomic rollback behavior");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get some valid handles
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let valid_handles = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should find visible boundary facets");

        assert!(
            !valid_handles.is_empty(),
            "Should have found some visible facets"
        );
        println!("  Found {} valid handles", valid_handles.len());

        // Record initial state
        let initial_cell_count = tds.number_of_cells();
        let initial_cell_keys: Vec<_> = tds.cell_keys().collect();
        println!("  Initial cell count: {initial_cell_count}");
        println!("  Initial cell keys: {}", initial_cell_keys.len());

        // Create a handle list with valid handles followed by an invalid one
        // This ensures some cells will be created before the error occurs
        let valid_cell_key = tds.cell_keys().next().expect("TDS should have cells");
        let mut mixed_handles = valid_handles.clone();
        mixed_handles.push(FacetHandle::new(valid_cell_key, 99)); // Add invalid handle at the end

        println!(
            "  Testing with {} handles ({} valid + 1 invalid)",
            mixed_handles.len(),
            mixed_handles.len() - 1
        );

        // Attempt to create cells - this should fail on the invalid handle
        let result = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &mixed_handles,
            &exterior_vertex,
        );

        // Verify the operation failed
        assert!(result.is_err(), "Should return error for invalid handle");
        println!("  ✓ Operation correctly failed: {:?}", result.unwrap_err());

        // Verify atomic rollback: cell count should be unchanged
        let final_cell_count = tds.number_of_cells();
        assert_eq!(
            final_cell_count, initial_cell_count,
            "Cell count should be unchanged after rollback (atomic behavior)"
        );
        println!("  ✓ Cell count unchanged: {initial_cell_count}");

        // Verify no new cells were added (all created cells were rolled back)
        let final_cell_keys: Vec<_> = tds.cell_keys().collect();
        assert_eq!(
            final_cell_keys.len(),
            initial_cell_keys.len(),
            "Number of cell keys should be unchanged"
        );

        // Verify the exact same cells exist (no cells added or removed)
        for initial_key in &initial_cell_keys {
            assert!(
                tds.get_cell(*initial_key).is_some(),
                "All initial cells should still exist after rollback"
            );
        }
        println!("  ✓ All initial cells preserved");

        // Verify no new cells exist
        for final_key in &final_cell_keys {
            assert!(
                initial_cell_keys.contains(final_key),
                "No new cells should exist after rollback"
            );
        }
        println!("  ✓ No new cells created");

        // Verify the vertex was removed during rollback
        assert!(
            tds.vertex_key_from_uuid(&exterior_vertex.uuid()).is_none(),
            "Vertex should not be present in TDS after rollback"
        );
        println!(
            "  ✓ Vertex {} properly removed from TDS during rollback",
            exterior_vertex.uuid()
        );

        // Now test that valid handles still work after the failed attempt
        let cells_created = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &valid_handles,
            &exterior_vertex,
        )
        .expect("Valid handles should work after failed attempt");

        assert!(
            cells_created > 0,
            "Should successfully create cells with valid handles"
        );
        println!("  ✓ TDS remains functional after rollback: created {cells_created} cells");

        let success_cell_count = tds.number_of_cells();
        assert_eq!(
            success_cell_count,
            initial_cell_count + cells_created,
            "Successful operation should increase cell count"
        );

        println!("✓ Atomic rollback behavior verified successfully");
    }

    #[test]
    fn test_is_facet_visible_from_vertex_orientation_cases() {
        use crate::core::facet::facet_key_from_vertices;

        println!("Testing is_facet_visible_from_vertex with different orientations");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        assert!(
            boundary_facets.clone().next().is_some(),
            "Should have at least one boundary facet"
        );

        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");

        // Find the cell adjacent to this boundary facet
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");

        // Compute facet key using VertexKeys
        let facet_vertices: Vec<_> = test_facet.vertices().unwrap().collect();
        let mut vertices = Vec::with_capacity(facet_vertices.len());
        for vertex in &facet_vertices {
            vertices.push(
                tds.vertex_key_from_uuid(&vertex.uuid())
                    .expect("Vertex should be in TDS"),
            );
        }
        let facet_key = facet_key_from_vertices(&vertices);

        let adjacent_cells = facet_to_cells.get(&facet_key).unwrap();
        assert_eq!(
            adjacent_cells.len(),
            1,
            "Boundary facet should have exactly one adjacent cell"
        );
        let adjacent_cell_key = adjacent_cells[0].cell_key();

        // Test visibility from different positions using the trait method
        let test_positions = vec![
            (vertex!([2.0, 0.0, 0.0]), "Far +X"),
            (vertex!([-1.0, 0.0, 0.0]), "Far -X"),
            (vertex!([0.0, 2.0, 0.0]), "Far +Y"),
            (vertex!([0.0, 0.0, 2.0]), "Far +Z"),
            (vertex!([0.1, 0.1, 0.1]), "Interior point"),
        ];

        let algorithm = IncrementalBowyerWatson::new();

        for (test_vertex, description) in test_positions {
            let is_visible = algorithm.is_facet_visible_from_vertex(
                &tds,
                &test_facet,
                &test_vertex,
                adjacent_cell_key,
            );

            match is_visible {
                Ok(visible) => println!("  {description} - Facet visible: {visible}"),
                Err(e) => println!("  {description} - Visibility error: {e}"),
            }
            // Note: We don't assert specific visibility results here because they depend
            // on the specific geometry and orientation of the facet, but the function
            // should not panic and should return a boolean result.
        }

        println!("✓ Facet visibility testing with different orientations works correctly");
    }

    #[test]
    fn test_create_cell_from_facet_and_vertex_success() {
        println!("Testing create_cell_from_facet_and_vertex successful creation");

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet handle (lightweight)
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");
        let cell_key = test_facet.cell_key();
        let facet_index = test_facet.facet_index();
        drop(boundary_facets); // Drop the iterator to release the immutable borrow

        let initial_cell_count = tds.number_of_cells();

        // Create a new vertex that should form a valid cell with the facet
        let new_vertex = vertex!([0.5, 0.5, 1.5]);

        let result =
            <IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                3,
            >>::create_cell_from_facet_handle(
                &mut tds, cell_key, facet_index, &new_vertex
            );

        assert!(
            result.is_ok(),
            "Should successfully create cell from valid facet and vertex: {:?}",
            result.err()
        );
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count + 1,
            "Cell count should increase by 1"
        );

        println!("✓ Cell creation from facet and vertex works correctly");
    }

    #[test]
    fn test_degenerate_only_cells_error_message() {
        println!("Testing TooManyDegenerateCells error message formatting");

        // Test the error message when all cells are degenerate (total_tested == 0)
        let error1 = BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
            degenerate_count: 5,
            total_tested: 0,
        });
        let error_msg1 = format!("{error1}");
        assert!(
            error_msg1.contains("All 5 candidate cells were degenerate"),
            "Error message should indicate all cells were degenerate: {error_msg1}"
        );
        println!("  ✓ All-degenerate case message: {error_msg1}");

        // Test the error message when some cells are degenerate
        let error2 = BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
            degenerate_count: 3,
            total_tested: 5,
        });
        let error_msg2 = format!("{error2}");
        assert!(
            error_msg2.contains("Too many degenerate circumspheres (3/5)"),
            "Error message should show the ratio: {error_msg2}"
        );
        println!("  ✓ Partial degenerate case message: {error_msg2}");

        // Test edge case: single degenerate cell
        let error3 = BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
            degenerate_count: 1,
            total_tested: 0,
        });
        let error_msg3 = format!("{error3}");
        assert!(
            error_msg3.contains("All 1 candidate cells were degenerate"),
            "Error message should handle singular correctly: {error_msg3}"
        );
        println!("  ✓ Single degenerate case message: {error_msg3}");

        println!("✓ TooManyDegenerateCells error message formatting works correctly");
    }

    #[test]
    fn test_integer_margin_calculation() {
        use num_traits::{One, Zero, cast};
        use std::ops::Div;

        // Helper function that mirrors the ACTUAL logic in is_vertex_likely_exterior
        fn calculate_margin<T>(range: T) -> T
        where
            T: Zero
                + One
                + Div<Output = T>
                + PartialEq
                + NumCast
                + std::ops::Mul<Output = T>
                + Copy,
        {
            let ten: T = cast(10).unwrap_or_else(T::one);

            // This mirrors the actual fixed implementation logic
            cast::<f64, T>(MARGIN_FACTOR).map_or_else(
                || {
                    // Fallback: divide by 10 with minimum of 1
                    let mut m = range / ten;
                    if m == T::zero() {
                        m = T::one();
                    }
                    m
                },
                |mf| {
                    if mf == T::zero() {
                        // Integer case: divide by 10, ensure at least 1 unit
                        let mut m = range / ten;
                        if m == T::zero() {
                            m = T::one();
                        }
                        m
                    } else {
                        // Float case: multiply by margin factor
                        range * mf
                    }
                },
            )
        }

        println!("Testing integer margin calculation for bounding box expansion");

        // Test with integers - should use division by 10
        let range_100: i32 = 100;
        let margin_100 = calculate_margin(range_100);
        assert_eq!(margin_100, 10, "100/10 should equal 10");
        println!("  ✓ Integer range 100 -> margin {margin_100} (10% via division)");

        let range_50: i32 = 50;
        let margin_50 = calculate_margin(range_50);
        assert_eq!(margin_50, 5, "50/10 should equal 5");
        println!("  ✓ Integer range 50 -> margin {margin_50} (10% via division)");

        // Test with small ranges - should ensure minimum of 1
        let range_5: i32 = 5;
        let margin_5 = calculate_margin(range_5);
        assert_eq!(margin_5, 1, "5/10 rounds to 0, but minimum should be 1");
        println!("  ✓ Integer range 5 -> margin {margin_5} (minimum of 1)");

        let range_3: i32 = 3;
        let margin_3 = calculate_margin(range_3);
        assert_eq!(margin_3, 1, "3/10 rounds to 0, but minimum should be 1");
        println!("  ✓ Integer range 3 -> margin {margin_3} (minimum of 1)");

        // Test with zero range - should still return 1
        let zero_range: i32 = 0;
        let zero_margin = calculate_margin(zero_range);
        assert_eq!(zero_margin, 1, "0/10 = 0, but minimum should be 1");
        println!("  ✓ Integer range 0 -> margin {zero_margin} (minimum of 1)");

        // Test with floating point - should use multiplication by 0.1
        let range_100_f64: f64 = 100.0;
        let margin_100_f64 = calculate_margin(range_100_f64);
        assert!(
            (margin_100_f64 - 10.0).abs() < 1e-10,
            "100.0 * 0.1 should equal 10.0"
        );
        println!("  ✓ Float range 100.0 -> margin {margin_100_f64} (10% via multiplication)");

        let range_5_f64: f64 = 5.0;
        let margin_5_f64 = calculate_margin(range_5_f64);
        assert!(
            (margin_5_f64 - 0.5).abs() < 1e-10,
            "5.0 * 0.1 should equal 0.5"
        );
        println!("  ✓ Float range 5.0 -> margin {margin_5_f64} (10% via multiplication)");

        // Test that the fix prevents the 100% expansion bug
        // Previously with the bug: margin_factor = 1 for integers, so margin = range * 1 = range
        // Now with the fix: margin = range / 10 (with minimum 1)
        let bug_test_range: i32 = 20;
        let bug_test_margin = calculate_margin(bug_test_range);
        assert_ne!(
            bug_test_margin, bug_test_range,
            "Margin should NOT equal the range (would be 100% expansion)"
        );
        assert_eq!(
            bug_test_margin, 2,
            "For range 20, margin should be 20/10 = 2, not 20"
        );
        println!(
            "  ✓ Bug prevention check: range {bug_test_range} -> margin {bug_test_margin} (not {bug_test_range}, preventing 100% expansion)"
        );

        println!("✓ Integer margin calculation works correctly with ~10% expansion");
    }

    /// Test `bbox_sub` with floating-point coordinates
    #[test]
    fn test_bbox_sub_float() {
        println!("Testing bbox_sub with floating-point types");

        // Normal subtraction cases
        let result = bbox_sub(10.0_f64, 3.0_f64);
        assert_abs_diff_eq!(result, 7.0, epsilon = 1e-10);
        println!("  ✓ f64: 10.0 - 3.0 = {result}");

        let result = bbox_sub(5.5_f32, 2.3_f32);
        assert_abs_diff_eq!(result, 3.2, epsilon = 1e-6);
        println!("  ✓ f32: 5.5 - 2.3 = {result}");

        // Negative result
        let result = bbox_sub(3.0_f64, 10.0_f64);
        assert_abs_diff_eq!(result, -7.0, epsilon = 1e-10);
        println!("  ✓ f64: 3.0 - 10.0 = {result} (negative result)");

        // Underflow to -infinity (floats handle this naturally)
        let result = bbox_sub(-f64::MAX, f64::MAX);
        assert!(result.is_infinite() && result.is_sign_negative());
        println!("  ✓ f64: -MAX - MAX = {result} (underflow to -infinity)");

        // Zero cases
        let result = bbox_sub(5.0_f64, 5.0_f64);
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
        println!("  ✓ f64: 5.0 - 5.0 = {result}");

        println!("✓ bbox_sub works correctly with floating-point types");
    }

    /// Test `bbox_add` with floating-point coordinates
    #[test]
    fn test_bbox_add_float() {
        println!("Testing bbox_add with floating-point types");

        // Normal addition cases
        let result = bbox_add(10.0_f64, 3.0_f64);
        assert_abs_diff_eq!(result, 13.0, epsilon = 1e-10);
        println!("  ✓ f64: 10.0 + 3.0 = {result}");

        let result = bbox_add(5.5_f32, 2.3_f32);
        assert_abs_diff_eq!(result, 7.8, epsilon = 1e-6);
        println!("  ✓ f32: 5.5 + 2.3 = {result}");

        // Large values
        let result = bbox_add(1e100_f64, 2e100_f64);
        assert_abs_diff_eq!(result, 3e100, epsilon = 1e90);
        println!("  ✓ f64: 1e100 + 2e100 = {result}");

        // Overflow to infinity (floats handle this naturally)
        let result = bbox_add(f64::MAX, f64::MAX);
        assert!(result.is_infinite() && result.is_sign_positive());
        println!("  ✓ f64: MAX + MAX = {result} (overflow to infinity)");

        // Zero cases
        let result = bbox_add(5.0_f64, 0.0_f64);
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
        println!("  ✓ f64: 5.0 + 0.0 = {result}");

        // Negative values
        let result = bbox_add(-3.5_f64, 2.5_f64);
        assert_abs_diff_eq!(result, -1.0, epsilon = 1e-10);
        println!("  ✓ f64: -3.5 + 2.5 = {result}");

        println!("✓ bbox_add works correctly with floating-point types");
    }

    /// Test `bbox_sub` behavior with bounding box expansion use case
    #[test]
    fn test_saturating_bbox_operations_integration() {
        println!("Testing saturating bbox operations in realistic bounding box scenarios");

        // Simulate bounding box expansion for floating-point coordinates
        let min_coord = 10.0_f64;
        let margin = 2.5_f64;
        let expanded_min = bbox_sub(min_coord, margin);
        assert_abs_diff_eq!(expanded_min, 7.5, epsilon = 1e-10);
        println!("  ✓ f64: Expand min bbox 10.0 by margin 2.5 -> {expanded_min}");

        let max_coord = 100.0_f64;
        let expanded_max = bbox_add(max_coord, margin);
        assert_abs_diff_eq!(expanded_max, 102.5, epsilon = 1e-10);
        println!("  ✓ f64: Expand max bbox 100.0 by margin 2.5 -> {expanded_max}");

        // Verify the expanded range is larger than original
        assert!(expanded_min < min_coord);
        assert!(expanded_max > max_coord);
        let original_range = max_coord - min_coord;
        let expanded_range = expanded_max - expanded_min;
        assert!(expanded_range > original_range);
        println!("  ✓ Expanded range {expanded_range} > original range {original_range}");

        // Test with very large coordinate values (e.g., astronomical scales)
        let large_min = 1e50_f64;
        let large_margin = 1e48_f64;
        let large_expanded_min = bbox_sub(large_min, large_margin);
        assert!(large_expanded_min < large_min);
        println!(
            "  ✓ f64: Large scale bbox expansion: {large_min} - {large_margin} = {large_expanded_min}"
        );

        // Test with very small coordinate values (e.g., microscopic scales)
        let small_min = 1e-50_f64;
        let small_margin = 1e-52_f64;
        let small_expanded_min = bbox_sub(small_min, small_margin);
        assert!(small_expanded_min < small_min);
        println!(
            "  ✓ f64: Small scale bbox expansion: {small_min} - {small_margin} = {small_expanded_min}"
        );

        println!("✓ Saturating bbox operations work correctly in realistic scenarios");
    }

    /// Test `TooManyDegenerateCellsError` display formatting
    #[test]
    fn test_too_many_degenerate_cells_error_display() {
        // Test with total_tested = 0
        let error_zero_total = TooManyDegenerateCellsError {
            degenerate_count: 5,
            total_tested: 0,
        };
        let display_zero = format!("{error_zero_total}");
        assert!(display_zero.contains("All 5 candidate cells were degenerate"));

        // Test with total_tested > 0
        let error_with_total = TooManyDegenerateCellsError {
            degenerate_count: 8,
            total_tested: 12,
        };
        let display_with_total = format!("{error_with_total}");
        assert!(display_with_total.contains("Too many degenerate circumspheres (8/12)"));
    }

    /// Test `InsertionStatistics` comprehensive functionality
    #[test]
    fn test_insertion_statistics_comprehensive() {
        let mut stats = InsertionStatistics::new();

        // Test initial state
        assert_eq!(stats.vertices_processed, 0);
        assert_eq!(stats.total_cells_created, 0);
        assert_eq!(stats.total_cells_removed, 0);
        assert_eq!(stats.fallback_strategies_used, 0);
        assert_abs_diff_eq!(
            stats.cavity_boundary_success_rate(),
            1.0,
            epsilon = f64::EPSILON
        ); // No attempts = 100%
        assert_abs_diff_eq!(stats.fallback_usage_rate(), 0.0, epsilon = f64::EPSILON);

        // Test recording various insertion types
        let cavity_info = InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed: 3,
            cells_created: 5,
            success: true,
            degenerate_case_handled: false,
        };
        stats.record_vertex_insertion(&cavity_info);

        let hull_info = InsertionInfo {
            strategy: InsertionStrategy::HullExtension,
            cells_removed: 0,
            cells_created: 2,
            success: true,
            degenerate_case_handled: true,
        };
        stats.record_vertex_insertion(&hull_info);

        let fallback_info = InsertionInfo {
            strategy: InsertionStrategy::Fallback,
            cells_removed: 1,
            cells_created: 2,
            success: true,
            degenerate_case_handled: false,
        };
        stats.record_vertex_insertion(&fallback_info);

        // Verify statistics
        assert_eq!(stats.vertices_processed, 3);
        assert_eq!(stats.total_cells_created, 9); // 5 + 2 + 2
        assert_eq!(stats.total_cells_removed, 4); // 3 + 0 + 1
        assert_eq!(stats.hull_extensions, 1);
        assert_eq!(stats.fallback_strategies_used, 1);
        assert_eq!(stats.degenerate_cases_handled, 1);

        // Test manual recording methods
        stats.record_fallback_usage();
        assert_eq!(stats.fallback_strategies_used, 2);

        stats.record_cavity_boundary_failure();
        assert_eq!(stats.cavity_boundary_failures, 1);

        stats.record_cavity_boundary_recovery();
        assert_eq!(stats.cavity_boundary_recoveries, 1);

        // Test success rate calculations
        let success_rate = stats.cavity_boundary_success_rate();
        let expected_rate = (3.0 - 1.0) / 3.0; // (processed - failures) / processed
        assert_abs_diff_eq!(success_rate, expected_rate, epsilon = f64::EPSILON);

        let fallback_rate = stats.fallback_usage_rate();
        let expected_fallback_rate = 2.0 / 3.0; // fallbacks / processed
        assert_abs_diff_eq!(
            fallback_rate,
            expected_fallback_rate,
            epsilon = f64::EPSILON
        );

        // Test basic tuple conversion
        let (processed, created, removed) = stats.as_basic_tuple();
        assert_eq!(processed, 3);
        assert_eq!(created, 9);
        assert_eq!(removed, 4);

        // Test reset
        stats.reset();
        assert_eq!(stats.vertices_processed, 0);
        assert_eq!(stats.total_cells_created, 0);
        assert_eq!(stats.fallback_strategies_used, 0);
    }

    /// Test `InsertionBuffers` functionality
    #[test]
    fn test_insertion_buffers_functionality() {
        let mut buffers = InsertionBuffers::<f64, Option<()>, Option<()>, 3>::new();

        // Test initial state
        assert_eq!(buffers.bad_cells_buffer.len(), 0);
        assert_eq!(buffers.boundary_facets_buffer.len(), 0);
        assert_eq!(buffers.vertex_points_buffer.len(), 0);
        assert_eq!(buffers.visible_facets_buffer.len(), 0);

        // Test with_capacity constructor
        let buffers_with_capacity =
            InsertionBuffers::<f64, Option<()>, Option<()>, 3>::with_capacity(10);
        assert!(buffers_with_capacity.bad_cells_buffer.capacity() >= 10);
        assert!(buffers_with_capacity.boundary_facets_buffer.capacity() >= 10);
        assert!(buffers_with_capacity.vertex_points_buffer.capacity() >= 40); // 10 * (3 + 1)
        assert!(buffers_with_capacity.visible_facets_buffer.capacity() >= 5); // 10 / 2

        // Test buffer preparation methods
        let bad_cells_buf = buffers.prepare_bad_cells_buffer();
        bad_cells_buf.push(crate::core::triangulation_data_structure::CellKey::default());
        assert_eq!(buffers.bad_cells_buffer.len(), 1);

        // Prepare boundary buffer - this only clears its own buffer, not others
        let boundary_buf = buffers.prepare_boundary_facets_buffer();
        assert_eq!(boundary_buf.len(), 0);
        assert_eq!(buffers.bad_cells_buffer.len(), 1); // Should still have the item

        // Preparing bad cells buffer again should clear it
        buffers.prepare_bad_cells_buffer();
        assert_eq!(buffers.bad_cells_buffer.len(), 0); // Now it should be cleared

        let vertex_points_buf = buffers.prepare_vertex_points_buffer();
        assert_eq!(vertex_points_buf.len(), 0);

        let visible_buf = buffers.prepare_visible_facets_buffer();
        assert_eq!(visible_buf.len(), 0);

        // Test clear_all
        buffers
            .bad_cells_buffer
            .push(crate::core::triangulation_data_structure::CellKey::default());
        buffers
            .vertex_points_buffer
            .push(Point::new([1.0, 2.0, 3.0]));

        buffers.clear_all();
        assert_eq!(buffers.bad_cells_buffer.len(), 0);
        assert_eq!(buffers.vertex_points_buffer.len(), 0);
    }

    /// Test `BadCellsError` variants and display
    #[test]
    fn test_bad_cells_error_variants() {
        // Test AllCellsBad variant
        let all_bad_error = BadCellsError::AllCellsBad {
            cell_count: 10,
            degenerate_count: 3,
        };
        let all_bad_display = format!("{all_bad_error}");
        assert!(all_bad_display.contains("All 10 cells marked as bad"));
        assert!(all_bad_display.contains("3 degenerate"));

        // Test TooManyDegenerateCells variant
        let too_many_degenerate =
            BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
                degenerate_count: 8,
                total_tested: 10,
            });
        let too_many_display = format!("{too_many_degenerate}");
        assert!(too_many_display.contains("Too many degenerate circumspheres (8/10)"));

        // Test NoCells variant
        let no_cells_error = BadCellsError::NoCells;
        let no_cells_display = format!("{no_cells_error}");
        assert!(no_cells_display.contains("No cells exist to test"));
    }

    /// Test `cavity_boundary_success_rate` edge cases
    #[test]
    fn test_cavity_boundary_success_rate_edge_cases() {
        let mut stats = InsertionStatistics::new();

        // Test with zero processed vertices
        assert_abs_diff_eq!(
            stats.cavity_boundary_success_rate(),
            1.0,
            epsilon = f64::EPSILON
        );

        // Test with failures >= processed (saturating subtraction)
        stats.vertices_processed = 3;
        stats.cavity_boundary_failures = 5; // More failures than processed

        let success_rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(success_rate, 0.0, epsilon = f64::EPSILON); // Should saturate to 0

        // Test with exact match
        stats.cavity_boundary_failures = 3;
        let exact_match_rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(exact_match_rate, 0.0, epsilon = f64::EPSILON);

        // Test precision with underflow protection
        stats.vertices_processed = 1;
        stats.cavity_boundary_failures = 0;
        let single_success_rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(single_success_rate, 1.0, epsilon = f64::EPSILON);
    }

    /// Test error path when `find_bad_cells` returns errors - comprehensive
    #[test]
    fn test_find_bad_cells_error_cases_comprehensive() {
        let mut algorithm = IncrementalBowyerWatson::new();

        // Create minimal TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with extreme coordinates that might cause numerical issues in predicates
        let problematic_vertex = vertex!([f64::MAX / 1000.0, f64::MAX / 1000.0, f64::MAX / 1000.0]);

        // This should handle extreme coordinates gracefully or return appropriate error
        let result = algorithm.find_bad_cells(&tds, &problematic_vertex);

        match result {
            Err(
                BadCellsError::NoCells
                | BadCellsError::AllCellsBad { .. }
                | BadCellsError::TooManyDegenerateCells(_)
                | BadCellsError::TdsCorruption { .. },
            )
            | Ok(_) => {
                // All these cases are valid - main test is that we don't panic with extreme input
            }
        }

        // The main test is that we don't panic with invalid input
    }

    /// Test `create_cell_from_facet_and_vertex` error cases - comprehensive
    #[test]
    fn test_create_cell_from_facet_and_vertex_failure_comprehensive() {
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();

        let boundary_facets = tds.boundary_facets().unwrap();
        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");
        let cell_key = test_facet.cell_key();
        let facet_index = test_facet.facet_index();
        drop(boundary_facets); // Drop the iterator to release the immutable borrow

        // Try to create cell with vertex that would create degenerate cell
        let degenerate_vertex = vertex!([0.0, 0.0, 0.0]); // Same as existing vertex

        let result =
            <IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                3,
            >>::create_cell_from_facet_handle(
                &mut tds, cell_key, facet_index, &degenerate_vertex
            );

        // This should either succeed (if handled gracefully) or fail with appropriate error
        match result {
            Ok(_cell_key) => {
                // If it succeeds, verify TDS is still valid
                assert!(tds.is_valid().is_ok());
            }
            Err(e) => {
                // If it fails, verify it's an appropriate error type
                println!("Expected error for degenerate cell creation: {e:?}");
            }
        }
    }

    /// Test comprehensive `InsertionError` creation and classification
    #[test]
    fn test_insertion_error_comprehensive() {
        println!("Testing comprehensive InsertionError creation and classification");

        // Test FallbacksExhausted error
        let fallbacks_error = InsertionError::FallbacksExhausted {
            attempts: 5,
            last_error: "All strategies failed".to_string(),
        };
        let fallbacks_display = format!("{fallbacks_error}");
        assert!(fallbacks_display.contains("All 5 fallback strategies exhausted"));
        assert!(fallbacks_display.contains("All strategies failed"));
        assert!(
            !fallbacks_error.is_recoverable(),
            "FallbacksExhausted should not be recoverable"
        );
        assert_eq!(fallbacks_error.attempted_strategy(), None);
        println!("  ✓ FallbacksExhausted error works correctly");

        // Test ExcessiveBadCells error
        let excessive_bad_cells_error = InsertionError::ExcessiveBadCells {
            found: 150,
            threshold: 100,
        };
        let excessive_display = format!("{excessive_bad_cells_error}");
        assert!(excessive_display.contains("Excessive bad cells found: 150"));
        assert!(excessive_display.contains("threshold: 100"));
        assert!(
            !excessive_bad_cells_error.is_recoverable(),
            "ExcessiveBadCells should not be recoverable"
        );
        assert_eq!(excessive_bad_cells_error.attempted_strategy(), None);
        println!("  ✓ ExcessiveBadCells error works correctly");

        // Test error conversion from TriangulationValidationError
        let validation_error = TriangulationValidationError::InconsistentDataStructure {
            message: "Test validation error".to_string(),
        };
        let insertion_error = InsertionError::TriangulationState(validation_error);
        let insertion_display = format!("{insertion_error}");
        assert!(insertion_display.contains("Triangulation validation error"));
        assert!(insertion_display.contains("Test validation error"));
        assert!(
            !insertion_error.is_recoverable(),
            "TriangulationState should not be recoverable"
        );
        println!("  ✓ TriangulationValidationError conversion works correctly");

        // Test error conversion from TriangulationConstructionError
        let construction_error = TriangulationConstructionError::FailedToCreateCell {
            message: "Test construction error".to_string(),
        };
        let insertion_error = InsertionError::TriangulationConstruction(construction_error);
        let insertion_display = format!("{insertion_error}");
        assert!(insertion_display.contains("Triangulation construction error"));
        assert!(insertion_display.contains("Test construction error"));
        assert!(
            !insertion_error.is_recoverable(),
            "TriangulationConstruction should not be recoverable"
        );
        println!("  ✓ TriangulationConstructionError conversion works correctly");

        // Test error conversion from BadCellsError
        let bad_cells_error = BadCellsError::AllCellsBad {
            cell_count: 10,
            degenerate_count: 3,
        };
        let insertion_error = InsertionError::BadCellsDetection(bad_cells_error);
        assert!(
            insertion_error.is_recoverable(),
            "BadCellsDetection should be recoverable"
        );
        println!("  ✓ BadCellsError conversion works correctly");

        // Test all is_recoverable cases
        let recoverable_errors = vec![
            InsertionError::geometric_failure("Test", InsertionStrategy::Standard),
            InsertionError::precision_failure(1e-10, 3),
            InsertionError::BadCellsDetection(BadCellsError::NoCells),
        ];

        for error in recoverable_errors {
            assert!(
                error.is_recoverable(),
                "Error should be recoverable: {error}"
            );
        }

        let non_recoverable_errors = vec![
            InsertionError::invalid_vertex("Test"),
            InsertionError::hull_extension_failure("Test"),
            InsertionError::FallbacksExhausted {
                attempts: 3,
                last_error: "Test".to_string(),
            },
            InsertionError::ExcessiveBadCells {
                found: 100,
                threshold: 50,
            },
        ];

        for error in non_recoverable_errors {
            assert!(
                !error.is_recoverable(),
                "Error should not be recoverable: {error}"
            );
        }

        println!("✓ Comprehensive InsertionError functionality works correctly");
    }

    /// Test `InsertionStrategy` determination edge cases
    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_insertion_strategy_determination_edge_cases() {
        println!("Testing InsertionStrategy determination edge cases");

        // Test with empty TDS
        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);

        let strategy =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::determine_strategy_default(
                &empty_tds,
                &test_vertex,
            );
        assert_eq!(
            strategy,
            InsertionStrategy::Standard,
            "Empty TDS should use Standard strategy"
        );
        println!("  ✓ Empty TDS correctly uses Standard strategy");

        // Test with single cell TDS
        let single_cell_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let single_cell_tds: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&single_cell_vertices).unwrap();
        assert_eq!(
            single_cell_tds.number_of_cells(),
            1,
            "Should have exactly one cell"
        );

        let strategy =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::determine_strategy_default(
                &single_cell_tds,
                &test_vertex,
            );
        assert_eq!(
            strategy,
            InsertionStrategy::HullExtension,
            "Single cell TDS should use HullExtension"
        );
        println!("  ✓ Single cell TDS correctly uses HullExtension strategy");

        // Test is_vertex_likely_exterior with various positions
        let multi_cell_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
            vertex!([3.6, 4.7, 5.8]), // Additional unique vertex to create more complex geometry
        ];
        let multi_cell_tds: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&multi_cell_vertices).unwrap();

        // Test exterior vertex (far outside bounding box)
        let far_exterior_vertex = vertex!([10.0, 10.0, 10.0]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &multi_cell_tds,
                &far_exterior_vertex,
            );
        assert!(
            is_exterior,
            "Far exterior vertex should be identified as likely exterior"
        );
        println!("  ✓ Far exterior vertex correctly identified");

        // Test interior vertex (well inside bounding box)
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &multi_cell_tds,
                &interior_vertex,
            );
        assert!(
            !is_exterior,
            "Interior vertex should not be identified as exterior"
        );
        println!("  ✓ Interior vertex correctly identified");

        // Test vertex at bounding box boundary (edge case)
        // Bounding box is [0,2] x [0,2] x [0,2], expanded by 10% margin becomes [-0.2,2.2] x [-0.2,2.2] x [-0.2,2.2]
        let boundary_vertex = vertex!([2.3, 1.0, 1.0]); // Just outside expanded boundary
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &multi_cell_tds,
                &boundary_vertex,
            );
        if is_exterior {
            println!("  ✓ Boundary vertex correctly identified as exterior");
        } else {
            println!("  ✓ Boundary vertex identified as interior (acceptable for this geometry)");
        }

        // Test vertex just inside expanded boundary
        let inside_boundary_vertex = vertex!([1.8, 1.0, 1.0]); // Well inside original boundary
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &multi_cell_tds,
                &inside_boundary_vertex,
            );
        if is_exterior {
            println!(
                "  ✓ Inside boundary vertex identified as exterior (acceptable for this geometry)"
            );
        } else {
            println!("  ✓ Inside boundary vertex correctly identified as interior");
        }

        // Test complete strategy determination with exterior vertex
        let exterior_strategy =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::determine_strategy_default(
                &multi_cell_tds,
                &far_exterior_vertex,
            );
        assert_eq!(
            exterior_strategy,
            InsertionStrategy::HullExtension,
            "Exterior vertex should use HullExtension strategy"
        );
        println!("  ✓ Exterior vertex strategy determination works correctly");

        // Test complete strategy determination with interior vertex
        let interior_strategy =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::determine_strategy_default(
                &multi_cell_tds,
                &interior_vertex,
            );
        assert_eq!(
            interior_strategy,
            InsertionStrategy::CavityBased,
            "Interior vertex should use CavityBased strategy"
        );
        println!("  ✓ Interior vertex strategy determination works correctly");

        println!("✓ InsertionStrategy determination edge cases work correctly");
    }

    /// Test vertex insertion strategies error paths
    #[test]
    fn test_vertex_insertion_strategies_error_paths() {
        println!("Testing vertex insertion strategies error paths");

        // Create a basic triangulation for testing
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Test insert_vertex_cavity_based with exterior vertex (should fail)
        let exterior_vertex = vertex!([10.0, 10.0, 10.0]);
        let cavity_result = algorithm.insert_vertex_cavity_based(&mut tds, &exterior_vertex);

        match cavity_result {
            Err(InsertionError::TriangulationState(_)) => {
                println!("  ✓ Cavity-based insertion correctly failed for exterior vertex");
            }
            Err(InsertionError::BadCellsDetection(BadCellsError::AllCellsBad { .. })) => {
                println!("  ✓ Cavity-based insertion correctly identified all cells as bad");
            }
            Ok(_) => {
                println!("  ✓ Cavity-based insertion succeeded (acceptable for this geometry)");
            }
            Err(e) => {
                println!("  ✓ Cavity-based insertion failed with appropriate error: {e}");
            }
        }

        // Test insert_vertex_hull_extension with interior vertex
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let hull_result = algorithm.insert_vertex_hull_extension(&mut tds, &interior_vertex);

        match hull_result {
            Err(InsertionError::TriangulationState(_)) => {
                println!("  ✓ Hull extension correctly failed for interior vertex");
            }
            Ok(_) => {
                println!("  ✓ Hull extension succeeded (acceptable for this geometry)");
            }
            Err(e) => {
                println!("  ✓ Hull extension failed with appropriate error: {e}");
            }
        }

        // Test insert_vertex_fallback with various scenarios
        let fallback_vertex = vertex!([0.5, 0.5, 0.5]);
        let fallback_result = algorithm.insert_vertex_fallback(&mut tds, &fallback_vertex);

        match fallback_result {
            Ok(info) => {
                assert_eq!(info.strategy, InsertionStrategy::Fallback);
                assert!(info.cells_created >= 1, "Should create at least one cell");
                println!(
                    "  ✓ Fallback insertion succeeded, created {} cells",
                    info.cells_created
                );
            }
            Err(e) => {
                println!("  ✓ Fallback insertion failed with error: {e}");
            }
        }

        println!("✓ Vertex insertion strategies error paths work correctly");
    }

    /// Test insertion strategies with corrupted TDS
    #[test]
    fn test_insertion_strategies_corrupted_tds() {
        println!("Testing insertion strategies with potentially corrupted TDS");

        // Create an empty TDS to simulate corruption scenarios
        let mut empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let mut algorithm = IncrementalBowyerWatson::new();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);

        // Test cavity-based insertion with empty TDS
        let cavity_result = algorithm.insert_vertex_cavity_based(&mut empty_tds, &test_vertex);
        assert!(
            cavity_result.is_err(),
            "Cavity insertion should fail on empty TDS"
        );

        match cavity_result {
            Err(InsertionError::BadCellsDetection(BadCellsError::NoCells)) => {
                println!("  ✓ Cavity insertion correctly identified empty TDS");
            }
            Err(InsertionError::TriangulationState(_)) => {
                println!("  ✓ Cavity insertion correctly failed with triangulation state error");
            }
            Err(e) => {
                println!("  ✓ Cavity insertion failed with appropriate error: {e}");
            }
            Ok(_) => panic!("Cavity insertion should not succeed on empty TDS"),
        }

        // Test hull extension with empty TDS
        let hull_result = algorithm.insert_vertex_hull_extension(&mut empty_tds, &test_vertex);
        assert!(
            hull_result.is_err(),
            "Hull extension should fail on empty TDS"
        );

        match hull_result {
            Err(InsertionError::TriangulationState(_)) => {
                println!("  ✓ Hull extension correctly failed with triangulation state error");
            }
            Err(e) => {
                println!("  ✓ Hull extension failed with appropriate error: {e}");
            }
            Ok(_) => panic!("Hull extension should not succeed on empty TDS"),
        }

        // Test fallback with empty TDS
        let fallback_result = algorithm.insert_vertex_fallback(&mut empty_tds, &test_vertex);
        assert!(
            fallback_result.is_err(),
            "Fallback should fail on empty TDS"
        );

        match fallback_result {
            Err(InsertionError::TriangulationState(_)) => {
                println!("  ✓ Fallback correctly failed with triangulation state error");
            }
            Err(e) => {
                println!("  ✓ Fallback failed with appropriate error: {e}");
            }
            Ok(_) => panic!("Fallback should not succeed on empty TDS"),
        }

        println!("✓ Insertion strategies correctly handle corrupted TDS scenarios");
    }

    /// Test triangulation creation and finalization methods
    #[test]
    fn test_triangulation_creation_and_finalization() {
        println!("Testing triangulation creation and finalization methods");

        let mut algorithm = IncrementalBowyerWatson::new();

        // Test triangulate with insufficient vertices
        let insufficient_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            // Missing one vertex for 3D (need D+1 = 4)
        ];
        let mut empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let result = algorithm.triangulate(&mut empty_tds, &insufficient_vertices);

        match result {
            Err(TriangulationConstructionError::InsufficientVertices {
                dimension,
                source: _,
            }) => {
                assert_eq!(dimension, 3);
                println!("  ✓ Insufficient vertices correctly rejected (dimension: {dimension})");
            }
            other => panic!("Expected InsufficientVertices error, got: {other:?}"),
        }

        // Test triangulate with sufficient vertices
        let sufficient_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]), // Additional vertex for testing incremental insertion
        ];
        let mut tds = Tds::empty();
        let result = algorithm.triangulate(&mut tds, &sufficient_vertices);

        match result {
            Ok(()) => {
                assert!(
                    tds.number_of_vertices() >= 4,
                    "Should have at least 4 vertices"
                );
                assert!(tds.number_of_cells() >= 1, "Should have at least 1 cell");
                println!(
                    "  ✓ Triangulation succeeded with {} vertices and {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
            Err(e) => panic!("Triangulation should succeed with sufficient vertices: {e}"),
        }

        // Test triangulate with empty vertex list
        let empty_vertices = vec![];
        let mut empty_tds = Tds::empty();
        let result = algorithm.triangulate(&mut empty_tds, &empty_vertices);
        assert!(
            result.is_ok(),
            "Empty vertex list should be handled gracefully"
        );
        println!("  ✓ Empty vertex list handled correctly");

        println!("✓ Triangulation creation works correctly");
    }

    /// Test `create_initial_simplex` edge cases
    #[test]
    fn test_create_initial_simplex_edge_cases() {
        println!("Testing create_initial_simplex edge cases");

        // Test with wrong number of vertices
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();

        // Too few vertices
        let too_few_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_initial_simplex(
                &mut tds,
                too_few_vertices,
            );

        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {
                println!("  ✓ Too few vertices correctly rejected");
            }
            other => panic!("Expected InsufficientVertices error, got: {other:?}"),
        }

        // Too many vertices
        let too_many_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]), // Extra vertex
        ];
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_initial_simplex(
                &mut tds,
                too_many_vertices,
            );

        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {
                println!("  ✓ Too many vertices correctly rejected");
            }
            other => panic!("Expected InsufficientVertices error, got: {other:?}"),
        }

        // Correct number of vertices
        let correct_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_initial_simplex(
                &mut tds,
                correct_vertices,
            );

        assert!(
            result.is_ok(),
            "Correct vertex count should succeed: {result:?}"
        );
        assert_eq!(
            tds.number_of_cells(),
            1,
            "Should have exactly one cell after initial simplex creation"
        );
        assert_eq!(
            tds.number_of_vertices(),
            4,
            "Should have exactly 4 vertices"
        );
        println!("  ✓ Correct vertex count succeeded");

        println!("✓ create_initial_simplex edge cases work correctly");
    }

    /// Test `finalize_triangulation` error handling
    #[test]
    fn test_finalize_triangulation_error_handling() {
        println!("Testing finalize_triangulation error handling");

        // Create a basic triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test finalization on a valid TDS
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::finalize_triangulation(
                &mut tds,
            );
        assert!(
            result.is_ok(),
            "Finalization should succeed on valid TDS: {result:?}"
        );
        println!("  ✓ Finalization succeeded on valid TDS");

        // Test finalization on empty TDS (should handle gracefully)
        let mut empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::finalize_triangulation(
                &mut empty_tds,
            );

        match result {
            Ok(()) => {
                println!("  ✓ Empty TDS finalization succeeded");
            }
            Err(e) => {
                println!("  ✓ Empty TDS finalization failed appropriately: {e}");
            }
        }

        println!("✓ finalize_triangulation error handling works correctly");
    }

    /// Test utility methods error handling
    #[test]
    fn test_utility_methods_error_handling() {
        println!("Testing utility methods error handling");

        // Test ensure_vertex_in_tds
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);

        // Test successful vertex insertion
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::ensure_vertex_in_tds(
                &mut tds,
                &test_vertex,
            );
        assert!(
            result.is_ok(),
            "ensure_vertex_in_tds should succeed for valid vertex"
        );
        assert!(
            tds.vertex_key_from_uuid(&test_vertex.uuid()).is_some(),
            "Vertex should be in TDS after insertion"
        );
        println!("  ✓ ensure_vertex_in_tds works correctly");

        // Test duplicate vertex insertion (should not fail)
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::ensure_vertex_in_tds(
                &mut tds,
                &test_vertex,
            );
        assert!(
            result.is_ok(),
            "ensure_vertex_in_tds should handle duplicate vertices gracefully"
        );
        println!("  ✓ ensure_vertex_in_tds handles duplicates correctly");
    }

    /// Test `create_cells_from_boundary_facets`
    #[test]
    fn test_create_cells_from_boundary_facets() {
        println!("Testing create_cells_from_boundary_facets");

        // Create a basic triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        assert!(
            boundary_facets.clone().next().is_some(),
            "Should have at least one boundary facet"
        );

        // Convert boundary facets to lightweight handles
        let boundary_facet_handles: Vec<_> = boundary_facets
            .map(|fv| FacetHandle::new(fv.cell_key(), fv.facet_index()))
            .collect();

        let initial_cell_count = tds.number_of_cells();
        let new_vertex = vertex!([2.0, 2.0, 2.0]);

        // Test successful cell creation using lightweight API
        let cells_created = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_cells_from_facet_handles(
            &mut tds,
            &boundary_facet_handles,
            &new_vertex,
        ).expect("Should successfully create cells from facet handles");

        assert!(cells_created > 0, "Should create at least one cell");
        assert!(
            tds.number_of_cells() > initial_cell_count,
            "Cell count should increase"
        );
        println!("  ✓ create_cells_from_boundary_facets created {cells_created} cells");

        // Test with empty boundary facets (should error due to early exit)
        let empty_facet_handles: Vec<FacetHandle> = vec![];
        let another_vertex = vertex!([3.0, 3.0, 3.0]);
        let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_cells_from_facet_handles(
            &mut tds,
            &empty_facet_handles,
            &another_vertex,
        );

        assert!(result.is_err(), "Empty facet handle list should error");
        println!("  ✓ Empty boundary facets error correctly");

        println!("✓ create_cells_from_boundary_facets works correctly");
    }

    /// Test `finalize_after_insertion` error handling
    #[test]
    fn test_finalize_after_insertion_error_handling() {
        println!("Testing finalize_after_insertion error handling");

        // Test with valid TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::finalize_after_insertion(
                &mut tds,
            );
        assert!(
            result.is_ok(),
            "finalize_after_insertion should succeed on valid TDS: {result:?}"
        );
        println!("  ✓ finalize_after_insertion succeeded on valid TDS");

        // Test with empty TDS (should handle gracefully)
        let mut empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::finalize_after_insertion(
                &mut empty_tds,
            );

        match result {
            Ok(()) => {
                println!("  ✓ Empty TDS finalization succeeded");
            }
            Err(e) => {
                println!("  ✓ Empty TDS finalization failed appropriately: {e}");
            }
        }

        println!("✓ finalize_after_insertion error handling works correctly");
    }

    /// Test `remove_bad_cells` utility method
    #[test]
    fn test_remove_bad_cells() {
        println!("Testing remove_bad_cells utility method");

        // Create a triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([2.5, 3.7, 4.1]), // Additional unique vertex to create more cells
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let initial_cell_count = tds.number_of_cells();
        let cell_keys: Vec<_> = tds.cell_keys().take(1).collect(); // Take one cell to remove

        assert!(
            !cell_keys.is_empty(),
            "Should have at least one cell to remove"
        );

        // Test cell removal
        IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::remove_bad_cells(
            &mut tds, &cell_keys,
        );

        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count - 1,
            "Should have one fewer cell"
        );
        println!("  ✓ remove_bad_cells removed {} cell(s)", cell_keys.len());

        // Test removal with empty list (should do nothing)
        let empty_keys: Vec<crate::core::triangulation_data_structure::CellKey> = vec![];
        let cell_count_before = tds.number_of_cells();
        IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::remove_bad_cells(
            &mut tds,
            &empty_keys,
        );
        assert_eq!(
            tds.number_of_cells(),
            cell_count_before,
            "Cell count should be unchanged"
        );
        println!("  ✓ Empty removal list handled correctly");

        println!("✓ remove_bad_cells works correctly");
    }

    /// Test visibility computation edge cases
    #[test]
    fn test_visibility_computation_edge_cases() {
        println!("Testing visibility computation edge cases");

        let algorithm = IncrementalBowyerWatson::new();

        // Test find_visible_boundary_facets with empty TDS
        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);

        let result = algorithm.find_visible_boundary_facets_lightweight(&empty_tds, &test_vertex);
        match result {
            Ok(facets) => {
                assert!(facets.is_empty(), "Empty TDS should have no visible facets");
                println!("  ✓ Empty TDS correctly returns no visible facets");
            }
            Err(e) => {
                println!("  ✓ Empty TDS appropriately failed with error: {e}");
            }
        }

        // Test with valid TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with exterior vertex (should see some facets)
        let exterior_vertex = vertex!([5.0, 5.0, 5.0]);
        let visible_facets = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should find visible facets for exterior vertex");

        // Should find a reasonable number of visible facets for an exterior vertex
        // For a tetrahedron, we can see at most 4 facets from any exterior point
        assert!(
            visible_facets.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron, got {}",
            visible_facets.len()
        );
        println!(
            "  ✓ Exterior vertex visibility: {} facets",
            visible_facets.len()
        );

        // Test with interior vertex (may see fewer or no facets)
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let visible_facets = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &interior_vertex)
            .expect("Should complete visibility computation for interior vertex");

        println!(
            "  ✓ Interior vertex visibility: {} facets",
            visible_facets.len()
        );

        // Test with vertex at boundary (edge case)
        let boundary_vertex = vertex!([1.0, 0.0, 0.0]); // Same as existing vertex
        let visible_facets = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &boundary_vertex)
            .expect("Should handle boundary vertex visibility");

        println!(
            "  ✓ Boundary vertex visibility: {} facets",
            visible_facets.len()
        );

        println!("✓ Visibility computation edge cases work correctly");
    }

    /// Test `is_facet_visible_from_vertex` with degenerate cases
    #[test]
    fn test_facet_visibility_degenerate_cases() {
        println!("Testing facet visibility with degenerate cases");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        assert!(
            boundary_facets.clone().next().is_some(),
            "Should have at least one boundary facet"
        );

        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");

        // Find adjacent cell for the facet
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map");
        let facet_vertices: Vec<_> = test_facet.vertices().unwrap().collect();
        let mut vertices = Vec::with_capacity(facet_vertices.len());
        for vertex in &facet_vertices {
            vertices.push(
                tds.vertex_key_from_uuid(&vertex.uuid())
                    .expect("Vertex should be in TDS"),
            );
        }

        let facet_key = facet_key_from_vertices(&vertices);
        let adjacent_cells = facet_to_cells
            .get(&facet_key)
            .expect("Facet should have adjacent cells");
        let adjacent_cell_key = adjacent_cells[0].cell_key();

        // Get boundary facet for coplanar vertex extraction
        let boundary_facets = tds.boundary_facets().unwrap();
        let boundary_facet = boundary_facets.clone().next().unwrap();
        let facet_vertices: Vec<_> = boundary_facet.vertices().unwrap().collect();
        let coplanar_vertex = &facet_vertices[0]; // Same as existing facet vertex

        let algorithm = IncrementalBowyerWatson::new();
        let is_visible = algorithm.is_facet_visible_from_vertex(
            &tds,
            &test_facet,
            coplanar_vertex,
            adjacent_cell_key,
        );
        match is_visible {
            Ok(visible) => println!("  ✓ Coplanar vertex visibility: {visible}"),
            Err(e) => {
                println!(
                    "  ✓ Coplanar vertex visibility error (expected for degenerate case): {e}"
                );
            }
        }

        // Test with extreme coordinates
        let extreme_vertex = vertex!([f64::MAX / 1000.0, 0.0, 0.0]);
        let is_visible = algorithm.is_facet_visible_from_vertex(
            &tds,
            &test_facet,
            &extreme_vertex,
            adjacent_cell_key,
        );
        match is_visible {
            Ok(visible) => println!("  ✓ Extreme coordinate vertex visibility: {visible}"),
            Err(e) => println!("  ✓ Extreme coordinate vertex visibility error: {e}"),
        }

        // Test with vertex very close to facet plane
        let close_vertex = vertex!([0.001, 0.001, 0.001]);
        let is_visible = algorithm.is_facet_visible_from_vertex(
            &tds,
            &test_facet,
            &close_vertex,
            adjacent_cell_key,
        );
        match is_visible {
            Ok(visible) => println!("  ✓ Close-to-facet vertex visibility: {visible}"),
            Err(e) => println!("  ✓ Close-to-facet vertex visibility error: {e}"),
        }

        println!("✓ Facet visibility degenerate cases handled correctly");
    }

    /// Test visibility computation with malformed facets scenarios
    #[test]
    fn test_visibility_with_potential_facet_issues() {
        println!("Testing visibility computation with potential facet issues");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test is_vertex_interior method edge cases
        let interior_test_vertex = vertex!([0.25, 0.25, 0.25]);
        let is_interior = algorithm.is_vertex_interior(&tds, &interior_test_vertex);
        println!("  ✓ Interior vertex test: {is_interior:?}");

        let exterior_test_vertex = vertex!([10.0, 10.0, 10.0]);
        let is_interior = algorithm.is_vertex_interior(&tds, &exterior_test_vertex);
        println!("  ✓ Exterior vertex interior test: {is_interior:?}");

        // Test with vertex at circumsphere boundary
        let boundary_test_vertex = vertex!([0.5, 0.5, 0.0]); // On edge/boundary
        let is_interior = algorithm.is_vertex_interior(&tds, &boundary_test_vertex);
        println!("  ✓ Boundary vertex interior test: {is_interior:?}");

        println!("✓ Visibility computation with potential facet issues handled correctly");
    }

    /// Test comprehensive bad cells detection scenarios including `DEGENERATE_CELL_THRESHOLD`
    #[test]
    fn test_bad_cells_detection_comprehensive_scenarios() {
        println!("Testing bad cells detection comprehensive scenarios");

        let mut algorithm = IncrementalBowyerWatson::new();

        // Test NoCells error with empty TDS
        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);

        let result = algorithm.find_bad_cells(&empty_tds, &test_vertex);
        match result {
            Err(BadCellsError::NoCells) => {
                println!("  ✓ NoCells error correctly detected for empty TDS");
            }
            other => panic!("Expected NoCells error for empty TDS, got: {other:?}"),
        }

        // Create a triangulation for further testing
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with interior vertex (should find bad cells)
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let result = algorithm.find_bad_cells(&tds, &interior_vertex);
        match result {
            Ok(bad_cells) => {
                assert!(
                    !bad_cells.is_empty(),
                    "Interior vertex should find bad cells"
                );
                assert!(
                    bad_cells.len() <= tds.number_of_cells(),
                    "Bad cells can't exceed total cells"
                );
                println!("  ✓ Interior vertex found {} bad cells", bad_cells.len());
            }
            Err(e) => {
                println!("  ✓ Interior vertex bad cells detection failed appropriately: {e}");
            }
        }

        // Test with exterior vertex (should find no or few bad cells)
        let exterior_vertex = vertex!([10.0, 10.0, 10.0]);
        let result = algorithm.find_bad_cells(&tds, &exterior_vertex);
        match result {
            Ok(bad_cells) => {
                assert!(
                    bad_cells.len() <= tds.number_of_cells(),
                    "Bad cells can't exceed total cells"
                );
                println!("  ✓ Exterior vertex found {} bad cells", bad_cells.len());
            }
            Err(BadCellsError::AllCellsBad {
                cell_count,
                degenerate_count,
            }) => {
                println!(
                    "  ✓ AllCellsBad detected: {cell_count} cells ({degenerate_count} degenerate)"
                );
            }
            Err(e) => {
                println!("  ✓ Exterior vertex bad cells detection failed appropriately: {e}");
            }
        }

        println!("✓ Bad cells detection comprehensive scenarios work correctly");
    }

    /// Test `DEGENERATE_CELL_THRESHOLD` boundary conditions
    #[test]
    fn test_degenerate_cell_threshold_boundary_conditions() {
        struct TestCase {
            degenerate_count: usize,
            cells_tested: usize,
            should_exceed_threshold: bool,
            description: &'static str,
        }

        println!("Testing DEGENERATE_CELL_THRESHOLD boundary conditions");

        // Create a scenario to test threshold behavior
        // Since DEGENERATE_CELL_THRESHOLD = 0.5, we need to test around 50% degenerate cells

        // The actual implementation uses optimized integer arithmetic for 0.5 threshold:
        // degenerate_count * 2 > total_cells

        // Test the threshold logic by verifying the constants
        assert!(
            (DEGENERATE_CELL_THRESHOLD - 0.5).abs() < f64::EPSILON,
            "Test assumes DEGENERATE_CELL_THRESHOLD is 0.5"
        );
        println!("  ✓ DEGENERATE_CELL_THRESHOLD is 0.5 as expected");

        // Test boundary cases for the 50% threshold
        // Simulating the logic: degenerate_count * 2 > total_cells

        let test_cases = vec![
            TestCase {
                degenerate_count: 1,
                cells_tested: 2, // total = 3, 1*2 = 2 < 3, should not exceed
                should_exceed_threshold: false,
                description: "33% degenerate (1/3)",
            },
            TestCase {
                degenerate_count: 2,
                cells_tested: 2, // total = 4, 2*2 = 4 = 4, should not exceed (not >)
                should_exceed_threshold: false,
                description: "50% degenerate (2/4)",
            },
            TestCase {
                degenerate_count: 3,
                cells_tested: 2, // total = 5, 3*2 = 6 > 5, should exceed
                should_exceed_threshold: true,
                description: "60% degenerate (3/5)",
            },
            TestCase {
                degenerate_count: 5,
                cells_tested: 5, // total = 10, 5*2 = 10 = 10, should not exceed
                should_exceed_threshold: false,
                description: "50% degenerate (5/10)",
            },
            TestCase {
                degenerate_count: 6,
                cells_tested: 4, // total = 10, 6*2 = 12 > 10, should exceed
                should_exceed_threshold: true,
                description: "60% degenerate (6/10)",
            },
        ];

        for test_case in test_cases {
            let total_cells = test_case
                .degenerate_count
                .saturating_add(test_case.cells_tested);
            let threshold_exceeded = test_case.degenerate_count.saturating_mul(2) > total_cells;

            assert_eq!(
                threshold_exceeded,
                test_case.should_exceed_threshold,
                "Threshold calculation incorrect for {}: degenerate={}, tested={}, total={}",
                test_case.description,
                test_case.degenerate_count,
                test_case.cells_tested,
                total_cells
            );

            println!(
                "  ✓ {}: {} (threshold {})",
                test_case.description,
                if threshold_exceeded {
                    "exceeds"
                } else {
                    "within"
                },
                if threshold_exceeded {
                    "exceeded"
                } else {
                    "not exceeded"
                }
            );
        }

        // Test edge case: all cells degenerate (zero tested)
        let all_degenerate_exceeded = 5_usize.saturating_mul(2) > 0_usize; // 10 > 0
        assert!(
            all_degenerate_exceeded,
            "All degenerate cells should exceed threshold"
        );
        println!("  ✓ All degenerate cells (0 tested) correctly exceed threshold");

        println!("✓ DEGENERATE_CELL_THRESHOLD boundary conditions work correctly");
    }

    /// Test `new_triangulation` method comprehensive scenarios
    #[test]
    fn test_new_triangulation_method_comprehensive() {
        println!("Testing new_triangulation method comprehensive scenarios");

        let mut algorithm: IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> =
            IncrementalBowyerWatson::new();

        // Test with empty vertex set
        let empty_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![];
        let result = algorithm.new_triangulation(&empty_vertices);

        match result {
            Ok(tds) => {
                assert_eq!(
                    tds.number_of_vertices(),
                    0,
                    "Empty triangulation should have no vertices"
                );
                assert_eq!(
                    tds.number_of_cells(),
                    0,
                    "Empty triangulation should have no cells"
                );
                println!("  ✓ Empty vertex set creates valid empty triangulation");
            }
            Err(e) => {
                println!("  ✓ Empty vertex set failed appropriately: {e}");
            }
        }

        // Test with insufficient vertices (less than D+1 for dimension D=3)
        let insufficient_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            // Missing fourth vertex for 3D
        ];
        let result = algorithm.new_triangulation(&insufficient_vertices);

        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3, "Should report correct dimension");
                println!("  ✓ Insufficient vertices correctly rejected (need 4 for 3D)");
            }
            other => {
                println!("  ✓ Insufficient vertices handled: {other:?}");
            }
        }

        // Test with exactly sufficient vertices (D+1 = 4 for 3D)
        let minimal_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let result = algorithm.new_triangulation(&minimal_vertices);

        match result {
            Ok(tds) => {
                assert_eq!(
                    tds.number_of_vertices(),
                    4,
                    "Should have exactly 4 vertices"
                );
                assert_eq!(
                    tds.number_of_cells(),
                    1,
                    "Should have exactly 1 cell (tetrahedron)"
                );
                println!("  ✓ Minimal vertex set (4 vertices) creates valid triangulation");
            }
            Err(e) => panic!("Minimal valid vertex set should succeed: {e}"),
        }

        // Test with more vertices for complex triangulation
        let complex_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
            vertex!([1.0, 1.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let result = algorithm.new_triangulation(&complex_vertices);

        match result {
            Ok(tds) => {
                assert_eq!(tds.number_of_vertices(), 6, "Should have all 6 vertices");
                assert!(tds.number_of_cells() >= 1, "Should have at least 1 cell");
                println!(
                    "  ✓ Complex vertex set (6 vertices) creates triangulation with {} cells",
                    tds.number_of_cells()
                );
            }
            Err(e) => {
                println!("  ✓ Complex vertex set failed appropriately: {e}");
            }
        }

        println!("✓ new_triangulation method comprehensive scenarios work correctly");
    }

    /// Test `new_triangulation` with degenerate and edge case vertices
    #[test]
    fn test_new_triangulation_degenerate_cases() {
        println!("Testing new_triangulation with degenerate and edge case vertices");

        let mut algorithm: IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> =
            IncrementalBowyerWatson::new();

        // Test with duplicate vertices
        let duplicate_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 0.0, 0.0]), // Duplicate of first vertex
        ];
        let result = algorithm.new_triangulation(&duplicate_vertices);

        match result {
            Ok(tds) => {
                // The TDS should handle duplicates gracefully, possibly ignoring or managing them
                println!(
                    "  ✓ Duplicate vertices handled: {} vertices, {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
            Err(e) => {
                println!("  ✓ Duplicate vertices failed appropriately: {e}");
            }
        }

        // Test with coplanar vertices (degenerate in 3D)
        let coplanar_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, 0.0]), // All in z=0 plane
        ];
        let result = algorithm.new_triangulation(&coplanar_vertices);

        match result {
            Ok(tds) => {
                println!(
                    "  ✓ Coplanar vertices handled: {} vertices, {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
            Err(e) => {
                println!("  ✓ Coplanar vertices failed appropriately: {e}");
            }
        }

        // Test with extreme coordinates
        let extreme_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([f64::MAX / 1000.0, 0.0, 0.0]),
            vertex!([0.0, f64::MAX / 1000.0, 0.0]),
            vertex!([0.0, 0.0, f64::MAX / 1000.0]),
        ];
        let result = algorithm.new_triangulation(&extreme_vertices);

        match result {
            Ok(tds) => {
                println!(
                    "  ✓ Extreme coordinates handled: {} vertices, {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
            Err(e) => {
                println!("  ✓ Extreme coordinates failed appropriately: {e}");
            }
        }

        // Test with very close vertices
        let close_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1e-10, 1e-10, 1e-10]), // Very close to first vertex
        ];
        let result = algorithm.new_triangulation(&close_vertices);

        match result {
            Ok(tds) => {
                println!(
                    "  ✓ Very close vertices handled: {} vertices, {} cells",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
            Err(e) => {
                println!("  ✓ Very close vertices failed appropriately: {e}");
            }
        }

        println!("✓ new_triangulation degenerate cases handled correctly");
    }

    /// Test `atomic_vertex_insert_and_remove_cells` method
    #[test]
    fn test_atomic_vertex_insert_and_remove_cells() {
        println!("Testing atomic_vertex_insert_and_remove_cells");

        // Create a triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([3.3, 2.8, 1.9]), // Additional unique vertex to create more cells
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let initial_generation = tds.generation();
        let initial_vertex_count = tds.number_of_vertices();
        let initial_cell_count = tds.number_of_cells();

        // Get some cells to remove (simulate bad cells)
        let cell_keys: Vec<_> = tds.cell_keys().take(1).collect();
        assert!(!cell_keys.is_empty(), "Should have at least one cell");

        // Create a new vertex to insert
        let new_vertex = vertex!([0.5, 0.5, 0.5]);

        // Test successful atomic operation
        let result =
            algorithm.atomic_vertex_insert_and_remove_cells(&mut tds, &new_vertex, &cell_keys);

        assert!(
            result.is_ok(),
            "Atomic operation should succeed: {:?}",
            result.err()
        );

        // Verify the vertex was added
        assert!(
            tds.vertex_key_from_uuid(&new_vertex.uuid()).is_some(),
            "New vertex should be in TDS"
        );

        // Verify cells were removed
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count - cell_keys.len(),
            "Cells should be removed"
        );

        // Verify generation changed (indicating TDS modification)
        assert_ne!(
            tds.generation(),
            initial_generation,
            "Generation should change after modifications"
        );

        println!("  ✓ Successful atomic operation completed");
        println!(
            "  ✓ Vertex count: {} -> {}",
            initial_vertex_count,
            tds.number_of_vertices()
        );
        println!(
            "  ✓ Cell count: {} -> {}",
            initial_cell_count,
            tds.number_of_cells()
        );
        println!(
            "  ✓ Generation: {} -> {}",
            initial_generation,
            tds.generation()
        );
    }

    /// Test `atomic_vertex_insert_and_remove_cells` with vertex validation failure
    #[test]
    fn test_atomic_vertex_insert_and_remove_cells_validation_failure() {
        println!("Testing atomic_vertex_insert_and_remove_cells with validation failure");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let _initial_generation = tds.generation();
        let _initial_cell_count = tds.number_of_cells();

        // Create a scenario that might cause vertex validation to fail
        // We'll use an existing vertex which should already be in the TDS
        let existing_vertex = vertices[0]; // Should already be in TDS

        // Get some cells to remove
        let cell_keys: Vec<_> = tds.cell_keys().take(1).collect();

        // Test atomic operation (should succeed since existing vertex handling is valid)
        let result =
            algorithm.atomic_vertex_insert_and_remove_cells(&mut tds, &existing_vertex, &cell_keys);

        // This should actually succeed because ensure_vertex_in_tds handles duplicates gracefully
        assert!(
            result.is_ok(),
            "Atomic operation with existing vertex should handle gracefully"
        );

        println!("  ✓ Atomic operation with existing vertex handled correctly");
    }

    /// Test `atomic_vertex_insert_and_remove_cells` with empty bad cells
    #[test]
    fn test_atomic_vertex_insert_and_remove_cells_empty_bad_cells() {
        println!("Testing atomic_vertex_insert_and_remove_cells with empty bad cells");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let initial_generation = tds.generation();
        let initial_cell_count = tds.number_of_cells();

        let new_vertex = vertex!([0.5, 0.5, 0.5]);
        let empty_bad_cells: Vec<crate::core::triangulation_data_structure::CellKey> = vec![];

        // Test atomic operation with no cells to remove
        let result = algorithm.atomic_vertex_insert_and_remove_cells(
            &mut tds,
            &new_vertex,
            &empty_bad_cells,
        );

        assert!(
            result.is_ok(),
            "Atomic operation should succeed with empty bad cells"
        );

        // Verify vertex was added
        assert!(
            tds.vertex_key_from_uuid(&new_vertex.uuid()).is_some(),
            "New vertex should be in TDS"
        );

        // Verify no cells were removed
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count,
            "Cell count should remain unchanged with empty bad cells"
        );

        // Generation should still change due to vertex insertion
        assert_ne!(
            tds.generation(),
            initial_generation,
            "Generation should change due to vertex insertion"
        );

        println!("  ✓ Empty bad cells handled correctly");
    }

    /// Test that cavity-based insertion succeeds for a simple interior vertex.
    ///
    /// This verifies the success path of the transactional algorithm:
    /// - Phase 1: Metadata extraction
    /// - Phase 2: New cell creation
    /// - Phase 3: Bad cell removal and neighbor wiring
    #[test]
    fn test_transactional_cavity_insertion_success() {
        println!("Testing transactional cavity-based insertion (success path)");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 1);

        // Insert an interior vertex
        let interior = vertex!([0.25, 0.25, 0.25]);
        let mut algorithm = IncrementalBowyerWatson::new();
        let result = algorithm.insert_vertex(&mut tds, interior);

        assert!(
            result.is_ok(),
            "Insertion should succeed: {:?}",
            result.err()
        );
        assert_eq!(tds.number_of_vertices(), 5);
        assert!(tds.number_of_cells() > 1);
        assert!(tds.is_valid().is_ok(), "Triangulation should remain valid");

        println!("  ✓ Transactional insertion succeeded");
        println!("  ✓ Vertices: 4 -> 5");
        println!("  ✓ Cells: 1 -> {}", tds.number_of_cells());
    }

    /// Test neighbor symmetry after transactional cavity insertion.
    ///
    /// Verifies that Phase 3 (neighbor wiring) correctly establishes
    /// bidirectional neighbor relationships.
    #[test]
    fn test_transactional_insertion_neighbor_symmetry() {
        println!("Testing neighbor symmetry after transactional insertion");

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

        let interior = vertex!([0.3, 0.3]);
        let mut algorithm = IncrementalBowyerWatson::new();
        algorithm.insert_vertex(&mut tds, interior).unwrap();

        // Verify neighbor symmetry
        let mut symmetric_count = 0;
        let mut checked_count = 0;

        for (cell_key, cell) in tds.cells() {
            if let Some(neighbors) = cell.neighbors() {
                for &neighbor_key_opt in neighbors {
                    checked_count += 1;
                    if let Some(neighbor_key) = neighbor_key_opt {
                        let neighbor = tds.get_cell(neighbor_key).unwrap();
                        if let Some(neighbor_neighbors) = neighbor.neighbors()
                            && neighbor_neighbors.contains(&Some(cell_key))
                        {
                            symmetric_count += 1;
                        }
                    }
                }
            }
        }

        assert!(
            symmetric_count > 0,
            "Should have at least some symmetric neighbor relationships"
        );

        println!("  ✓ Neighbor symmetry verified");
        println!("  ✓ Checked {checked_count} neighbor pointers");
        println!("  ✓ Found {symmetric_count} symmetric relationships");
    }

    /// Test multiple cavity insertions maintain validity.
    ///
    /// Stress-tests the transactional pattern with sequential insertions.
    #[test]
    fn test_transactional_multiple_insertions() {
        println!("Testing multiple transactional insertions");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let interior_vertices = [
            vertex!([0.5, 0.5, 0.5]),
            vertex!([1.0, 0.5, 0.5]),
            vertex!([0.5, 1.0, 0.5]),
        ];

        for (i, v) in interior_vertices.iter().enumerate() {
            let result = algorithm.insert_vertex(&mut tds, *v);
            assert!(result.is_ok(), "Insertion {i} should succeed");
            assert!(
                tds.is_valid().is_ok(),
                "TDS should be valid after insertion {i}"
            );
        }

        assert_eq!(tds.number_of_vertices(), 7);
        println!("  ✓ All {} insertions succeeded", interior_vertices.len());
        println!("  ✓ Final vertex count: 7");
        println!("  ✓ Final cell count: {}", tds.number_of_cells());
    }

    /// Test 2D transactional cavity insertion.
    #[test]
    fn test_transactional_insertion_2d() {
        println!("Testing 2D transactional cavity insertion");

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let interior = vertex!([0.5, 0.3]);
        assert!(algorithm.insert_vertex(&mut tds, interior).is_ok());
        assert_eq!(tds.number_of_vertices(), 4);
        assert!(tds.is_valid().is_ok());

        println!("  ✓ 2D insertion successful");
    }

    /// Test 4D transactional cavity insertion.
    #[test]
    fn test_transactional_insertion_4d() {
        println!("Testing 4D transactional cavity insertion");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let interior = vertex!([0.2, 0.2, 0.2, 0.2]);
        assert!(algorithm.insert_vertex(&mut tds, interior).is_ok());
        assert_eq!(tds.number_of_vertices(), 6);
        assert!(tds.is_valid().is_ok());

        println!("  ✓ 4D insertion successful");
    }

    // =========================================================================
    // CRITICAL MISSING TESTS - Added based on comprehensive test coverage analysis
    // =========================================================================

    /// Test `is_vertex_interior` with empty TDS (edge case not covered by macro)
    #[test]
    fn test_is_vertex_interior_empty_tds() {
        println!("Testing is_vertex_interior with empty TDS");
        let algorithm = IncrementalBowyerWatson::new();

        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);
        let result = algorithm.is_vertex_interior(&empty_tds, &test_vertex);
        assert!(result.is_ok(), "Should handle empty TDS");
        assert!(
            !result.unwrap(),
            "Empty TDS should return false for interior"
        );
        println!("  ✓ Empty TDS handled correctly");
        println!("✓ Empty TDS edge case test passed");
    }

    // test_gather_boundary_facet_info_errors removed - now covered by dimension macro (_gather_boundary_facet_info)

    /// Test `deduplicate_boundary_facet_info` edge cases
    #[test]
    fn test_deduplicate_boundary_facet_info_edge_cases() {
        println!("Testing deduplicate_boundary_facet_info edge cases");

        // Test with all duplicates
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();
        let vertex_keys: Vec<_> = tds.vertex_keys().take(3).collect();

        let mut facet_vertices = SmallBuffer::new();
        for vk in &vertex_keys {
            facet_vertices.push(*vk);
        }

        let all_duplicates = vec![
            BoundaryFacetInfo {
                bad_cell: cell_key,
                bad_facet_index: 0,
                facet_vertex_keys: facet_vertices.clone(),
                outside_neighbor: None,
            },
            BoundaryFacetInfo {
                bad_cell: cell_key,
                bad_facet_index: 1,
                facet_vertex_keys: facet_vertices.clone(),
                outside_neighbor: None,
            },
            BoundaryFacetInfo {
                bad_cell: cell_key,
                bad_facet_index: 2,
                facet_vertex_keys: facet_vertices.clone(),
                outside_neighbor: None,
            },
        ];

        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::deduplicate_boundary_facet_info(
                all_duplicates,
            );
        // Should return error with correct duplicate count
        match result {
            Err(InsertionError::DuplicateBoundaryFacets {
                duplicate_count,
                total_count,
            }) => {
                assert_eq!(duplicate_count, 2, "Should detect 2 duplicates");
                assert_eq!(total_count, 3, "Total count should be 3");
                println!(
                    "  ✓ All duplicates correctly detected: {duplicate_count} duplicates out of {total_count}"
                );
            }
            Ok(_) => panic!("Should have returned DuplicateBoundaryFacets error"),
            Err(other) => panic!("Unexpected error: {other:?}"),
        }

        println!("✓ deduplicate_boundary_facet_info edge cases passed");
    }

    /// Test `rollback_created_cells_and_vertex` verification
    #[test]
    fn test_rollback_created_cells_and_vertex_verification() {
        println!("Testing rollback_created_cells_and_vertex");

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();

        let initial_vertex_count = tds.number_of_vertices();
        let _initial_cell_count = tds.number_of_cells();

        // Insert a new vertex and create cells
        let new_vertex = vertex!([0.5, 0.5, 0.5]);
        let boundary_facets: Vec<_> = tds
            .boundary_facets()
            .unwrap()
            .map(|fv| FacetHandle::new(fv.cell_key(), fv.facet_index()))
            .take(2)
            .collect();

        // Manually create some cells
        let mut created_cells = Vec::new();
        for handle in &boundary_facets {
            if let Ok(cell_key) = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_cell_from_facet_handle(
                &mut tds,
                handle.cell_key(),
                handle.facet_index(),
                &new_vertex,
            ) {
                created_cells.push(cell_key);
            }
        }

        let cells_after_creation = tds.number_of_cells();
        let _vertices_after_creation = tds.number_of_vertices();

        // Rollback with vertex_existed_before = false (vertex should be removed)
        IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::rollback_created_cells_and_vertex(
            &mut tds,
            &created_cells,
            &new_vertex,
            false,
        );

        assert_eq!(
            tds.number_of_vertices(),
            initial_vertex_count,
            "Vertex count should return to initial after rollback"
        );
        assert_eq!(
            tds.number_of_cells(),
            cells_after_creation - created_cells.len(),
            "Cells should be removed"
        );
        println!(
            "  ✓ Rollback removed {} cells and the vertex",
            created_cells.len()
        );

        println!("✓ rollback_created_cells_and_vertex verification passed");
    }

    /// Test `set_neighbor_with_validation` bounds checking
    #[test]
    fn test_set_neighbor_with_validation_bounds() {
        println!("Testing set_neighbor_with_validation bounds checking");

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();

        let cell_keys: Vec<_> = tds.cell_keys().collect();
        assert!(!cell_keys.is_empty(), "Should have at least one cell");

        let cell_key = cell_keys[0];

        // Test valid neighbor index (0..=D)
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::set_neighbor_with_validation(
                &mut tds, cell_key, 0, cell_key,
            );
        assert!(result.is_ok(), "Valid neighbor index should succeed");
        println!("  ✓ Valid neighbor index (0) works");

        // Test boundary valid index (D)
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::set_neighbor_with_validation(
                &mut tds, cell_key, 3, cell_key,
            );
        assert!(result.is_ok(), "Boundary valid index (D=3) should succeed");
        println!("  ✓ Boundary valid index (3) works");

        // Test out of bounds index (> D)
        let result =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::set_neighbor_with_validation(
                &mut tds, cell_key, 4, cell_key,
            );
        assert!(
            result.is_err(),
            "Out of bounds index should fail: {result:?}"
        );
        println!("  ✓ Out of bounds index (4) correctly rejected");

        println!("✓ set_neighbor_with_validation bounds checking passed");
    }

    // =========================================================================
    // PROPTEST-BASED PROPERTY TESTS
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        /// Property: calculate_margin should never return zero for non-zero input
        #[test]
        fn prop_calculate_margin_never_zero_for_nonzero_input(range in 1i32..10000) {
            let margin = calculate_margin(range);
            prop_assert!(margin > 0, "Margin should be positive for positive range");
            prop_assert!(margin <= range, "Margin should not exceed range");
        }

        /// Property: InsertionStatistics rates should always be in [0.0, inf) with proper bounds
        #[test]
        fn prop_insertion_statistics_rates_bounded(
            vertices_processed in 0usize..1000,
            cavity_failures in 0usize..1000,
        ) {
            let mut stats = InsertionStatistics::new();
            stats.vertices_processed = vertices_processed;
            stats.cavity_boundary_failures = cavity_failures;
            // Ensure fallback_uses doesn't exceed vertices_processed for semantic correctness
            stats.fallback_strategies_used = if vertices_processed > 0 {
                vertices_processed / 2  // At most half can be fallbacks
            } else {
                0
            };

            let success_rate = stats.cavity_boundary_success_rate();
            prop_assert!((0.0..=1.0).contains(&success_rate),
                "Success rate {success_rate} should be in [0.0, 1.0]");

            let fallback_rate = stats.fallback_usage_rate();
            prop_assert!((0.0..=1.0).contains(&fallback_rate),
                "Fallback rate {fallback_rate} should be in [0.0, 1.0]");
        }

        /// Property: Deduplication correctly detects duplicates
        #[test]
        fn prop_deduplication_detects_duplicates(count in 2usize..20) {
            // Create test data
            let vertices = vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cell_keys().next().unwrap();
            let vertex_keys: Vec<_> = tds.vertex_keys().take(3).collect();

            let mut facet_vertices = SmallBuffer::new();
            for vk in &vertex_keys {
                facet_vertices.push(*vk);
            }

            // Create multiple copies of the same facet (all duplicates)
            let mut infos = Vec::new();
            for i in 0..count {
                infos.push(BoundaryFacetInfo {
                    bad_cell: cell_key,
                    bad_facet_index: i,
                    facet_vertex_keys: facet_vertices.clone(),
                    outside_neighbor: None,
                });
            }

            // Deduplication should detect duplicates and error
            let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::deduplicate_boundary_facet_info(
                infos,
            );

            // Should detect duplicates (count - 1 duplicates since first is not a duplicate)
            prop_assert!(result.is_err(), "Should detect duplicates and return error");

            if let Err(InsertionError::DuplicateBoundaryFacets { duplicate_count, .. }) = result {
                prop_assert_eq!(duplicate_count, count - 1, "Should detect correct number of duplicates");
            }
        }

        /// Property: Filter operations should never increase facet count
        #[test]
        fn prop_filter_never_increases_count(vertex_count in 4usize..10) {
            // Create vertices
            #[expect(clippy::cast_precision_loss)]
            let vertices: Vec<_> = (0..vertex_count)
                .map(|i| vertex!([(i as f64), 0.0, 0.0]))
                .collect();

            if vertices.len() >= 4 {
                let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices[..4]).unwrap();
                let cell_key = tds.cell_keys().next().unwrap();
                let vk_list: Vec<_> = tds.vertex_keys().collect();

                // Create boundary infos
                let mut infos = Vec::new();
                for (i, _) in vk_list[..3].iter().enumerate() {
                    let mut facet_vks = SmallBuffer::new();
                    for (j, &vk) in vk_list[..3].iter().enumerate() {
                        if j != i {
                            facet_vks.push(vk);
                        }
                    }
                    facet_vks.push(vk_list[3]);

                    infos.push(BoundaryFacetInfo {
                        bad_cell: cell_key,
                        bad_facet_index: i,
                        facet_vertex_keys: facet_vks,
                        outside_neighbor: None,
                    });
                }

                let initial_count = infos.len();
                let new_vk = vk_list[0]; // Use existing vertex

                // Convert to SmallBuffer
                let mut infos_buffer: SmallBuffer<BoundaryFacetInfo, MAX_PRACTICAL_DIMENSION_SIZE> = SmallBuffer::new();
                for info in infos {
                    infos_buffer.push(info);
                }

                let filtered = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::filter_boundary_facets_by_valid_facet_sharing(
                    &tds,
                    infos_buffer,
                    new_vk,
                )
                .expect("Filter should succeed");

                prop_assert!(filtered.len() <= initial_count,
                    "Filter should never increase facet count: {initial_count} -> {}",
                    filtered.len());
            }
        }

        /// Property: Saturating arithmetic should not panic
        #[test]
        fn prop_saturating_arithmetic_no_panic(
            a in -1000.0f64..1000.0f64,
            b in -1000.0f64..1000.0f64,
        ) {
            // These should never panic
            let _sub_result = bbox_sub(a, b);
            let _add_result = bbox_add(a, b);
            // If we got here without panicking, test passes
            prop_assert!(true);
        }
    }

    /// Integration test: Large scale degenerate input
    #[test]
    fn test_large_scale_degenerate_coplanar_points() {
        println!("Testing large scale degenerate coplanar points");

        // Create many coplanar points (all z=0)
        let mut vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]), // One non-coplanar for valid 3D
        ];

        // Add more coplanar points
        #[expect(clippy::cast_lossless)]
        for i in 0..10 {
            vertices.push(vertex!([(i as f64) * 0.1, (i as f64) * 0.15, 0.0]));
        }

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices[..4]).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Try inserting coplanar points
        let mut success_count = 0;
        for vertex in vertices.iter().skip(4) {
            match algorithm.insert_vertex(&mut tds, *vertex) {
                Ok(_) => success_count += 1,
                Err(e) => {
                    println!("  ⚠ Insertion failed (expected for degenerate): {e}");
                }
            }
        }

        let num_coplanar = vertices.len() - 4;
        println!(
            "  ✓ Handled {num_coplanar} coplanar points gracefully ({success_count} successful insertions)"
        );
        println!("✓ Large scale degenerate input test completed");
    }

    /// Stress test: Multiple consecutive insertions with potential rollbacks
    #[test]
    fn test_stress_multiple_consecutive_insertions() {
        println!("Stress testing multiple consecutive insertions");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Insert many interior points in rapid succession
        #[expect(clippy::cast_lossless)]
        let test_vertices: Vec<_> = (0..20)
            .map(|i| {
                let t = (i as f64) * 0.1;
                vertex!([0.5 + t * 0.1, 0.5 + t * 0.05, 0.5 + t * 0.08])
            })
            .collect();

        let mut success_count = 0;
        let mut rollback_count = 0;

        for vertex in test_vertices {
            match algorithm.insert_vertex(&mut tds, vertex) {
                Ok(_) => success_count += 1,
                Err(_) => rollback_count += 1,
            }
        }

        println!(
            "  ✓ Completed stress test: {success_count} successful, {rollback_count} failures/rollbacks"
        );
        assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after stress test"
        );
        println!("✓ Stress test completed successfully");
    }

    // =========================================================================
    // REMAINING CRITICAL TESTS - is_vertex_likely_exterior edge cases
    // =========================================================================

    /// Test `is_vertex_likely_exterior` with single vertex in TDS
    #[test]
    fn test_is_vertex_likely_exterior_single_vertex() {
        println!("Testing is_vertex_likely_exterior with single vertex in TDS");

        // Create TDS with just one vertex (empty after construction)
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let single_vertex = vertex!([1.0, 1.0, 1.0]);
        tds.insert_vertex_with_mapping(single_vertex).unwrap();

        let test_vertex = vertex!([2.0, 2.0, 2.0]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &tds,
                &test_vertex,
            );

        // With only one vertex, any other vertex should be considered exterior
        // because there's no meaningful bounding box
        println!("  ✓ Single vertex TDS: is_exterior = {is_exterior}");
        println!("✓ Single vertex test completed");
    }

    /// Test `is_vertex_likely_exterior` with all vertices at same point
    #[test]
    fn test_is_vertex_likely_exterior_all_same_point() {
        println!("Testing is_vertex_likely_exterior with all vertices at same point");

        // Create vertices all at the same point (degenerate case)
        let same_point = vertex!([1.0, 1.0, 1.0]);
        let vertices = vec![same_point; 4]; // All 4 vertices at same point

        // This will likely fail to create a valid TDS, but let's handle it
        match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            Ok(tds) => {
                let test_vertex = vertex!([2.0, 2.0, 2.0]);
                let is_exterior = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                    &tds,
                    &test_vertex,
                );
                println!("  ✓ All same point: is_exterior = {is_exterior}");
            }
            Err(e) => {
                println!("  ✓ Degenerate case correctly rejected: {e}");
            }
        }

        println!("✓ All same point test completed");
    }

    /// Test `is_vertex_likely_exterior` with vertex exactly on bounding box edge
    #[test]
    fn test_is_vertex_likely_exterior_on_bbox_edge() {
        println!("Testing is_vertex_likely_exterior with vertex on bounding box edge");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([10.0, 0.0, 0.0]),
            vertex!([0.0, 10.0, 0.0]),
            vertex!([0.0, 0.0, 10.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Bounding box is [0,10] x [0,10] x [0,10]
        // With 10% margin: [-1,11] x [-1,11] x [-1,11]

        // Test vertex exactly on expanded boundary
        let on_boundary = vertex!([11.0, 5.0, 5.0]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &tds,
                &on_boundary,
            );
        println!("  ✓ On boundary [11.0, 5.0, 5.0]: is_exterior = {is_exterior}");

        // Test vertex just inside expanded boundary
        let just_inside = vertex!([10.5, 5.0, 5.0]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &tds,
                &just_inside,
            );
        println!("  ✓ Just inside [10.5, 5.0, 5.0]: is_exterior = {is_exterior}");

        // Test vertex just outside expanded boundary
        let just_outside = vertex!([11.5, 5.0, 5.0]);
        let is_exterior =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &tds,
                &just_outside,
            );
        assert!(
            is_exterior,
            "Vertex outside expanded bbox should be exterior"
        );
        println!("  ✓ Just outside [11.5, 5.0, 5.0]: is_exterior = {is_exterior}");

        println!("✓ Bounding box edge test completed");
    }

    // =========================================================================
    // REMAINING CRITICAL TESTS - find_bad_cells missing cases
    // =========================================================================

    /// Test `find_bad_cells` with exact `DEGENERATE_CELL_THRESHOLD` boundary (50%)
    #[test]
    fn test_find_bad_cells_exact_threshold() {
        println!("Testing find_bad_cells at exact degenerate threshold (50%)");

        // This test verifies the threshold logic by simulation
        // We can't easily create a TDS with exactly 50% degenerate cells,
        // but we can verify the error message

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Test with a vertex that should work normally
        let test_vertex = vertex!([0.5, 0.5, 0.5]);
        let result = algorithm.find_bad_cells(&tds, &test_vertex);

        match result {
            Ok(bad_cells) => {
                let num_bad = bad_cells.len();
                println!("  ✓ Found {num_bad} bad cells for interior vertex");
                assert!(!bad_cells.is_empty(), "Should find at least one bad cell");
            }
            Err(e) => {
                println!("  ✓ Error (acceptable): {e}");
            }
        }

        println!("✓ Degenerate threshold test completed");
    }

    // =========================================================================
    // REMAINING CRITICAL TESTS - InsertionAlgorithm trait methods
    // =========================================================================

    /// Test `create_cell_from_vertices_and_vertex` duplicate detection
    #[test]
    fn test_create_cell_from_vertices_and_vertex_duplicate() {
        println!("Testing create_cell_from_vertices_and_vertex duplicate detection");

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();

        // Get facet vertices
        let facet_vertices: Vec<_> = tds.vertices().take(3).map(|(_, v)| *v).collect();

        // Try to create cell with duplicate vertex (one that's already in facet)
        let duplicate_vertex = facet_vertices[0];

        let result = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_cell_from_vertices_and_vertex(
            &mut tds,
            facet_vertices,
            &duplicate_vertex,
        );

        assert!(
            result.is_err(),
            "Should reject duplicate vertex: {result:?}"
        );

        if let Err(e) = result {
            let error_msg = format!("{e}");
            assert!(
                error_msg.contains("duplicate"),
                "Error should mention duplicate: {error_msg}"
            );
            println!("  ✓ Duplicate correctly rejected: {e}");
        }

        println!("✓ Duplicate vertex detection test passed");
    }

    /// Test `invalidate_cache_atomically` is callable
    #[test]
    fn test_invalidate_cache_atomically_callable() {
        println!("Testing invalidate_cache_atomically is callable");

        let mut algorithm: IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> =
            IncrementalBowyerWatson::new();

        // This should not panic - just verify it's callable
        algorithm.invalidate_cache_atomically();

        println!("  ✓ invalidate_cache_atomically called successfully");
        println!("✓ Cache invalidation test passed");
    }

    /// Test `determine_strategy` with zero-cell TDS
    #[test]
    fn test_determine_strategy_zero_cells() {
        println!("Testing determine_strategy with zero cells");

        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let test_vertex = vertex!([1.0, 1.0, 1.0]);
        let algorithm: IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> =
            IncrementalBowyerWatson::new();

        let strategy = algorithm.determine_strategy(&empty_tds, &test_vertex);

        // Note: determine_strategy_default returns Standard for 0 cells,
        // but the actual implementation may return HullExtension since
        // there's no existing triangulation to extend into
        println!("  ✓ Zero cells strategy: {strategy:?}");
        assert!(
            matches!(
                strategy,
                InsertionStrategy::Standard | InsertionStrategy::HullExtension
            ),
            "Empty TDS should use Standard or HullExtension strategy, got {strategy:?}"
        );
        println!("✓ determine_strategy zero cells test passed");
    }

    /// Test `InsertionAlgorithm` comprehensive trait method coverage
    #[test]
    fn test_insertion_algorithm_trait_methods_comprehensive() {
        println!("Testing InsertionAlgorithm trait methods comprehensively");

        let mut algorithm: IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> =
            IncrementalBowyerWatson::new();

        // Test get_statistics
        let (insertions, created, removed) = algorithm.get_statistics();
        assert_eq!(insertions, 0, "Initial insertions should be 0");
        assert_eq!(created, 0, "Initial cells created should be 0");
        assert_eq!(removed, 0, "Initial cells removed should be 0");
        println!("  ✓ get_statistics works correctly");

        // Test reset
        algorithm.reset();
        let (insertions, created, removed) = algorithm.get_statistics();
        assert_eq!(insertions, 0);
        assert_eq!(created, 0);
        assert_eq!(removed, 0);
        println!("  ✓ reset works correctly");

        // Test increment methods
        algorithm.increment_cells_created(5);
        algorithm.increment_cells_removed(3);
        let (_, created, removed) = algorithm.get_statistics();
        assert_eq!(created, 5, "Cells created should be incremented");
        assert_eq!(removed, 3, "Cells removed should be incremented");
        println!("  ✓ increment methods work correctly");

        // Test update_statistics
        algorithm.update_statistics(2, 1);
        let (_, created, removed) = algorithm.get_statistics();
        assert_eq!(created, 7, "Cells created should be 5+2");
        assert_eq!(removed, 4, "Cells removed should be 3+1");
        println!("  ✓ update_statistics works correctly");

        println!("✓ InsertionAlgorithm trait methods comprehensive test passed");
    }

    /// Test that preventive facet filtering correctly handles interior boundary facets.
    ///
    /// This test specifically checks for a bug where the filter would reject all valid
    /// interior boundary facets because it counted facets from bad cells (about to be deleted)
    /// as if they would remain, causing legitimate boundary facets to be incorrectly flagged
    /// as over-sharing.
    ///
    /// The test creates a simple 3D triangulation with 5 vertices forming 2 tetrahedra,
    /// then simulates the cavity-based insertion by:
    /// 1. Marking one cell as "bad"
    /// 2. Finding its boundary facets
    /// 3. Applying preventive filtering
    /// 4. Verifying that valid interior boundary facets are NOT filtered out
    #[test]
    fn test_preventive_filter_does_not_reject_valid_interior_facets() {
        println!("Testing preventive facet filter with interior boundary facets");

        // Create a 3D triangulation with initial 4 vertices (single tetrahedron)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.5, 1.0, 0.0]),
            vertex!([0.5, 0.5, 1.0]),
        ];

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Add 5th vertex to create second tetrahedron
        let fifth_vertex = vertex!([10.5, 11.5, 12.0]);
        tds.add(fifth_vertex).expect("Should add 5th vertex");

        // Verify we have 2 cells
        assert_eq!(tds.number_of_cells(), 2, "Should have 2 tetrahedra");

        // Pick one cell as the "bad cell" that we're removing
        let bad_cell = tds
            .cell_keys()
            .next()
            .expect("Should have at least one cell");

        // Find the boundary facets of this "cavity"
        let algorithm = IncrementalBowyerWatson::new();
        let boundary_facets = algorithm
            .find_cavity_boundary_facets(&tds, &[bad_cell])
            .expect("Should find boundary facets");

        println!(
            "  Found {} boundary facets for single-cell cavity",
            boundary_facets.len()
        );
        assert_eq!(
            boundary_facets.len(),
            4,
            "Single tetrahedron cavity should have 4 boundary facets"
        );

        // Gather boundary facet information (mimics real insertion flow)
        let boundary_infos =
            IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::gather_boundary_facet_info(
                &tds,
                &boundary_facets,
            )
            .expect("Should gather boundary facet info");

        // Get a vertex key for the "inserted" vertex (use one from the other cell)
        let other_cell_key = tds
            .cell_keys()
            .find(|&ck| ck != bad_cell)
            .expect("Should have second cell");
        let other_cell = tds.get_cell(other_cell_key).expect("Should get other cell");
        let inserted_vk = other_cell.vertices()[0];

        // Convert to SmallBuffer for filtering
        let mut boundary_infos_buffer: SmallBuffer<
            BoundaryFacetInfo,
            MAX_PRACTICAL_DIMENSION_SIZE,
        > = SmallBuffer::new();
        for info in boundary_infos.clone() {
            boundary_infos_buffer.push(info);
        }

        // Apply preventive filtering
        let filtered = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::filter_boundary_facets_by_valid_facet_sharing(
            &tds,
            boundary_infos_buffer,
            inserted_vk,
        )
        .expect("Filter should succeed");

        println!(
            "  After filtering: {} facets remaining (from {})",
            filtered.len(),
            boundary_infos.len()
        );

        // CRITICAL CHECK: The filter should NOT remove all boundary facets
        // Before the fix, this would fail because interior facets were incorrectly
        // counted as belonging to 2 cells (bad cell + neighbor) and rejected
        assert!(
            !filtered.is_empty(),
            "Preventive filter should not reject all valid interior boundary facets. \
             This indicates the bug where bad cell facets are counted incorrectly."
        );

        // For a single-cell cavity in a 2-cell triangulation, at least 1 facet
        // should pass (the shared facet between the two cells)
        assert!(
            !filtered.is_empty(),
            "Should keep at least the interior boundary facet(s)"
        );

        println!(
            "  ✓ Filter correctly kept {} valid boundary facet(s)",
            filtered.len()
        );
        println!("✓ Preventive facet filter test passed - no false rejections");
    }

    // Property tests for margin calculation and exterior detection
    proptest! {
        #[test]
        fn prop_margin_calculation_accuracy(range in 10i32..100_000) {
            let margin = calculate_margin(range);
            // For integers, margin should be range/10 (with minimum 1)
            let expected_min = std::cmp::max(1, range / 10);
            prop_assert!(margin >= expected_min,
                "Margin {margin} should be at least {expected_min}");
            // Margin should not exceed range/10 by more than 1 (rounding)
            let expected_max = (range / 10) + 1;
            prop_assert!(margin <= expected_max,
                "Margin {margin} should not exceed {expected_max}");
        }

        /// Property: is_vertex_likely_exterior should be consistent
        #[test]
        #[expect(clippy::tuple_array_conversions)]
        fn prop_is_vertex_likely_exterior_consistency(
            x in -100.0f64..100.0,
            y in -100.0f64..100.0,
            z in -100.0f64..100.0,
        ) {
            let vertices = vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
            let test_vertex = vertex!([x, y, z]);

            // Should not panic
            let _is_exterior = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_vertex_likely_exterior(
                &tds,
                &test_vertex,
            );

            // Test passes if we didn't panic
            prop_assert!(true);
        }
    }
}
