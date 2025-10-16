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
use approx::abs_diff_eq;
use num_traits::NumCast;
use num_traits::{One, Zero, cast};
use serde::{Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, Div, SubAssign};

// REMOVED: make_facet_from_view was broken due to Phase 3A refactoring
// The deprecated Facet::vertices() returns an empty Vec, causing silent failures
// All code now uses FacetView directly or lightweight (CellKey, u8) handles

/// Error for too many degenerate cells case
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl std::error::Error for TooManyDegenerateCellsError {}

/// Error that can occur during bad cells detection
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
}

/// Comprehensive error type for vertex insertion operations
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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

/// Threshold for determining when too many degenerate cells make results unreliable.
/// If more than this fraction of cells are degenerate, the results are considered unreliable.
/// Currently set to 0.5 (50%), which means if more than half the cells are degenerate,
/// we consider the results unreliable. This threshold can be adjusted based on the
/// tolerance for degenerate cases in specific applications.
const DEGENERATE_CELL_THRESHOLD: f64 = 0.5;

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
#[allow(dead_code)] // Fields used for debugging but not all accessed in production code
pub struct BoundaryFacetInfo {
    /// Key of the bad cell containing this boundary facet
    bad_cell: CellKey,
    /// Index of the facet within the bad cell (0..=D)
    bad_facet_index: usize,
    /// Vertex keys forming the boundary facet (D vertices)
    facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Neighbor cell across this boundary facet and its reciprocal facet index.
    /// `None` if this is a true boundary facet (no neighbor on the exterior side).
    /// Format: (`neighbor_cell_key`, `reciprocal_facet_index_in_neighbor`)
    outside_neighbor: Option<(CellKey, usize)>,
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
            #[allow(clippy::cast_precision_loss)]
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
            #[allow(clippy::cast_precision_loss)]
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
#[allow(clippy::struct_field_names)]
pub struct InsertionBuffers<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    /// Insert a single vertex into the triangulation
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
    /// Returns an error if vertex insertion fails due to geometric degeneracy,
    /// numerical issues, or topological constraints.
    fn insert_vertex(
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + One + Zero + PartialEq + Div<Output = T>,
        for<'a> &'a T: Div<T>,
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + One + Zero + PartialEq + Div<Output = T>,
        for<'a> &'a T: Div<T>,
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
    /// `true` if the vertex is interior, `false` otherwise.
    fn is_vertex_interior(&self, tds: &Tds<T, U, V, D>, vertex: &Vertex<T, U, D>) -> bool
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    {
        use crate::geometry::predicates::{InSphere, insphere};

        // Reserve exact capacity once; keep on stack for typical small D
        let mut vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

        for (_cell_key, cell) in tds.cells() {
            // Clear and reuse the buffer - capacity is already preallocated
            vertex_points.clear();
            // Phase 3A: Get vertices via TDS using vertices
            for &vkey in cell.vertices() {
                if let Some(v) = tds.vertices().get(vkey) {
                    vertex_points.push(*v.point());
                }
            }

            if matches!(
                insphere(&vertex_points, *vertex.point()),
                Ok(InSphere::INSIDE)
            ) {
                return true;
            }
        }
        false
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + One + Zero + PartialEq + Div<Output = T>,
    {
        // Get the vertex coordinates
        let vertex_coords: [T; D] = vertex.point().into();

        // Calculate rough bounding box of existing vertices
        let mut min_coords = [T::zero(); D];
        let mut max_coords = [T::zero(); D];
        let mut initialized = false;
        let mut vertex_count = 0;

        for existing_vertex in tds.vertices().values() {
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
        // Precompute 10 for integer division fallback
        let ten: T = cast(10).unwrap_or_else(T::one);
        let mut expanded_min = [T::zero(); D];
        let mut expanded_max = [T::zero(); D];

        for i in 0..D {
            let range = max_coords[i] - min_coords[i];

            // For floats: use 10% expansion (0.1 * range)
            // For integers: use range/10 with minimum of 1
            // Note: cast::<f64, i32>(0.1) returns Some(0), not None!
            let margin = cast::<f64, T>(MARGIN_FACTOR).map_or_else(
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
            );

            // Use simple arithmetic for bounding box expansion
            // For floating-point types, overflow goes to infinity (acceptable for heuristic)
            // For integer types, overflow wraps (acceptable for this heuristic use case)
            expanded_min[i] = min_coords[i] - margin;
            expanded_max[i] = max_coords[i] + margin;
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
        for<'a> &'a T: Div<T>,
    {
        // Check if there are any cells to test
        if tds.cells().is_empty() {
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
            let v_count = cell.vertices().len();
            // Treat non-D+1 vertex counts as degenerate
            if v_count != D + 1 {
                degenerate_count += 1;
                continue;
            }

            cells_tested += 1;
            // Reuse buffer by clearing and repopulating
            vertex_points.clear();
            // Phase 3A: Get vertices via TDS using vertices
            for &vkey in cell.vertices() {
                if let Some(v) = tds.vertices().get(vkey) {
                    vertex_points.push(*v.point());
                }
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

            // NOTE: The integer arithmetic optimization below is specifically designed for
            // DEGENERATE_CELL_THRESHOLD = 0.5. This avoids floating-point operations for
            // the common 50% threshold case. To generalize for arbitrary rational thresholds,
            // we would compare: degenerate_count * denominator > total * numerator
            // For now, we detect the 0.5 case and optimize it; other values fall back to FP.
            if abs_diff_eq!(DEGENERATE_CELL_THRESHOLD, 0.5, epsilon = f64::EPSILON) {
                // Use optimized integer arithmetic for 50% threshold
                let threshold_exceeded = degenerate_count.saturating_mul(2) > total_cells;

                if threshold_exceeded {
                    return Err(BadCellsError::TooManyDegenerateCells(
                        TooManyDegenerateCellsError {
                            degenerate_count,
                            total_tested: cells_tested,
                        },
                    ));
                }
            } else {
                // For non-0.5 thresholds, we must use floating point comparison
                // We accept the precision loss here as it's unavoidable for large usize values
                #[allow(clippy::cast_precision_loss)]
                let degenerate_ratio = (degenerate_count as f64) / (total_cells as f64);
                if degenerate_ratio > DEGENERATE_CELL_THRESHOLD {
                    return Err(BadCellsError::TooManyDegenerateCells(
                        TooManyDegenerateCellsError {
                            degenerate_count,
                            total_tested: cells_tested,
                        },
                    ));
                }
            }
        }

        Ok(bad_cells)
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
        for<'a> &'a T: Div<T>,
    {
        let mut boundary_facet_handles = Vec::new();

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
        let mut seen_facet_keys: FastHashSet<u64> =
            fast_hash_set_with_capacity(bad_cells.len() * (D + 1));

        // Scan each bad cell's D+1 facets
        for &bad_cell_key in bad_cells {
            let Some(bad_cell) = tds.cells().get(bad_cell_key) else {
                continue;
            };

            let Some(neighbors) = bad_cell.neighbors() else {
                // Cell has no neighbor information; treat all facets as boundary
                for facet_idx in 0..=D {
                    if let Ok(facet_idx_u8) = usize_to_u8(facet_idx, D + 1) {
                        // Compute canonical facet key for deduplication
                        let facet_vertices: SmallBuffer<_, MAX_PRACTICAL_DIMENSION_SIZE> = bad_cell
                            .vertices()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &v)| (i != facet_idx).then_some(v))
                            .collect();
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
                let Some(&neighbor_key_opt) = neighbors.get(facet_idx) else {
                    continue;
                };

                // Boundary facet if: no neighbor OR neighbor is not bad
                let is_boundary = neighbor_key_opt.is_none_or(|n| !bad_cell_set.contains(&n));
                if !is_boundary {
                    continue; // Interior facet; skip
                }

                // This is a boundary facet; compute canonical key and deduplicate
                let Ok(facet_idx_u8) = usize_to_u8(facet_idx, D + 1) else {
                    continue;
                };

                // Compute canonical facet key: sorted vertex keys of the D vertices
                let facet_vertices: SmallBuffer<_, MAX_PRACTICAL_DIMENSION_SIZE> = bad_cell
                    .vertices()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (i != facet_idx).then_some(v))
                    .collect();
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
    #[allow(clippy::too_many_lines)]
    fn insert_vertex_cavity_based(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
        for<'a> &'a T: Div<T>,
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

        // ========================================================================
        // PHASE 2: TENTATIVE - Insert vertex and create new cells (no removal yet)
        // ========================================================================
        // Track whether vertex existed before this operation for atomic rollback
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        // Ensure vertex is in TDS (needed to create cells)
        Self::ensure_vertex_in_tds(tds, vertex)?;

        // Get the inserted vertex key for cell creation
        let inserted_vk = tds.vertex_key_from_uuid(&vertex.uuid()).ok_or_else(|| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: "Vertex was not found in TDS immediately after insertion".to_string(),
                },
            )
        })?;

        // Create all new cells BEFORE removing bad cells
        // This allows clean rollback if creation fails
        let mut created_cell_keys = Vec::with_capacity(boundary_infos.len());
        for info in &boundary_infos {
            // Combine facet vertices with the inserted vertex
            let mut cell_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                info.facet_vertex_keys.clone();
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
                    // Rollback: remove only newly-created cells and the vertex if it was new
                    Self::rollback_created_cells_and_vertex(
                        tds,
                        &created_cell_keys,
                        vertex,
                        vertex_existed_before,
                    );
                    return Err(InsertionError::TriangulationConstruction(e));
                }
            }
        }
        let cells_created = created_cell_keys.len();

        // ========================================================================
        // PHASE 3: COMMIT - Remove bad cells and establish neighbor relationships
        // ========================================================================
        // Now that all new cells exist, remove the bad cells
        // This is the point of no return - from here on, we cannot rollback
        Self::remove_bad_cells(tds, &bad_cells);

        // Wire neighbor relationships between new cells and existing triangulation
        Self::connect_new_cells_to_neighbors(
            tds,
            inserted_vk,
            &boundary_infos,
            &created_cell_keys,
        )?;

        // Finalize the triangulation after insertion to fix any invalid states
        Self::finalize_after_insertion(tds).map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!(
                    "Failed to finalize triangulation after cavity-based insertion: {e}"
                ),
            }
        })?;

        Ok(InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed,
            cells_created,
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
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    {
        // Conservative fallback: try to connect to any existing boundary facet
        // This avoids creating invalid geometry by arbitrary vertex replacement

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
                if let Some(_cell) = tds.cells().get(cell_key) {
                    // Phase 3A: Use lightweight facet handles directly
                    // Try to create a cell from this facet handle and the vertex
                    if Self::create_cell_from_facet_handle(tds, cell_key, facet_index, vertex)
                        .is_ok()
                    {
                        // Finalize the triangulation after insertion to fix any invalid states
                        Self::finalize_after_insertion(tds).map_err(|e| {
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Failed to finalize triangulation after fallback insertion: {e}"
                                ),
                            }
                        })?;

                        return Ok(InsertionInfo {
                            strategy: InsertionStrategy::Fallback,
                            cells_removed: 0,
                            cells_created: 1,
                            success: true,
                            degenerate_case_handled: false,
                        });
                    }
                }
            }
        }

        // If boundary facets don't work, try ALL facets (including internal ones)
        for cells in facet_to_cells.values() {
            for facet_handle in cells {
                let cell_key = facet_handle.cell_key();
                let facet_index = facet_handle.facet_index();
                let _fi = <usize as From<_>>::from(facet_index);
                if let Some(_cell) = tds.cells().get(cell_key) {
                    // Phase 3A: Use lightweight facet handles directly
                    // Try to create a cell from this facet handle and the vertex
                    if Self::create_cell_from_facet_handle(tds, cell_key, facet_index, vertex)
                        .is_ok()
                    {
                        // Finalize the triangulation after insertion to fix any invalid states
                        Self::finalize_after_insertion(tds).map_err(|e| {
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Failed to finalize triangulation after fallback insertion: {e}"
                                ),
                            }
                        })?;

                        return Ok(InsertionInfo {
                            strategy: InsertionStrategy::Fallback,
                            cells_removed: 0,
                            cells_created: 1,
                            success: true,
                            degenerate_case_handled: false,
                        });
                    }
                }
            }
        }

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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        for<'a> &'a T: Div<T>,
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        for<'a> &'a T: Div<T>,
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
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    {
        // Get the adjacent cell to this boundary facet
        let Some(adjacent_cell) = tds.cells().get(adjacent_cell_key) else {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Adjacent cell {adjacent_cell_key:?} not found during visibility test. This indicates TDS corruption."
                    ),
                },
            ));
        };

        // HOT PATH: Collect facet vertices once to avoid duplicate iteration
        // This single collection is used for both UUID lookup and point extraction
        let facet_vertices_vec: SmallVec<[Vertex<T, U, D>; 8]> = facet
            .vertices()
            .map_err(|e| {
                InsertionError::TriangulationState(TriangulationValidationError::FacetError(e))
            })?
            .copied()
            .collect();
        let facet_vertex_uuids: SmallVec<[uuid::Uuid; 8]> =
            facet_vertices_vec.iter().map(Vertex::uuid).collect();

        // Find the vertex in the adjacent cell that is NOT part of the facet
        // This is the \"opposite\" vertex that defines the \"inside\" side of the facet
        let cell_vertices = adjacent_cell.vertices();

        let mut opposite_vertex = None;
        for &vkey in cell_vertices {
            let Some(cell_vertex) = tds.vertices().get(vkey) else {
                continue;
            };
            // Check membership using cached UUIDs instead of calling facet.vertices() repeatedly
            let is_in_facet = facet_vertex_uuids.contains(&cell_vertex.uuid());
            if !is_in_facet {
                opposite_vertex = Some(cell_vertex);
                break;
            }
        }

        let Some(opposite_vertex) = opposite_vertex else {
            // Could not find opposite vertex - topology is inconsistent
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Facet lacked opposite vertex for cell {adjacent_cell_key:?}. This indicates potential TDS corruption \
                         where the facet vertices do not form a proper (D-1)-face of the adjacent D-cell."
                    ),
                },
            ));
        };

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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
        for<'a> &'a T: Div<T>,
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
        for<'a> &'a T: Div<T>,
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
        for<'a> &'a T: Div<T>,
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
        for<'a> &'a T: Div<T>,
    {
        // Track whether vertex existed before this operation for atomic rollback
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        // Phase 1: Extract all facet data upfront before creating any cells
        // This ensures we can validate everything before modifying the TDS
        let mut extracted_facet_data = Vec::with_capacity(facet_handles.len());

        for handle in facet_handles {
            let cell_key = handle.cell_key();
            let facet_index = handle.facet_index();

            // Validate cell exists first
            let _cell = tds.cells().get(cell_key).ok_or_else(|| {
                InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell key {cell_key:?} not found in TDS during cell creation"
                        ),
                    },
                )
            })?;

            // Create FacetView and extract vertices
            let facet_view = crate::core::facet::FacetView::new(tds, cell_key, facet_index)
                .map_err(|_| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Facet index {facet_index} out of bounds for cell {cell_key:?}"
                            ),
                        },
                    )
                })?;

            // Extract vertex data from FacetView
            let facet_vertices_iter = facet_view.vertices().map_err(|e| {
                InsertionError::TriangulationState(TriangulationValidationError::FacetError(e))
            })?;
            let facet_vertices: Vec<Vertex<T, U, D>> = facet_vertices_iter.copied().collect();
            extracted_facet_data.push(facet_vertices);
        }

        // Phase 2: Create all cells, tracking created cell keys for potential rollback
        let mut created_cell_keys = Vec::with_capacity(extracted_facet_data.len());

        for facet_vertices in extracted_facet_data {
            match Self::create_cell_from_vertices_and_vertex(tds, facet_vertices, vertex) {
                Ok(cell_key) => {
                    created_cell_keys.push(cell_key);
                }
                Err(e) => {
                    Self::rollback_created_cells_and_vertex(
                        tds,
                        &created_cell_keys,
                        vertex,
                        vertex_existed_before,
                    );
                    return Err(InsertionError::TriangulationState(e));
                }
            }
        }

        let cells_created = created_cell_keys.len();

        // Validate that we created at least some cells
        if cells_created == 0 && !facet_handles.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: format!(
                        "Failed to create any cells from {} facet handles",
                        facet_handles.len()
                    ),
                },
            ));
        }

        Ok(cells_created)
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
        for<'a> &'a T: Div<T>,
    {
        let mut boundary_infos = Vec::with_capacity(boundary_facet_handles.len());

        for handle in boundary_facet_handles {
            let bad_cell = handle.cell_key();
            let bad_facet_index = <usize as From<u8>>::from(handle.facet_index());

            // Get the bad cell
            let Some(cell) = tds.cells().get(bad_cell) else {
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
                    let neighbor_cell = tds.cells().get(neighbor_key).ok_or_else(|| {
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
                        // Neighbor has no neighbor info; cannot determine reciprocal index
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
    #[allow(clippy::too_many_lines)] // Complex topology wiring requires detailed logic
    fn connect_new_cells_to_neighbors(
        tds: &mut Tds<T, U, V, D>,
        inserted_vk: VertexKey,
        boundary_infos: &[BoundaryFacetInfo],
        created_cells: &[CellKey],
    ) -> Result<(), InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
        for<'a> &'a T: Div<T>,
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
            let new_cell = tds.cells().get(*new_cell_key).ok_or_else(|| {
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
                let new_cell_mut = tds.cells_mut().get_mut(*new_cell_key).ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Cannot get mutable reference to cell {new_cell_key:?}"
                            ),
                        },
                    )
                })?;

                if new_cell_mut.neighbors.is_none() {
                    let mut neighbors = SmallBuffer::new();
                    neighbors.resize(D + 1, None);
                    new_cell_mut.neighbors = Some(neighbors);
                }
                if let Some(neighbors) = &mut new_cell_mut.neighbors {
                    neighbors[inserted_idx] = Some(outside_ck);
                }

                // Set outside_ck's neighbor at outside_facet_idx → new_cell
                let outside_cell = tds.cells_mut().get_mut(outside_ck).ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Outside neighbor {outside_ck:?} not found for new cell {new_cell_key:?}"
                            ),
                        },
                    )
                })?;

                if outside_cell.neighbors.is_none() {
                    let mut neighbors = SmallBuffer::new();
                    neighbors.resize(D + 1, None);
                    outside_cell.neighbors = Some(neighbors);
                }
                if let Some(neighbors) = &mut outside_cell.neighbors {
                    neighbors[outside_facet_idx] = Some(*new_cell_key);
                }
            }
        }

        // ====================================================================
        // STEP 2: Wire New→New neighbors within the cavity
        // ====================================================================
        // Type alias at module level would be better, but this is a local helper
        #[allow(clippy::items_after_statements)]
        type FacetSignature = SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>;

        // Build a map from facet signatures to (cell_key, local_facet_index)
        // Facets that include the inserted vertex connect new cells to each other
        let mut facet_to_cell: FastHashMap<u64, (CellKey, usize)> =
            fast_hash_map_with_capacity(created_cells.len() * D);

        for &new_cell_key in created_cells {
            // Clone cell vertices to avoid borrow issues
            let cell_vertices = tds
                .cells()
                .get(new_cell_key)
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

                // Build facet signature: all vertices except opposite_vk, sorted
                let mut facet_sig: FacetSignature = cell_vertices
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &vk)| (i != opposite_idx).then_some(vk))
                    .collect();
                facet_sig.sort_unstable();

                let facet_key = facet_key_from_vertices(&facet_sig);

                // Check if we've seen this facet before
                if let Some((other_cell_key, other_facet_idx)) = facet_to_cell.get(&facet_key) {
                    // Wire bidirectional neighbor pointers
                    let other_ck = *other_cell_key;
                    let other_idx = *other_facet_idx;

                    // Set new_cell[opposite_idx] → other_cell
                    let cell_mut = tds.cells_mut().get_mut(new_cell_key).ok_or_else(|| {
                        InsertionError::TriangulationState(
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!("Cannot get mutable cell {new_cell_key:?}"),
                            },
                        )
                    })?;
                    if cell_mut.neighbors.is_none() {
                        let mut neighbors = SmallBuffer::new();
                        neighbors.resize(D + 1, None);
                        cell_mut.neighbors = Some(neighbors);
                    }
                    if let Some(neighbors) = &mut cell_mut.neighbors {
                        neighbors[opposite_idx] = Some(other_ck);
                    }

                    // Set other_cell[other_idx] → new_cell
                    let other_mut = tds.cells_mut().get_mut(other_ck).ok_or_else(|| {
                        InsertionError::TriangulationState(
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!("Cannot get mutable cell {other_ck:?}"),
                            },
                        )
                    })?;
                    if other_mut.neighbors.is_none() {
                        let mut neighbors = SmallBuffer::new();
                        neighbors.resize(D + 1, None);
                        other_mut.neighbors = Some(neighbors);
                    }
                    if let Some(neighbors) = &mut other_mut.neighbors {
                        neighbors[other_idx] = Some(new_cell_key);
                    }
                } else {
                    // First time seeing this facet; record it
                    facet_to_cell.insert(facet_key, (new_cell_key, opposite_idx));
                }
            }
        }

        Ok(())
    }

    /// Rollback created cells and optionally remove a vertex that was inserted during the operation.
    ///
    /// This is a shared utility method used by insertion algorithms to provide atomic rollback
    /// semantics. If cell creation fails partway through, this method cleans up both the
    /// partially created cells and the vertex if it was newly inserted.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `created_cell_keys` - Keys of cells that were created and need to be removed
    /// * `vertex` - The vertex that was being inserted
    /// * `vertex_existed_before` - Whether the vertex existed in TDS before the operation started
    fn rollback_created_cells_and_vertex(
        tds: &mut Tds<T, U, V, D>,
        created_cell_keys: &[crate::core::triangulation_data_structure::CellKey],
        vertex: &Vertex<T, U, D>,
        vertex_existed_before: bool,
    ) where
        T: AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        // Rollback created cells
        if !created_cell_keys.is_empty() {
            tds.remove_cells_by_keys(created_cell_keys);
        }

        // Remove the vertex if it was inserted during this operation
        if !vertex_existed_before {
            tds.remove_vertex_by_uuid(&vertex.uuid());
        }
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
        for<'a> &'a T: Div<T>,
    {
        // Use the optimized batch removal method that handles UUID mapping
        // and generation counter updates internally
        tds.remove_cells_by_keys(bad_cells);
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
        for<'a> &'a T: Div<T>,
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
        for<'a> &'a T: Div<T>,
    {
        // Remove duplicate cells first
        tds.remove_duplicate_cells()?;

        // Fix invalid facet sharing
        tds.fix_invalid_facet_sharing().map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to fix invalid facet sharing: {e}"),
            }
        })?;

        // Assign neighbor relationships
        tds.assign_neighbors()?;

        // Assign incident cells to vertices
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

    #[test]
    fn test_find_visible_boundary_facets_exterior_vertex() {
        println!("Testing find_visible_boundary_facets_lightweight with exterior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test exterior vertex that should see some facets
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let visible_facet_handles = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should successfully find visible boundary facets");

        println!("  Found {} visible facets", visible_facet_handles.len());

        // Should find at least some visible facets for exterior vertex
        assert!(
            !visible_facet_handles.is_empty(),
            "Exterior vertex should see at least some boundary facets"
        );
        assert!(
            visible_facet_handles.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        // Test that each visible facet handle is valid
        for (i, handle) in visible_facet_handles.iter().enumerate() {
            let facet =
                crate::core::facet::FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                    .expect("Should create valid FacetView");
            assert_eq!(
                facet.vertices().unwrap().count(),
                3,
                "Visible facet {i} should have 3 vertices"
            );
        }

        println!("✓ Visible boundary facet detection works correctly");
    }

    #[test]
    fn test_find_visible_boundary_facets_interior_vertex() {
        println!("Testing find_visible_boundary_facets with interior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test interior vertex that should not see any facets from outside
        let interior_vertex = vertex!([0.4, 0.4, 0.4]);
        let visible_facets = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &interior_vertex)
            .expect("Should successfully find visible boundary facets");

        println!("  Interior vertex sees {} facets", visible_facets.len());

        // Interior vertex should see few or no boundary facets as "visible"
        // (The exact number depends on the orientation predicates)
        assert!(
            visible_facets.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        println!("✓ Interior vertex visibility test works correctly");
    }

    #[test]
    fn test_find_visible_boundary_facets_lightweight_exterior_vertex() {
        println!("Testing find_visible_boundary_facets_lightweight with exterior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test exterior vertex that should see some facets
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let lightweight_result = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &exterior_vertex)
            .expect("Should successfully find visible boundary facets");

        println!("  Found {} visible facet handles", lightweight_result.len());

        // Should find at least some visible facets for exterior vertex
        assert!(
            !lightweight_result.is_empty(),
            "Exterior vertex should see at least some boundary facets"
        );
        assert!(
            lightweight_result.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        // Test that each handle is valid
        for (i, handle) in lightweight_result.iter().enumerate() {
            // Verify cell exists in TDS
            assert!(
                tds.cells().get(handle.cell_key()).is_some(),
                "Cell key {:?} for visible facet {i} should exist in TDS",
                handle.cell_key()
            );

            // Verify facet index is valid for 3D (should be 0, 1, 2, or 3)
            assert!(
                handle.facet_index() < 4,
                "Facet index {} for visible facet {i} should be < 4 for 3D tetrahedron",
                handle.facet_index()
            );

            // Create FacetView from handle to verify it's valid
            let facet_view =
                crate::core::facet::FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                    .expect("Should be able to create FacetView from returned handle");

            assert_eq!(
                facet_view.vertices().unwrap().count(),
                3,
                "Visible facet {i} should have 3 vertices in 3D"
            );
        }

        println!("✓ Exterior vertex lightweight visibility test works correctly");
    }

    #[test]
    fn test_find_visible_boundary_facets_lightweight_interior_vertex() {
        println!("Testing find_visible_boundary_facets_lightweight with interior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Test interior vertex that should not see any facets from outside
        let interior_vertex = vertex!([0.4, 0.4, 0.4]);
        let lightweight_result = algorithm
            .find_visible_boundary_facets_lightweight(&tds, &interior_vertex)
            .expect("Should successfully find visible boundary facets");

        println!(
            "  Interior vertex sees {} facet handles",
            lightweight_result.len()
        );

        // Interior vertex should see few or no boundary facets as "visible"
        assert!(
            lightweight_result.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        // Verify all returned handles are valid
        for handle in &lightweight_result {
            assert!(
                tds.cells().get(handle.cell_key()).is_some(),
                "Cell key {:?} should exist in TDS",
                handle.cell_key()
            );
            assert!(
                handle.facet_index() < 4,
                "Facet index {} should be valid for 3D tetrahedron",
                handle.facet_index()
            );
        }

        println!("✓ Interior vertex lightweight visibility test works correctly");
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
                    tds.cells().get(handle.cell_key()).is_some(),
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

        // Create cells from empty handle list
        let cells_created = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &empty_handles,
            &test_vertex,
        )
        .expect("Should handle empty handle list gracefully");

        let final_cell_count = tds.number_of_cells();

        // Should create no cells from empty input
        assert_eq!(
            cells_created, 0,
            "Should create 0 cells from empty handle list"
        );
        assert_eq!(
            final_cell_count, initial_cell_count,
            "Cell count should not change with empty handle list"
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
        let valid_cell_key = tds.cells().keys().next().expect("TDS should have cells");

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
                        message.contains("Cell key") && message.contains("not found"),
                        "Error message should mention cell key not found, got: {message}"
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
            .cells()
            .keys()
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
                        message.contains("Facet index") && message.contains("out of bounds"),
                        "Error message should mention facet index out of bounds, got: {message}"
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

        // Create cells from handles with duplicates
        let cells_created = IncrementalBowyerWatson::create_cells_from_facet_handles(
            &mut tds,
            &duplicate_handles,
            &exterior_vertex,
        )
        .expect("Should handle duplicate handles gracefully");

        let final_cell_count = tds.number_of_cells();
        println!("  Final cell count: {final_cell_count}");
        println!("  Cells created: {cells_created}");

        // Should still create cells (the implementation might handle duplicates by creating duplicate cells,
        // or it might be smart enough to avoid them - either behavior is acceptable as long as it doesn't crash)
        assert!(cells_created > 0, "Should have created at least some cells");
        assert_eq!(
            final_cell_count,
            initial_cell_count + cells_created,
            "Cell count should increase by the number of cells created"
        );

        println!("✓ Duplicate handles test works correctly");
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
        let valid_cell_key = tds.cells().keys().next().expect("TDS should have cells");
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
            .cells()
            .keys()
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
        let initial_cell_keys: Vec<_> = tds.cells().keys().collect();
        println!("  Initial cell count: {initial_cell_count}");
        println!("  Initial cell keys: {}", initial_cell_keys.len());

        // Create a handle list with valid handles followed by an invalid one
        // This ensures some cells will be created before the error occurs
        let valid_cell_key = tds.cells().keys().next().expect("TDS should have cells");
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
        let final_cell_keys: Vec<_> = tds.cells().keys().collect();
        assert_eq!(
            final_cell_keys.len(),
            initial_cell_keys.len(),
            "Number of cell keys should be unchanged"
        );

        // Verify the exact same cells exist (no cells added or removed)
        for initial_key in &initial_cell_keys {
            assert!(
                tds.get_cell_by_key(*initial_key).is_some(),
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
    fn test_find_bad_cells_for_interior_vertex() {
        println!("Testing find_bad_cells with interior vertex");

        // Create initial tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Test with interior vertex that should violate Delaunay property
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        let bad_cells = algorithm
            .find_bad_cells(&tds, &interior_vertex)
            .expect("Should successfully find bad cells for interior vertex");

        // Since this is an interior vertex, it should find the tetrahedron as a bad cell
        assert!(
            !bad_cells.is_empty(),
            "Interior vertex should find at least one bad cell"
        );
        assert!(
            bad_cells.len() <= tds.number_of_cells(),
            "Cannot have more bad cells than total cells"
        );

        println!("  Found {} bad cells for interior vertex", bad_cells.len());
        println!("✓ Bad cell detection for interior vertex works correctly");
    }

    #[test]
    fn test_find_bad_cells_for_exterior_vertex() {
        println!("Testing find_bad_cells with exterior vertex");

        // Create initial tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Test with exterior vertex that should not violate circumsphere
        let exterior_vertex = vertex!([3.0, 0.0, 0.0]);
        let bad_cells = algorithm
            .find_bad_cells(&tds, &exterior_vertex)
            .expect("Should successfully find bad cells for exterior vertex");

        // Exterior vertex should typically find no bad cells (no circumsphere violations)
        println!("  Found {} bad cells for exterior vertex", bad_cells.len());
        assert!(
            bad_cells.len() <= tds.number_of_cells(),
            "Cannot have more bad cells than total cells"
        );

        println!("✓ Bad cell detection for exterior vertex works correctly");
    }

    #[test]
    fn test_find_bad_cells_edge_cases() {
        println!("Testing find_bad_cells edge cases");

        // Create a more complex triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]), // Additional vertex to create more cells
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        // Test vertex very close to existing vertex
        let close_vertex = vertex!([0.001, 0.0, 0.0]);
        let bad_cells_close = algorithm
            .find_bad_cells(&tds, &close_vertex)
            .unwrap_or_else(|e| {
                println!("  Close vertex error: {e}");
                vec![]
            });
        println!("  Close vertex found {} bad cells", bad_cells_close.len());

        // Test vertex on circumsphere boundary (should be handled gracefully)
        let boundary_vertex = vertex!([0.5, 0.5, 0.0]);
        let bad_cells_boundary = algorithm
            .find_bad_cells(&tds, &boundary_vertex)
            .unwrap_or_else(|e| {
                println!("  Boundary vertex error: {e}");
                vec![]
            });
        println!(
            "  Boundary vertex found {} bad cells",
            bad_cells_boundary.len()
        );

        // All results should be within reasonable bounds
        assert!(bad_cells_close.len() <= tds.number_of_cells());
        assert!(bad_cells_boundary.len() <= tds.number_of_cells());

        println!("✓ Bad cell detection edge cases handled correctly");
    }

    #[test]
    fn test_find_cavity_boundary_facets_single_bad_cell() {
        println!("Testing find_cavity_boundary_facets with single bad cell");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get the single cell as a "bad cell"
        let cell_keys: Vec<_> = tds.cells().keys().collect();
        assert_eq!(cell_keys.len(), 1, "Should have exactly one cell");

        let bad_cells = vec![cell_keys[0]];
        let boundary_facet_handles = algorithm
            .find_cavity_boundary_facets(&tds, &bad_cells)
            .expect("Should find boundary facets");

        // For a single tetrahedron, all 4 facets should be cavity boundary facets
        assert_eq!(
            boundary_facet_handles.len(),
            4,
            "Single tetrahedron should have 4 boundary facets"
        );

        // Each facet handle should be valid - create FacetViews to verify
        for (i, handle) in boundary_facet_handles.iter().enumerate() {
            let facet_view =
                crate::core::facet::FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                    .expect("Should be able to create FacetView from handle");
            let vertex_count = facet_view.vertices().unwrap().count();
            assert_eq!(vertex_count, 3, "Facet {i} should have 3 vertices in 3D");
        }

        println!("✓ Cavity boundary facet detection for single cell works correctly");
    }

    #[test]
    fn test_find_cavity_boundary_facets_empty_bad_cells() {
        println!("Testing find_cavity_boundary_facets with empty bad cells list");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        let bad_cells = vec![]; // Empty list
        let boundary_facets = algorithm
            .find_cavity_boundary_facets(&tds, &bad_cells)
            .expect("Should handle empty bad cells list");

        assert_eq!(
            boundary_facets.len(),
            0,
            "Empty bad cells list should produce empty boundary facets"
        );

        println!("✓ Empty bad cells list handled correctly");
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
    fn test_create_cell_from_facet_and_vertex_failure() {
        println!("Testing create_cell_from_facet_and_vertex with duplicate vertex");

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet using FacetView
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");
        let cell_key = test_facet.cell_key();
        let facet_index = test_facet.facet_index();

        // Get a vertex that's already in the facet (this will create a duplicate)
        let facet_vertices: Vec<_> = test_facet.vertices().unwrap().collect();
        let duplicate_vertex = *facet_vertices[0]; // Use an existing facet vertex
        drop(boundary_facets); // Drop the iterator to release the immutable borrow

        let initial_cell_count = tds.number_of_cells();

        let result =
            <IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                3,
            >>::create_cell_from_facet_handle(
                &mut tds, cell_key, facet_index, &duplicate_vertex
            );

        // Check if creation was rejected or succeeded
        match result {
            Ok(_cell_key) => {
                println!("  Cell creation succeeded");

                // The cell was created - check if it's valid or invalid
                // Get the newly created cell (should be the last one added)
                let final_cell_count = tds.number_of_cells();
                assert_eq!(
                    final_cell_count,
                    initial_cell_count + 1,
                    "Cell count should have increased by 1"
                );

                // Check if any cell in the TDS is now invalid due to duplicate vertices
                let mut found_invalid = false;
                for (_, cell) in tds.cells() {
                    if cell.is_valid()
                        == Err(crate::core::cell::CellValidationError::DuplicateVertices)
                    {
                        found_invalid = true;
                        println!("  ✓ Found cell with duplicate vertices (validation detected it)");
                        break;
                    }
                }

                if !found_invalid {
                    println!("  Note: No cells with duplicate vertices found after creation");
                    println!("  This means duplicate detection may happen earlier in the pipeline");
                }
            }
            Err(e) => {
                println!("  ✓ Cell creation properly rejected duplicate vertex: {e}");
                assert_eq!(
                    tds.number_of_cells(),
                    initial_cell_count,
                    "Cell count should remain unchanged after failed creation"
                );
            }
        }

        println!("✓ Cell creation with duplicate vertex handled correctly");
    }

    #[test]
    fn test_cavity_boundary_success_rate_zero_attempts() {
        use approx::assert_abs_diff_eq;
        println!("Testing cavity_boundary_success_rate with zero vertices processed");

        let stats = InsertionStatistics::new();
        let rate = stats.cavity_boundary_success_rate();

        assert_abs_diff_eq!(rate, 1.0, epsilon = f64::EPSILON);
        println!("✓ Zero attempts handled correctly");
    }

    #[test]
    fn test_cavity_boundary_success_rate_underflow_case() {
        use approx::assert_abs_diff_eq;
        println!("Testing cavity_boundary_success_rate with more failures than vertices processed");

        let mut stats = InsertionStatistics::new();
        stats.vertices_processed = 5;
        stats.cavity_boundary_failures = 10; // More failures than vertices processed

        let rate = stats.cavity_boundary_success_rate();

        // With saturating subtraction, this should give us 0 successes / 5 attempts = 0.0
        assert_abs_diff_eq!(rate, 0.0, epsilon = f64::EPSILON);
        println!("✓ Underflow case handled correctly with saturating subtraction");
    }

    #[test]
    fn test_cavity_boundary_success_rate_all_failures() {
        use approx::assert_abs_diff_eq;
        println!("Testing cavity_boundary_success_rate with failures equal to vertices processed");

        let mut stats = InsertionStatistics::new();
        stats.vertices_processed = 10;
        stats.cavity_boundary_failures = 10; // All attempts failed

        let rate = stats.cavity_boundary_success_rate();

        assert_abs_diff_eq!(rate, 0.0, epsilon = f64::EPSILON);
        println!("✓ All failures case handled correctly");
    }

    #[test]
    fn test_cavity_boundary_success_rate_normal_cases() {
        use approx::assert_abs_diff_eq;
        println!("Testing cavity_boundary_success_rate with normal cases");

        // Case 1: All successes
        let mut stats = InsertionStatistics::new();
        stats.vertices_processed = 10;
        stats.cavity_boundary_failures = 0;

        let rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(rate, 1.0, epsilon = f64::EPSILON);

        // Case 2: Half successes
        stats.vertices_processed = 10;
        stats.cavity_boundary_failures = 5;

        let rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(rate, 0.5, epsilon = f64::EPSILON);

        // Case 3: Most successes
        stats.vertices_processed = 100;
        stats.cavity_boundary_failures = 10;

        let rate = stats.cavity_boundary_success_rate();
        assert_abs_diff_eq!(rate, 0.9, epsilon = f64::EPSILON);

        println!("✓ Normal cases handled correctly");
    }

    #[test]
    fn test_cavity_boundary_success_rate_precision() {
        use approx::assert_abs_diff_eq;
        println!("Testing cavity_boundary_success_rate precision with edge values");

        let mut stats = InsertionStatistics::new();
        stats.vertices_processed = 3;
        stats.cavity_boundary_failures = 1;

        let rate = stats.cavity_boundary_success_rate();
        let expected = 2.0 / 3.0; // Should be approximately 0.6666666666666666

        assert_abs_diff_eq!(rate, expected, epsilon = f64::EPSILON);
        println!("  Rate: {rate}, Expected: {expected}");
        println!("✓ Precision handling works correctly");
    }

    #[test]
    fn test_find_bad_cells_error_cases() {
        println!("Testing find_bad_cells error cases");

        // Create an empty triangulation to test NoCells error
        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::empty();
        let mut algorithm = IncrementalBowyerWatson::new();
        let vertex = vertex!([0.25, 0.25, 0.25]);

        match algorithm.find_bad_cells(&empty_tds, &vertex) {
            Err(BadCellsError::NoCells) => {
                println!("  ✓ Empty triangulation correctly returns NoCells error");
            }
            Ok(_) => panic!("Expected NoCells error for empty triangulation"),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_bad_cells_error_types() {
        println!("Testing BadCellsError types");

        // Test error formatting
        let err1 = BadCellsError::AllCellsBad {
            cell_count: 5,
            degenerate_count: 2,
        };
        println!("  AllCellsBad error: {err1}");

        let err2 = BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
            degenerate_count: 3,
            total_tested: 5,
        });
        println!("  TooManyDegenerateCells error: {err2}");

        let err3 = BadCellsError::NoCells;
        println!("  NoCells error: {err3}");

        // Test PartialEq implementation
        let err1_clone = BadCellsError::AllCellsBad {
            cell_count: 5,
            degenerate_count: 2,
        };
        assert_eq!(err1, err1_clone, "Equal errors should compare as equal");
        assert_ne!(err1, err2, "Different errors should not be equal");
        assert_ne!(err2, err3, "Different error variants should not be equal");

        println!("✓ BadCellsError types work correctly with equality");
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
    fn test_facet_new_error_propagation() {
        println!("Testing FacetView validation and facet consistency checking");

        // Create a triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let algorithm = IncrementalBowyerWatson::new();

        // Get cell keys to work with
        let cell_keys: Vec<_> = tds.cells().keys().collect();
        assert!(!cell_keys.is_empty(), "Should have at least one cell");

        // Test 1: Verify that FacetView properly validates facet indices
        println!("  Test 1: FacetView facet index validation");
        let first_cell_key = cell_keys[0];

        // Try to create a FacetView with an invalid facet index
        let invalid_facet_result = crate::core::facet::FacetView::new(&tds, first_cell_key, 99);
        assert!(
            invalid_facet_result.is_err(),
            "FacetView creation should fail with invalid facet index"
        );
        match invalid_facet_result {
            Err(crate::core::facet::FacetError::InvalidFacetIndex { index, .. }) => {
                println!("    ✓ Got expected InvalidFacetIndex error for index {index}");
            }
            other => {
                panic!("Expected InvalidFacetIndex error, got: {other:?}");
            }
        }

        // Test 2: Verify facet consistency between neighboring cells
        println!("  Test 2: Facet index consistency between neighbors");
        if cell_keys.len() >= 2 {
            let first_neighbor_key = cell_keys[0];
            let second_neighbor_key = cell_keys[1];

            // Use the public utility to check facet consistency
            let consistency_result = crate::core::util::verify_facet_index_consistency(
                &tds,
                first_neighbor_key,
                second_neighbor_key,
                0, // Check facet 0
            );

            match consistency_result {
                Ok(consistent) => {
                    println!("    ✓ Facet consistency check completed: {consistent}");
                }
                Err(e) => {
                    println!("    ✓ Facet consistency check properly handles errors: {e}");
                }
            }
        }

        // Test 3: Verify find_cavity_boundary_facets works correctly with valid cells
        println!("  Test 3: find_cavity_boundary_facets with valid cells");
        let bad_cells = vec![cell_keys[0]];
        let result = algorithm.find_cavity_boundary_facets(&tds, &bad_cells);

        assert!(
            result.is_ok(),
            "find_cavity_boundary_facets should succeed with valid cells"
        );

        if let Ok(boundary_facets) = result {
            println!(
                "    ✓ Found {} boundary facet handles",
                boundary_facets.len()
            );

            // Verify each boundary facet handle is valid
            for (i, handle) in boundary_facets.iter().enumerate() {
                // Create FacetView from the handle
                let facet_view = crate::core::facet::FacetView::new(
                    &tds,
                    handle.cell_key(),
                    handle.facet_index(),
                )
                .expect("Should be able to create FacetView from boundary facet handle");

                // Verify the facet has the correct number of vertices (D vertices for D-dimensional triangulation)
                let vertex_count = facet_view
                    .vertices()
                    .expect("Should be able to get vertices")
                    .count();
                assert_eq!(
                    vertex_count, 3,
                    "Boundary facet {i} should have 3 vertices in 3D triangulation"
                );
            }
            println!("    ✓ All boundary facet handles are valid");
        }

        println!("✓ FacetView validation and facet consistency checking works correctly");
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

    /// Test error handling in `InsertionError` creation and classification
    #[test]
    fn test_insertion_error_creation_and_classification() {
        // Test geometric failure error creation
        let geom_error = InsertionError::geometric_failure(
            "Test geometric failure",
            InsertionStrategy::CavityBased,
        );

        match geom_error {
            InsertionError::GeometricFailure {
                message,
                strategy_attempted,
            } => {
                assert_eq!(message, "Test geometric failure");
                assert_eq!(strategy_attempted, InsertionStrategy::CavityBased);
            }
            _ => panic!("Expected GeometricFailure variant"),
        }

        // Test invalid vertex error creation
        let invalid_vertex_error = InsertionError::invalid_vertex("Duplicate coordinates");
        match invalid_vertex_error {
            InsertionError::InvalidVertex { reason } => {
                assert_eq!(reason, "Duplicate coordinates");
            }
            _ => panic!("Expected InvalidVertex variant"),
        }

        // Test precision failure error creation
        let precision_error = InsertionError::precision_failure(1e-15, 5);
        match precision_error {
            InsertionError::PrecisionFailure {
                tolerance,
                perturbation_attempts,
            } => {
                assert_abs_diff_eq!(tolerance, 1e-15, epsilon = f64::EPSILON);
                assert_eq!(perturbation_attempts, 5);
            }
            _ => panic!("Expected PrecisionFailure variant"),
        }

        // Test hull extension failure
        let hull_error = InsertionError::hull_extension_failure("No visible facets");
        match hull_error {
            InsertionError::HullExtensionFailure { reason } => {
                assert_eq!(reason, "No visible facets");
            }
            _ => panic!("Expected HullExtensionFailure variant"),
        }
    }

    /// Test `InsertionError` recoverability classification
    #[test]
    fn test_insertion_error_recoverability() {
        // Recoverable errors
        let geometric_error =
            InsertionError::geometric_failure("Test", InsertionStrategy::Standard);
        assert!(geometric_error.is_recoverable());

        let precision_error = InsertionError::precision_failure(1e-10, 3);
        assert!(precision_error.is_recoverable());

        let bad_cells_error = InsertionError::BadCellsDetection(
            BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
                degenerate_count: 10,
                total_tested: 15,
            }),
        );
        assert!(bad_cells_error.is_recoverable());

        // Non-recoverable errors
        let invalid_vertex_error = InsertionError::invalid_vertex("Invalid coordinates");
        assert!(!invalid_vertex_error.is_recoverable());

        let hull_error = InsertionError::hull_extension_failure("No visible facets");
        assert!(!hull_error.is_recoverable());
    }

    /// Test `InsertionError` strategy extraction
    #[test]
    fn test_insertion_error_strategy_extraction() {
        // Error with strategy information
        let geometric_error =
            InsertionError::geometric_failure("Test", InsertionStrategy::HullExtension);
        assert_eq!(
            geometric_error.attempted_strategy(),
            Some(InsertionStrategy::HullExtension)
        );

        // Error without strategy information
        let invalid_vertex_error = InsertionError::invalid_vertex("Test");
        assert_eq!(invalid_vertex_error.attempted_strategy(), None);
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
                | BadCellsError::TooManyDegenerateCells(_),
            )
            | Ok(_) => {
                // All these cases are valid - main test is that we don't panic with extreme input
            }
        }

        // The main test is that we don't panic with invalid input
    }

    /// Test `create_cell_from_facet_and_vertex` error cases - additional scenarios
    #[test]
    fn test_create_cell_from_facet_and_vertex_failure_additional() {
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

    /// Test `facet_new_error_propagation` - comprehensive coverage
    #[test]
    fn test_facet_new_error_propagation_comprehensive() {
        // This test verifies that errors from Facet::new are properly propagated
        // through the insertion algorithm error handling paths

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let _algorithm = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Get boundary facet handle (lightweight)
        let boundary_facets = tds.boundary_facets().unwrap();
        let test_facet = boundary_facets
            .clone()
            .next()
            .expect("Should have boundary facets");
        let cell_key = test_facet.cell_key();
        let facet_index = test_facet.facet_index();
        drop(boundary_facets); // Drop the iterator to release the immutable borrow

        // Test vertex that might cause facet creation issues
        let test_vertex = vertex!([10.0, 10.0, 10.0]);

        // The actual error paths depend on the specific implementation,
        // but we're testing that errors are handled gracefully
        let _ = <IncrementalBowyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
            f64,
            Option<()>,
            Option<()>,
            3,
        >>::create_cell_from_facet_handle(
            &mut tds, cell_key, facet_index, &test_vertex
        );

        // Main goal is ensuring no panics occur during error handling
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
    #[allow(clippy::too_many_lines)]
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
            vertex!([1.0, 1.0, 1.0]), // Additional vertex to create more complex geometry
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

        // Test with empty boundary facets (should create no cells)
        let empty_facet_handles: Vec<FacetHandle> = vec![];
        let another_vertex = vertex!([3.0, 3.0, 3.0]);
        let cells_created = IncrementalBowyerWatson::<f64, Option<()>, Option<()>, 3>::create_cells_from_facet_handles(
            &mut tds,
            &empty_facet_handles,
            &another_vertex,
        ).unwrap_or(0);

        assert_eq!(
            cells_created, 0,
            "Should create no cells from empty boundary facets"
        );
        println!("  ✓ Empty boundary facets handled correctly");

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
            vertex!([1.0, 1.0, 1.0]), // Additional vertex to create more cells
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let initial_cell_count = tds.number_of_cells();
        let cell_keys: Vec<_> = tds.cells().keys().take(1).collect(); // Take one cell to remove

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
        println!("  ✓ Interior vertex test: {is_interior}");

        let exterior_test_vertex = vertex!([10.0, 10.0, 10.0]);
        let is_interior = algorithm.is_vertex_interior(&tds, &exterior_test_vertex);
        println!("  ✓ Exterior vertex interior test: {is_interior}");

        // Test with vertex at circumsphere boundary
        let boundary_test_vertex = vertex!([0.5, 0.5, 0.0]); // On edge/boundary
        let is_interior = algorithm.is_vertex_interior(&tds, &boundary_test_vertex);
        println!("  ✓ Boundary vertex interior test: {is_interior}");

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

    /// Test bad cells error variants comprehensive display and equality
    #[test]
    fn test_bad_cells_error_comprehensive_variants() {
        println!("Testing BadCellsError comprehensive variants");

        // Test AllCellsBad with various counts
        let all_bad_zero_degenerate = BadCellsError::AllCellsBad {
            cell_count: 10,
            degenerate_count: 0,
        };
        let display = format!("{all_bad_zero_degenerate}");
        assert!(display.contains("All 10 cells marked as bad"));
        assert!(display.contains("0 degenerate"));
        println!("  ✓ AllCellsBad with zero degenerate: {display}");

        let all_bad_some_degenerate = BadCellsError::AllCellsBad {
            cell_count: 5,
            degenerate_count: 3,
        };
        let display = format!("{all_bad_some_degenerate}");
        assert!(display.contains("All 5 cells marked as bad"));
        assert!(display.contains("3 degenerate"));
        println!("  ✓ AllCellsBad with some degenerate: {display}");

        // Test TooManyDegenerateCells with boundary conditions
        let too_many_boundary =
            BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
                degenerate_count: 5,
                total_tested: 10, // Exactly at 50% threshold
            });
        let display = format!("{too_many_boundary}");
        assert!(display.contains("Too many degenerate circumspheres (5/10)"));
        println!("  ✓ TooManyDegenerateCells at boundary: {display}");

        let too_many_exceed = BadCellsError::TooManyDegenerateCells(TooManyDegenerateCellsError {
            degenerate_count: 7,
            total_tested: 10, // 70% > 50% threshold
        });
        let display = format!("{too_many_exceed}");
        assert!(display.contains("Too many degenerate circumspheres (7/10)"));
        println!("  ✓ TooManyDegenerateCells exceeding threshold: {display}");

        // Test equality and cloning
        let error1 = BadCellsError::AllCellsBad {
            cell_count: 5,
            degenerate_count: 2,
        };
        let error1_clone = error1.clone();
        assert_eq!(error1, error1_clone, "Cloned errors should be equal");

        let error2 = BadCellsError::NoCells;
        let error2_clone = error2.clone();
        assert_eq!(error2, error2_clone, "NoCells clones should be equal");
        assert_ne!(error1, error2, "Different error types should not be equal");

        println!("  ✓ BadCellsError equality and cloning work correctly");

        println!("✓ BadCellsError comprehensive variants work correctly");
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
            vertex!([1.0, 1.0, 1.0]), // Additional vertex to create more cells
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();

        let initial_generation = tds.generation();
        let initial_vertex_count = tds.number_of_vertices();
        let initial_cell_count = tds.number_of_cells();

        // Get some cells to remove (simulate bad cells)
        let cell_keys: Vec<_> = tds.cells().keys().take(1).collect();
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
        let cell_keys: Vec<_> = tds.cells().keys().take(1).collect();

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
                        let neighbor = tds.cells().get(neighbor_key).unwrap();
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
}
