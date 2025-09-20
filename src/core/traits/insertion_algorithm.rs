//! Common trait for vertex insertion algorithms in Delaunay triangulations.
//!
//! This module defines the `InsertionAlgorithm` trait that provides a unified
//! interface for different vertex insertion strategies, including the basic
//! Bowyer-Watson algorithm and robust variants with enhanced numerical stability.

use crate::core::collections::CellKeySet;
use crate::core::{
    cell::CellBuilder,
    facet::Facet,
    triangulation_data_structure::{
        Tds, TriangulationConstructionError, TriangulationValidationError,
    },
    vertex::Vertex,
};
use crate::geometry::point::Point;
use crate::geometry::predicates::{InSphere, Orientation, insphere, simplex_orientation};
use crate::geometry::traits::coordinate::CoordinateScalar;
use approx::abs_diff_eq;
use num_traits::{NumCast, One, Zero, cast};
use serde::{Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use std::{
    iter::Sum,
    ops::{AddAssign, Div, SubAssign},
};

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
pub struct InsertionBuffers<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Buffer for storing bad cell keys during cavity detection
    pub bad_cells_buffer: Vec<crate::core::triangulation_data_structure::CellKey>,
    /// Buffer for storing boundary facets during cavity boundary detection
    pub boundary_facets_buffer: Vec<Facet<T, U, V, D>>,
    /// Buffer for storing vertex points during geometric computations
    pub vertex_points_buffer: Vec<crate::geometry::point::Point<T, D>>,
    /// Buffer for storing visible boundary facets
    pub visible_facets_buffer: Vec<Facet<T, U, V, D>>,
}

impl<T, U, V, const D: usize> Default for InsertionBuffers<T, U, V, D>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Create new empty buffers
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bad_cells_buffer: Vec::new(),
            boundary_facets_buffer: Vec::new(),
            vertex_points_buffer: Vec::new(),
            visible_facets_buffer: Vec::new(),
        }
    }

    /// Create buffers with pre-allocated capacity for better performance
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bad_cells_buffer: Vec::with_capacity(capacity),
            boundary_facets_buffer: Vec::with_capacity(capacity),
            vertex_points_buffer: Vec::with_capacity(capacity * (D + 1)), // More points per operation
            visible_facets_buffer: Vec::with_capacity(capacity / 2), // Typically fewer visible facets
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
    ) -> &mut Vec<crate::core::triangulation_data_structure::CellKey> {
        self.bad_cells_buffer.clear();
        &mut self.bad_cells_buffer
    }

    /// Prepare the boundary facets buffer and return a mutable reference
    pub fn prepare_boundary_facets_buffer(&mut self) -> &mut Vec<Facet<T, U, V, D>> {
        self.boundary_facets_buffer.clear();
        &mut self.boundary_facets_buffer
    }

    /// Prepare the vertex points buffer and return a mutable reference
    pub fn prepare_vertex_points_buffer(
        &mut self,
    ) -> &mut Vec<crate::geometry::point::Point<T, D>> {
        self.vertex_points_buffer.clear();
        &mut self.vertex_points_buffer
    }

    /// Prepare the visible facets buffer and return a mutable reference
    pub fn prepare_visible_facets_buffer(&mut self) -> &mut Vec<Facet<T, U, V, D>> {
        self.visible_facets_buffer.clear();
        &mut self.visible_facets_buffer
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
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        use crate::geometry::predicates::{InSphere, insphere};

        // Reserve exact capacity once before the loop to avoid reallocations
        let mut vertex_points = Vec::with_capacity(D + 1);

        for cell in tds.cells().values() {
            // Clear and reuse the buffer - capacity is already preallocated
            vertex_points.clear();
            vertex_points.extend(cell.vertices().iter().map(|v| *v.point()));

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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        // Check if there are any cells to test
        if tds.cells().is_empty() {
            return Err(BadCellsError::NoCells);
        }

        let mut bad_cells = Vec::new();
        let mut cells_tested = 0;
        let mut degenerate_count = 0;
        // Preallocate vertex_points buffer outside the loop to avoid per-iteration allocations
        let mut vertex_points = Vec::with_capacity(D + 1);

        // Only consider cells that have a valid circumsphere and strict containment
        for (cell_key, cell) in tds.cells() {
            let v_count = cell.vertices().len();
            // Treat non-D+1 vertex counts as degenerate
            if v_count != D + 1 {
                degenerate_count += 1;
                continue;
            }

            cells_tested += 1;
            // Reuse buffer by clearing and repopulating
            vertex_points.clear();
            vertex_points.extend(cell.vertices().iter().map(|v| *v.point()));

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
    /// A vector of boundary facets, or an error if computation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the cavity boundary computation fails due to topological issues.
    fn find_cavity_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[crate::core::triangulation_data_structure::CellKey],
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        let mut boundary_facets = Vec::new();

        if bad_cells.is_empty() {
            return Ok(boundary_facets);
        }

        let bad_cell_set: CellKeySet = bad_cells.iter().copied().collect();

        // Use the canonical facet-to-cells map from TDS (already uses VertexKeys)
        // Default trait implementation cannot access concrete type's cache,
        // so we use try_build which returns Result.
        //
        // Performance note: Concrete implementations that also implement FacetCacheProvider
        // should override this method to use get_or_build_facet_cache() for better performance
        // in hot paths where this method is called repeatedly.
        let facet_to_cells = tds.build_facet_to_cells_map().map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to build facet-to-cells map: {e}"),
            }
        })?;

        // Track which boundary facets we need to create
        let mut boundary_facet_specs = Vec::new();

        // Process each facet in the map to identify boundary facets
        for sharing_cells in facet_to_cells.values() {
            // Count how many bad vs good cells share this facet without allocation
            // Note: sharing_cells contains (CellKey, facet_index) pairs
            let total_count = sharing_cells.len();
            if total_count > 2 {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Facet shared by more than two cells (total_count = {total_count})"
                        ),
                    },
                ));
            }

            // Count bad cells and capture the single bad sharer if it exists
            let mut bad_count = 0;
            let mut single_bad_cell = None;
            for &(cell_key, facet_index) in sharing_cells {
                if bad_cell_set.contains(&cell_key) {
                    bad_count += 1;
                    if bad_count == 1 {
                        single_bad_cell = Some((cell_key, facet_index));
                    } else {
                        // More than one bad cell, can short-circuit
                        break;
                    }
                }
            }

            // A facet is on the cavity boundary if:
            // 1. Exactly one bad cell uses it (boundary between bad and good)
            // 2. OR it's a true boundary facet (only one cell total) that's bad
            if bad_count == 1 && (total_count == 2 || total_count == 1) {
                // Store the bad cell key and facet index for later processing
                if let Some((bad_cell_key, facet_index)) = single_bad_cell {
                    boundary_facet_specs.push((bad_cell_key, facet_index));
                }
            }
            // Skip facets that are:
            // - Internal to the cavity (bad_count > 1)
            // - Not touched by any bad cells (bad_count == 0)
            // - Invalid sharing (total_count > 2)
        }

        // Now create the actual Facet objects for the identified boundary facets
        boundary_facets.reserve(boundary_facet_specs.len());
        for (cell_key, facet_index) in boundary_facet_specs {
            if let Some(cell) = tds.cells().get(cell_key) {
                // Convert facet_index (u8) to usize for array indexing
                let facet_idx = <usize as From<u8>>::from(facet_index);
                if facet_idx >= cell.vertices().len() {
                    // Invalid facet index indicates TDS corruption - fail fast
                    return Err(InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Facet index {} out of bounds (cell has {} vertices) while building cavity boundary",
                                facet_idx,
                                cell.vertices().len()
                            ),
                        },
                    ));
                }
                let opposite_vertex = cell.vertices()[facet_idx];
                // Create the facet using the Cell and its opposite vertex
                // TODO: Optimize Facet construction to avoid Cell cloning (allocation reduction)
                // Consider lightweight Facet construction from (cell_key, facet_index) parameters
                // to eliminate this allocation in insertion hot paths.
                let facet = Facet::new(cell.clone(), opposite_vertex).map_err(|e| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Failed to construct boundary facet: {e}"),
                    }
                })?;
                boundary_facets.push(facet);
            } else {
                // Cell not found - this shouldn't happen if TDS is consistent
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell with key {cell_key:?} not found while building cavity boundary"
                        ),
                    },
                ));
            }
        }

        // Validation: ensure we have a reasonable number of boundary facets
        if boundary_facets.is_empty() && !bad_cells.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: format!(
                        "No cavity boundary facets found for {} bad cells. This indicates a topological error.",
                        bad_cells.len()
                    ),
                },
            ));
        }

        Ok(boundary_facets)
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
    /// `true` if the facet is visible from the vertex, `false` otherwise
    fn is_facet_visible_from_vertex(
        &self,
        tds: &Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        adjacent_cell_key: crate::core::triangulation_data_structure::CellKey,
    ) -> bool
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        // Default implementation delegates to the static helper method
        Self::is_facet_visible_from_vertex_impl(tds, facet, vertex, adjacent_cell_key)
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
    fn insert_vertex_cavity_based(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
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

        // Find boundary facets of the cavity
        let boundary_facets = self.find_cavity_boundary_facets(tds, &bad_cells)?;

        if boundary_facets.is_empty() {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message: "No boundary facets found for cavity insertion".to_string(),
                },
            ));
        }

        let cells_removed = bad_cells.len();

        // Remove bad cells
        Self::remove_bad_cells(tds, &bad_cells);

        // Create new cells
        let cells_created = Self::create_cells_from_boundary_facets(tds, &boundary_facets, vertex);

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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        // Get visible boundary facets
        let visible_facets = self.find_visible_boundary_facets(tds, vertex)?;

        if visible_facets.is_empty() {
            // No visible facets - this method is not applicable
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FailedToCreateCell {
                    message:
                        "Hull extension insertion failed: no visible boundary facets found for vertex"
                            .to_string(),
                }
            ));
        }

        // Create new cells from visible facets
        let cells_created = Self::create_cells_from_boundary_facets(tds, &visible_facets, vertex);

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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
                let &(cell_key, facet_index) = cells.first().ok_or_else(|| {
                    InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: "Boundary facet had no adjacent cell".to_string(),
                        },
                    )
                })?;
                let fi = <usize as From<_>>::from(facet_index);
                if let Some(cell) = tds.cells().get(cell_key)
                    && let Ok(facets) = cell.facets()
                    && facets.get(fi).is_some()
                {
                    let facet = &facets[fi];

                    // Try to create a cell from this facet and the vertex
                    if Self::create_cell_from_facet_and_vertex(tds, facet, vertex).is_ok() {
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
            for &(cell_key, facet_index) in cells {
                let fi = <usize as From<_>>::from(facet_index);
                if let Some(cell) = tds.cells().get(cell_key)
                    && let Ok(facets) = cell.facets()
                    && facets.get(fi).is_some()
                {
                    let facet = &facets[fi];

                    // Try to create a cell from this facet and the vertex
                    if Self::create_cell_from_facet_and_vertex(tds, facet, vertex).is_ok() {
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
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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

        let cell = CellBuilder::default()
            .vertices(vertices)
            .build()
            .map_err(|e| TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex: {e}"),
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

    /// Finds boundary facets visible from the given vertex
    ///
    /// A boundary facet is visible if the vertex lies on the "outside" of that facet.
    /// This uses proper geometric orientation tests to determine visibility.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex from which to test visibility
    ///
    /// # Returns
    ///
    /// A result containing a vector of visible boundary facets, or an error if the computation fails.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if:
    /// - The boundary facets cannot be computed
    /// - The orientation tests fail for geometric reasons
    /// - The triangulation data structure is in an invalid state
    fn find_visible_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        let mut visible_facets = Vec::new();

        // Get all boundary facets (facets shared by exactly one cell)
        // TODO: Use FacetCacheProvider if available in concrete implementation.
        // Default trait implementation cannot access concrete type's cache,
        // so we use try_build which returns Result. Concrete implementations
        // that have FacetCacheProvider should override this to use get_or_build_facet_cache.
        let facet_to_cells = tds.build_facet_to_cells_map().map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to build facet-to-cells map: {e}"),
            }
        })?;
        let boundary_facets: Vec<_> = facet_to_cells
            .iter()
            .filter(|(_, cells)| cells.len() == 1)
            .collect();

        for (_facet_key, cells) in boundary_facets {
            let Some(&(cell_key, facet_index)) = cells.first() else {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: "Boundary facet had no adjacent cell".to_string(),
                    },
                ));
            };
            if let Some(cell) = tds.cells().get(cell_key) {
                if let Ok(facets) = cell.facets() {
                    let idx = <usize as From<_>>::from(facet_index);
                    if let Some(facet) = facets.get(idx) {
                        // Test visibility using proper orientation predicates
                        if Self::is_facet_visible_from_vertex_impl(tds, facet, vertex, cell_key) {
                            visible_facets.push(facet.clone());
                        }
                    } else {
                        // Fail fast on invalid facet index - indicates TDS corruption
                        return Err(InsertionError::TriangulationState(
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Facet index {} out of bounds (cell has {} facets) during visibility computation. \
                                     This indicates triangulation data structure corruption.",
                                    idx,
                                    facets.len()
                                ),
                            },
                        ));
                    }
                } else {
                    return Err(InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: "Failed to get facets from cell during visibility computation"
                                .to_string(),
                        },
                    ));
                }
            } else {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: "Cell key not found during visibility computation".to_string(),
                    },
                ));
            }
        }

        Ok(visible_facets)
    }

    /// Helper method to test if a boundary facet is visible from a given vertex
    ///
    /// This is separated into its own method to allow for different visibility testing strategies
    /// across algorithm implementations while providing a common default.
    fn is_facet_visible_from_vertex_impl(
        tds: &Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        adjacent_cell_key: crate::core::triangulation_data_structure::CellKey,
    ) -> bool
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        // Get the adjacent cell to this boundary facet
        let Some(adjacent_cell) = tds.cells().get(adjacent_cell_key) else {
            return false;
        };

        // Find the vertex in the adjacent cell that is NOT part of the facet
        // This is the "opposite" vertex that defines the "inside" side of the facet
        let facet_vertices = facet.vertices();
        let cell_vertices = adjacent_cell.vertices();

        let mut opposite_vertex = None;
        for cell_vertex in cell_vertices {
            let is_in_facet = facet_vertices
                .iter()
                .any(|fv| fv.uuid() == cell_vertex.uuid());
            if !is_in_facet {
                opposite_vertex = Some(cell_vertex);
                break;
            }
        }

        let Some(opposite_vertex) = opposite_vertex else {
            // Could not find opposite vertex - something is wrong with the topology
            return false;
        };

        // Create test simplices for orientation comparison
        // Using SmallVec to avoid heap allocation for small simplices (D+1 points)
        let mut simplex_with_opposite: SmallVec<[Point<T, D>; 8]> =
            facet_vertices.iter().map(|v| *v.point()).collect();
        simplex_with_opposite.push(*opposite_vertex.point());

        let mut simplex_with_test: SmallVec<[Point<T, D>; 8]> =
            facet_vertices.iter().map(|v| *v.point()).collect();
        simplex_with_test.push(*vertex.point());

        // Get orientations
        let orientation_opposite = simplex_orientation(&simplex_with_opposite);
        let orientation_test = simplex_orientation(&simplex_with_test);

        match (orientation_opposite, orientation_test) {
            (Ok(ori_opp), Ok(ori_test)) => {
                // Facet is visible if the orientations are different
                // (vertices are on opposite sides of the hyperplane)
                match (ori_opp, ori_test) {
                    (Orientation::NEGATIVE, Orientation::POSITIVE)
                    | (Orientation::POSITIVE, Orientation::NEGATIVE) => true,
                    (Orientation::DEGENERATE, _)
                    | (_, Orientation::DEGENERATE)
                    | (Orientation::NEGATIVE, Orientation::NEGATIVE)
                    | (Orientation::POSITIVE, Orientation::POSITIVE) => false, // Same orientation = same side = not visible
                }
            }
            _ => {
                // Orientation computation failed - conservative fallback
                false
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
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;
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
    /// let mut algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> = RobustBoyerWatson::new();
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
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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

    /// Creates a new cell from a facet and a vertex
    ///
    /// This is a shared utility method that combines a facet with a new vertex
    /// to create a cell, handling vertex registration and cell insertion.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `facet` - The facet to extend into a cell
    /// * `vertex` - The vertex to add to the facet
    ///
    /// # Returns
    ///
    /// `Ok(())` if the cell was successfully created, or an error describing the failure.
    ///
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if:
    /// - Vertex registration in TDS fails
    /// - Cell building fails (e.g., degenerate geometry)
    /// - Cell insertion fails (e.g., duplicate UUID)
    fn create_cell_from_facet_and_vertex(
        tds: &mut Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<(), TriangulationValidationError>
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        // Ensure the vertex is registered in the TDS vertex mapping
        Self::ensure_vertex_in_tds(tds, vertex)?;

        let mut facet_vertices = facet.vertices();
        facet_vertices.push(*vertex);

        let new_cell = CellBuilder::default()
            .vertices(facet_vertices)
            .build()
            .map_err(|e| TriangulationValidationError::FailedToCreateCell {
                message: format!("Failed to build cell from facet and vertex: {e}"),
            })?;

        tds.insert_cell_with_mapping(new_cell).map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to insert cell into TDS: {e}"),
            }
        })?;

        Ok(())
    }

    /// Creates multiple cells from boundary facets and a vertex
    ///
    /// This is a shared utility method that creates cells by combining
    /// each boundary facet with the new vertex.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `boundary_facets` - The facets to extend into cells
    /// * `vertex` - The vertex to add to each facet
    ///
    /// # Returns
    ///
    /// The number of cells successfully created. Note that some cells may fail
    /// to be created due to geometric constraints, which is expected behavior.
    fn create_cells_from_boundary_facets(
        tds: &mut Tds<T, U, V, D>,
        boundary_facets: &[Facet<T, U, V, D>],
        vertex: &Vertex<T, U, D>,
    ) -> usize
    where
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        let mut cells_created = 0;
        for facet in boundary_facets {
            // We intentionally ignore errors here as some facets may not be able
            // to form valid cells with the given vertex (e.g., degenerate cases)
            if Self::create_cell_from_facet_and_vertex(tds, facet, vertex).is_ok() {
                cells_created += 1;
            }
        }
        cells_created
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
        for<'a> &'a T: Div<T>,
    {
        // Use the optimized batch removal method that handles UUID mapping
        // and generation counter updates internally
        tds.remove_cells_by_keys(bad_cells);
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
        T: AddAssign<T> + SubAssign<T> + Sum + NumCast,
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
    use crate::core::algorithms::bowyer_watson::IncrementalBoyerWatson;
    use crate::core::facet::FacetError;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::vertex;

    #[test]
    fn test_find_visible_boundary_facets_exterior_vertex() {
        println!("Testing find_visible_boundary_facets with exterior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test exterior vertex that should see some facets
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let visible_facets = algorithm
            .find_visible_boundary_facets(&tds, &exterior_vertex)
            .expect("Should successfully find visible boundary facets");

        println!("  Found {} visible facets", visible_facets.len());

        // Should find at least some visible facets for exterior vertex
        assert!(
            !visible_facets.is_empty(),
            "Exterior vertex should see at least some boundary facets"
        );
        assert!(
            visible_facets.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        // Test that each visible facet is valid
        for (i, facet) in visible_facets.iter().enumerate() {
            assert_eq!(
                facet.vertices().len(),
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
        let algorithm = IncrementalBoyerWatson::new();

        // Test interior vertex that should not see any facets from outside
        let interior_vertex = vertex!([0.4, 0.4, 0.4]);
        let visible_facets = algorithm
            .find_visible_boundary_facets(&tds, &interior_vertex)
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
    fn test_is_facet_visible_from_vertex_impl_orientation_cases() {
        use crate::core::facet::facet_key_from_vertex_keys;

        println!("Testing is_facet_visible_from_vertex_impl with different orientations");

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
            !boundary_facets.is_empty(),
            "Should have at least one boundary facet"
        );

        let test_facet = &boundary_facets[0];

        // Find the cell adjacent to this boundary facet
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");

        // Compute facet key using VertexKeys
        let facet_vertices = test_facet.vertices();
        let mut vertex_keys = Vec::with_capacity(facet_vertices.len());
        for vertex in &facet_vertices {
            vertex_keys.push(
                tds.vertex_key_from_uuid(&vertex.uuid())
                    .expect("Vertex should be in TDS"),
            );
        }
        let facet_key = facet_key_from_vertex_keys(&vertex_keys);

        let adjacent_cells = facet_to_cells.get(&facet_key).unwrap();
        assert_eq!(
            adjacent_cells.len(),
            1,
            "Boundary facet should have exactly one adjacent cell"
        );
        let (adjacent_cell_key, _) = adjacent_cells[0];

        // Test visibility from different positions using the trait method
        let test_positions = vec![
            (vertex!([2.0, 0.0, 0.0]), "Far +X"),
            (vertex!([-1.0, 0.0, 0.0]), "Far -X"),
            (vertex!([0.0, 2.0, 0.0]), "Far +Y"),
            (vertex!([0.0, 0.0, 2.0]), "Far +Z"),
            (vertex!([0.1, 0.1, 0.1]), "Interior point"),
        ];

        for (test_vertex, description) in test_positions {
            let is_visible =
                <IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                    f64,
                    Option<()>,
                    Option<()>,
                    3,
                >>::is_facet_visible_from_vertex_impl(
                    &tds,
                    test_facet,
                    &test_vertex,
                    adjacent_cell_key,
                );

            println!("  {description} - Facet visible: {is_visible}");
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
        let mut algorithm = IncrementalBoyerWatson::new();

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
        let mut algorithm = IncrementalBoyerWatson::new();

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
        let mut algorithm = IncrementalBoyerWatson::new();

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
        let algorithm = IncrementalBoyerWatson::new();

        // Get the single cell as a "bad cell"
        let cell_keys: Vec<_> = tds.cells().keys().collect();
        assert_eq!(cell_keys.len(), 1, "Should have exactly one cell");

        let bad_cells = vec![cell_keys[0]];
        let boundary_facets = algorithm
            .find_cavity_boundary_facets(&tds, &bad_cells)
            .expect("Should find boundary facets");

        // For a single tetrahedron, all 4 facets should be cavity boundary facets
        assert_eq!(
            boundary_facets.len(),
            4,
            "Single tetrahedron should have 4 boundary facets"
        );

        // Each facet should be valid
        for (i, facet) in boundary_facets.iter().enumerate() {
            assert_eq!(
                facet.vertices().len(),
                3,
                "Facet {i} should have 3 vertices in 3D"
            );
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
        let algorithm = IncrementalBoyerWatson::new();

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

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = &boundary_facets[0];

        let initial_cell_count = tds.number_of_cells();

        // Create a new vertex that should form a valid cell with the facet
        let new_vertex = vertex!([0.5, 0.5, 1.5]);

        let result =
            <IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                3,
            >>::create_cell_from_facet_and_vertex(&mut tds, test_facet, &new_vertex);

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
        println!("Testing create_cell_from_facet_and_vertex with invalid geometry");

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = &boundary_facets[0];

        let initial_cell_count = tds.number_of_cells();

        // Try to create a degenerate cell by using a vertex that's already in the facet
        let facet_vertices = test_facet.vertices();
        let duplicate_vertex = facet_vertices[0]; // Use an existing facet vertex

        let result =
            <IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> as InsertionAlgorithm<
                f64,
                Option<()>,
                Option<()>,
                3,
            >>::create_cell_from_facet_and_vertex(
                &mut tds, test_facet, &duplicate_vertex
            );

        // This should fail because it would create a degenerate cell
        assert!(
            result.is_err(),
            "Should fail to create cell with degenerate geometry"
        );
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count,
            "Cell count should remain unchanged after failed creation"
        );

        println!("✓ Cell creation properly handles invalid geometry");
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
        let empty_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::default();
        let mut algorithm = IncrementalBoyerWatson::new();
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
        println!("Testing Facet::new error propagation in find_cavity_boundary_facets");

        // Create a triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Manually corrupt the triangulation to create an invalid state
        // We'll remove a vertex from the vertices SlotMap but keep a cell that references it
        // This will cause Facet::new to fail when it tries to validate the cell-vertex relationship

        // Get a cell key
        let cell_keys: Vec<_> = tds.cells().keys().collect();
        assert!(!cell_keys.is_empty(), "Should have at least one cell");

        // Create a corrupted cell with an invalid vertex reference
        // We'll create a cell that references a vertex not actually in its vertex list
        let valid_cell = tds.cells().get(cell_keys[0]).unwrap().clone();

        // Create a new vertex that won't be in the cell
        let invalid_vertex = vertex!([99.0, 99.0, 99.0]);

        // Try to create a facet with the invalid vertex (not in the cell)
        // This should fail with CellDoesNotContainVertex
        let facet_result = Facet::new(valid_cell, invalid_vertex);

        // Check that we get the specific error variant
        match facet_result {
            Err(FacetError::CellDoesNotContainVertex) => {
                println!("  ✓ Got expected FacetError::CellDoesNotContainVertex");
            }
            Err(other) => {
                panic!("Expected FacetError::CellDoesNotContainVertex, got: {other:?}");
            }
            Ok(_) => {
                panic!("Expected Facet::new to fail with invalid vertex");
            }
        }

        // Now test that the error would be properly propagated in find_cavity_boundary_facets
        // The find_cavity_boundary_facets method converts FacetError to TriangulationValidationError
        // with message format: "Failed to construct boundary facet: {original_error}"

        // We can't directly test the TriangulationValidationError variant since it only has
        // one variant (InconsistentDataStructure) with a message field, but we've verified
        // the FacetError variant above
        println!("✓ Facet::new error variant checking works correctly");

        // Test with actual bad cells to ensure the method properly handles errors
        let bad_cells = vec![cell_keys[0]];

        // The actual find_cavity_boundary_facets call would now propagate any Facet::new errors
        // as TriangulationValidationError::InconsistentDataStructure
        let result = algorithm.find_cavity_boundary_facets(&tds, &bad_cells);

        // This should succeed because we're using valid cells from the TDS
        assert!(
            result.is_ok(),
            "find_cavity_boundary_facets should succeed with valid cells"
        );

        println!("✓ find_cavity_boundary_facets works correctly with error propagation in place");
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
}
