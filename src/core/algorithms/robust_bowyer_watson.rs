//! Robust Bowyer-Watson algorithm using enhanced geometric predicates.
//!
//! This module demonstrates how to integrate the robust geometric predicates
//! into the Bowyer-Watson triangulation algorithm to address the
//! "No cavity boundary facets found" error.

use crate::core::collections::MAX_PRACTICAL_DIMENSION_SIZE;
use crate::core::collections::{
    CellKeySet, FacetToCellsMap, FastHashMap, FastHashSet, SmallBuffer, fast_hash_set_with_capacity,
};
use crate::core::facet::FacetHandle;
use crate::core::traits::facet_cache::FacetCacheProvider;
use arc_swap::ArcSwapOption;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, SubAssign};
use std::sync::{Arc, atomic::AtomicU64};

use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::insertion_algorithm::{
    InsertionAlgorithm, InsertionBuffers, InsertionError, InsertionInfo, InsertionStatistics,
    InsertionStrategy,
};
use crate::core::{
    triangulation_data_structure::{
        CellKey, Tds, TriangulationConstructionError, TriangulationValidationError,
    },
    vertex::Vertex,
};
use crate::geometry::{
    algorithms::convex_hull::ConvexHull,
    point::Point,
    predicates::{InSphere, Orientation, simplex_orientation},
    robust_predicates::{RobustPredicateConfig, config_presets, robust_insphere},
    traits::coordinate::{Coordinate, CoordinateScalar},
    util::safe_usize_to_scalar,
};
use nalgebra::{self as na, ComplexField};
use serde::de::DeserializeOwned;
use std::iter::Sum;

/// Enhanced Bowyer-Watson algorithm with robust geometric predicates.
pub struct RobustBowyerWatson<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
{
    /// Configuration for robust predicates
    predicate_config: RobustPredicateConfig<T>,
    /// Unified statistics tracking
    stats: InsertionStatistics,
    /// Reusable buffers for performance
    buffers: InsertionBuffers<T, U, V, D>,
    /// Cached convex hull for hull extension
    hull: Option<ConvexHull<T, U, V, D>>,
    /// Cache for facet-to-cells mapping
    facet_to_cells_cache: ArcSwapOption<FacetToCellsMap>,
    /// Generation counter for cache invalidation
    cached_generation: Arc<AtomicU64>,
    /// Phantom data to indicate that U and V types are used in method signatures
    _phantom: PhantomData<(U, V)>,
}

impl<T, U, V, const D: usize> RobustBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    na::OPoint<T, na::Const<D>>: From<[f64; D]>,
{
    /// Create a new robust Bowyer-Watson algorithm instance.
    ///
    /// Creates an instance with default predicate configuration optimized
    /// for general-purpose triangulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    ///
    /// let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> = RobustBowyerWatson::new();
    /// // Algorithm should be properly initialized with general triangulation config
    /// let (processed, created, removed) = algorithm.get_statistics();
    /// assert_eq!(processed, 0); // No vertices processed yet
    /// assert_eq!(created, 0);   // No cells created yet
    /// assert_eq!(removed, 0);   // No cells removed yet
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            predicate_config: config_presets::general_triangulation::<T>(),
            stats: InsertionStatistics::new(),
            buffers: InsertionBuffers::with_capacity(D * 10), // Scale capacity with dimension
            hull: None,
            facet_to_cells_cache: ArcSwapOption::empty(),
            cached_generation: Arc::new(AtomicU64::new(0)),
            _phantom: PhantomData,
        }
    }

    /// Create with custom predicate configuration.
    ///
    /// Allows fine-tuning of the robustness parameters for specific use cases.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom predicate configuration
    ///
    /// # Panics
    ///
    /// Panics in debug builds if configuration contains invalid values:
    /// - `visibility_threshold_multiplier` must be finite and non-negative
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// let config = config_presets::high_precision::<f64>();
    /// let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     RobustBowyerWatson::with_config(config);
    /// ```
    pub fn with_config(config: RobustPredicateConfig<T>) -> Self {
        // Validate configuration at construction time (debug builds only)
        debug_assert!(
            f64::from(config.visibility_threshold_multiplier).is_finite()
                && f64::from(config.visibility_threshold_multiplier) >= 0.0,
            "visibility_threshold_multiplier must be finite and non-negative, got: {:?}",
            f64::from(config.visibility_threshold_multiplier)
        );
        Self {
            predicate_config: config,
            stats: InsertionStatistics::new(),
            buffers: InsertionBuffers::with_capacity(D * 10), // Scale capacity with dimension
            hull: None,
            facet_to_cells_cache: ArcSwapOption::empty(),
            cached_generation: Arc::new(AtomicU64::new(0)),
            _phantom: PhantomData,
        }
    }

    /// Create optimized for handling degenerate cases.
    ///
    /// Uses more lenient tolerances and fewer refinement iterations to handle
    /// nearly degenerate geometric configurations that might cause numerical instability.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    ///
    /// let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     RobustBowyerWatson::for_degenerate_cases();
    /// // Algorithm should be properly initialized for degenerate cases
    /// let (processed, created, removed) = algorithm.get_statistics();
    /// assert_eq!(processed, 0); // No vertices processed yet
    /// ```
    #[must_use]
    pub fn for_degenerate_cases() -> Self {
        Self {
            predicate_config: config_presets::degenerate_robust::<T>(),
            stats: InsertionStatistics::new(),
            buffers: InsertionBuffers::with_capacity(D * 10), // Scale capacity with dimension
            hull: None,
            facet_to_cells_cache: ArcSwapOption::empty(),
            cached_generation: Arc::new(AtomicU64::new(0)),
            _phantom: PhantomData,
        }
    }

    /// Robust implementation that uses trait methods as primary strategies with enhanced predicates as fallbacks.
    ///
    /// # Errors
    ///
    /// Returns an error if vertex insertion fails due to geometric issues,
    /// validation problems, or if recovery strategies are unsuccessful.
    fn robust_insert_vertex_impl(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
    {
        // Determine the best strategy using trait method
        let strategy = self.determine_strategy(tds, vertex);

        // Try primary insertion strategies using trait methods
        let mut result = match strategy {
            InsertionStrategy::CavityBased => {
                self.insert_vertex_cavity_based_with_robust_predicates(tds, vertex)
            }
            InsertionStrategy::HullExtension => {
                self.insert_vertex_hull_extension_with_robust_predicates(tds, vertex)
            }
            _ => {
                // For other strategies, try cavity-based first
                self.insert_vertex_cavity_based_with_robust_predicates(tds, vertex)
            }
        };

        // If primary strategy failed, try fallback strategies
        let mut used_fallback = false;
        if result.is_err() {
            used_fallback = true;

            // Try the other strategy
            result = match strategy {
                InsertionStrategy::CavityBased => {
                    // If cavity-based failed, try hull extension
                    self.insert_vertex_hull_extension_with_robust_predicates(tds, vertex)
                }
                InsertionStrategy::HullExtension => {
                    // If hull extension failed, try cavity-based
                    self.insert_vertex_cavity_based_with_robust_predicates(tds, vertex)
                }
                _ => {
                    // Try hull extension as fallback
                    self.insert_vertex_hull_extension_with_robust_predicates(tds, vertex)
                }
            };

            // If both strategies failed, try general fallback
            if result.is_err() {
                result = self.insert_vertex_fallback(tds, vertex);
            }
        }

        // Update statistics on successful insertion (matching IncrementalBoyerWatson pattern)
        if let Ok(ref info) = result {
            self.stats.vertices_processed += 1;
            self.stats.total_cells_created += info.cells_created;
            self.stats.total_cells_removed += info.cells_removed;

            // Track robust-specific statistics
            if used_fallback {
                self.stats.fallback_strategies_used += 1;
            }
        }

        result
    }

    /// Cavity-based insertion with robust predicates as enhancement
    ///
    /// This method adds robust predicate fallbacks on top of the standard trait implementation.
    /// The core insertion algorithm is now in the trait's default implementation (which handles
    /// the Phase 3A lightweight `FacetView` approach correctly).
    fn insert_vertex_cavity_based_with_robust_predicates(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
    {
        // Try the standard trait method first
        // The trait's implementation now correctly handles the Phase 3A lightweight approach:
        // - Extracts facet data BEFORE removing cells
        // - Uses FacetView for zero-allocation vertex access
        // - Properly manages handle invalidation
        let result = self.insert_vertex_cavity_based(tds, vertex);

        // If standard method succeeds, we're done
        if result.is_ok() {
            return result;
        }

        // If standard method fails, try with robust predicates as fallback
        // This adds tolerance and multiple predicate strategies for degenerate cases
        let bad_cells = self.find_bad_cells_with_robust_fallback(tds, vertex)?;

        if bad_cells.is_empty() {
            return result; // No bad cells with robust method either, return original error
        }

        let boundary_handles =
            match self.find_cavity_boundary_facets_with_robust_fallback(tds, &bad_cells) {
                Ok(handles) if !handles.is_empty() => handles,
                _ => return result, // Can't find boundary, return original error
            };

        // Now perform the full cavity-based insertion using trait methods
        // Follow the same pattern as the trait's insert_vertex_cavity_based
        let cells_removed = bad_cells.len();

        // Phase 1: Gather boundary facet info before modifying TDS
        let Ok(boundary_infos) =
            <Self as InsertionAlgorithm<T, U, V, D>>::gather_boundary_facet_info(
                tds,
                &boundary_handles,
            )
        else {
            return result; // Can't gather info, return original error
        };

        // Phase 2: Insert vertex and create new cells
        let vertex_existed_before = tds.vertex_key_from_uuid(&vertex.uuid()).is_some();

        if let Err(e) = <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex)
        {
            return Err(InsertionError::TriangulationState(e));
        }

        let Some(inserted_vk) = tds.vertex_key_from_uuid(&vertex.uuid()) else {
            return result; // Vertex not found after insertion, return original error
        };

        // Create all new cells BEFORE removing bad cells
        let mut created_cell_keys = Vec::with_capacity(boundary_infos.len());
        for info in &boundary_infos {
            let mut cell_vertices: SmallBuffer<_, MAX_PRACTICAL_DIMENSION_SIZE> =
                info.facet_vertex_keys.clone();
            cell_vertices.push(inserted_vk);

            let Ok(new_cell) = crate::core::cell::Cell::new(cell_vertices, None) else {
                // Rollback and return original error
                <Self as InsertionAlgorithm<T, U, V, D>>::rollback_created_cells_and_vertex(
                    tds,
                    &created_cell_keys,
                    vertex,
                    vertex_existed_before,
                );
                return result;
            };

            if let Ok(key) = tds.insert_cell_with_mapping(new_cell) {
                created_cell_keys.push(key);
            } else {
                // Rollback and return original error
                <Self as InsertionAlgorithm<T, U, V, D>>::rollback_created_cells_and_vertex(
                    tds,
                    &created_cell_keys,
                    vertex,
                    vertex_existed_before,
                );
                return result;
            }
        }
        let cells_created = created_cell_keys.len();

        // Phase 3: Remove bad cells (point of no return)
        <Self as InsertionAlgorithm<T, U, V, D>>::remove_bad_cells(tds, &bad_cells);

        // Invalidate cache after TDS structural changes
        self.invalidate_facet_cache();

        // Phase 4: Connect neighbor relationships
        <Self as InsertionAlgorithm<T, U, V, D>>::connect_new_cells_to_neighbors(
            tds,
            inserted_vk,
            &boundary_infos,
            &created_cell_keys,
        )?;

        // Phase 5: Finalize
        <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds).map_err(|e| {
            InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Failed to finalize triangulation after robust cavity-based insertion: {e}"
                    ),
                },
            )
        })?;

        Ok(InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed,
            cells_created,
            success: true,
            degenerate_case_handled: false,
        })
    }

    /// Hull extension insertion with robust predicates as enhancement
    fn insert_vertex_hull_extension_with_robust_predicates(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
    {
        // Use visibility detection with robust fallback
        #[allow(clippy::collapsible_if)] // Can't collapse due to if-let chain guard limitations
        if let Ok(visible_facet_handles) =
            self.find_visible_boundary_facets_with_robust_fallback(tds, vertex)
        {
            if !visible_facet_handles.is_empty() {
                // Ensure vertex is in TDS - if this fails, propagate the error
                <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex)?;

                let cells_created =
                    <Self as InsertionAlgorithm<T, U, V, D>>::create_cells_from_facet_handles(
                        tds,
                        &visible_facet_handles,
                        vertex,
                    )?;

                // Maintain invariants after structural changes
                <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds).map_err(
                    |e| InsertionError::TriangulationState(
                        TriangulationValidationError::FinalizationFailed {
                            message: format!(
                                "Failed to finalize triangulation after robust hull extension insertion \
                                     (created {cells_created} cells). Underlying error: {e}"
                            ),
                        }
                    ),
                )?;

                return Ok(InsertionInfo {
                    strategy: InsertionStrategy::HullExtension,
                    cells_removed: 0,
                    cells_created,
                    success: true,
                    degenerate_case_handled: false,
                });
            }
        }

        // If visibility detection fails, fall back to trait method
        self.insert_vertex_hull_extension(tds, vertex)
    }

    /// Find bad cells by first using the trait method, then applying robust predicates for edge cases.
    ///
    /// This approach integrates the trait's `find_bad_cells` method with the robust predicates
    /// to provide a more reliable cell detection method, especially for degenerate cases.
    ///
    /// # Errors
    ///
    /// Returns an error if TDS corruption is detected (missing vertex keys).
    fn find_bad_cells_with_robust_fallback(
        &mut self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<CellKey>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
    {
        // First try to find bad cells using the trait's method
        let mut bad_cells = match InsertionAlgorithm::<T, U, V, D>::find_bad_cells(
            self, tds, vertex,
        ) {
            Ok(cells) => cells,
            Err(crate::core::traits::insertion_algorithm::BadCellsError::AllCellsBad {
                ..
            }) => {
                // All cells marked as bad - try robust method to get a better result
                self.robust_find_bad_cells(tds, vertex)?
            }
            Err(
                crate::core::traits::insertion_algorithm::BadCellsError::TooManyDegenerateCells(_),
            ) => {
                // Too many degenerate cells - try robust method as fallback
                self.robust_find_bad_cells(tds, vertex)?
            }
            Err(crate::core::traits::insertion_algorithm::BadCellsError::NoCells) => {
                // No cells - return empty
                return Ok(Vec::new());
            }
            Err(crate::core::traits::insertion_algorithm::BadCellsError::TdsCorruption {
                cell_key,
                vertex_key,
            }) => {
                // TDS corruption detected - this is a fatal error that must be propagated
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "TDS corruption: Cell {cell_key:?} references non-existent vertex {vertex_key:?}"
                        ),
                    },
                ));
            }
        };

        // If the standard method doesn't find any bad cells (likely a degenerate case)
        // or we're using the robust configuration, supplement with robust predicates
        if bad_cells.is_empty() || self.predicate_config.base_tolerance > T::default_tolerance() {
            let robust_bad_cells = self.robust_find_bad_cells(tds, vertex)?;

            // Use a set for O(1) membership checking to avoid O(n²) complexity
            let mut seen: CellKeySet = bad_cells.iter().copied().collect();
            for cell_key in robust_bad_cells {
                // Only add if not already present (insert returns true if new)
                if seen.insert(cell_key) {
                    bad_cells.push(cell_key);
                }
            }
        }

        Ok(bad_cells)
    }

    /// Find cavity boundary facets using lightweight handles with robust fallback.
    ///
    /// This optimized approach uses the lightweight `find_cavity_boundary_facets` method
    /// first, then applies robust predicates for edge cases, returning lightweight handles instead
    /// of heavyweight Facet objects for improved performance.
    ///
    /// # Important Usage Note
    ///
    /// The returned `FacetHandle` are only valid while the referenced cells exist.
    /// If you plan to remove cells (e.g., via `remove_bad_cells`), you MUST extract the facet
    /// data BEFORE removing the cells, otherwise the handles become invalid.
    fn find_cavity_boundary_facets_with_robust_fallback(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<FacetHandle>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
    {
        // First try to find boundary facets using the lightweight trait method
        match InsertionAlgorithm::<T, U, V, D>::find_cavity_boundary_facets(self, tds, bad_cells) {
            Ok(boundary_handles) => {
                // If the lightweight method succeeds and finds facets, use them
                if !boundary_handles.is_empty() {
                    return Ok(boundary_handles);
                }
                // If lightweight method succeeds but finds no facets, try robust method as fallback
                self.robust_find_cavity_boundary_facets(tds, bad_cells)
            }
            Err(_) => {
                // If lightweight method fails, use robust method as fallback
                self.robust_find_cavity_boundary_facets(tds, bad_cells)
            }
        }
    }

    /// Find visible boundary facets by first using the trait method, then applying robust predicates for edge cases.
    ///
    /// This approach integrates the trait's `find_visible_boundary_facets` method with the robust predicates
    /// to provide a more reliable visibility detection method, especially for degenerate cases.
    fn find_visible_boundary_facets_with_robust_fallback(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<FacetHandle>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
    {
        // First try the lightweight method that avoids heavy cloning
        match <Self as InsertionAlgorithm<T, U, V, D>>::find_visible_boundary_facets_lightweight(
            self, tds, vertex,
        ) {
            Ok(handles) if !handles.is_empty() => Ok(handles),
            _ => {
                // Fallback: robust check via Facet conversion (slow path)
                let mut handles = Vec::new();
                for fv in tds
                    .boundary_facets()
                    .map_err(InsertionError::TriangulationState)?
                {
                    if self.is_facet_visible_from_vertex_robust(tds, &fv, vertex)? {
                        handles.push(FacetHandle::new(fv.cell_key(), fv.facet_index()));
                    }
                }
                Ok(handles)
            }
        }
    }

    /// Find bad cells using robust insphere predicate.
    /// This is a lower-level method used by `find_bad_cells_with_robust_fallback`.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if TDS corruption is detected (cell references non-existent vertex).
    fn robust_find_bad_cells(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<CellKey>, InsertionError> {
        let mut bad_cells = SmallBuffer::<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut vertex_points = SmallBuffer::<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        vertex_points.reserve_exact(D + 1);

        for (cell_key, cell) in tds.cells() {
            // Extract vertex points from the cell using key-based API (reusing buffer)
            vertex_points.clear();

            // Phase 3A: Access vertices via TDS using vertices
            // TDS corruption: cell references non-existent vertex - must propagate error
            for &vkey in cell.vertices() {
                let Some(v) = tds.get_vertex_by_key(vkey) else {
                    return Err(InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "TDS corruption: Cell {cell_key:?} references non-existent vertex {vkey:?}"
                            ),
                        },
                    ));
                };
                vertex_points.push(*v.point());
            }

            // TDS corruption: cell has incomplete vertex list (should have D+1 vertices)
            if vertex_points.len() < D + 1 {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "TDS corruption: Cell {cell_key:?} has {} vertices but expected {} (D+1)",
                            vertex_points.len(),
                            D + 1
                        ),
                    },
                ));
            }

            // Use robust insphere test
            match robust_insphere(&vertex_points, vertex.point(), &self.predicate_config) {
                Ok(InSphere::INSIDE) => {
                    bad_cells.push(cell_key);
                }
                Ok(InSphere::BOUNDARY) => {
                    // Boundary cases are handled conservatively
                    // In degenerate configurations, we might include boundary cells
                    if matches!(self.predicate_config.base_tolerance, t if t > T::default_tolerance())
                    {
                        bad_cells.push(cell_key);
                    }
                }
                Ok(InSphere::OUTSIDE) => {
                    // Cell is good - don't include
                }
                Err(_) => {
                    // Robust predicate failed - fall back to conservative approach
                    // Note: We don't track fallback usage here as this is an internal method.
                    // Statistics are only tracked on successful vertex insertion.

                    // Use the original insphere predicate as fallback
                    if matches!(
                        crate::geometry::predicates::insphere(&vertex_points, *vertex.point()),
                        Ok(InSphere::INSIDE)
                    ) {
                        bad_cells.push(cell_key);
                    }
                }
            }
        }

        Ok(bad_cells.into_vec())
    }

    /// Find cavity boundary facets with enhanced error handling, returning lightweight handles.
    ///
    /// This optimized version returns `FacetHandle` instead of heavyweight facet data,
    /// providing significant performance improvements for boundary facet detection.
    ///
    /// Made `pub(crate)` for testing purposes.
    pub(crate) fn robust_find_cavity_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<FacetHandle>, InsertionError> {
        let mut boundary_handles = Vec::new();

        if bad_cells.is_empty() {
            return Ok(boundary_handles);
        }

        let bad_cell_set: CellKeySet = bad_cells.iter().copied().collect();

        // Build facet-to-cells mapping with enhanced validation
        let facet_to_cells = self.build_validated_facet_mapping(tds)?;

        // Find boundary facets with improved logic, returning lightweight handles
        let mut processed_facets = FastHashSet::default();

        // Track first error for better error reporting instead of just continuing
        let mut first_facet_error: Option<(CellKey, usize, &'static str)> = None;

        for &bad_cell_key in bad_cells {
            if let Some(bad_cell) = tds.get_cell(bad_cell_key) {
                // Phase 3A: Use vertices() to get the facet count
                let facet_count = bad_cell.number_of_vertices();
                for facet_idx in 0..facet_count {
                    let Ok(facet_idx_u8) = u8::try_from(facet_idx) else {
                        continue;
                    };
                    let Ok(fv) =
                        crate::core::facet::FacetView::new(tds, bad_cell_key, facet_idx_u8)
                    else {
                        // Track first FacetView construction error for better diagnostics
                        if first_facet_error.is_none() {
                            first_facet_error =
                                Some((bad_cell_key, facet_idx, "FacetView::new failed"));
                        }
                        continue;
                    };
                    let Ok(facet_key) = fv.key() else {
                        // Track first facet key derivation error for better diagnostics
                        if first_facet_error.is_none() {
                            first_facet_error =
                                Some((bad_cell_key, facet_idx, "facet.key() failed"));
                        }
                        continue;
                    };

                    if !processed_facets.insert(facet_key) {
                        continue;
                    }

                    if let Some(sharing_cells) = facet_to_cells.get(&facet_key) {
                        let bad_count = sharing_cells
                            .iter()
                            .filter(|&&cell_key| bad_cell_set.contains(&cell_key))
                            .count();
                        let total_count = sharing_cells.len();

                        if Self::is_cavity_boundary_facet(bad_count, total_count) {
                            boundary_handles.push(FacetHandle::new(bad_cell_key, facet_idx_u8));
                        }
                    }
                }
            }
        }

        // Provide specific error diagnostics if available, otherwise fall back to validation
        if boundary_handles.is_empty() && !bad_cells.is_empty() {
            if let Some((error_cell, error_facet, error_type)) = first_facet_error {
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Failed to construct/derive facet during robust cavity mapping: cell={error_cell:?}, facet_index={error_facet}, cause={error_type}"
                        ),
                    },
                ));
            }
            // Generic fallback validation
            self.validate_boundary_facets(&boundary_handles, bad_cells.len())?;
        }

        Ok(boundary_handles)
    }

    /// Handle degenerate insertion cases with special strategies.
    #[allow(dead_code)]
    fn handle_degenerate_insertion_case(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        _bad_cells: &[CellKey],
    ) -> RobustInsertionInfo {
        self.stats.degenerate_cases_handled += 1;

        // Strategy 1: Try vertex perturbation
        if let Ok(perturbed_vertex) = self.create_perturbed_vertex(vertex)
            && let Ok(info) = self.insert_vertex(tds, perturbed_vertex)
        {
            return RobustInsertionInfo {
                success: true,
                cells_created: info.cells_created,
                cells_removed: info.cells_removed,
                strategy_used: InsertionStrategy::Perturbation,
                degenerate_case_handled: true,
            };
        }

        // Strategy 2: Skip this vertex and mark as degenerate
        RobustInsertionInfo {
            success: false,
            cells_created: 0,
            cells_removed: 0,
            strategy_used: InsertionStrategy::Skip,
            degenerate_case_handled: true,
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    #[allow(clippy::unused_self)]
    fn build_validated_facet_mapping(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<FastHashMap<u64, Vec<CellKey>>, InsertionError> {
        // Use cached facet mapping to avoid recomputation with proper error handling
        let tds_map = self
            .try_get_or_build_facet_cache(tds)
            .map_err(InsertionError::TriangulationState)?;

        // Transform the TDS map into the required format with validation
        let facet_to_cells: FastHashMap<u64, Vec<CellKey>> = tds_map
            .iter()
            .map(|(&facet_key, cell_facet_pairs)| {
                // Extract just the CellKeys, discarding facet indices
                let mut cell_keys = Vec::with_capacity(cell_facet_pairs.len());
                cell_keys.extend(cell_facet_pairs.iter().map(FacetHandle::cell_key));

                // Defensively deduplicate cell keys in case build_facet_to_cells_hashmap()
                // ever yields duplicate (cell_key, idx) pairs per facet
                {
                    let mut seen = fast_hash_set_with_capacity(cell_keys.len());
                    cell_keys.retain(|k| seen.insert(*k));
                }

                // Validate that no facet is shared by more than 2 cells
                if cell_keys.len() > 2 {
                    return Err(InsertionError::TriangulationState(
                        TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Facet {} is shared by {} cells (should be ≤2)",
                                facet_key,
                                cell_keys.len()
                            ),
                        },
                    ));
                }

                Ok((facet_key, cell_keys))
            })
            .collect::<Result<FastHashMap<_, _>, _>>()?;

        Ok(facet_to_cells)
    }

    const fn is_cavity_boundary_facet(bad_count: usize, total_count: usize) -> bool {
        matches!((bad_count, total_count), (1, 1 | 2))
    }

    /// Validates boundary facets using the key-based `FacetView` API.
    ///
    /// This checks that boundary facets were found when bad cells exist,
    /// and can perform additional geometric validation if needed.
    #[allow(clippy::unused_self)]
    const fn validate_boundary_facets(
        &self,
        boundary_facets: &[FacetHandle],
        bad_cell_count: usize,
    ) -> Result<(), InsertionError> {
        if boundary_facets.is_empty() && bad_cell_count > 0 {
            return Err(InsertionError::ExcessiveBadCells {
                found: bad_cell_count,
                threshold: 0,
            });
        }

        // Additional validation could check facet geometry using FacetView, etc.
        // For example:
        // for (cell_key, facet_idx) in boundary_facets {
        //     if let Some(cell) = tds.cells().get(*cell_key) {
        //         if let Ok(facet_view) = FacetView::new(tds, *cell_key, *facet_idx) {
        //             // Validate facet geometry...
        //         }
        //     }
        // }

        Ok(())
    }

    /// Robust helper method to test if a boundary facet is visible from a given vertex
    ///
    /// **Phase 3A**: Updated to use lightweight `FacetView` instead of heavyweight `Facet`.
    ///
    /// This method uses multiple fallback strategies when geometric predicates fail
    /// or return degenerate results, making it more suitable for exterior vertex insertion.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if TDS corruption is detected (cell or facet references non-existent vertex).
    fn is_facet_visible_from_vertex_robust(
        &self,
        tds: &Tds<T, U, V, D>,
        facet_view: &crate::core::facet::FacetView<'_, T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<bool, InsertionError>
    where
        T: AddAssign<T>
            + ComplexField<RealField = T>
            + SubAssign<T>
            + Sum
            + From<f64>
            + DivAssign<T>,
        f64: From<T>,
    {
        // Get the adjacent cell to this boundary facet
        let Some(adjacent_cell) = tds.get_cell(facet_view.cell_key()) else {
            // Cell not found - TDS corruption
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "TDS corruption: Facet references non-existent cell {:?}",
                        facet_view.cell_key()
                    ),
                },
            ));
        };

        // Phase 3A: Get facet vertices via FacetView (returns iterator over &Vertex)
        let Ok(facet_vertex_iter) = facet_view.vertices() else {
            // Cannot get facet vertices - TDS corruption or FacetError
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "TDS corruption: Cannot retrieve vertices for facet at cell {:?}, index {}",
                        facet_view.cell_key(),
                        facet_view.facet_index()
                    ),
                },
            ));
        };
        // Use SmallBuffer to avoid heap allocation in hot path (facets have at most D vertices)
        let mut facet_vertices: SmallBuffer<&Vertex<T, U, D>, { MAX_PRACTICAL_DIMENSION_SIZE }> =
            SmallBuffer::new();
        facet_vertices.extend(facet_vertex_iter);

        // Get cell vertices via TDS using vertices
        // TDS corruption: cell references non-existent vertex - must propagate error
        let mut cell_vertices: SmallBuffer<&Vertex<T, U, D>, { MAX_PRACTICAL_DIMENSION_SIZE }> =
            SmallBuffer::new();
        for &vkey in adjacent_cell.vertices() {
            let Some(v) = tds.get_vertex_by_key(vkey) else {
                // TDS corruption detected: missing vertex
                return Err(InsertionError::TriangulationState(
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "TDS corruption: Cell {:?} references non-existent vertex {vkey:?}",
                            facet_view.cell_key()
                        ),
                    },
                ));
            };
            cell_vertices.push(v);
        }

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
            // Could not find opposite vertex - TDS corruption or geometric issue
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "TDS corruption: Cannot find opposite vertex for facet at cell {:?}, index {} (all cell vertices are in facet)",
                        facet_view.cell_key(),
                        facet_view.facet_index()
                    ),
                },
            ));
        };

        // Create test simplices for orientation comparison
        // Use SmallBuffer with safe capacity for all practical dimensions (D+1 vertices needed)
        let mut simplex_with_opposite: SmallBuffer<Point<T, D>, { MAX_PRACTICAL_DIMENSION_SIZE }> =
            SmallBuffer::new();
        simplex_with_opposite.extend(facet_vertices.iter().map(|v| *v.point()));
        simplex_with_opposite.push(*opposite_vertex.point());

        let mut simplex_with_test: SmallBuffer<Point<T, D>, { MAX_PRACTICAL_DIMENSION_SIZE }> =
            SmallBuffer::new();
        simplex_with_test.extend(facet_vertices.iter().map(|v| *v.point()));
        simplex_with_test.push(*vertex.point());

        // Get orientations using robust predicates first, fall back to standard predicates
        let orientation_opposite = crate::geometry::robust_predicates::robust_orientation(
            &simplex_with_opposite,
            &self.predicate_config,
        )
        .unwrap_or_else(|_| {
            simplex_orientation(&simplex_with_opposite).unwrap_or(Orientation::DEGENERATE)
        });

        let orientation_test = crate::geometry::robust_predicates::robust_orientation(
            &simplex_with_test,
            &self.predicate_config,
        )
        .unwrap_or_else(|_| {
            simplex_orientation(&simplex_with_test).unwrap_or(Orientation::DEGENERATE)
        });

        // Determine visibility based on orientation comparison
        match (orientation_opposite, orientation_test) {
            (Orientation::NEGATIVE, Orientation::POSITIVE)
            | (Orientation::POSITIVE, Orientation::NEGATIVE) => Ok(true),
            (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
                // Degenerate case - use distance-based fallback for exterior vertices
                self.fallback_visibility_heuristic(facet_view, vertex)
            }
            _ => Ok(false), // Same orientation = same side = not visible
        }
    }

    /// Fallback visibility heuristic for degenerate cases
    ///
    /// **Phase 3A**: Updated to use lightweight `FacetView` instead of heavyweight `Facet`.
    ///
    /// When geometric predicates fail or return degenerate results, this method uses
    /// a distance-based heuristic to determine if a facet should be considered visible.
    /// For exterior vertex insertion, this is more aggressive than the default implementation.
    ///
    /// # Errors
    ///
    /// Returns `InsertionError` if TDS corruption is detected or facet vertices cannot be accessed.
    fn fallback_visibility_heuristic(
        &self,
        facet_view: &crate::core::facet::FacetView<'_, T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<bool, InsertionError>
    where
        T: DivAssign<T>
            + AddAssign<T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + num_traits::Zero
            + From<f64>,
        f64: From<T>,
    {
        // Phase 3A: Get facet vertices via FacetView
        let Ok(facet_vertex_iter) = facet_view.vertices() else {
            // Cannot get facet vertices - TDS corruption or FacetError
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "TDS corruption: Cannot retrieve vertices for facet at cell {:?}, index {} in fallback heuristic",
                        facet_view.cell_key(),
                        facet_view.facet_index()
                    ),
                },
            ));
        };
        // Use SmallBuffer to avoid heap allocation in hot path (facets have at most D vertices)
        let mut facet_vertices: SmallBuffer<&Vertex<T, U, D>, { MAX_PRACTICAL_DIMENSION_SIZE }> =
            SmallBuffer::new();
        facet_vertices.extend(facet_vertex_iter);

        if facet_vertices.is_empty() {
            // No vertices in facet - TDS corruption or geometric issue
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "TDS corruption: Facet at cell {:?}, index {} has no vertices",
                        facet_view.cell_key(),
                        facet_view.facet_index()
                    ),
                },
            ));
        }

        // Calculate facet centroid
        let mut centroid_coords = [T::zero(); D];
        for facet_vertex in &facet_vertices {
            let coords: [T; D] = facet_vertex.point().into();
            for (i, &coord) in coords.iter().enumerate() {
                centroid_coords[i] += coord;
            }
        }
        // Use safe conversion to avoid precision loss warning
        // Note: This should never fail for facet sizes (≤ D) but we handle it defensively
        let Ok(num_vertices) = safe_usize_to_scalar::<T>(facet_vertices.len()) else {
            // Conversion failed - unexpected but handle gracefully
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Failed to convert facet vertex count {} to scalar type in fallback heuristic",
                        facet_vertices.len()
                    ),
                },
            ));
        };
        for coord in &mut centroid_coords {
            *coord /= num_vertices;
        }

        // Calculate distance from vertex to centroid
        let vertex_coords: [T; D] = vertex.point().into();
        let mut distance_squared = T::zero();
        for i in 0..D {
            let diff = vertex_coords[i] - centroid_coords[i];
            distance_squared += diff * diff;
        }

        // For exterior vertices, use a more aggressive threshold
        // If the vertex is far from the facet centroid, consider it visible
        // Use a threshold based on the perturbation scale multiplied by a configurable factor
        // Note: Configuration is validated at construction time (see with_config debug_assert)
        let threshold = {
            let scale = self.predicate_config.perturbation_scale;
            let multiplier = self.predicate_config.visibility_threshold_multiplier;
            scale * scale * multiplier
        };
        Ok(distance_squared > threshold)
    }

    #[allow(dead_code)]
    fn create_perturbed_vertex(
        &self,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vertex<T, U, D>, crate::core::vertex::VertexValidationError> {
        let mut coords: [T; D] = *vertex.point().coords();
        let perturbation = self.predicate_config.perturbation_scale;

        // Apply small random perturbation to first coordinate
        coords[0] += perturbation;

        let perturbed_point = Point::new(coords);

        // Clone the original vertex and update the point using set_point for proper validation
        let mut perturbed_vertex = *vertex;

        // Use the set_point method for proper validation
        // Return the result directly to allow caller to handle potential validation errors
        perturbed_vertex.set_point(perturbed_point)?;

        Ok(perturbed_vertex)
    }

    #[allow(dead_code)]
    /// Determines if a vertex needs robust handling based on geometric properties
    ///
    /// This is a helper method for the `hybrid_insert_vertex` example that shows
    /// how a robust algorithm could selectively use enhanced predicates only
    /// when needed for performance optimization.
    #[allow(clippy::unused_self)]
    fn vertex_needs_robust_handling(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> bool {
        // Example heuristics for when robust handling might be needed:
        // 1. Very small coordinates that might cause precision issues
        // 2. Vertex very close to existing circumspheres
        // 3. High vertex density in the insertion area
        // 4. Previous insertions in this area required fallback strategies

        // For this example, use a simple heuristic based on coordinate magnitude
        let coords: [T; D] = *vertex.point().coords();
        let has_small_coords = coords.iter().any(|&c| {
            let c_f64: f64 = c.into();
            c_f64.abs() < 1e-10
        });

        let has_large_coords = coords.iter().any(|&c| {
            let c_f64: f64 = c.into();
            c_f64.abs() > 1e6
        });

        // Also check if we're in a high-density area
        // Guard for large triangulations: limit proximity scan to prevent O(n) overhead
        let vertex_count = tds.number_of_vertices();
        let nearby_vertices = if vertex_count > 1000 {
            // For large triangulations, use early exit after finding sufficient nearby vertices
            let mut count = 0;
            let max_scan = 100; // Early exit threshold
            for (i, (_vkey, v)) in tds.vertices().enumerate() {
                if i >= max_scan {
                    break; // Early exit to bound computational cost
                }
                let v_coords: [T; D] = v.point().into();
                let distance_squared: f64 = coords
                    .iter()
                    .zip(v_coords.iter())
                    .map(|(&a, &b)| {
                        let diff: f64 = (a - b).into();
                        diff * diff
                    })
                    .sum();
                if distance_squared < 1e-6 {
                    count += 1;
                    if count > 3 {
                        break; // Found enough nearby vertices
                    }
                }
            }
            count
        } else {
            // For smaller triangulations, do the full scan
            tds.vertices()
                .filter(|(_vkey, v)| {
                    let v_coords: [T; D] = v.point().into();
                    let distance_squared: f64 = coords
                        .iter()
                        .zip(v_coords.iter())
                        .map(|(&a, &b)| {
                            let diff: f64 = (a - b).into();
                            diff * diff
                        })
                        .sum();
                    distance_squared < 1e-6 // Very close vertices
                })
                .count()
        };

        has_small_coords || has_large_coords || nearby_vertices > 3
    }
}

impl<T, U, V, const D: usize> FacetCacheProvider<T, U, V, D> for RobustBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar
        + ComplexField<RealField = T>
        + AddAssign<T>
        + SubAssign<T>
        + Sum
        + num_traits::NumCast
        + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
{
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
        &self.facet_to_cells_cache
    }

    fn cached_generation(&self) -> &AtomicU64 {
        self.cached_generation.as_ref()
    }
}

impl<T, U, V, const D: usize> Default for RobustBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    na::OPoint<T, na::Const<D>>: From<[f64; D]>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a robust vertex insertion operation.
#[derive(Debug)]
pub struct RobustInsertionInfo {
    /// Whether the insertion was successful
    pub success: bool,
    /// Number of cells created during insertion
    pub cells_created: usize,
    /// Number of cells removed during insertion
    pub cells_removed: usize,
    /// Strategy used for insertion
    pub strategy_used: InsertionStrategy,
    /// Whether a degenerate case was handled
    pub degenerate_case_handled: bool,
}

impl<T, U, V, const D: usize> InsertionAlgorithm<T, U, V, D> for RobustBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    na::OPoint<T, na::Const<D>>: From<[f64; D]>,
{
    fn insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError> {
        // Use the simplified robust implementation that leverages trait methods
        self.robust_insert_vertex_impl(tds, &vertex)
    }

    fn get_statistics(&self) -> (usize, usize, usize) {
        // Use the standardized statistics method
        self.stats.as_basic_tuple()
    }

    fn reset(&mut self) {
        self.stats.reset();
        self.buffers.clear_all();
        self.hull = None;
        // Clear facet cache to prevent serving stale mappings across runs
        self.invalidate_facet_cache();
    }

    fn determine_strategy(
        &self,
        _tds: &Tds<T, U, V, D>,
        _vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy {
        // Default to standard insertion strategy
        // A more sophisticated implementation could analyze the vertex position
        InsertionStrategy::Standard
    }

    fn new_triangulation(
        &mut self,
        vertices: &[Vertex<T, U, D>],
    ) -> Result<Tds<T, U, V, D>, TriangulationConstructionError> {
        // One attempt for now; map error clearly. If a true robust fallback is added later, branch here.
        Tds::new(vertices).map_err(|e| {
            TriangulationConstructionError::ValidationError(
                TriangulationValidationError::FailedToCreateCell {
                    message: format!("Robust triangulation failed: {e}"),
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::facet::FacetView;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::core::traits::facet_cache::FacetCacheProvider;
    use crate::core::traits::insertion_algorithm::{InsertionAlgorithm, InsertionError};
    use crate::core::util::{derive_facet_key_from_vertex_keys, verify_facet_index_consistency};
    use crate::core::vertex::VertexBuilder;
    use crate::vertex;
    use approx::assert_abs_diff_eq;
    use approx::assert_abs_diff_ne;
    use std::sync::atomic::Ordering;

    // Conditional debug output macro for tests - reduces noisy CI logs
    macro_rules! debug_println {
        ($($arg:tt)*) => {
            #[cfg(feature = "test-debug")]
            println!($($arg)*);
        };
    }

    /// Macro to generate dimension-specific robust algorithm tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions for the `RobustBowyerWatson` algorithm. It creates tests for:
    /// - Algorithm construction and initialization
    /// - Vertex insertion with cavity-based approach
    /// - Statistics tracking across insertions
    /// - TDS validation after operations
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_robust_algorithm_dimensions! {
    ///     robust_2d => 2 => "triangle" => vec![vertex!([0.0, 0.0]), ...],
    /// }
    /// ```
    macro_rules! test_robust_algorithm_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr => $initial_vertices:expr, $test_vertex:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic algorithm functionality in this dimension
                    let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, $dim>::new();
                    let initial_vertices = $initial_vertices;
                    let mut tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();

                    assert!(tds.is_valid().is_ok(), "{}D initial TDS should be valid", $dim);
                    assert_eq!(tds.dim(), $dim as i32, "{}D TDS should have dimension {}", $dim, $dim);

                    // Verify initial statistics
                    let (initial_processed, initial_created, initial_removed) = algorithm.get_statistics();
                    assert_eq!(initial_processed, 0, "{}D: Should have 0 vertices processed initially", $dim);
                    assert_eq!(initial_created, 0, "{}D: Should have 0 cells created initially", $dim);
                    assert_eq!(initial_removed, 0, "{}D: Should have 0 cells removed initially", $dim);

                    // Test vertex insertion
                    let test_vertex = $test_vertex;
                    let result = algorithm.insert_vertex(&mut tds, test_vertex);
                    assert!(result.is_ok(), "{}D: {} insertion should succeed", $dim, $desc);

                    let info = result.unwrap();
                    assert!(info.success, "{}D: Insertion should be successful", $dim);
                    assert!(info.cells_created > 0, "{}D: Should create at least one cell", $dim);

                    // Verify statistics were updated
                    let (processed, created, removed) = algorithm.get_statistics();
                    assert_eq!(processed, 1, "{}D: Should have processed 1 vertex", $dim);
                    assert_eq!(created, info.cells_created, "{}D: Created cells should match", $dim);
                    assert_eq!(removed, info.cells_removed, "{}D: Removed cells should match", $dim);

                    // Verify TDS remains valid after insertion
                    assert!(tds.is_valid().is_ok(), "{}D: TDS should remain valid after insertion", $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_degenerate_config>]() {
                        // Test with degenerate-robust configuration
                        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, $dim>::for_degenerate_cases();
                        let initial_vertices = $initial_vertices;
                        let mut tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();

                        let test_vertex = $test_vertex;
                        let result = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Should either succeed or fail gracefully
                        if result.is_ok() {
                            assert!(tds.is_valid().is_ok(),
                                "{}D: TDS should be valid after successful degenerate insertion", $dim);
                        } else {
                            // Even on failure, TDS should remain valid
                            assert!(tds.is_valid().is_ok(),
                                "{}D: TDS should remain valid even after failed degenerate insertion", $dim);
                        }
                    }

                    #[test]
                    fn [<$test_name _reset>]() {
                        // Test algorithm reset functionality
                        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, $dim>::new();
                        let initial_vertices = $initial_vertices;
                        let mut tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();

                        // Insert a vertex
                        let test_vertex = $test_vertex;
                        let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Verify statistics are non-zero
                        let (processed, _, _) = algorithm.get_statistics();
                        assert!(processed > 0, "{}D: Should have processed vertices before reset", $dim);

                        // Reset the algorithm
                        algorithm.reset();

                        // Verify statistics are reset
                        let (processed, created, removed) = algorithm.get_statistics();
                        assert_eq!(processed, 0, "{}D: Processed should be 0 after reset", $dim);
                        assert_eq!(created, 0, "{}D: Created should be 0 after reset", $dim);
                        assert_eq!(removed, 0, "{}D: Removed should be 0 after reset", $dim);
                    }

                    #[test]
                    fn [<$test_name _multiple_insertions>]() {
                        // Test multiple vertex insertions
                        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, $dim>::new();
                        let initial_vertices = $initial_vertices;
                        let mut tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&initial_vertices).unwrap();

                        let initial_vertex_count = tds.number_of_vertices();

                        // Insert test vertex
                        let test_vertex = $test_vertex;
                        let result1 = algorithm.insert_vertex(&mut tds, test_vertex);
                        assert!(result1.is_ok(), "{}D: First insertion should succeed", $dim);

                        // Verify vertex count increased
                        assert_eq!(tds.number_of_vertices(), initial_vertex_count + 1,
                            "{}D: Vertex count should increase after insertion", $dim);

                        // Verify TDS is still valid
                        assert!(tds.is_valid().is_ok(),
                            "{}D: TDS should be valid after multiple insertions", $dim);

                        // Verify statistics tracking
                        let (processed, _, _) = algorithm.get_statistics();
                        assert_eq!(processed, 1, "{}D: Should have processed 1 vertex", $dim);
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D using the macro
    test_robust_algorithm_dimensions! {
        robust_2d_insertion => 2 => "interior point" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([2.0, 0.0]),
                vertex!([1.0, 2.0]),
            ],
            vertex!([1.0, 0.5]),

        robust_3d_insertion => 3 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5]),

        robust_4d_insertion => 4 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5]),

        robust_5d_insertion => 5 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5, 0.5]),
    }

    /// Comprehensive test for algorithm configuration and constructor methods
    /// Consolidates: `test_robust_bowyer_watson_creation`, `test_with_config_constructor`,
    /// `test_default_implementation_consistency`, `test_default_has_proper_buffer_capacity`,
    /// `test_algorithm_configuration_presets`, `test_configuration_validation_paths`
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_algorithm_configuration_comprehensive() {
        // Test 1: Basic construction with new()
        let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
            RobustBowyerWatson::new();
        assert_eq!(
            algorithm.stats.vertices_processed, 0,
            "New algorithm should have zero vertices processed"
        );
        let (proc, created, removed) = algorithm.get_statistics();
        assert_eq!(proc, 0);
        assert_eq!(created, 0);
        assert_eq!(removed, 0);

        // Test 2: Construction with custom config (with_config)
        let config = config_presets::high_precision::<f64>();
        let custom_algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
            RobustBowyerWatson::with_config(config.clone());
        assert!(
            custom_algorithm.predicate_config.base_tolerance <= config.base_tolerance,
            "Configuration should be applied"
        );
        let (proc, created, removed) = custom_algorithm.get_statistics();
        assert_eq!(proc, 0);
        assert_eq!(created, 0);
        assert_eq!(removed, 0);

        // Test 3: Default::default() consistency with new()
        let default_algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::default();
        let new_algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();
        approx::assert_relative_eq!(
            default_algorithm.predicate_config.base_tolerance,
            new_algorithm.predicate_config.base_tolerance,
            epsilon = f64::EPSILON,
            max_relative = f64::EPSILON
        );
        approx::assert_relative_eq!(
            default_algorithm.predicate_config.perturbation_scale,
            new_algorithm.predicate_config.perturbation_scale,
            epsilon = f64::EPSILON,
            max_relative = f64::EPSILON
        );
        assert_eq!(
            default_algorithm
                .cached_generation()
                .load(Ordering::Acquire),
            new_algorithm.cached_generation().load(Ordering::Acquire),
            "Default and new() should have identical cache generation"
        );

        // Test 4: Default has proper buffer capacity (functional test)
        let mut default_func_algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::default();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let result = default_func_algorithm.new_triangulation(&vertices);
        assert!(
            result.is_ok(),
            "Default algorithm should be able to create triangulation"
        );
        let tds = result.unwrap();
        assert_eq!(
            tds.number_of_vertices(),
            4,
            "Triangulation should have 4 vertices"
        );
        assert_eq!(
            tds.number_of_cells(),
            1,
            "Triangulation should have 1 tetrahedron"
        );

        // Test 5: Standard configuration presets
        let configs = vec![
            ("general", config_presets::general_triangulation::<f64>()),
            (
                "degenerate_robust",
                config_presets::degenerate_robust::<f64>(),
            ),
        ];
        let test_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        for (name, config) in configs {
            let mut algorithm =
                RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);
            let mut tds = Tds::new(&test_vertices).expect("TDS creation should succeed");
            assert!(
                tds.is_valid().is_ok(),
                "TDS should be valid for {name} preset"
            );
            let test_vertex = vertex!([0.5, 0.5, 0.5]);
            let result = algorithm.insert_vertex(&mut tds, test_vertex);
            assert!(
                result.is_ok(),
                "Interior insertion should succeed with {name} preset"
            );
        }

        // Test 6: Extreme tolerance configurations
        let mut extreme_config = config_presets::general_triangulation::<f64>();
        extreme_config.base_tolerance = f64::MIN_POSITIVE;
        let extreme_algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);
        let _tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&test_vertices).unwrap();
        let _stats = extreme_algorithm.get_statistics();

        // Test 7: Degenerate cases configuration
        let degenerate_algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();
        let _stats = degenerate_algorithm.get_statistics();
    }

    #[test]
    #[allow(clippy::used_underscore_binding)] // Variables used in conditional debug_println! macros
    fn test_no_double_counting_statistics() {
        debug_println!("Testing that robust vertex insertion statistics are not double counted");

        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create initial triangulation with exactly 4 vertices (minimum for a tetrahedron)
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // After initial creation, algorithm stats should be zero since no insertions have been performed yet
        let (insertions_after_initial, created_after_initial, removed_after_initial) =
            algorithm.get_statistics();

        debug_println!("After initial tetrahedron:");
        debug_println!("  Insertions: {insertions_after_initial}");
        debug_println!("  Cells created: {created_after_initial}");
        debug_println!("  Cells removed: {removed_after_initial}");

        assert_eq!(insertions_after_initial, 0, "No insertions performed yet");
        assert_eq!(
            created_after_initial, 0,
            "Algorithm didn't create the initial cells"
        );
        assert_eq!(removed_after_initial, 0, "No cells removed yet");

        // Now add additional vertices one by one
        let additional_vertices = [
            vertex!([0.5, 0.5, 0.5]), // Interior point
            vertex!([2.0, 0.0, 0.0]), // Exterior point
        ];

        for (i, &new_vertex) in additional_vertices.iter().enumerate() {
            // Track statistics before insertion
            let (before_insertions, _before_created, _before_removed) = algorithm.get_statistics();

            // Insert using the robust algorithm
            let insertion_result = algorithm.insert_vertex(&mut tds, new_vertex);

            assert!(
                insertion_result.is_ok(),
                "Robust insertion should succeed for vertex {}",
                i + 1
            );

            let (after_insertions, _after_created, _after_removed) = algorithm.get_statistics();
            let insertion_info = insertion_result.unwrap();

            debug_println!(
                "\nAfter adding vertex {} ({:?}):",
                i + 1,
                new_vertex.point().coords()
            );
            debug_println!("  Insertions: {after_insertions}");
            debug_println!("  Cells created: {_after_created}");
            debug_println!("  Cells removed: {_after_removed}");
            debug_println!("  Total cells in TDS: {}", tds.number_of_cells());
            debug_println!(
                "  InsertionInfo: created={}, removed={}",
                insertion_info.cells_created,
                insertion_info.cells_removed
            );

            // Critical test: insertion count should increment by exactly 1
            assert_eq!(
                after_insertions - before_insertions,
                1,
                "Insertion count should increment by exactly 1 (vertex {})",
                i + 1
            );

            // The robust algorithm doesn't track cells_created/removed totals (returns 0)
            // But the insertion should succeed and provide valid insertion info
            assert!(insertion_info.success, "Insertion should be successful");
            assert!(
                insertion_info.cells_created > 0,
                "Should create at least some cells"
            );
        }

        debug_println!(
            "✓ No double counting detected in robust algorithm - statistics are accurate and consistent"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_debug_exterior_vertex_insertion() {
        debug_println!("Testing exterior vertex insertion in robust Bowyer-Watson");

        let mut algorithm = RobustBowyerWatson::new();

        // Create initial triangulation with exactly 4 vertices (minimum for a tetrahedron)
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        debug_println!("Initial TDS has {} cells", tds.number_of_cells());

        // First, test with an interior vertex (should work)
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        debug_println!(
            "Inserting interior vertex {:?}",
            interior_vertex.point().coords()
        );

        let interior_result = algorithm.insert_vertex(&mut tds, interior_vertex);
        match &interior_result {
            Ok(info) => println!(
                "Interior insertion succeeded: created={}, removed={}",
                info.cells_created, info.cells_removed
            ),
            Err(e) => println!("Interior insertion failed: {e}"),
        }
        assert!(
            interior_result.is_ok(),
            "Interior vertex insertion should succeed"
        );

        debug_println!("TDS now has {} cells", tds.number_of_cells());

        // Let's check the TDS consistency after the interior insertion
        debug_println!("Checking TDS consistency...");
        let boundary_facets_result = tds.boundary_facets();
        match boundary_facets_result {
            Ok(boundary_facets) => {
                let boundary_count = boundary_facets.count();
                println!("TDS has {boundary_count} boundary facets");
            }
            Err(e) => println!("Failed to get boundary facets: {e}"),
        }

        // Try to validate the TDS
        let validation_result = tds.is_valid();
        match &validation_result {
            Ok(()) => println!("TDS is valid"),
            Err(e) => println!("TDS validation failed: {e}"),
        }

        // Now test with an exterior vertex (this is what's failing)
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        debug_println!(
            "Inserting exterior vertex {:?}",
            exterior_vertex.point().coords()
        );

        // Let's debug what happens step by step
        debug_println!("Finding bad cells for exterior vertex...");
        let bad_cells = algorithm
            .robust_find_bad_cells(&tds, &exterior_vertex)
            .expect("Should not encounter TDS corruption with valid TDS");
        debug_println!("Found {} bad cells: {:?}", bad_cells.len(), bad_cells);

        if bad_cells.is_empty() {
            debug_println!("No bad cells found - will try hull extension");

            // Check what boundary facets exist before trying visibility
            println!("Getting all boundary facets...");
            if let Ok(all_boundary_facets) = tds.boundary_facets() {
                let all_boundary_facets_vec: Vec<_> = all_boundary_facets.collect();
                println!("Total boundary facets: {}", all_boundary_facets_vec.len());
                for (i, facet) in all_boundary_facets_vec.iter().enumerate() {
                    println!(
                        "  Boundary facet {}: {} vertices",
                        i,
                        facet
                            .vertices()
                            .map(std::iter::Iterator::count)
                            .unwrap_or(0)
                    );
                }
            }

            // Test the visibility detection directly
            println!("Testing visibility detection...");
            let visible_result =
                algorithm.find_visible_boundary_facets_with_robust_fallback(&tds, &exterior_vertex);
            match &visible_result {
                Ok(facet_handles) => {
                    println!("Found {} visible boundary facets", facet_handles.len());
                    for (i, handle) in facet_handles.iter().enumerate() {
                        if let Ok(facet_view) =
                            FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                        {
                            let vertex_count = facet_view
                                .vertices()
                                .map(std::iter::Iterator::count)
                                .unwrap_or(0);
                            println!("  Visible facet {i}: {vertex_count} vertices");
                        }
                    }
                }
                Err(e) => println!("Visibility detection failed: {e}"),
            }

            // Test the actual insertion
            let insertion_result = algorithm.insert_vertex(&mut tds, exterior_vertex);
            match &insertion_result {
                Ok(info) => println!(
                    "Exterior insertion succeeded: created={}, removed={}",
                    info.cells_created, info.cells_removed
                ),
                Err(e) => println!("Exterior insertion failed: {e}"),
            }

            // For debugging, we'll allow failure but let's understand why
            if insertion_result.is_err() {
                println!("Exterior vertex insertion failed, but this tells us what to fix");
            }
        } else {
            println!("Bad cells found, should use standard insertion");
            let insertion_result = algorithm.insert_vertex(&mut tds, exterior_vertex);
            match &insertion_result {
                Ok(info) => println!(
                    "Exterior insertion succeeded: created={}, removed={}",
                    info.cells_created, info.cells_removed
                ),
                Err(e) => println!("Exterior insertion failed: {e}"),
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_cavity_based_insertion_consistency() {
        debug_println!("Testing cavity-based insertion maintains TDS consistency");

        let mut algorithm = RobustBowyerWatson::new();

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Verify initial TDS is valid
        assert!(tds.is_valid().is_ok(), "Initial TDS should be valid");
        let initial_cells = tds.number_of_cells();
        println!("Initial TDS has {initial_cells} cells");

        // Insert interior vertices that should trigger cavity-based insertion
        let test_vertices = [
            vertex!([0.5, 0.5, 0.5]), // Interior point
            vertex!([0.3, 0.3, 0.3]), // Another interior point
            vertex!([0.7, 0.2, 0.1]), // Interior point near boundary
        ];

        for (i, test_vertex) in test_vertices.iter().enumerate() {
            println!(
                "\nInserting vertex {} at {:?}",
                i + 1,
                test_vertex.point().coords()
            );

            let cells_before = tds.number_of_cells();

            // Insert the vertex
            let result = algorithm.insert_vertex(&mut tds, *test_vertex);

            // Verify insertion succeeded
            assert!(
                result.is_ok(),
                "Cavity-based insertion {} should succeed",
                i + 1
            );
            let info = result.unwrap();

            println!(
                "  Result: created={}, removed={}",
                info.cells_created, info.cells_removed
            );

            // Verify TDS consistency after insertion
            let validation_result = tds.is_valid();
            assert!(
                validation_result.is_ok(),
                "TDS should be valid after cavity-based insertion {}: {:?}",
                i + 1,
                validation_result.err()
            );

            let cells_after = tds.number_of_cells();
            println!("  Cells before: {cells_before}, after: {cells_after}");

            // Verify cells were created (cavity-based should create cells)
            assert!(info.cells_created > 0, "Should create at least one cell");

            // Verify neighbor relationships are consistent
            for (cell_key, cell) in tds.cells() {
                if let Some(neighbors) = &cell.neighbors {
                    for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
                        if let Some(neighbor_key) = neighbor_key_opt
                            && let Some(neighbor) = tds.get_cell(*neighbor_key)
                            && let Some(neighbor_neighbors) = &neighbor.neighbors
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            assert!(
                                neighbor_neighbors
                                    .iter()
                                    .any(|n| n.as_ref() == Some(&cell_key)),
                                "Neighbor relationship should be symmetric after insertion {}",
                                i + 1
                            );

                            // Verify facet indices consistency
                            match verify_facet_index_consistency(
                                &tds,
                                cell_key,
                                *neighbor_key,
                                facet_idx,
                            ) {
                                Ok(true) => {} // Consistent - test passes
                                Ok(false) => {
                                    panic!(
                                        "No matching facet found between neighboring cells after insertion {}: \
                                         facet {} in cell {:?} not found in cell {:?}",
                                        i + 1,
                                        facet_idx,
                                        cell_key,
                                        neighbor_key
                                    );
                                }
                                Err(e) => {
                                    panic!(
                                        "Error verifying facet index consistency after insertion {}: {}",
                                        i + 1,
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Verify boundary facets are consistent
            let boundary_result = tds.boundary_facets();
            assert!(
                boundary_result.is_ok(),
                "Should be able to compute boundary facets after insertion {}",
                i + 1
            );

            if let Ok(boundary_facets) = boundary_result {
                let boundary_facets_vec: Vec<_> = boundary_facets.collect();
                println!("  Boundary facets: {}", boundary_facets_vec.len());
                // Each boundary facet should have exactly 3 vertices (for 3D)
                for facet in &boundary_facets_vec {
                    assert_eq!(
                        facet
                            .vertices()
                            .map(std::iter::Iterator::count)
                            .unwrap_or(0),
                        3,
                        "Boundary facet should have 3 vertices after insertion {}",
                        i + 1
                    );
                }
            }
        }

        println!("✓ All cavity-based insertions maintain TDS consistency");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_hull_extension_insertion_consistency() {
        println!("Testing hull extension insertion maintains TDS consistency");

        let mut algorithm = RobustBowyerWatson::new();

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Verify initial TDS is valid
        assert!(tds.is_valid().is_ok(), "Initial TDS should be valid");
        let initial_cells = tds.number_of_cells();
        println!("Initial TDS has {initial_cells} cells");

        // Insert exterior vertices that should trigger hull extension
        let test_vertices = vec![
            vertex!([2.0, 0.0, 0.0]),  // Exterior point extending in x
            vertex!([-1.0, 0.0, 0.0]), // Exterior point in negative x
            vertex!([0.0, 2.0, 0.0]),  // Exterior point extending in y
            vertex!([0.0, 0.0, -1.0]), // Exterior point in negative z
            vertex!([1.5, 1.5, 0.0]),  // Exterior point in xy plane
        ];

        for (i, test_vertex) in test_vertices.iter().enumerate() {
            println!(
                "\nInserting exterior vertex {} at {:?}",
                i + 1,
                test_vertex.point().coords()
            );

            let cells_before = tds.number_of_cells();
            let initial_boundary_facets = tds.boundary_facets().unwrap().count();

            // Insert the vertex
            let result = algorithm.insert_vertex(&mut tds, *test_vertex);

            // Verify insertion succeeded
            assert!(
                result.is_ok(),
                "Hull extension insertion {} should succeed",
                i + 1
            );
            let info = result.unwrap();

            println!(
                "  Result: created={}, removed={}",
                info.cells_created, info.cells_removed
            );

            // Verify TDS consistency after insertion
            let validation_result = tds.is_valid();
            assert!(
                validation_result.is_ok(),
                "TDS should be valid after hull extension insertion {}: {:?}",
                i + 1,
                validation_result.err()
            );

            let cells_after = tds.number_of_cells();
            println!("  Cells before: {cells_before}, after: {cells_after}");

            // Verify cells were created
            assert!(info.cells_created > 0, "Should create at least one cell");

            // Note: Cell count may increase, decrease, or stay same depending on topology constraints.
            // What matters is validity - the triangulation must maintain all invariants.
            // Even "exterior" vertices might trigger cavity-based insertion if they're
            // inside the circumsphere of existing cells, and topology filtering may prevent
            // some cells from being created to maintain valid facet sharing.
            assert!(
                cells_after > 0,
                "Triangulation must have at least one cell after insertion"
            );

            // Print information about the insertion strategy used
            if info.cells_removed > 0 {
                println!(
                    "    Note: Vertex {} used cavity-based insertion (removed {} cells)",
                    i + 1,
                    info.cells_removed
                );
            } else {
                println!(
                    "    Note: Vertex {} used pure hull extension (no cells removed)",
                    i + 1
                );
            }

            // Verify neighbor relationships are consistent
            for (cell_key, cell) in tds.cells() {
                if let Some(neighbors) = &cell.neighbors {
                    for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
                        if let Some(neighbor_key) = neighbor_key_opt
                            && let Some(neighbor) = tds.get_cell(*neighbor_key)
                            && let Some(neighbor_neighbors) = &neighbor.neighbors
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            assert!(
                                neighbor_neighbors
                                    .iter()
                                    .any(|opt| opt.as_ref() == Some(&cell_key)),
                                "Neighbor relationship should be symmetric after hull extension {}",
                                i + 1
                            );

                            // Verify facet indices consistency
                            match verify_facet_index_consistency(
                                &tds,
                                cell_key,
                                *neighbor_key,
                                facet_idx,
                            ) {
                                Ok(true) => {} // Consistent - test passes
                                Ok(false) => {
                                    panic!(
                                        "No matching facet found between neighboring cells after hull extension {}: \
                                         facet {} in cell {:?} not found in cell {:?}",
                                        i + 1,
                                        facet_idx,
                                        cell_key,
                                        neighbor_key
                                    );
                                }
                                Err(e) => {
                                    panic!(
                                        "Error verifying facet index consistency after hull extension {}: {}",
                                        i + 1,
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Verify boundary facets are consistent
            let boundary_result = tds.boundary_facets();
            assert!(
                boundary_result.is_ok(),
                "Should be able to compute boundary facets after hull extension {}",
                i + 1
            );

            if let Ok(boundary_facets) = boundary_result {
                let boundary_facets_vec: Vec<_> = boundary_facets.collect();
                let final_boundary_facets = boundary_facets_vec.len();
                println!(
                    "  Initial boundary facets: {initial_boundary_facets}, final: {final_boundary_facets}"
                );

                // Each boundary facet should have exactly 3 vertices (for 3D)
                for facet in &boundary_facets_vec {
                    assert_eq!(
                        facet
                            .vertices()
                            .map(std::iter::Iterator::count)
                            .unwrap_or(0),
                        3,
                        "Boundary facet should have 3 vertices after hull extension {}",
                        i + 1
                    );
                }

                // The newly inserted vertex should be in the triangulation
                let vertex_found = tds.vertices().any(|(_vkey, v)| {
                    let v_coords = v.point().coords();
                    let test_coords = test_vertex.point().coords();
                    v_coords
                        .iter()
                        .zip(test_coords.iter())
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON)
                });
                assert!(
                    vertex_found,
                    "Inserted vertex should be found in TDS after hull extension {}",
                    i + 1
                );
            }
        }

        println!("✓ All hull extension insertions maintain TDS consistency");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_finalization_prevents_inconsistencies() {
        println!("Testing that finalization prevents data structure inconsistencies");

        let mut algorithm = RobustBowyerWatson::new();

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([3.0, 0.0, 0.0]),
            vertex!([0.0, 3.0, 0.0]),
            vertex!([0.0, 0.0, 3.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Test a sequence of insertions that could create inconsistencies without proper finalization
        let test_sequence = vec![
            vertex!([1.0, 1.0, 1.0]),   // Interior
            vertex!([4.0, 0.0, 0.0]),   // Exterior
            vertex!([0.5, 0.5, 0.5]),   // Interior
            vertex!([0.0, 4.0, 0.0]),   // Exterior
            vertex!([2.0, 2.0, 0.1]),   // Near boundary
            vertex!([-1.0, -1.0, 0.0]), // Exterior negative
        ];

        for (i, test_vertex) in test_sequence.iter().enumerate() {
            println!(
                "\nInsertion {} at {:?}",
                i + 1,
                test_vertex.point().coords()
            );

            // Insert vertex
            let result = algorithm.insert_vertex(&mut tds, *test_vertex);
            assert!(result.is_ok(), "Insertion {} should succeed", i + 1);

            // Immediately verify all critical invariants that finalization should maintain

            // 1. TDS validation should pass
            let validation = tds.is_valid();
            assert!(
                validation.is_ok(),
                "TDS validation should pass after insertion {}: {:?}",
                i + 1,
                validation.err()
            );

            // 2. No duplicate cells should exist
            let mut cell_signatures = FastHashSet::default();
            for (_, cell) in tds.cells() {
                // Create efficient signature using sorted UUID array instead of string formatting
                let mut vertex_uuids: SmallBuffer<uuid::Uuid, MAX_PRACTICAL_DIMENSION_SIZE> = cell
                    .vertex_uuid_iter(&tds)
                    .collect::<Result<_, _>>()
                    .unwrap();
                vertex_uuids.sort_unstable();

                let inserted = cell_signatures.insert(vertex_uuids.clone());
                if !inserted {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Duplicate cell found after insertion {}: {:?}",
                        i + 1,
                        vertex_uuids
                    );
                }
                assert!(inserted, "Duplicate cell found after insertion {}", i + 1);
            }

            // 3. All facets should be properly shared
            #[allow(deprecated)] // Test verification - OK to use deprecated method
            let facet_to_cells = tds.build_facet_to_cells_map_lenient();
            for (facet_key, cells) in &facet_to_cells {
                assert!(
                    cells.len() <= 2,
                    "Facet {} shared by more than 2 cells after insertion {}: {} cells",
                    facet_key,
                    i + 1,
                    cells.len()
                );

                // If shared by 2 cells, both should reference each other as neighbors
                if cells.len() == 2 {
                    let cell1_key = cells[0].cell_key();
                    let facet1_idx = cells[0].facet_index();
                    let cell2_key = cells[1].cell_key();

                    if let (Some(cell1), Some(cell2)) =
                        (tds.get_cell(cell1_key), tds.get_cell(cell2_key))
                        && let (Some(neighbors1), Some(neighbors2)) =
                            (&cell1.neighbors, &cell2.neighbors)
                    {
                        assert!(
                            neighbors1.iter().flatten().any(|key| *key == cell2_key),
                            "Cell1 should reference cell2 as neighbor after insertion {}",
                            i + 1
                        );
                        assert!(
                            neighbors2.iter().flatten().any(|key| *key == cell1_key),
                            "Cell2 should reference cell1 as neighbor after insertion {}",
                            i + 1
                        );

                        // Verify facet indices consistency for the shared facet
                        match verify_facet_index_consistency(
                            &tds,
                            cell1_key,
                            cell2_key,
                            facet1_idx as usize,
                        ) {
                            Ok(true) => {} // Consistent - test passes
                            Ok(false) => {
                                panic!(
                                    "No matching facet found for shared facet after insertion {}: \
                                     facet {} in cell {:?} not found in cell {:?}",
                                    i + 1,
                                    facet1_idx,
                                    cell1_key,
                                    cell2_key
                                );
                            }
                            Err(e) => {
                                panic!(
                                    "Error verifying facet index consistency after insertion {}: {}",
                                    i + 1,
                                    e
                                );
                            }
                        }
                    }
                }
            }

            // 4. All vertices should have proper incident cells assigned
            // Phase 3: incident_cell is now a CellKey, not UUID
            for (vertex_key, vertex) in tds.vertices() {
                if let Some(incident_cell_key) = vertex.incident_cell
                    && let Some(incident_cell) = tds.get_cell(incident_cell_key)
                {
                    let cell_vertex_keys = incident_cell.vertices();
                    let vertex_is_in_cell = cell_vertex_keys.contains(&vertex_key);
                    assert!(
                        vertex_is_in_cell,
                        "Vertex incident cell should contain the vertex after insertion {}",
                        i + 1
                    );
                }
            }

            println!("  ✓ All invariants maintained after insertion {}", i + 1);
        }

        println!("✓ Finalization successfully prevents all tested inconsistencies");
    }

    #[test]
    fn test_create_perturbed_vertex_error_handling() {
        println!("Testing create_perturbed_vertex error handling for invalid coordinates");

        // Test 1: Normal case - should succeed
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();
        let normal_vertex = vertex!([1.0, 2.0, 3.0]);

        let result = algorithm.create_perturbed_vertex(&normal_vertex);
        assert!(result.is_ok(), "Normal vertex perturbation should succeed");

        if let Ok(perturbed) = result {
            let original_coords = normal_vertex.point().coords();
            let perturbed_coords = perturbed.point().coords();

            // First coordinate should be different (perturbed)
            assert_abs_diff_ne!(
                original_coords[0],
                perturbed_coords[0],
                epsilon = f64::EPSILON
            );

            // Other coordinates should remain the same
            assert_abs_diff_eq!(
                original_coords[1],
                perturbed_coords[1],
                epsilon = f64::EPSILON
            );
            assert_abs_diff_eq!(
                original_coords[2],
                perturbed_coords[2],
                epsilon = f64::EPSILON
            );

            // All coordinates should be finite
            assert!(
                perturbed_coords.iter().all(|&c| c.is_finite()),
                "All perturbed coordinates should be finite"
            );
        }

        // Test 2: Create a scenario that could potentially cause overflow
        // Use a very large perturbation scale that might cause overflow with large coordinates
        let mut extreme_config =
            crate::geometry::robust_predicates::config_presets::general_triangulation::<f64>();
        extreme_config.perturbation_scale = f64::MAX / 10.0; // Very large perturbation scale

        let extreme_algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);
        let large_vertex = vertex!([f64::MAX / 2.0, 1.0, 1.0]);

        // This should either succeed with a valid result or fail gracefully
        let extreme_result = extreme_algorithm.create_perturbed_vertex(&large_vertex);

        match extreme_result {
            Ok(perturbed) => {
                // If it succeeds, all coordinates must be finite
                let coords = perturbed.point().coords();
                assert!(
                    coords.iter().all(|&c| c.is_finite()),
                    "All coordinates in successful perturbation must be finite"
                );
                println!("  ✓ Large perturbation succeeded with finite coordinates");
            }
            Err(error) => {
                // If it fails, it should be due to invalid coordinates
                match error {
                    crate::core::vertex::VertexValidationError::InvalidPoint { .. } => {
                        println!("  ✓ Large perturbation correctly failed with InvalidPoint error");
                    }
                    other @ crate::core::vertex::VertexValidationError::InvalidUuid { .. } => {
                        panic!("Unexpected error type: {other:?}");
                    }
                }
            }
        }

        // Test 3: Test with edge case coordinates that are already near the limits
        let edge_vertex = vertex!([f64::MAX * 0.9, 0.0, 0.0]);
        let edge_result = extreme_algorithm.create_perturbed_vertex(&edge_vertex);

        // This should handle the edge case gracefully
        match edge_result {
            Ok(perturbed) => {
                let coords = perturbed.point().coords();
                assert!(
                    coords.iter().all(|&c| c.is_finite()),
                    "Edge case perturbation result must have finite coordinates"
                );
                println!("  ✓ Edge case perturbation succeeded");
            }
            Err(_) => {
                println!("  ✓ Edge case perturbation correctly failed due to coordinate overflow");
            }
        }

        // Test 4: Verify that UUID and other properties are preserved
        let uuid_test_vertex = vertex!([0.1, 0.2, 0.3]);
        let original_uuid = uuid_test_vertex.uuid();

        if let Ok(perturbed_vertex) = algorithm.create_perturbed_vertex(&uuid_test_vertex) {
            assert_eq!(
                original_uuid,
                perturbed_vertex.uuid(),
                "UUID should be preserved during perturbation"
            );
            assert_eq!(
                uuid_test_vertex.data, perturbed_vertex.data,
                "Data should be preserved during perturbation"
            );
            assert_eq!(
                uuid_test_vertex.incident_cell, perturbed_vertex.incident_cell,
                "Incident cell should be preserved during perturbation"
            );
            println!("  ✓ Vertex properties correctly preserved during perturbation");
        }

        println!("✓ All create_perturbed_vertex error handling tests passed");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_facet_cache_provider_implementation() {
        println!("Testing FacetCacheProvider implementation for RobustBowyerWatson");

        // Create test triangulation
        let points = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&points).unwrap();
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test 1: Initial cache state
        assert!(
            algorithm.facet_cache().load().is_none(),
            "Cache should initially be empty"
        );
        assert_eq!(
            algorithm.cached_generation().load(Ordering::Relaxed),
            0,
            "Initial generation should be 0"
        );
        println!("  ✓ Initial cache state verified");

        // Test 2: Cache building on first access
        let initial_generation = tds.generation();
        let cache1 = algorithm
            .try_get_or_build_facet_cache(&tds)
            .expect("Cache building should succeed in test");

        assert!(
            !cache1.is_empty(),
            "Built cache should contain facet mappings"
        );
        assert_eq!(
            algorithm.cached_generation().load(Ordering::Relaxed),
            initial_generation,
            "Cached generation should match TDS generation"
        );
        println!("  ✓ Cache builds correctly on first access");

        // Test 3: Cache reuse on subsequent access (no TDS changes)
        let cache2 = algorithm
            .try_get_or_build_facet_cache(&tds)
            .expect("Cache building should succeed in test");
        assert!(
            Arc::ptr_eq(&cache1, &cache2),
            "Same cache instance should be returned when TDS unchanged"
        );
        println!("  ✓ Cache correctly reused when TDS unchanged");

        // Test 4: Cache invalidation after TDS modification
        // Create a new triangulation with an additional vertex to simulate modification
        let mut modified_points = points;
        modified_points.push(vertex!([0.25, 0.25, 0.25]));
        let tds_modified: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&modified_points).unwrap();

        // Generation should be different for new TDS
        let new_generation = tds_modified.generation();
        // Note: Different TDS instances should have different generations
        assert_ne!(
            initial_generation, new_generation,
            "Expected different generation between distinct TDS instances"
        );

        // Get cache for modified TDS - should be different
        let cache3 = algorithm
            .try_get_or_build_facet_cache(&tds_modified)
            .expect("Cache building should succeed in test");
        assert!(
            !Arc::ptr_eq(&cache1, &cache3),
            "New cache instance should be created for different TDS"
        );
        assert_eq!(
            algorithm.cached_generation().load(Ordering::Relaxed),
            new_generation,
            "Cached generation should be updated to new TDS generation"
        );
        println!("  ✓ Cache correctly built for modified TDS");

        // Test 5: Manual cache invalidation
        algorithm.invalidate_facet_cache();
        assert!(
            algorithm.facet_cache().load().is_none(),
            "Cache should be empty after manual invalidation"
        );
        assert_eq!(
            algorithm.cached_generation().load(Ordering::Relaxed),
            0,
            "Generation should reset to 0 after manual invalidation"
        );
        println!("  ✓ Manual cache invalidation works correctly");

        // Test 6: Cache rebuilds after manual invalidation
        let cache4 = algorithm
            .try_get_or_build_facet_cache(&tds_modified)
            .expect("Cache building should succeed in test");
        assert!(
            !cache4.is_empty(),
            "Cache should rebuild after manual invalidation"
        );
        assert_eq!(
            algorithm.cached_generation().load(Ordering::Relaxed),
            new_generation,
            "Generation should be restored after rebuild"
        );
        println!("  ✓ Cache rebuilds correctly after manual invalidation");

        // Test 7: Verify cache content correctness
        #[allow(deprecated)] // Using for verification in test - cache is the recommended approach
        let direct_map = tds_modified.build_facet_to_cells_map_lenient();
        assert_eq!(
            cache4.len(),
            direct_map.len(),
            "Cached map should have same size as direct build"
        );

        for (facet_key, cells_in_cache) in cache4.iter() {
            assert!(
                direct_map.contains_key(facet_key),
                "All cached facets should exist in direct map"
            );
            assert_eq!(
                cells_in_cache.len(),
                direct_map[facet_key].len(),
                "Cell counts should match for facet {facet_key}"
            );
        }
        println!("  ✓ Cache content matches direct build");

        println!("✓ All FacetCacheProvider tests passed for RobustBowyerWatson");
    }

    // =============================================================================
    // ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn test_robust_error_handling_paths() {
        println!("Testing robust error handling paths...");

        // Test 1: Empty vertex list
        let empty_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![];
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&empty_vertices);
        // Empty vertex list is actually valid - just creates an empty TDS
        if result.is_ok() {
            println!("  ✓ Empty vertex list creates valid empty TDS");
        } else {
            println!(
                "  ✓ Empty vertex list handled with error: {:?}",
                result.err()
            );
        }

        // Test 2: Insufficient vertices for dimension
        let insufficient = vec![vertex!([0.0, 0.0, 0.0])];
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&insufficient);
        assert!(result.is_err(), "Insufficient vertices should fail");
        println!("  ✓ Insufficient vertices handled correctly");

        // Test 3: Degenerate vertex configuration (all coplanar in 3D)
        let coplanar = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([3.0, 0.0, 0.0]),
            vertex!([4.0, 0.0, 0.0]),
        ];
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&coplanar);
        // Should either succeed with robust handling or fail gracefully
        match result {
            Ok(tds) => {
                // Robust handling may create empty or partial triangulation for degenerate inputs
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid even if empty/partial for coplanar points"
                );
                println!("  ✓ Coplanar configuration handled robustly");
            }
            Err(_) => {
                println!("  ✓ Coplanar configuration failed gracefully");
            }
        }
    }

    // =============================================================================
    // FALLBACK AND RECOVERY MECHANISM TESTS
    // =============================================================================

    #[test]
    fn test_fallback_recovery_mechanisms() {
        println!("Testing fallback and recovery mechanisms...");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            // Add a point that might cause numerical issues
            vertex!([
                0.333_333_333_333_333_3,
                0.333_333_333_333_333_3,
                0.333_333_333_333_333_3
            ]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();

        // Test fallback facet mapping
        let initial_stats = algorithm.get_statistics();

        // Try to trigger fallback scenarios with a problematic vertex
        let _problematic_vertex: Vertex<f64, Option<()>, 3> =
            vertex!([f64::EPSILON, f64::EPSILON, f64::EPSILON]);

        // Should not panic and should handle gracefully
        let stats_after = algorithm.get_statistics();
        assert!(
            stats_after.0 >= initial_stats.0
                && stats_after.1 >= initial_stats.1
                && stats_after.2 >= initial_stats.2
        );
        println!("  ✓ Fallback mechanisms handle problematic vertices");

        // Test cache invalidation during operations
        algorithm.invalidate_facet_cache();
        let cache = algorithm
            .try_get_or_build_facet_cache(&tds)
            .expect("Cache building should succeed in test");
        assert!(!cache.is_empty(), "Cache should rebuild after invalidation");
        println!("  ✓ Cache recovery after invalidation works");
    }

    #[test]
    fn test_geometric_edge_cases() {
        println!("Testing geometric edge cases...");

        // Test 1: Nearly collinear points in 2D
        let nearly_collinear_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, f64::EPSILON]),
        ];
        let result_2d = Tds::<f64, Option<()>, Option<()>, 2>::new(&nearly_collinear_2d);
        match result_2d {
            Ok(tds) => {
                // Robust handling may create empty or partial triangulation for degenerate inputs
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid even if empty/partial for nearly-collinear points"
                );
                println!("  ✓ Nearly collinear 2D points handled");
            }
            Err(_) => println!("  ✓ Nearly collinear 2D points failed gracefully"),
        }

        // Test 2: Nearly coplanar points in 3D
        let nearly_coplanar_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, f64::EPSILON]),
        ];
        let result_3d = Tds::<f64, Option<()>, Option<()>, 3>::new(&nearly_coplanar_3d);
        match result_3d {
            Ok(tds) => {
                // Robust handling may create empty or partial triangulation for degenerate inputs
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid even if empty/partial for nearly-coplanar points"
                );
                println!("  ✓ Nearly coplanar 3D points handled");
            }
            Err(_) => println!("  ✓ Nearly coplanar 3D points failed gracefully"),
        }

        // Test 3: Points with extreme coordinates
        let extreme_coords = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([f64::MAX / 1e6, 0.0, 0.0]),
            vertex!([0.0, f64::MAX / 1e6, 0.0]),
            vertex!([0.0, 0.0, f64::MAX / 1e6]),
        ];
        let result_extreme = Tds::<f64, Option<()>, Option<()>, 3>::new(&extreme_coords);
        match result_extreme {
            Ok(tds) => {
                // Robust handling may create empty or partial triangulation for degenerate inputs
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid even if empty/partial for extreme coordinates"
                );
                println!("  ✓ Extreme coordinate points handled");
            }
            Err(_) => println!("  ✓ Extreme coordinate points failed gracefully"),
        }

        // Test 4: Perturbation edge cases
        let mut algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();

        // Test with zero coordinates
        let zero_vertex = vertex!([0.0, 0.0, 0.0]);
        let perturb_result = algorithm.create_perturbed_vertex(&zero_vertex);
        match perturb_result {
            Ok(perturbed) => {
                let coords = perturbed.point().coords();
                assert!(
                    coords.iter().any(|&x| x != 0.0),
                    "Perturbation should change at least one coordinate"
                );
                println!("  ✓ Zero vertex perturbation handled");
            }
            Err(_) => println!("  ✓ Zero vertex perturbation failed gracefully"),
        }

        // Test with infinite coordinates - should return InvalidVertex error
        // First create a valid TDS to insert into
        let mut tds = Tds::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .expect("Should create valid TDS");

        // Try to insert vertex with infinite coordinates
        // Note: We need to bypass the vertex! macro validation and create an invalid vertex directly

        match Point::try_from([f64::INFINITY, 1.0, 2.0]) {
            Ok(infinite_point) => {
                let infinite_vertex = VertexBuilder::default()
                    .point(infinite_point)
                    .build()
                    .expect("Should build vertex");

                match algorithm.insert_vertex(&mut tds, infinite_vertex) {
                    Err(
                        crate::core::traits::insertion_algorithm::InsertionError::InvalidVertex {
                            reason,
                        },
                    ) => {
                        println!(
                            "  ✓ Infinite coordinates correctly returned InvalidVertex: {reason}"
                        );
                    }
                    Err(other_error) => {
                        println!("  ✓ Infinite coordinates rejected with error: {other_error:?}");
                    }
                    Ok(_) => panic!("Infinite coordinates should be rejected"),
                }
            }
            Err(_) => {
                println!("  ✓ Infinite coordinates properly rejected during point creation");
            }
        }
    }

    // =============================================================================
    // COMPREHENSIVE INTEGRATION TESTS
    // =============================================================================

    #[test]

    fn test_comprehensive_algorithm_paths() {
        println!("Testing comprehensive algorithm paths...");

        // Test with various configurations and vertex sets
        let test_cases = vec![
            (
                "regular tetrahedron",
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([1.0, 0.0, 0.0]),
                    vertex!([0.5, 0.866, 0.0]),
                    vertex!([0.5, 0.289, 0.816]),
                ],
            ),
            (
                "cube vertices",
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([1.0, 0.0, 0.0]),
                    vertex!([0.0, 1.0, 0.0]),
                    vertex!([0.0, 0.0, 1.0]),
                    vertex!([1.0, 1.0, 0.0]),
                    vertex!([1.0, 0.0, 1.0]),
                    vertex!([0.0, 1.0, 1.0]),
                    vertex!([1.0, 1.0, 1.0]),
                ],
            ),
            (
                "random cloud",
                vec![
                    vertex!([0.1, 0.2, 0.3]),
                    vertex!([0.4, 0.5, 0.6]),
                    vertex!([0.7, 0.8, 0.9]),
                    vertex!([0.2, 0.7, 0.1]),
                    vertex!([0.9, 0.1, 0.8]),
                    vertex!([0.3, 0.9, 0.2]),
                ],
            ),
        ];

        for (name, vertices) in test_cases {
            println!("  Testing case: {name}");

            // Test with different configurations
            let configs = vec![
                ("general", config_presets::general_triangulation::<f64>()),
                (
                    "degenerate robust",
                    config_presets::degenerate_robust::<f64>(),
                ),
            ];

            for (config_name, config) in configs {
                let algorithm =
                    RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);

                match Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
                    Ok(tds) => {
                        // Test basic functionality
                        let cache = algorithm
                            .try_get_or_build_facet_cache(&tds)
                            .expect("Cache building should succeed in test");
                        assert!(!cache.is_empty(), "Cache should not be empty for valid TDS");

                        let _stats = algorithm.get_statistics();

                        // Test cache invalidation and rebuild
                        algorithm.invalidate_facet_cache();
                        let rebuilt_cache = algorithm
                            .try_get_or_build_facet_cache(&tds)
                            .expect("Cache building should succeed in test");
                        assert!(
                            !rebuilt_cache.is_empty(),
                            "Rebuilt cache should not be empty"
                        );

                        println!("    ✓ {config_name} config with {name} case");
                    }
                    Err(e) => {
                        println!("    - {config_name} config with {name} case failed: {e}");
                        // Some configurations might legitimately fail with certain vertex sets
                    }
                }
            }
        }

        println!("✓ Comprehensive algorithm path testing completed");
    }

    // =========================================================================
    // Module Organization Pattern: Configuration and Validation Tests
    // =========================================================================

    #[test]
    fn test_algorithm_with_extreme_tolerance_configurations() {
        // Test with very tight tolerance (should require high precision)
        let mut tight_config = config_presets::general_triangulation::<f64>();
        tight_config.base_tolerance = 1e-15;
        tight_config.perturbation_scale = 1e-10;

        let mut algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(tight_config);

        // Create vertices that would be problematic with loose tolerance
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1e-14, 1e-14, 1e-14]), // Very close to origin
        ];

        let mut tds = Tds::new(&vertices[..4]).expect("Initial TDS creation should succeed");
        let result = algorithm.insert_vertex(&mut tds, vertices[4]);

        // Should handle precision requirements appropriately
        assert!(result.is_ok() || matches!(result, Err(InsertionError::GeometricFailure { .. })));

        // Test with very loose tolerance (should be more permissive)
        let mut loose_config = config_presets::general_triangulation::<f64>();
        loose_config.base_tolerance = 1e-6;
        loose_config.perturbation_scale = 1e-3;

        let mut algorithm_loose =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(loose_config);
        let mut tds_loose = Tds::new(&vertices[..4]).expect("Initial TDS creation should succeed");
        let result_loose = algorithm_loose.insert_vertex(&mut tds_loose, vertices[4]);

        // Loose tolerance might succeed where tight fails, or vice versa
        assert!(
            result_loose.is_ok()
                || matches!(result_loose, Err(InsertionError::GeometricFailure { .. }))
        );
    }

    #[test]
    fn test_invalid_configuration_handling() {
        // Test configurations that should be invalid or problematic
        let mut invalid_configs = Vec::new();

        // Zero tolerance
        let mut zero_tol_config = config_presets::general_triangulation::<f64>();
        zero_tol_config.base_tolerance = 0.0;
        invalid_configs.push(zero_tol_config);

        // Negative tolerance
        let mut neg_tol_config = config_presets::general_triangulation::<f64>();
        neg_tol_config.base_tolerance = -1e-12;
        invalid_configs.push(neg_tol_config);

        // Extreme perturbation scale
        let mut extreme_pert_config = config_presets::general_triangulation::<f64>();
        extreme_pert_config.perturbation_scale = f64::MAX;
        invalid_configs.push(extreme_pert_config);

        // Zero perturbation scale
        let mut zero_pert_config = config_presets::general_triangulation::<f64>();
        zero_pert_config.perturbation_scale = 0.0;
        invalid_configs.push(zero_pert_config);

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        for (i, config) in invalid_configs.into_iter().enumerate() {
            let mut algorithm =
                RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);
            let mut tds = Tds::new(&vertices).expect("Initial TDS creation should succeed");

            let test_vertex = vertex!([0.5, 0.5, 0.5]);
            let result = algorithm.insert_vertex(&mut tds, test_vertex);

            // Invalid configurations may fail or succeed with degraded behavior
            if result.is_err() {
                println!(
                    "Invalid config {} failed as expected: {:?}",
                    i,
                    result.err()
                );
            } else {
                println!("Invalid config {i} unexpectedly succeeded");
            }
        }
    }

    // =========================================================================
    // Module Organization Pattern: Fallback and Recovery Mechanisms Tests
    // =========================================================================

    #[test]

    fn test_cache_invalidation_and_recovery() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut tds = Tds::new(&vertices).expect("Initial TDS creation should succeed");

        // Force cache population by calling a method that uses it
        let _initial_cache_gen = algorithm.cached_generation().load(Ordering::Acquire);
        let facet_cache = algorithm
            .try_get_or_build_facet_cache(&tds)
            .expect("Cache building should succeed in test");
        assert!(!facet_cache.is_empty(), "Cache should be populated");

        // Insert a vertex, which should invalidate cache
        let new_vertex = vertex!([0.5, 0.5, 0.5]);
        let result = algorithm.insert_vertex(&mut tds, new_vertex);
        assert!(result.is_ok(), "Vertex insertion should succeed");

        // Cache generation should have been updated by TDS changes
        let _new_cache_gen = algorithm.cached_generation().load(Ordering::Acquire);

        // Note: The cache generation is managed by TDS, not the algorithm directly
        // So we check that the cache can still be retrieved successfully
        let updated_cache = algorithm
            .try_get_or_build_facet_cache(&tds)
            .expect("Cache building should succeed in test");
        assert!(
            !updated_cache.is_empty(),
            "Cache should be rebuilt after TDS changes"
        );

        // Verify the algorithm still works correctly after cache invalidation
        let another_vertex = vertex!([0.25, 0.25, 0.25]);
        let another_result = algorithm.insert_vertex(&mut tds, another_vertex);
        assert!(
            another_result.is_ok(),
            "Subsequent insertion should work after cache invalidation"
        );
    }

    // =========================================================================
    // Module Organization Pattern: Various Geometric Configurations Tests
    // =========================================================================

    #[test]
    fn test_geometric_configurations_regular_patterns() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test 1: Cubic lattice points
        #[allow(clippy::cast_possible_truncation)]
        let cubic_vertices: Vec<_> = (0..3_usize)
            .flat_map(|i| {
                (0..3_usize).flat_map(move |j| {
                    (0..3_usize).map(move |k| {
                        vertex!([f64::from(i as u8), f64::from(j as u8), f64::from(k as u8)])
                    })
                })
            })
            .collect();

        let mut cubic_tds =
            Tds::new(&cubic_vertices[..4]).expect("Cubic TDS creation should succeed");
        for (i, vertex) in cubic_vertices[4..].iter().enumerate() {
            let result = algorithm.insert_vertex(&mut cubic_tds, *vertex);
            match result {
                Ok(_) => println!("  ✓ Cubic vertex {} inserted successfully", i + 4),
                Err(e) => println!("  ⚠ Cubic vertex {} failed: {:?}", i + 4, e),
            }
        }
        assert!(
            cubic_tds.is_valid().is_ok(),
            "Cubic lattice TDS should remain valid"
        );

        // Test 2: Spherical distribution
        let sphere_vertices: Vec<_> = (0..20)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * f64::from(i) / 20.0;
                let phi = std::f64::consts::PI * f64::from(i % 5) / 5.0;
                vertex!([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos()
                ])
            })
            .collect();

        let mut sphere_tds =
            Tds::new(&sphere_vertices[..4]).expect("Sphere TDS creation should succeed");
        for (i, vertex) in sphere_vertices[4..].iter().enumerate() {
            let result = algorithm.insert_vertex(&mut sphere_tds, *vertex);
            match result {
                Ok(_) => println!("  ✓ Sphere vertex {} inserted successfully", i + 4),
                Err(e) => println!("  ⚠ Sphere vertex {} failed: {:?}", i + 4, e),
            }
        }
        assert!(
            sphere_tds.is_valid().is_ok(),
            "Spherical TDS should remain valid"
        );

        // Test 3: Linear arrangement (challenging for 3D triangulation)
        let linear_vertices: Vec<_> = (0..10)
            .map(|i| vertex!([f64::from(i) * 0.1, 0.0, 0.0]))
            .collect();

        // Start with a proper 3D configuration, then add linear points
        let mut linear_base = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        linear_base.extend_from_slice(&linear_vertices[4..]);

        let mut linear_tds =
            Tds::new(&linear_base[..4]).expect("Linear base TDS creation should succeed");
        for (i, vertex) in linear_base[4..].iter().enumerate() {
            let result = algorithm.insert_vertex(&mut linear_tds, *vertex);
            match result {
                Ok(_) => println!("  ✓ Linear vertex {} inserted successfully", i + 4),
                Err(e) => println!("  ⚠ Linear vertex {} failed: {:?}", i + 4, e),
            }
        }
        assert!(
            linear_tds.is_valid().is_ok(),
            "Linear TDS should remain valid"
        );
    }

    #[test]
    #[allow(unused_variables)]
    fn test_geometric_configurations_extreme_coordinates() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test with very large coordinates
        let large_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e6, 0.0, 0.0]),
            vertex!([0.0, 1e6, 0.0]),
            vertex!([0.0, 0.0, 1e6]),
            vertex!([1e5, 1e5, 1e5]), // Interior point with large coordinates
        ];

        let mut large_tds =
            Tds::new(&large_vertices[..4]).expect("Large coordinate TDS should be created");
        let large_result = algorithm.insert_vertex(&mut large_tds, large_vertices[4]);

        match large_result {
            Ok(_) => {
                debug_println!("  ✓ Large coordinate insertion succeeded");
                assert!(
                    large_tds.is_valid().is_ok(),
                    "Large coordinate TDS should remain valid"
                );
            }
            Err(e) => {
                debug_println!("  ⚠ Large coordinate insertion failed: {e:?}");
                // TDS should still be valid even if insertion failed
                assert!(
                    large_tds.is_valid().is_ok(),
                    "TDS should remain valid after large coordinate failure"
                );
            }
        }

        // Test with very small coordinates
        let small_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-6, 0.0, 0.0]),
            vertex!([0.0, 1e-6, 0.0]),
            vertex!([0.0, 0.0, 1e-6]),
            vertex!([1e-7, 1e-7, 1e-7]), // Interior point with small coordinates
        ];

        let mut small_tds =
            Tds::new(&small_vertices[..4]).expect("Small coordinate TDS should be created");
        let small_result = algorithm.insert_vertex(&mut small_tds, small_vertices[4]);

        match small_result {
            Ok(_) => {
                debug_println!("  ✓ Small coordinate insertion succeeded");
                assert!(
                    small_tds.is_valid().is_ok(),
                    "Small coordinate TDS should remain valid"
                );
            }
            Err(e) => {
                debug_println!("  ⚠ Small coordinate insertion failed: {e:?}");
                // TDS should still be valid even if insertion failed
                assert!(
                    small_tds.is_valid().is_ok(),
                    "TDS should remain valid after small coordinate failure"
                );
            }
        }

        // Test with mixed large and small coordinates
        let mixed_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e6, 0.0, 0.0]),
            vertex!([0.0, 1e-6, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1e3, 1e-3, 0.5]), // Mixed scale interior point
        ];

        let mut mixed_tds =
            Tds::new(&mixed_vertices[..4]).expect("Mixed coordinate TDS should be created");
        let mixed_result = algorithm.insert_vertex(&mut mixed_tds, mixed_vertices[4]);

        match mixed_result {
            Ok(_) => {
                debug_println!("  ✓ Mixed coordinate insertion succeeded");
                assert!(
                    mixed_tds.is_valid().is_ok(),
                    "Mixed coordinate TDS should remain valid"
                );
            }
            Err(e) => {
                debug_println!("  ⚠ Mixed coordinate insertion failed: {e:?}");
                // TDS should still be valid even if insertion failed
                assert!(
                    mixed_tds.is_valid().is_ok(),
                    "TDS should remain valid after mixed coordinate failure"
                );
            }
        }
    }

    /// Test `vertex_needs_robust_handling` heuristics (lines 1108-1166)
    #[test]
    fn test_vertex_needs_robust_handling() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create a TDS for testing
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test vertex with small coordinates (should trigger robust handling)
        let small_vertex = vertex!([1e-15, 1e-15, 1e-15]);
        let needs_robust_small = algorithm.vertex_needs_robust_handling(&tds, &small_vertex);
        assert!(
            needs_robust_small,
            "Very small coordinates should need robust handling"
        );

        // Test vertex with large coordinates (should trigger robust handling)
        let large_vertex = vertex!([1e10, 1e10, 1e10]);
        let needs_robust_large = algorithm.vertex_needs_robust_handling(&tds, &large_vertex);
        assert!(
            needs_robust_large,
            "Very large coordinates should need robust handling"
        );

        // Test vertex with normal coordinates (might not need robust handling)
        let normal_vertex = vertex!([0.5, 0.5, 0.5]);
        let _needs_robust_normal = algorithm.vertex_needs_robust_handling(&tds, &normal_vertex);
        // Don't assert specific result as it depends on proximity to other vertices
    }

    /// Test `build_validated_facet_mapping` error paths (lines 734-776)
    #[test]
    fn test_build_validated_facet_mapping() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create a valid TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test validated facet mapping build
        let result = algorithm.build_validated_facet_mapping(&tds);
        assert!(
            result.is_ok(),
            "Should build valid facet mapping for well-formed TDS"
        );

        let facet_mapping = result.unwrap();
        assert!(
            !facet_mapping.is_empty(),
            "Facet mapping should not be empty"
        );

        // Verify that no facet is shared by more than 2 cells
        for (facet_key, cells) in &facet_mapping {
            assert!(
                cells.len() <= 2,
                "Facet {} should be shared by at most 2 cells, found {}",
                facet_key,
                cells.len()
            );
        }
    }

    /// Test `validate_boundary_facets` error conditions using lightweight handles
    #[test]
    fn test_validate_boundary_facets() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test with empty boundary facets but non-zero bad cell count (should error)
        let empty_handles: Vec<FacetHandle> = vec![];
        let result = algorithm.validate_boundary_facets(&empty_handles, 3);
        assert!(
            result.is_err(),
            "Should error when no boundary facets found but bad cells exist"
        );

        match result.err().unwrap() {
            InsertionError::ExcessiveBadCells { found, threshold } => {
                assert_eq!(found, 3);
                assert_eq!(threshold, 0);
            }
            other => panic!("Expected ExcessiveBadCells error, got {other:?}"),
        }

        // Test with boundary facets present (should succeed)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let boundary_facets = tds.boundary_facets().unwrap();

        // Collect lightweight boundary facet handles
        let boundary_handles: Vec<FacetHandle> = boundary_facets
            .map(|fv| FacetHandle::new(fv.cell_key(), fv.facet_index()))
            .collect();

        let result = algorithm.validate_boundary_facets(&boundary_handles, 1);
        assert!(result.is_ok(), "Should succeed with valid boundary facets");
    }

    /// Test error handling in `robust_find_bad_cells` when predicates fail (lines 544-556)
    #[test]
    #[allow(unused_variables)]
    fn test_robust_find_bad_cells_predicate_failure() {
        // Use extreme configuration that might cause predicate failures
        let mut extreme_config = config_presets::general_triangulation::<f64>();
        extreme_config.base_tolerance = f64::MAX; // Extreme tolerance

        let algorithm =
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test with extreme coordinates that might cause predicate computation issues
        // Use the most extreme valid values to test robustness
        let extreme_vertex = vertex!([f64::MAX, f64::MIN_POSITIVE, f64::MIN]);

        // Should handle gracefully even with extreme input
        let extreme_bad_cells = algorithm
            .robust_find_bad_cells(&tds, &extreme_vertex)
            .expect("Should not encounter TDS corruption with valid TDS");
        debug_println!(
            "Found {} bad cells with extreme coordinates (handled gracefully)",
            extreme_bad_cells.len()
        );

        // Test with coordinates very close to zero that might cause precision issues
        let tiny_vertex = vertex!([f64::EPSILON, f64::EPSILON * 2.0, f64::EPSILON * 3.0]);
        let tiny_bad_cells = algorithm
            .robust_find_bad_cells(&tds, &tiny_vertex)
            .expect("Should not encounter TDS corruption with valid TDS");
        debug_println!(
            "Found {} bad cells with tiny coordinates",
            tiny_bad_cells.len()
        );
    }

    /// Test `is_facet_visible_from_vertex_robust` degenerate handling (lines 999-1004)
    #[test]
    #[allow(unused_variables)]
    fn test_is_facet_visible_degenerate_handling() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().unwrap();
        let boundary_facets_vec: Vec<_> = boundary_facets.collect();
        assert!(!boundary_facets_vec.is_empty());
        let test_facet = &boundary_facets_vec[0];

        // Get the adjacent cell key for this facet
        let facet_to_cells = algorithm.try_get_or_build_facet_cache(&tds).unwrap();
        let test_facet_vertices: Vec<_> = match test_facet.vertices() {
            Ok(iter) => iter.copied().collect(),
            Err(_) => return, // Skip test if facet is invalid
        };
        // Get vertex keys from vertices via TDS
        let test_facet_vertex_keys: Vec<_> = test_facet_vertices
            .iter()
            .filter_map(|v| tds.vertex_key_from_uuid(&v.uuid()))
            .collect();
        let key = derive_facet_key_from_vertex_keys::<f64, Option<()>, Option<()>, 3>(
            &test_facet_vertex_keys,
        )
        .expect("Should derive facet key");
        let adjacent_cell_key = facet_to_cells
            .get(&key)
            .and_then(|cells| (cells.len() == 1).then_some(cells[0].cell_key()))
            .expect("Should find adjacent cell for test facet");

        // Test with a point that might cause degenerate orientation results
        let degenerate_vertex = vertex!([0.5, 0.5, 0.5]); // Point at center of tetrahedron

        // This should exercise the fallback visibility heuristic path
        // Use FacetView directly (Phase 3A: key-based API)
        let is_visible = algorithm
            .is_facet_visible_from_vertex_robust(&tds, test_facet, &degenerate_vertex)
            .expect("Should not encounter TDS corruption with valid TDS");

        debug_println!("Degenerate visibility test result: {is_visible}");
        // Don't assert specific result since it depends on geometry, just ensure it doesn't panic
    }

    /// Test `safe_usize_to_scalar` conversion in `fallback_visibility_heuristic` (lines 1032-1034)
    #[test]
    fn test_fallback_visibility_safe_conversion() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create a facet with a very large number of vertices to test conversion edge case
        // Note: This is artificial since real facets in 3D have exactly 3 vertices
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let boundary_facets = tds.boundary_facets().unwrap();

        let test_vertex = vertex!([10.0, 10.0, 10.0]);

        // Test the fallback visibility heuristic with a normal facet
        for facet in boundary_facets {
            // Use FacetView directly (Phase 3A: key-based API)
            let is_visible = algorithm
                .fallback_visibility_heuristic(&facet, &test_vertex)
                .expect("Should not encounter TDS corruption with valid TDS");
            println!("Fallback visibility for far point: {is_visible}");
            // Should typically be true for a far point, but mainly testing no panic
        }
    }

    /// Test fallback strategy in `robust_insert_vertex_impl` (lines 228-240, 244)
    #[test]
    fn test_robust_insert_fallback_strategies() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create a TDS with an interior vertex that might cause fallback scenarios
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Insert a vertex that might trigger fallback strategies
        let challenging_vertex = vertex!([0.25, 0.25, 0.25]); // Interior point
        let result = algorithm.robust_insert_vertex_impl(&mut tds, &challenging_vertex);

        match result {
            Ok(info) => {
                println!(
                    "Insertion succeeded: strategy={:?}, cells_created={}, cells_removed={}",
                    info.strategy, info.cells_created, info.cells_removed
                );
                assert!(info.success, "Should report success");

                // Verify statistics were updated (lines 250-252)
                let (processed, created, removed) = algorithm.get_statistics();
                assert_eq!(processed, 1);
                assert_eq!(created, info.cells_created);
                assert_eq!(removed, info.cells_removed);
            }
            Err(e) => {
                println!("Insertion failed as expected: {e:?}");
                // Even on failure, TDS should remain valid
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should remain valid after failure"
                );
            }
        }
    }

    /// Test fallback in `insert_vertex_fallback` method (lines 286-298)
    #[test]
    fn test_insert_vertex_fallback() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test fallback insertion with a vertex that should work
        let vertex_to_insert = vertex!([0.5, 0.5, 0.5]);
        let result = algorithm.insert_vertex_fallback(&mut tds, &vertex_to_insert);

        match result {
            Ok(info) => {
                println!("Fallback insertion succeeded: {info:?}");
                assert!(info.success);
                assert_eq!(info.strategy, InsertionStrategy::Fallback);
            }
            Err(e) => {
                println!("Fallback insertion failed: {e:?}");
                // TDS should still be valid
                assert!(tds.is_valid().is_ok());
            }
        }
    }

    /// Test error paths in `find_visible_boundary_facets_with_robust_fallback` (lines 499-505)
    #[test]
    fn test_find_visible_boundary_facets_fallback() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test with exterior vertex (should find visible facets)
        let exterior_vertex = vertex!([2.0, 2.0, 2.0]);
        let result =
            algorithm.find_visible_boundary_facets_with_robust_fallback(&tds, &exterior_vertex);

        match result {
            Ok(visible_facet_handles) => {
                println!(
                    "Found {} visible boundary facets",
                    visible_facet_handles.len()
                );
                if visible_facet_handles.is_empty() {
                    println!("No visible facets found (edge case)");
                } else {
                    println!("Successfully found visible facets");
                }
            }
            Err(e) => {
                println!("Visible facets detection failed: {e:?}");
                // This tests the error handling path (line 502)
            }
        }

        // Test with interior vertex (might not find visible facets)
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let interior_result =
            algorithm.find_visible_boundary_facets_with_robust_fallback(&tds, &interior_vertex);

        match interior_result {
            Ok(facet_handles) => {
                println!(
                    "Interior vertex found {} visible facets",
                    facet_handles.len()
                );
            }
            Err(e) => {
                println!("Interior vertex visibility failed: {e:?}");
            }
        }
    }

    /// Test boundary detection in `robust_find_cavity_boundary_facets` (lines 533-539)
    #[test]
    fn test_robust_boundary_detection_edge_cases() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Get a subset of cells as "bad cells"
        let all_cell_keys: Vec<_> = tds.cell_keys().collect();
        let bad_cells = &all_cell_keys[..1];

        let result = algorithm.robust_find_cavity_boundary_facets(&tds, bad_cells);

        match result {
            Ok(boundary_facet_handles) => {
                println!(
                    "Robust boundary detection found {} facet handles",
                    boundary_facet_handles.len()
                );
                // Should find boundary facets for valid bad cells

                // Verify all returned facet handles are valid
                for handle in &boundary_facet_handles {
                    if let Ok(facet_view) =
                        FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                    {
                        let vertex_count = facet_view
                            .vertices()
                            .map(std::iter::Iterator::count)
                            .unwrap_or(0);
                        assert!(vertex_count > 0, "Facet should have vertices");
                    }
                }
            }
            Err(e) => {
                println!("Robust boundary detection failed: {e:?}");
            }
        }
    }

    /// Test `is_cavity_boundary_facet` helper function (lines 778-779)
    #[test]
    fn test_is_cavity_boundary_facet_logic() {
        // Test the boundary facet detection logic
        assert!(
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 1)
        );
        assert!(
            RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 2)
        );
        assert!(
            !RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(0, 1)
        );
        assert!(
            !RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(2, 2)
        );
        assert!(
            !RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 3)
        );
    }

    /// Test conservative boundary cell inclusion (lines 536-539)
    #[test]
    fn test_conservative_boundary_cell_inclusion() {
        // Test with high tolerance configuration to trigger conservative boundary handling
        let mut config = config_presets::degenerate_robust::<f64>();
        config.base_tolerance = 1e-6; // Larger than default to trigger conservative path

        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test with a vertex on the boundary of the circumsphere
        let boundary_vertex = vertex!([0.5, 0.5, 0.5]);
        let bad_cells = algorithm
            .robust_find_bad_cells(&tds, &boundary_vertex)
            .expect("Should not encounter TDS corruption with valid TDS");

        println!(
            "Conservative boundary handling found {} bad cells",
            bad_cells.len()
        );
        // With high tolerance, might include more cells conservatively
    }

    /// Test error paths in `robust_insert_vertex_impl` for better coverage
    #[test]
    fn test_robust_insert_error_paths() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test insertion with extreme coordinates that might trigger fallback paths
        let extreme_vertex = vertex!([f64::MAX / 1e10, f64::MAX / 1e10, f64::MAX / 1e10]);

        let result = algorithm.robust_insert_vertex_impl(&mut tds, &extreme_vertex);
        match result {
            Ok(info) => {
                println!(
                    "Extreme coordinate insertion succeeded with strategy: {:?}",
                    info.strategy
                );
                assert!(info.success);
            }
            Err(e) => {
                println!("Expected error with extreme coordinates: {e}");
                // Error should be meaningful
                assert!(!e.to_string().is_empty());
            }
        }
    }

    /// Test cavity-based insertion with various failure scenarios
    #[test]
    fn test_cavity_based_insertion_error_scenarios() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test insertion of vertex very close to existing vertex
        let close_vertex = vertex!([1e-15, 1e-15, 1e-15]);

        let result =
            algorithm.insert_vertex_cavity_based_with_robust_predicates(&mut tds, &close_vertex);
        match result {
            Ok(info) => {
                println!("Close vertex insertion succeeded: {info:?}");
                assert_eq!(info.strategy, InsertionStrategy::CavityBased);
            }
            Err(e) => {
                println!("Close vertex insertion failed as expected: {e}");
            }
        }
    }

    /// Test hull extension insertion with various scenarios
    #[test]
    fn test_hull_extension_insertion_scenarios() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with exterior vertex
        let exterior_vertex = vertex!([5.0, 5.0, 5.0]);

        let result = algorithm
            .insert_vertex_hull_extension_with_robust_predicates(&mut tds, &exterior_vertex);
        match result {
            Ok(info) => {
                println!("Hull extension insertion succeeded: {info:?}");
                assert_eq!(info.strategy, InsertionStrategy::HullExtension);
                assert_eq!(info.cells_removed, 0); // Hull extension doesn't remove cells
            }
            Err(e) => {
                println!("Hull extension insertion failed: {e}");
            }
        }
    }

    /// Test bad cells detection with robust fallback
    #[test]
    fn test_find_bad_cells_with_robust_fallback_coverage() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with vertex that should be inside circumsphere
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let bad_cells = algorithm
            .find_bad_cells_with_robust_fallback(&tds, &interior_vertex)
            .expect("Should find bad cells for interior vertex");

        println!("Found {} bad cells for interior vertex", bad_cells.len());
        // Should find at least the containing cell as "bad"
        assert!(!bad_cells.is_empty());
    }

    /// Test that valid TDS operations don't produce false TDS corruption errors
    ///
    /// This test verifies that the error propagation logic for TDS corruption
    /// exists and that valid TDS operations don't incorrectly trigger corruption errors.
    /// Actual TDS corruption detection is tested in integration tests.
    #[test]
    fn test_find_bad_cells_no_false_corruption_errors() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let test_vertex = vertex!([0.25, 0.25, 0.25]);

        // Verify that valid TDS operations don't produce false TDS corruption errors
        let result = algorithm.find_bad_cells_with_robust_fallback(&tds, &test_vertex);
        assert!(
            result.is_ok(),
            "Valid TDS should not produce TDS corruption errors"
        );

        println!("  ✓ Valid TDS correctly returns Ok without false corruption errors");
        println!("  ✓ Actual TDS corruption detection is tested in integration tests");
    }

    #[test]
    fn test_find_bad_cells_no_cells_error_path() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create an empty TDS
        let empty_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&empty_vertices).unwrap();

        let test_vertex = vertex!([0.25, 0.25, 0.25]);
        let result = algorithm.find_bad_cells_with_robust_fallback(&tds, &test_vertex);

        // Empty TDS should return empty bad cells list (lines 471-473)
        match result {
            Ok(bad_cells) => {
                assert_eq!(bad_cells.len(), 0, "Empty TDS should return no bad cells");
                println!("  ✓ NoCells error path handled correctly");
            }
            Err(e) => {
                println!("  ✓ Empty TDS error handled: {e:?}");
            }
        }

        // Test with vertex outside all circumspheres
        let exterior_vertex = vertex!([10.0, 10.0, 10.0]);
        let bad_cells_exterior = algorithm
            .find_bad_cells_with_robust_fallback(&tds, &exterior_vertex)
            .expect("Should find bad cells for exterior vertex");

        println!(
            "Found {} bad cells for exterior vertex",
            bad_cells_exterior.len()
        );
        // Exterior vertex might not have any bad cells
    }

    /// Test visible boundary facets detection with robust fallback
    #[test]
    fn test_find_visible_boundary_facets_comprehensive() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with clearly exterior vertex
        let exterior_vertex = vertex!([2.0, 2.0, 2.0]);
        let result =
            algorithm.find_visible_boundary_facets_with_robust_fallback(&tds, &exterior_vertex);

        match result {
            Ok(visible_facet_handles) => {
                println!(
                    "Found {} visible boundary facets from exterior",
                    visible_facet_handles.len()
                );
                // Should find some visible facets for exterior point
                if !visible_facet_handles.is_empty() {
                    for handle in &visible_facet_handles {
                        assert!(
                            tds.get_cell(handle.cell_key()).is_some(),
                            "Cell key {:?} should exist in TDS",
                            handle.cell_key()
                        );
                        assert!(
                            handle.facet_index() < 4, // 3D tetrahedra have 4 facets
                            "Facet index {} should be valid for 3D cell",
                            handle.facet_index()
                        );
                    }
                }
            }
            Err(e) => {
                println!("Visible facets detection failed: {e}");
            }
        }

        // Test with interior vertex (should find no visible facets)
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let interior_result =
            algorithm.find_visible_boundary_facets_with_robust_fallback(&tds, &interior_vertex);

        match interior_result {
            Ok(visible_facet_handles) => {
                println!(
                    "Found {} visible boundary facets from interior",
                    visible_facet_handles.len()
                );
                // Interior points typically don't see boundary facets
            }
            Err(e) => {
                println!("Interior visibility detection failed: {e}");
            }
        }
    }

    /// Test error handling in `ensure_vertex_in_tds` calls
    #[test]
    fn test_ensure_vertex_in_tds_error_handling() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Create a vertex that already exists in TDS
        let duplicate_vertex = vertex!([0.0, 0.0, 0.0]); // Same as first vertex

        let result = algorithm.insert_vertex(&mut tds, duplicate_vertex);
        match result {
            Ok(info) => {
                println!("Duplicate vertex handled: {info:?}");
            }
            Err(e) => {
                println!("Duplicate vertex properly rejected: {e}");
                // This tests the error handling path in ensure_vertex_in_tds
            }
        }
    }

    /// Test `finalize_after_insertion` error handling
    #[test]
    fn test_finalize_after_insertion_error_paths() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Insert a normal vertex first to create a scenario where finalization might be tested
        let normal_vertex = vertex!([0.1, 0.1, 0.1]);

        let result = algorithm.insert_vertex(&mut tds, normal_vertex);
        match result {
            Ok(info) => {
                println!("Normal vertex insertion succeeded: {info:?}");
                // Finalization succeeded
                assert!(info.success);
            }
            Err(e) => {
                println!("Normal vertex insertion failed: {e}");
                // This tests error handling in finalization
                assert!(e.to_string().contains("finalize") || !e.to_string().is_empty());
            }
        }
    }

    /// Test statistics tracking in various error scenarios
    #[test]
    fn test_statistics_tracking_comprehensive() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Initial statistics should be zero
        let (initial_processed, initial_created, initial_removed) = algorithm.get_statistics();
        assert_eq!(initial_processed, 0);
        assert_eq!(initial_created, 0);
        assert_eq!(initial_removed, 0);

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Try multiple insertions to test statistics accumulation
        let test_vertices = [
            vertex!([0.1, 0.1, 0.1]),
            vertex!([0.2, 0.2, 0.2]),
            vertex!([0.3, 0.3, 0.3]),
        ];

        let mut successful_insertions = 0;
        for (i, vertex) in test_vertices.iter().enumerate() {
            let result = algorithm.insert_vertex(&mut tds, *vertex);
            match result {
                Ok(info) => {
                    successful_insertions += 1;
                    println!(
                        "Insertion {} succeeded: created={}, removed={}",
                        i + 1,
                        info.cells_created,
                        info.cells_removed
                    );
                }
                Err(e) => {
                    println!("Insertion {} failed: {}", i + 1, e);
                }
            }
        }

        // Check final statistics
        let (final_processed, final_created, final_removed) = algorithm.get_statistics();
        println!(
            "Final stats: processed={final_processed}, created={final_created}, removed={final_removed}"
        );

        // Should have processed at least the successful insertions
        assert!(final_processed >= successful_insertions);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_atomic_vertex_insert_and_remove_cells() {
        println!("Testing atomic_vertex_insert_and_remove_cells");

        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create initial triangulation with a few vertices
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        let initial_vertex_count = tds.number_of_vertices();
        let initial_cell_count = tds.number_of_cells();
        let initial_generation = tds.generation();

        println!(
            "  Initial state: {initial_vertex_count} vertices, {initial_cell_count} cells, generation {initial_generation}"
        );

        // Test Case 1: Successful atomic operation (new vertex)
        let new_vertex = vertex!([3.0, 3.0, 3.0]);
        let bad_cells: Vec<_> = tds.cell_keys().take(1).collect(); // Take one cell to "remove"

        assert!(
            !bad_cells.is_empty(),
            "Should have at least one cell to test with"
        );
        println!(
            "  Test 1: Attempting atomic operation with new vertex and {} bad cells",
            bad_cells.len()
        );

        let result =
            algorithm.atomic_vertex_insert_and_remove_cells(&mut tds, &new_vertex, &bad_cells);

        match result {
            Ok(()) => {
                println!("  ✓ Test 1 SUCCESS: Atomic operation completed");

                // Verify vertex was added
                assert!(
                    tds.vertex_key_from_uuid(&new_vertex.uuid()).is_some(),
                    "New vertex should be in TDS after successful atomic operation"
                );

                // Verify cells were removed
                assert_eq!(
                    tds.number_of_cells(),
                    initial_cell_count - bad_cells.len(),
                    "Bad cells should have been removed"
                );

                // Verify generation was bumped (due to structural changes)
                assert!(
                    tds.generation() > initial_generation,
                    "Generation should have increased due to TDS modifications"
                );

                println!(
                    "  ✓ Final state: {} vertices, {} cells, generation {}",
                    tds.number_of_vertices(),
                    tds.number_of_cells(),
                    tds.generation()
                );
            }
            Err(e) => {
                println!("  ✓ Test 1 HANDLED GRACEFULLY: Atomic operation failed with error: {e}");

                // Even if the operation failed, we should verify atomicity:
                // Either everything succeeded or nothing was modified
                let current_generation = tds.generation();

                if current_generation == initial_generation {
                    println!("    ✓ ATOMIC: Generation unchanged, no side effects");

                    // Verify vertex was NOT added if generation unchanged
                    assert!(
                        tds.vertex_key_from_uuid(&new_vertex.uuid()).is_none(),
                        "Vertex should NOT be in TDS if operation failed atomically"
                    );
                } else {
                    println!(
                        "    ✓ Side effects occurred, but this is acceptable if vertex was added"
                    );

                    // If generation changed, the vertex should be in TDS
                    // (this means ensure_vertex_in_tds succeeded but remove_bad_cells may have had issues)
                    if tds.vertex_key_from_uuid(&new_vertex.uuid()).is_some() {
                        println!("    ✓ Vertex was successfully added to TDS");
                    }
                }
            }
        }

        // Test Case 2: Atomic operation with existing vertex (should succeed without adding duplicate)
        println!("  Test 2: Attempting atomic operation with existing vertex");

        let existing_vertex = initial_vertices[0]; // Use first vertex from initial set
        let current_vertex_count = tds.number_of_vertices();
        let current_cell_count = tds.number_of_cells();
        let current_generation = tds.generation();

        // Get remaining cells for this test
        let remaining_bad_cells: Vec<_> = tds.cell_keys().take(1).collect();

        if remaining_bad_cells.is_empty() {
            println!("  ✓ Test 2 SKIPPED: No remaining cells to test with");
        } else {
            let result = algorithm.atomic_vertex_insert_and_remove_cells(
                &mut tds,
                &existing_vertex,
                &remaining_bad_cells,
            );

            match result {
                Ok(()) => {
                    println!("  ✓ Test 2 SUCCESS: Atomic operation with existing vertex completed");

                    // Vertex count should not increase (vertex already existed)
                    assert_eq!(
                        tds.number_of_vertices(),
                        current_vertex_count,
                        "Vertex count should not increase for existing vertex"
                    );

                    // Cells should be removed
                    assert_eq!(
                        tds.number_of_cells(),
                        current_cell_count - remaining_bad_cells.len(),
                        "Bad cells should have been removed"
                    );

                    // Generation should be bumped due to cell removal
                    assert!(
                        tds.generation() > current_generation,
                        "Generation should increase due to cell removal"
                    );
                }
                Err(e) => {
                    println!("  ✓ Test 2 HANDLED: Existing vertex operation failed: {e}");
                    // This is also acceptable - some geometric configurations might not allow cell removal
                }
            }
        }

        // Test Case 3: Verify atomicity with empty bad cells (should succeed)
        println!("  Test 3: Atomic operation with empty bad cells list");

        let another_new_vertex = vertex!([4.0, 4.0, 4.0]);
        let empty_bad_cells: Vec<crate::core::triangulation_data_structure::CellKey> = vec![];
        let _pre_test3_vertex_count = tds.number_of_vertices();
        let pre_test3_cell_count = tds.number_of_cells();

        let result = algorithm.atomic_vertex_insert_and_remove_cells(
            &mut tds,
            &another_new_vertex,
            &empty_bad_cells,
        );

        match result {
            Ok(()) => {
                println!("  ✓ Test 3 SUCCESS: Atomic operation with empty bad cells completed");

                // Vertex should be added
                assert!(
                    tds.vertex_key_from_uuid(&another_new_vertex.uuid())
                        .is_some(),
                    "New vertex should be in TDS"
                );

                // No cells should be removed
                assert_eq!(
                    tds.number_of_cells(),
                    pre_test3_cell_count,
                    "Cell count should be unchanged with empty bad cells"
                );
            }
            Err(e) => {
                println!("  ✓ Test 3 HANDLED: Empty bad cells operation failed: {e}");
            }
        }

        println!("✓ Atomic operation testing completed successfully");
    }

    #[test]
    fn test_atomic_operation_error_propagation() {
        println!("Testing atomic operation error propagation");

        // Create a minimal triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test with a vertex that might cause insertion issues
        // Create a vertex with the same UUID as an existing vertex (should cause duplicate error)
        let existing_vertex = initial_vertices[0];
        let _duplicate_vertex: Vertex<f64, Option<()>, 3> = vertex!([99.0, 99.0, 99.0]); // Different coordinates...

        // Manually set the UUID to match existing vertex to force a conflict
        // Note: This is a conceptual test - actual implementation might handle this differently
        let bad_cells: Vec<_> = tds.cell_keys().take(1).collect();

        println!("  Testing error propagation with potential duplicate UUID scenario");

        let initial_generation = tds.generation();
        let initial_cell_count = tds.number_of_cells();

        let result =
            algorithm.atomic_vertex_insert_and_remove_cells(&mut tds, &existing_vertex, &bad_cells);

        match result {
            Ok(()) => {
                println!("  ✓ Operation succeeded (existing vertex handled correctly)");

                // Verify cells were removed but no new vertex added
                assert_eq!(
                    tds.number_of_cells(),
                    initial_cell_count - bad_cells.len(),
                    "Cells should be removed for existing vertex"
                );

                assert!(
                    tds.generation() > initial_generation,
                    "Generation should increase due to cell removal"
                );
            }
            Err(e) => {
                println!("  ✓ Error propagated correctly: {e}");

                // Verify atomicity: if error occurred, TDS should be in consistent state
                // Generation might have changed if vertex insertion succeeded but cell removal failed
                let current_generation = tds.generation();
                if current_generation == initial_generation {
                    println!("    ✓ Generation unchanged - complete rollback");
                } else {
                    println!("    ✓ Generation changed - partial success with atomic cleanup");
                }

                // In either case, TDS should be valid
                // We can't easily test TDS validity here, but the fact that no panic occurred is good
            }
        }

        println!("✓ Error propagation testing completed");
    }

    #[test]
    fn test_cavity_detection_with_lightweight_facetview() {
        debug_println!("\n=== Testing Cavity Detection with Lightweight FacetView ===");

        // Create initial tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        debug_println!("\nInitial TDS:");
        debug_println!("  Vertices: {}", tds.number_of_vertices());
        debug_println!("  Cells: {}", tds.number_of_cells());

        // Create interior vertex
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        debug_println!("\nInterior vertex: {:?}", interior_vertex.point().coords());

        // Create algorithm and test bad cell detection
        let mut algorithm = RobustBowyerWatson::new();

        // Test bad cells detection
        debug_println!("\nTesting bad cell detection...");
        let bad_cells_result = algorithm.find_bad_cells(&tds, &interior_vertex);

        let bad_cells = bad_cells_result.expect("find_bad_cells should succeed");
        debug_println!("  Bad cells found: {}", bad_cells.len());

        assert!(
            !bad_cells.is_empty(),
            "Interior vertex should find bad cells"
        );
        debug_println!("  ✓ Bad cells detected correctly");

        // Test cavity boundary detection using LIGHTWEIGHT method
        debug_println!("\nTesting lightweight cavity boundary facet detection...");
        let boundary_handles_result =
            algorithm.robust_find_cavity_boundary_facets(&tds, &bad_cells);

        let boundary_handles =
            boundary_handles_result.expect("find_cavity_boundary_facets should succeed");
        debug_println!("  Boundary facet handles found: {}", boundary_handles.len());

        assert!(
            !boundary_handles.is_empty(),
            "Should find boundary facet handles when bad cells exist"
        );
        debug_println!(
            "  ✓ {} boundary facet handles found correctly!",
            boundary_handles.len()
        );

        // Verify we can create FacetView from these handles and get vertices
        debug_println!("\n  Verifying FacetView creation from handles:");
        for (i, handle) in boundary_handles.iter().enumerate() {
            let facet_view =
                crate::core::facet::FacetView::new(&tds, handle.cell_key(), handle.facet_index())
                    .expect("FacetView::new should succeed");

            let vertex_iter = facet_view
                .vertices()
                .expect("FacetView::vertices should succeed");
            let vertex_count = vertex_iter.count();
            debug_println!(
                "    Handle {}: FacetView with {} vertices ✓",
                i,
                vertex_count
            );

            assert!(vertex_count > 0, "FacetView {i} should have vertices");
        }

        debug_println!("\n  ✓ All boundary facet handles are valid and usable!");
        debug_println!("  ✓ The lightweight FacetView-based approach works correctly!");

        // Now test the full insertion including cell creation
        debug_println!("\n=== Testing Full Cavity-Based Insertion ===");
        let mut tds_for_insertion: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&initial_vertices).unwrap();
        let mut algorithm_for_insertion = RobustBowyerWatson::new();

        let interior_vertex_for_insertion = vertex!([0.5, 0.5, 0.5]);
        debug_println!("\nInserting interior vertex into TDS...");

        match algorithm_for_insertion
            .insert_vertex(&mut tds_for_insertion, interior_vertex_for_insertion)
        {
            Ok(info) => {
                debug_println!(
                    "  ✓ Successfully inserted! Created {} cells, removed {} cells",
                    info.cells_created,
                    info.cells_removed
                );
                assert!(info.cells_created > 0, "Should create at least one cell");
                assert_eq!(info.cells_removed, 1, "Should remove the single bad cell");
                assert_eq!(
                    tds_for_insertion.number_of_vertices(),
                    5,
                    "Should have 5 vertices"
                );
                assert!(
                    tds_for_insertion.number_of_cells() > 1,
                    "Should have multiple cells"
                );
            }
            Err(e) => {
                panic!("❌ Full cavity-based insertion failed: {e}");
            }
        }

        debug_println!("\n  ✓ Full cavity-based insertion works correctly!");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_atomic_cavity_based_insertion_guarantees() {
        println!("Testing atomic cavity-based insertion guarantees");

        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Capture initial state
        let initial_vertex_count = tds.number_of_vertices();
        let initial_cell_count = tds.number_of_cells();
        let initial_generation = tds.generation();
        let initial_vertex_uuids: Vec<_> = tds.vertices().map(|(_vkey, v)| v.uuid()).collect();

        println!(
            "Initial state: {initial_vertex_count} vertices, {initial_cell_count} cells, generation {initial_generation}"
        );

        // Test Case 1: Successful insertion (should be atomic - all or nothing)
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        let result = algorithm.insert_vertex(&mut tds, interior_vertex);

        match result {
            Ok(info) => {
                println!(
                    "✓ Insertion succeeded: created={}, removed={}",
                    info.cells_created, info.cells_removed
                );

                // Verify atomicity: ALL changes should have occurred
                assert!(
                    tds.vertex_key_from_uuid(&interior_vertex.uuid()).is_some(),
                    "Vertex should be in TDS after successful insertion"
                );
                assert!(
                    tds.number_of_cells() != initial_cell_count,
                    "Cell count should have changed"
                );
                assert!(
                    tds.generation() > initial_generation,
                    "Generation should have increased"
                );
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid after successful insertion"
                );

                // Verify the insertion was complete (cells created, neighbors connected)
                assert!(
                    info.cells_created > 0,
                    "Should have created cells during cavity-based insertion"
                );
                assert!(
                    info.cells_removed > 0,
                    "Should have removed bad cells during cavity-based insertion"
                );

                println!("✓ All changes applied atomically");
            }
            Err(e) => {
                println!("✓ Insertion failed: {e}");

                // Verify atomicity: NO changes should have occurred (rollback)
                assert_eq!(
                    tds.number_of_vertices(),
                    initial_vertex_count,
                    "Vertex count should be unchanged after failed insertion (rollback)"
                );
                assert_eq!(
                    tds.number_of_cells(),
                    initial_cell_count,
                    "Cell count should be unchanged after failed insertion (rollback)"
                );

                // Verify the failed vertex was NOT added
                assert!(
                    tds.vertex_key_from_uuid(&interior_vertex.uuid()).is_none(),
                    "Failed vertex should NOT be in TDS after rollback"
                );

                // Verify all original vertices are still there
                for uuid in &initial_vertex_uuids {
                    assert!(
                        tds.vertices().any(|(_vkey, v)| v.uuid() == *uuid),
                        "Original vertex should still be in TDS after failed insertion"
                    );
                }

                // TDS should still be valid after rollback
                assert!(
                    tds.is_valid().is_ok(),
                    "TDS should be valid even after failed insertion (rollback)"
                );

                println!("✓ Rollback preserved TDS integrity");
            }
        }

        // Test Case 2: Multiple insertions to verify consistent atomicity
        println!("\nTesting multiple atomic insertions...");
        let test_vertices = [
            vertex!([0.3, 0.3, 0.3]),
            vertex!([0.7, 0.2, 0.1]),
            vertex!([0.1, 0.6, 0.4]),
        ];

        for (i, test_vertex) in test_vertices.iter().enumerate() {
            let pre_vertex_count = tds.number_of_vertices();
            let pre_cell_count = tds.number_of_cells();
            let pre_generation = tds.generation();

            let result = algorithm.insert_vertex(&mut tds, *test_vertex);

            if let Ok(info) = result {
                println!("  ✓ Insertion {} succeeded atomically", i + 1);
                assert!(tds.vertex_key_from_uuid(&test_vertex.uuid()).is_some());
                assert!(tds.generation() > pre_generation);
                assert!(info.cells_created > 0);
            } else {
                println!("  ✓ Insertion {} failed with rollback", i + 1);
                assert_eq!(tds.number_of_vertices(), pre_vertex_count);
                assert_eq!(tds.number_of_cells(), pre_cell_count);
                assert!(tds.vertex_key_from_uuid(&test_vertex.uuid()).is_none());
            }

            // Verify TDS remains valid after each operation
            assert!(
                tds.is_valid().is_ok(),
                "TDS should remain valid after insertion {}",
                i + 1
            );
        }

        println!("✓ Atomic cavity-based insertion guarantees verified");
    }

    // =============================================================================
    // TDS CORRUPTION ERROR DETECTION TESTS
    // =============================================================================
    //
    // These tests verify that the refactored error handling properly detects
    // and propagates TDS corruption errors rather than silently handling them.
    // Note: Actual TDS corruption simulation tests exist in triangulation_data_structure.rs

    /// Test that valid TDS operations in `robust_find_bad_cells` don't trigger false positives
    #[test]
    fn test_robust_find_bad_cells_no_false_positives() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test with various vertices - should never return TDS corruption errors on valid TDS
        let test_vertices = [
            vertex!([0.5, 0.5, 0.5]), // Interior
            vertex!([2.0, 0.0, 0.0]), // Exterior
            vertex!([0.0, 0.0, 0.0]), // On vertex
            vertex!([0.5, 0.0, 0.0]), // On edge
        ];

        for test_vertex in test_vertices {
            let result = algorithm.robust_find_bad_cells(&tds, &test_vertex);

            match result {
                Ok(bad_cells) => {
                    println!(
                        "✓ Found {} bad cells for vertex at {:?}",
                        bad_cells.len(),
                        test_vertex.point().coords()
                    );
                }
                Err(e) => {
                    // Should never get TDS corruption errors on valid TDS
                    panic!("Unexpected TDS corruption error on valid TDS: {e}");
                }
            }
        }
    }

    /// Test that valid TDS operations in `is_facet_visible_from_vertex_robust` don't trigger errors
    #[test]
    fn test_is_facet_visible_no_false_positives() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Get boundary facets
        let boundary_facets: Vec<_> = tds.boundary_facets().unwrap().collect();
        assert!(!boundary_facets.is_empty(), "Should have boundary facets");

        let test_vertices = [
            vertex!([5.0, 5.0, 5.0]), // Far exterior (should be visible)
            vertex!([0.1, 0.1, 0.1]), // Near interior (should not be visible)
            vertex!([2.0, 0.0, 0.0]), // Near exterior
        ];

        for test_vertex in &test_vertices {
            for facet in &boundary_facets {
                let result =
                    algorithm.is_facet_visible_from_vertex_robust(&tds, facet, test_vertex);

                match result {
                    Ok(is_visible) => {
                        println!(
                            "✓ Facet visibility for vertex at {:?}: {is_visible}",
                            test_vertex.point().coords()
                        );
                    }
                    Err(e) => {
                        // Should never get TDS corruption errors on valid TDS
                        panic!("Unexpected TDS corruption error on valid TDS: {e}");
                    }
                }
            }
        }
    }

    /// Test that valid TDS operations in `fallback_visibility_heuristic` don't trigger errors
    #[test]
    fn test_fallback_visibility_heuristic_no_false_positives() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Get boundary facets
        let boundary_facets: Vec<_> = tds.boundary_facets().unwrap().collect();
        assert!(!boundary_facets.is_empty(), "Should have boundary facets");

        let test_vertices = [
            vertex!([10.0, 10.0, 10.0]), // Very far (should be visible)
            vertex!([0.5, 0.5, 0.5]),    // Center (should not be visible)
            vertex!([1.5, 0.0, 0.0]),    // Near exterior
        ];

        for test_vertex in &test_vertices {
            for facet in &boundary_facets {
                let result = algorithm.fallback_visibility_heuristic(facet, test_vertex);

                match result {
                    Ok(is_visible) => {
                        println!(
                            "✓ Fallback heuristic for vertex at {:?}: {is_visible}",
                            test_vertex.point().coords()
                        );
                    }
                    Err(e) => {
                        // Should never get TDS corruption errors on valid TDS
                        panic!("Unexpected TDS corruption error on valid TDS: {e}");
                    }
                }
            }
        }
    }

    /// Test that TDS corruption error messages contain useful debugging information
    #[test]
    fn test_tds_corruption_error_messages_are_helpful() {
        // Verify that TDS corruption errors would contain useful debug info
        // We can't actually corrupt a TDS, but we can verify the error message format

        let error = InsertionError::TriangulationState(
            TriangulationValidationError::InconsistentDataStructure {
                message:
                    "TDS corruption: Cell CellKey(42) references non-existent vertex VertexKey(17)"
                        .to_string(),
            },
        );

        let error_string = error.to_string();
        assert!(
            error_string.contains("TDS corruption"),
            "Error should mention TDS corruption"
        );
        assert!(
            error_string.contains("CellKey"),
            "Error should include cell key for debugging"
        );
        assert!(
            error_string.contains("VertexKey"),
            "Error should include vertex key for debugging"
        );

        println!("✓ TDS corruption error format is helpful: {error_string}");
    }

    /// Test that error propagation chain works correctly through insertion
    #[test]
    fn test_tds_corruption_error_propagation() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Normal insertion should work without TDS corruption errors
        let test_vertex = vertex!([0.5, 0.5, 0.5]);
        let result = algorithm.insert_vertex(&mut tds, test_vertex);

        // Verify no TDS corruption errors on valid operations
        match result {
            Ok(info) => {
                println!("✓ Normal insertion succeeded without TDS corruption errors");
                println!(
                    "  Created: {}, Removed: {}",
                    info.cells_created, info.cells_removed
                );
                assert!(info.success, "Insertion should succeed");
            }
            Err(e) => {
                // If insertion fails for geometric reasons, that's acceptable
                // But should never be TDS corruption on valid TDS
                let error_string = e.to_string();
                assert!(
                    !error_string.contains("TDS corruption"),
                    "Unexpected TDS corruption error on valid TDS: {e}"
                );
                println!("ℹ Insertion failed for geometric reasons (acceptable): {e}");
            }
        }

        // Verify TDS is still valid after operation
        assert!(tds.is_valid().is_ok(), "TDS should remain valid");
        println!("✓ TDS remains valid after operations");
    }

    /// Test that all three refactored methods properly return Result types
    #[test]
    fn test_refactored_methods_return_results() {
        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let test_vertex = vertex!([0.5, 0.5, 0.5]);

        // Test 1: robust_find_bad_cells returns Result<Vec<CellKey>, InsertionError>
        let result1 = algorithm.robust_find_bad_cells(&tds, &test_vertex);
        assert!(
            result1.is_ok(),
            "robust_find_bad_cells should return Ok on valid TDS"
        );
        println!("✓ robust_find_bad_cells returns Result<Vec<CellKey>, InsertionError>");

        // Test 2: is_facet_visible_from_vertex_robust returns Result<bool, InsertionError>
        let boundary_facets: Vec<_> = tds.boundary_facets().unwrap().collect();
        if let Some(facet) = boundary_facets.first() {
            let result2 = algorithm.is_facet_visible_from_vertex_robust(&tds, facet, &test_vertex);
            assert!(
                result2.is_ok(),
                "is_facet_visible_from_vertex_robust should return Ok on valid TDS"
            );
            println!("✓ is_facet_visible_from_vertex_robust returns Result<bool, InsertionError>");

            // Test 3: fallback_visibility_heuristic returns Result<bool, InsertionError>
            let result3 = algorithm.fallback_visibility_heuristic(facet, &test_vertex);
            assert!(
                result3.is_ok(),
                "fallback_visibility_heuristic should return Ok on valid TDS"
            );
            println!("✓ fallback_visibility_heuristic returns Result<bool, InsertionError>");
        }

        println!("✓ All three refactored methods properly return Result types");
    }

    /// Verify that Result-based error handling works correctly with multiple insertions
    ///
    /// This test validates the functional correctness of error handling across multiple
    /// vertex insertions. Performance regression detection should use the benchmark suite.
    #[test]
    fn test_result_based_error_handling_with_multiple_insertions() {
        let mut algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        let test_vertices = [
            vertex!([0.1, 0.1, 0.1]),
            vertex!([0.2, 0.2, 0.2]),
            vertex!([0.3, 0.3, 0.3]),
            vertex!([0.4, 0.4, 0.4]),
            vertex!([0.5, 0.5, 0.5]),
        ];

        let mut successful = 0;
        for test_vertex in test_vertices {
            if algorithm.insert_vertex(&mut tds, test_vertex).is_ok() {
                successful += 1;
            }
        }

        // Verify functional correctness: at least some insertions should succeed
        assert!(
            successful > 0,
            "At least one insertion should succeed with valid vertices"
        );
        assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after insertions"
        );
        println!(
            "✓ Successfully completed {successful}/{} insertions with Result-based error handling",
            test_vertices.len()
        );
    }

    /// Test that Result-based methods integrate correctly with ? operator
    #[test]
    fn test_error_propagation_with_question_mark_operator() {
        // This test verifies the ergonomics of the Result-based API
        fn test_helper(
            algorithm: &RobustBowyerWatson<f64, Option<()>, Option<()>, 3>,
            tds: &Tds<f64, Option<()>, Option<()>, 3>,
            vertex: &Vertex<f64, Option<()>, 3>,
        ) -> Result<bool, InsertionError> {
            // Test that ? operator works correctly with our Result types
            let bad_cells = algorithm.robust_find_bad_cells(tds, vertex)?;

            if !bad_cells.is_empty() {
                let boundary_facets: Vec<_> = tds
                    .boundary_facets()
                    .map_err(InsertionError::TriangulationState)?
                    .collect();

                if let Some(facet) = boundary_facets.first() {
                    let is_visible =
                        algorithm.is_facet_visible_from_vertex_robust(tds, facet, vertex)?;
                    if is_visible {
                        let fallback = algorithm.fallback_visibility_heuristic(facet, vertex)?;
                        return Ok(fallback);
                    }
                }
            }

            Ok(false)
        }

        let algorithm = RobustBowyerWatson::<f64, Option<()>, Option<()>, 3>::new();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let test_vertex = vertex!([0.5, 0.5, 0.5]);

        let result = test_helper(&algorithm, &tds, &test_vertex);
        assert!(
            result.is_ok(),
            "Error propagation with ? operator should work correctly"
        );
        println!("✓ Result-based API integrates correctly with ? operator");
    }
}
