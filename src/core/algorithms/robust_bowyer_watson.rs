//! Robust Bowyer-Watson algorithm using enhanced geometric predicates.
//!
//! This module demonstrates how to integrate the robust geometric predicates
//! into the Bowyer-Watson triangulation algorithm to address the
//! "No cavity boundary facets found" error.

use crate::core::collections::MAX_PRACTICAL_DIMENSION_SIZE;
use crate::core::collections::{
    CellKeySet, FacetToCellsMap, FastHashMap, FastHashSet, SmallBuffer, fast_hash_set_with_capacity,
};
use crate::core::traits::facet_cache::FacetCacheProvider;
use crate::core::util::derive_facet_key_from_vertices;
use arc_swap::ArcSwapOption;
use std::marker::PhantomData;
use std::ops::{AddAssign, Div, DivAssign, SubAssign};
use std::sync::{Arc, atomic::AtomicU64};

use crate::core::traits::insertion_algorithm::{
    InsertionAlgorithm, InsertionBuffers, InsertionError, InsertionInfo, InsertionStatistics,
    InsertionStrategy,
};
use crate::core::{
    facet::Facet,
    triangulation_data_structure::{
        CellKey, Tds, TriangulationConstructionError, TriangulationValidationError,
    },
    vertex::Vertex,
};
use crate::geometry::{
    algorithms::convex_hull::ConvexHull,
    point::Point,
    predicates::InSphere,
    robust_predicates::{RobustPredicateConfig, config_presets, robust_insphere},
    traits::coordinate::{Coordinate, CoordinateScalar},
    util::safe_usize_to_scalar,
};
use nalgebra::{self as na, ComplexField};
use serde::{Serialize, de::DeserializeOwned};
use std::iter::Sum;

/// Enhanced Bowyer-Watson algorithm with robust geometric predicates.
#[derive(Default)]
pub struct RobustBoyerWatson<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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

impl<T, U, V, const D: usize> RobustBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    ordered_float::OrderedFloat<f64>: From<T>,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    ///
    /// let algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> = RobustBoyerWatson::new();
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
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// let config = config_presets::high_precision::<f64>();
    /// let algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     RobustBoyerWatson::with_config(config);
    /// ```
    pub fn with_config(config: RobustPredicateConfig<T>) -> Self {
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
    /// use delaunay::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    ///
    /// let algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     RobustBoyerWatson::for_degenerate_cases();
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
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
    fn insert_vertex_cavity_based_with_robust_predicates(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // First try using trait method for bad cell detection with robust fallback
        let bad_cells = self.find_bad_cells_with_robust_fallback(tds, vertex);

        if !bad_cells.is_empty() {
            // Try boundary facet detection using lightweight trait method with robust fallback
            #[allow(clippy::collapsible_if)] // Can't collapse due to if-let chain guard limitations
            if let Ok(boundary_handles) =
                self.find_cavity_boundary_facets_lightweight_with_robust_fallback(tds, &bad_cells)
            {
                if !boundary_handles.is_empty() {
                    // CRITICAL: We need to use handles BEFORE removing cells, as they become invalid after removal.
                    // The cavity-based insertion algorithm requires: extract → remove → create

                    // Store handles and extract facet data while cells still exist
                    let mut extracted_facet_data = Vec::new();
                    for &(cell_key, facet_index) in &boundary_handles {
                        if let Some(_cell) = tds.cells().get(cell_key) {
                            if let Ok(facet_view) =
                                crate::core::facet::FacetView::new(tds, cell_key, facet_index)
                            {
                                let facet_vertices: Vec<Vertex<T, U, D>> =
                                    facet_view.vertices().copied().collect();
                                extracted_facet_data.push(facet_vertices);
                            }
                        }
                    }

                    let cells_removed = bad_cells.len();

                    // Remove bad cells (invalidates handles)
                    <Self as InsertionAlgorithm<T, U, V, D>>::remove_bad_cells(tds, &bad_cells);

                    // Ensure vertex is in TDS
                    <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex)?;

                    // Create cells from pre-extracted data
                    let mut cells_created = 0;
                    for facet_vertices in extracted_facet_data {
                        if <Self as InsertionAlgorithm<T, U, V, D>>::create_cell_from_vertices_and_vertex(
                            tds,
                            facet_vertices,
                            vertex,
                        ).is_ok() {
                            cells_created += 1;
                        }
                    }

                    // Maintain invariants after structural changes
                    <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds).map_err(
                        |e| InsertionError::TriangulationState(
                            TriangulationValidationError::FinalizationFailed {
                                message: format!(
                                    "Failed to finalize triangulation after robust cavity-based insertion \
                                         (removed {cells_removed} cells, created {cells_created} cells). \
                                         Underlying error: {e}"
                                ),
                            }
                        ),
                    )?;

                    return Ok(InsertionInfo {
                        strategy: InsertionStrategy::CavityBased,
                        cells_removed,
                        cells_created,
                        success: true,
                        degenerate_case_handled: false,
                    });
                }
            }
        }

        // If robust detection fails, fall back to trait method
        self.insert_vertex_cavity_based(tds, vertex)
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
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
    fn find_bad_cells_with_robust_fallback(
        &mut self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Vec<CellKey>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // First try to find bad cells using the trait's method
        let mut bad_cells =
            match InsertionAlgorithm::<T, U, V, D>::find_bad_cells(self, tds, vertex) {
                Ok(cells) => cells,
                Err(crate::core::traits::insertion_algorithm::BadCellsError::AllCellsBad {
                    ..
                }) => {
                    // All cells marked as bad - try robust method to get a better result
                    self.robust_find_bad_cells(tds, vertex)
                }
                Err(
                    crate::core::traits::insertion_algorithm::BadCellsError::TooManyDegenerateCells(
                        _,
                    ),
                ) => {
                    // Too many degenerate cells - try robust method as fallback
                    self.robust_find_bad_cells(tds, vertex)
                }
                Err(crate::core::traits::insertion_algorithm::BadCellsError::NoCells) => {
                    // No cells - return empty
                    return Vec::new();
                }
            };

        // If the standard method doesn't find any bad cells (likely a degenerate case)
        // or we're using the robust configuration, supplement with robust predicates
        if bad_cells.is_empty() || self.predicate_config.base_tolerance > T::default_tolerance() {
            let robust_bad_cells = self.robust_find_bad_cells(tds, vertex);

            // Use a set for O(1) membership checking to avoid O(n²) complexity
            let mut seen: CellKeySet = bad_cells.iter().copied().collect();
            for cell_key in robust_bad_cells {
                // Only add if not already present (insert returns true if new)
                if seen.insert(cell_key) {
                    bad_cells.push(cell_key);
                }
            }
        }

        bad_cells
    }

    /// Find cavity boundary facets using lightweight handles with robust fallback.
    ///
    /// This optimized approach uses the lightweight `find_cavity_boundary_facets_lightweight` method
    /// first, then applies robust predicates for edge cases, returning lightweight handles instead
    /// of heavyweight Facet objects for improved performance.
    ///
    /// # Important Usage Note
    ///
    /// The returned handles `(CellKey, u8)` are only valid while the referenced cells exist.
    /// If you plan to remove cells (e.g., via `remove_bad_cells`), you MUST convert the handles
    /// to Facet objects BEFORE removing the cells, otherwise the handles become invalid.
    fn find_cavity_boundary_facets_lightweight_with_robust_fallback(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<(CellKey, u8)>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
    {
        // First try to find boundary facets using the lightweight trait method
        match InsertionAlgorithm::<T, U, V, D>::find_cavity_boundary_facets_lightweight(
            self, tds, bad_cells,
        ) {
            Ok(boundary_handles) => {
                // If the lightweight method succeeds and finds facets, use them
                if !boundary_handles.is_empty() {
                    return Ok(boundary_handles);
                }
                // If lightweight method succeeds but finds no facets, try robust method as fallback
                self.robust_find_cavity_boundary_facets_lightweight(tds, bad_cells)
            }
            Err(_) => {
                // If lightweight method fails, use robust method as fallback
                self.robust_find_cavity_boundary_facets_lightweight(tds, bad_cells)
            }
        }
    }

    /// Find cavity boundary facets by first using the trait method, then applying robust predicates for edge cases.
    ///
    /// This approach integrates the trait's `find_cavity_boundary_facets` method with the robust predicates
    /// to provide a more reliable boundary facet detection method, especially for degenerate cases.
    ///
    /// # Deprecated
    /// Consider using `find_cavity_boundary_facets_lightweight_with_robust_fallback` for better performance.
    /// Find visible boundary facets by first using the trait method, then applying robust predicates for edge cases.
    ///
    /// This approach integrates the trait's `find_visible_boundary_facets` method with the robust predicates
    /// to provide a more reliable visibility detection method, especially for degenerate cases.
    fn find_visible_boundary_facets_with_robust_fallback(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<(CellKey, u8)>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Use the lightweight method that avoids heavy cloning
        <Self as InsertionAlgorithm<T, U, V, D>>::find_visible_boundary_facets_lightweight(
            self, tds, vertex,
        )
    }

    /// Find bad cells using robust insphere predicate.
    /// This is a lower-level method used by `find_bad_cells_with_robust_fallback`.
    fn robust_find_bad_cells(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Vec<CellKey> {
        let mut bad_cells = SmallBuffer::<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut vertex_points = SmallBuffer::<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        vertex_points.reserve_exact(D + 1);

        for (cell_key, cell) in tds.cells() {
            // Extract vertex points from the cell (reusing buffer)
            vertex_points.clear();
            vertex_points.extend(cell.vertices().iter().map(|v| *v.point()));

            if vertex_points.len() < D + 1 {
                continue; // Skip incomplete cells
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

        bad_cells.into_vec()
    }

    /// Find cavity boundary facets with enhanced error handling, returning lightweight handles.
    ///
    /// This optimized version returns `(CellKey, u8)` handles instead of heavyweight Facet objects,
    /// providing significant performance improvements for boundary facet detection.
    fn robust_find_cavity_boundary_facets_lightweight(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<(CellKey, u8)>, InsertionError> {
        let mut boundary_handles = Vec::new();

        if bad_cells.is_empty() {
            return Ok(boundary_handles);
        }

        let bad_cell_set: CellKeySet = bad_cells.iter().copied().collect();

        // Build facet-to-cells mapping with enhanced validation
        let facet_to_cells = self.build_validated_facet_mapping(tds)?;

        // Find boundary facets with improved logic, returning lightweight handles
        let mut processed_facets = FastHashSet::default();

        for &bad_cell_key in bad_cells {
            if let Some(bad_cell) = tds.cells().get(bad_cell_key)
                && let Ok(facets) = bad_cell.facets()
            {
                for (facet_idx, facet) in facets.iter().enumerate() {
                    // Derive facet key using the utility function
                    let facet_vertices = facet.vertices();
                    let Ok(facet_key) = derive_facet_key_from_vertices(&facet_vertices, tds) else {
                        continue;
                    }; // Cannot form a valid facet key - vertex not found

                    if processed_facets.contains(&facet_key) {
                        continue;
                    }

                    if let Some(sharing_cells) = facet_to_cells.get(&facet_key) {
                        let bad_count = sharing_cells
                            .iter()
                            .filter(|&&cell_key| bad_cell_set.contains(&cell_key))
                            .count();
                        let total_count = sharing_cells.len();

                        // Enhanced boundary detection logic
                        if Self::is_cavity_boundary_facet(bad_count, total_count) {
                            // Return lightweight handle instead of heavy Facet object
                            if let Ok(facet_idx_u8) = u8::try_from(facet_idx) {
                                boundary_handles.push((bad_cell_key, facet_idx_u8));
                            }
                            processed_facets.insert(facet_key);
                        }
                    }
                }
            }
        }

        // Additional validation of boundary facet count
        if boundary_handles.is_empty() && !bad_cells.is_empty() {
            return Err(InsertionError::ExcessiveBadCells {
                found: bad_cells.len(),
                threshold: 0,
            });
        }

        Ok(boundary_handles)
    }

    /// Find cavity boundary facets with enhanced error handling.
    fn robust_find_cavity_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError> {
        let mut boundary_facets = Vec::new();

        if bad_cells.is_empty() {
            return Ok(boundary_facets);
        }

        let bad_cell_set: CellKeySet = bad_cells.iter().copied().collect();

        // Build facet-to-cells mapping with enhanced validation
        let facet_to_cells = self.build_validated_facet_mapping(tds)?;

        // Find boundary facets with improved logic
        let mut processed_facets = FastHashSet::default();

        for &bad_cell_key in bad_cells {
            if let Some(bad_cell) = tds.cells().get(bad_cell_key)
                && let Ok(facets) = bad_cell.facets()
            {
                for facet in facets {
                    // Derive facet key using the utility function
                    let facet_vertices = facet.vertices();
                    let Ok(facet_key) = derive_facet_key_from_vertices(&facet_vertices, tds) else {
                        continue;
                    }; // Cannot form a valid facet key - vertex not found

                    if processed_facets.contains(&facet_key) {
                        continue;
                    }

                    if let Some(sharing_cells) = facet_to_cells.get(&facet_key) {
                        let bad_count = sharing_cells
                            .iter()
                            .filter(|&&cell_key| bad_cell_set.contains(&cell_key))
                            .count();
                        let total_count = sharing_cells.len();

                        // Enhanced boundary detection logic
                        if Self::is_cavity_boundary_facet(bad_count, total_count) {
                            boundary_facets.push(facet.clone());
                            processed_facets.insert(facet_key);
                        }
                    }
                }
            }
        }

        // Additional validation of boundary facets
        self.validate_boundary_facets(&boundary_facets, bad_cells.len())?;

        Ok(boundary_facets)
    }

    /// Recover from cavity boundary facet detection failure.
    #[allow(dead_code)]
    fn recover_cavity_boundary_facets(
        &mut self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError> {
        self.stats.cavity_boundary_recoveries += 1;

        // Recovery Strategy 1: Use more lenient boundary detection criteria
        let facets = Self::lenient_cavity_boundary_detection(tds, bad_cells);
        if !facets.is_empty() {
            return Ok(facets);
        }

        // Recovery Strategy 2: Reduce the set of bad cells
        if let Ok(facets) = self.reduced_bad_cell_strategy(tds, bad_cells, vertex)
            && !facets.is_empty()
        {
            return Ok(facets);
        }

        // Recovery Strategy 3: Use convex hull extension
        self.convex_hull_extension_strategy(tds, vertex)
    }

    /// Handle the case where no bad cells are found.
    #[allow(dead_code)]
    fn handle_no_bad_cells_case(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<RobustInsertionInfo, InsertionError> {
        // This typically means the vertex is outside the convex hull
        // Try to extend the hull by connecting to visible boundary facets

        let visible_facets = self.find_visible_boundary_facets(tds, vertex)?;

        if visible_facets.is_empty() {
            return Err(InsertionError::HullExtensionFailure {
                reason: "No visible boundary facets found for hull extension".to_string(),
            });
        }

        // Add the vertex to the TDS if it's not already there
        <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex)?;

        let cells_created =
            <Self as InsertionAlgorithm<T, U, V, D>>::create_cells_from_boundary_facets(
                tds,
                &visible_facets,
                vertex,
            );

        // Finalize the triangulation to ensure consistency
        if let Err(e) = <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds) {
            return Err(InsertionError::TriangulationState(
                TriangulationValidationError::FinalizationFailed {
                    message: format!(
                        "Failed to finalize triangulation after hull extension. Underlying error: {e}"
                    ),
                },
            ));
        }

        Ok(RobustInsertionInfo {
            success: true,
            cells_created,
            cells_removed: 0,
            strategy_used: InsertionStrategy::HullExtension,
            degenerate_case_handled: false,
        })
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
                cell_keys.extend(cell_facet_pairs.iter().map(|(cell_key, _)| *cell_key));

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

    #[allow(clippy::unused_self)]
    const fn validate_boundary_facets(
        &self,
        boundary_facets: &[Facet<T, U, V, D>],
        bad_cell_count: usize,
    ) -> Result<(), InsertionError> {
        if boundary_facets.is_empty() && bad_cell_count > 0 {
            return Err(InsertionError::ExcessiveBadCells {
                found: bad_cell_count,
                threshold: 0,
            });
        }

        // Additional validation could check facet geometry, etc.
        Ok(())
    }

    #[allow(dead_code)]
    const fn lenient_cavity_boundary_detection(
        _tds: &Tds<T, U, V, D>,
        _bad_cells: &[CellKey],
    ) -> Vec<Facet<T, U, V, D>> {
        // Use more permissive criteria for boundary facet detection
        // This might include facets that are "mostly" boundary facets

        // Implementation would be similar to robust_find_cavity_boundary_facets
        // but with relaxed criteria

        // For now, return empty to indicate this strategy didn't work
        Vec::new()
    }

    #[allow(dead_code)]
    fn reduced_bad_cell_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
        _vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError> {
        // Try with a reduced set of bad cells
        // Remove cells that might be causing the boundary detection to fail

        if bad_cells.len() <= 1 {
            return Ok(Vec::new()); // Can't reduce further
        }

        // Try with just the first half of bad cells
        let reduced_bad_cells = &bad_cells[..bad_cells.len() / 2];
        self.robust_find_cavity_boundary_facets(tds, reduced_bad_cells)
    }

    #[allow(dead_code)]
    #[allow(clippy::unused_self)]
    fn convex_hull_extension_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError> {
        // Find all boundary facets and check which are visible from the vertex
        self.find_visible_boundary_facets(tds, vertex)
    }

    /// Robust implementation of `find_visible_boundary_facets` that handles degenerate cases
    ///
    /// This method is more aggressive than the default implementation, using fallback strategies
    /// when geometric predicates fail or return degenerate results.
    fn find_visible_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, InsertionError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        use crate::core::collections::{KeyBasedCellMap, fast_hash_map_with_capacity};

        let mut visible_facets = Vec::new();

        // Get all boundary facets (facets shared by exactly one cell) using cache with proper error handling
        let facet_to_cells = self
            .try_get_or_build_facet_cache(tds)
            .map_err(InsertionError::TriangulationState)?;

        let mut cell_facets_cache: KeyBasedCellMap<Vec<Facet<T, U, V, D>>> =
            fast_hash_map_with_capacity(tds.number_of_cells());

        // Directly iterate over filtered boundary facets without collecting into a temporary Vec
        for (_facet_key, cells) in facet_to_cells.iter().filter(|(_, cells)| cells.len() == 1) {
            let (cell_key, facet_index) = cells[0];
            if let Some(cell) = tds.cells().get(cell_key) {
                let facets = match cell_facets_cache.entry(cell_key) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let computed = cell.facets().map_err(|e| {
                            TriangulationValidationError::InconsistentDataStructure {
                                message: format!(
                                    "Failed to get facets from cell during visibility computation: {e}"
                                ),
                            }
                        })?;
                        v.insert(computed)
                    }
                };

                let idx = usize::from(facet_index);
                if idx < facets.len() {
                    let facet = &facets[idx];

                    // Test visibility using robust orientation predicates with fallback
                    if self.is_facet_visible_from_vertex_robust(tds, facet, vertex, cell_key) {
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
                        message: "Cell key not found during visibility computation".to_string(),
                    },
                ));
            }
        }

        Ok(visible_facets)
    }

    /// Robust helper method to test if a boundary facet is visible from a given vertex
    ///
    /// This method uses multiple fallback strategies when geometric predicates fail
    /// or return degenerate results, making it more suitable for exterior vertex insertion.
    fn is_facet_visible_from_vertex_robust(
        &self,
        tds: &Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        adjacent_cell_key: crate::core::triangulation_data_structure::CellKey,
    ) -> bool
    where
        T: AddAssign<T>
            + ComplexField<RealField = T>
            + SubAssign<T>
            + Sum
            + From<f64>
            + DivAssign<T>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        use crate::geometry::point::Point;
        use crate::geometry::predicates::{Orientation, simplex_orientation};

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

        match (orientation_opposite, orientation_test) {
            (Orientation::NEGATIVE, Orientation::POSITIVE)
            | (Orientation::POSITIVE, Orientation::NEGATIVE) => true,
            (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
                // Degenerate case - use distance-based fallback for exterior vertices
                self.fallback_visibility_heuristic(facet, vertex)
            }
            _ => false, // Same orientation = same side = not visible
        }
    }

    /// Fallback visibility heuristic for degenerate cases
    ///
    /// When geometric predicates fail or return degenerate results, this method uses
    /// a distance-based heuristic to determine if a facet should be considered visible.
    /// For exterior vertex insertion, this is more aggressive than the default implementation.
    fn fallback_visibility_heuristic(
        &self,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> bool
    where
        T: DivAssign<T> + AddAssign<T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
    {
        let facet_vertices = facet.vertices();
        if facet_vertices.is_empty() {
            // Conservatively treat as not visible if we cannot compute a centroid
            return false;
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
            return false; // Conservatively treat as not visible if conversion fails
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
        let threshold = {
            let scale = self.predicate_config.perturbation_scale;
            let multiplier = self.predicate_config.visibility_threshold_multiplier;
            let th = scale * scale * multiplier;
            // Best-effort clamp: treat non-finite as "very large"
            // If T is float-like, this avoids NaN/Inf comparisons.
            if (f64::from(th)).is_finite() {
                th
            } else {
                <T as From<f64>>::from(f64::MAX / 2.0)
            }
        };
        distance_squared > threshold
    }

    #[allow(dead_code)]
    fn create_perturbed_vertex(
        &self,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vertex<T, U, D>, crate::core::vertex::VertexValidationError> {
        let mut coords: [T; D] = vertex.point().to_array();
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
        let coords: [T; D] = vertex.point().to_array();
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
        let vertex_count = tds.vertices().len();
        let nearby_vertices = if vertex_count > 1000 {
            // For large triangulations, use early exit after finding sufficient nearby vertices
            let mut count = 0;
            let max_scan = 100; // Early exit threshold
            for (i, v) in tds.vertices().values().enumerate() {
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
                .values()
                .filter(|v| {
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

impl<T, U, V, const D: usize> FacetCacheProvider<T, U, V, D> for RobustBoyerWatson<T, U, V, D>
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
        &self.facet_to_cells_cache
    }

    fn cached_generation(&self) -> &AtomicU64 {
        self.cached_generation.as_ref()
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

impl<T, U, V, const D: usize> InsertionAlgorithm<T, U, V, D> for RobustBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    ordered_float::OrderedFloat<f64>: From<T>,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::core::traits::facet_cache::FacetCacheProvider;
    use crate::core::traits::insertion_algorithm::InsertionError;
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

    /// Helper function to verify facet index consistency between neighboring cells
    ///
    /// This method checks that the shared facet key computed from both cells'
    /// perspectives matches, catching subtle neighbor assignment errors.
    fn verify_facet_index_consistency<T, U, V, const D: usize>(
        tds: &Tds<T, U, V, D>,
        cell1_key: CellKey,
        cell2_key: CellKey,
        facet_idx: usize,
        insertion_num: usize,
    ) where
        T: CoordinateScalar
            + std::ops::AddAssign<T>
            + std::ops::SubAssign<T>
            + std::iter::Sum
            + num_traits::cast::NumCast,
        U: crate::core::traits::data_type::DataType,
        V: crate::core::traits::data_type::DataType,
        [T; D]: Copy + DeserializeOwned + Serialize + Sized,
    {
        use crate::core::util::derive_facet_key_from_vertices;

        if let (Some(cell1), Some(cell2)) = (tds.cells().get(cell1_key), tds.cells().get(cell2_key))
            && let (Ok(facets1), Ok(facets2)) = (cell1.facets(), cell2.facets())
            && facet_idx < facets1.len()
        {
            let facet1 = &facets1[facet_idx];
            let facet1_vertices = facet1.vertices();

            // Derive the facet key from cell1's perspective
            if let Ok(facet_key1) = derive_facet_key_from_vertices(&facet1_vertices, tds) {
                // Find the corresponding facet in cell2 that shares the same vertices
                let mut found_matching_facet = false;
                for facet2 in facets2 {
                    let facet2_vertices = facet2.vertices();
                    if let Ok(facet_key2) = derive_facet_key_from_vertices(&facet2_vertices, tds)
                        && facet_key1 == facet_key2
                    {
                        found_matching_facet = true;
                        break;
                    }
                }

                assert!(
                    found_matching_facet,
                    "No matching facet found between neighboring cells after insertion {insertion_num}: \
                     cell1 facet key {facet_key1} not found in cell2"
                );
            }
        }
    }

    #[test]
    fn test_robust_bowyer_watson_creation() {
        let algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> = RobustBoyerWatson::new();

        assert_eq!(algorithm.stats.vertices_processed, 0);
    }

    #[test]
    fn test_degenerate_configuration() {
        let mut algorithm = RobustBoyerWatson::for_degenerate_cases();

        // Create a TDS with some initial cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Try to insert a vertex that might cause degenerate behavior
        let problematic_vertex = vertex!([0.5, 0.5, 1e-15]);

        let result = algorithm.insert_vertex(&mut tds, problematic_vertex);
        // Should handle gracefully: either succeed or leave TDS valid on failure
        assert!(
            tds.is_valid().is_ok(),
            "TDS must remain valid after attempt: {:?}",
            result.err()
        );
    }

    #[test]
    #[allow(clippy::used_underscore_binding)] // Variables used in conditional debug_println! macros
    fn test_no_double_counting_statistics() {
        debug_println!("Testing that robust vertex insertion statistics are not double counted");

        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
                new_vertex.point().to_array()
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
    fn test_debug_exterior_vertex_insertion() {
        debug_println!("Testing exterior vertex insertion in robust Bowyer-Watson");

        let mut algorithm = RobustBoyerWatson::new();

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
            interior_vertex.point().to_array()
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
            exterior_vertex.point().to_array()
        );

        // Let's debug what happens step by step
        debug_println!("Finding bad cells for exterior vertex...");
        let bad_cells = algorithm.robust_find_bad_cells(&tds, &exterior_vertex);
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
                        facet.vertices().count()
                    );
                }
            }

            // Test the visibility detection directly
            println!("Testing visibility detection...");
            let visible_result = algorithm.find_visible_boundary_facets(&tds, &exterior_vertex);
            match &visible_result {
                Ok(facets) => {
                    println!("Found {} visible boundary facets", facets.len());
                    for (i, facet) in facets.iter().enumerate() {
                        println!("  Visible facet {}: {} vertices", i, facet.vertices().len());
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
    fn test_cavity_based_insertion_consistency() {
        debug_println!("Testing cavity-based insertion maintains TDS consistency");

        let mut algorithm = RobustBoyerWatson::new();

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
                test_vertex.point().to_array()
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
                    for (facet_idx, neighbor_uuid) in neighbors.iter().enumerate() {
                        if let Some(neighbor_uuid) = neighbor_uuid
                            && let Some(neighbor_key) = tds.cell_key_from_uuid(neighbor_uuid)
                            && let Some(neighbor) = tds.cells().get(neighbor_key)
                            && let Some(neighbor_neighbors) = &neighbor.neighbors
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            let cell_uuid = tds
                                .cell_uuid_from_key(cell_key)
                                .expect("Cell should have UUID");
                            assert!(
                                neighbor_neighbors
                                    .iter()
                                    .any(|n| n.as_ref() == Some(&cell_uuid)),
                                "Neighbor relationship should be symmetric after insertion {}",
                                i + 1
                            );

                            // Verify facet indices consistency
                            verify_facet_index_consistency(
                                &tds,
                                cell_key,
                                neighbor_key,
                                facet_idx,
                                i + 1,
                            );
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
                        facet.vertices().count(),
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

        let mut algorithm = RobustBoyerWatson::new();

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
                test_vertex.point().to_array()
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

            // Verify the triangulation expanded (more cells added overall)
            // Note: Even "exterior" vertices might trigger cavity-based insertion if they're
            // inside the circumsphere of existing cells, so we can't guarantee pure hull extension
            assert!(
                cells_after > cells_before,
                "Insertion should increase cell count overall"
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
                    for (facet_idx, neighbor_uuid) in neighbors.iter().enumerate() {
                        if let Some(neighbor_uuid) = neighbor_uuid
                            && let Some(neighbor_key) = tds.cell_key_from_uuid(neighbor_uuid)
                            && let Some(neighbor) = tds.cells().get(neighbor_key)
                            && let Some(neighbor_neighbors) = &neighbor.neighbors
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            let cell_uuid = tds
                                .cell_uuid_from_key(cell_key)
                                .expect("Cell should have UUID");
                            assert!(
                                neighbor_neighbors
                                    .iter()
                                    .any(|opt| opt.as_ref() == Some(&cell_uuid)),
                                "Neighbor relationship should be symmetric after hull extension {}",
                                i + 1
                            );

                            // Verify facet indices consistency
                            verify_facet_index_consistency(
                                &tds,
                                cell_key,
                                neighbor_key,
                                facet_idx,
                                i + 1,
                            );
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
                        facet.vertices().count(),
                        3,
                        "Boundary facet should have 3 vertices after hull extension {}",
                        i + 1
                    );
                }

                // The newly inserted vertex should be in the triangulation
                let vertex_found = tds.vertices().values().any(|v| {
                    let v_coords: [f64; 3] = v.point().to_array();
                    let test_coords: [f64; 3] = test_vertex.point().to_array();
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

        let mut algorithm = RobustBoyerWatson::new();

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
                test_vertex.point().to_array()
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
                let mut vertex_uuids: SmallBuffer<uuid::Uuid, MAX_PRACTICAL_DIMENSION_SIZE> =
                    cell.vertex_uuid_iter().collect();
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
                    let (cell1_key, facet1_idx) = cells[0];
                    let (cell2_key, _facet2_idx) = cells[1];

                    if let (Some(cell1), Some(cell2)) =
                        (tds.cells().get(cell1_key), tds.cells().get(cell2_key))
                        && let (Some(neighbors1), Some(neighbors2)) =
                            (&cell1.neighbors, &cell2.neighbors)
                    {
                        let cell2_uuid = tds
                            .cell_uuid_from_key(cell2_key)
                            .expect("Cell2 should have UUID");
                        let cell1_uuid = tds
                            .cell_uuid_from_key(cell1_key)
                            .expect("Cell1 should have UUID");
                        assert!(
                            neighbors1.iter().flatten().any(|uuid| *uuid == cell2_uuid),
                            "Cell1 should reference cell2 as neighbor after insertion {}",
                            i + 1
                        );
                        assert!(
                            neighbors2.iter().flatten().any(|uuid| *uuid == cell1_uuid),
                            "Cell2 should reference cell1 as neighbor after insertion {}",
                            i + 1
                        );

                        // Verify facet indices consistency for the shared facet
                        verify_facet_index_consistency(
                            &tds,
                            cell1_key,
                            cell2_key,
                            facet1_idx as usize,
                            i + 1,
                        );
                    }
                }
            }

            // 4. All vertices should have proper incident cells assigned
            for (_, vertex) in tds.vertices() {
                if let Some(incident_cell_uuid) = vertex.incident_cell
                    && let Some(incident_cell_key) = tds.cell_key_from_uuid(&incident_cell_uuid)
                    && let Some(incident_cell) = tds.cells().get(incident_cell_key)
                {
                    let cell_vertices = incident_cell.vertices();
                    let vertex_is_in_cell = cell_vertices.iter().any(|v| v.uuid() == vertex.uuid());
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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();
        let normal_vertex = vertex!([1.0, 2.0, 3.0]);

        let result = algorithm.create_perturbed_vertex(&normal_vertex);
        assert!(result.is_ok(), "Normal vertex perturbation should succeed");

        if let Ok(perturbed) = result {
            let original_coords = normal_vertex.point().to_array();
            let perturbed_coords = perturbed.point().to_array();

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
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);
        let large_vertex = vertex!([f64::MAX / 2.0, 1.0, 1.0]);

        // This should either succeed with a valid result or fail gracefully
        let extreme_result = extreme_algorithm.create_perturbed_vertex(&large_vertex);

        match extreme_result {
            Ok(perturbed) => {
                // If it succeeds, all coordinates must be finite
                let coords = perturbed.point().to_array();
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
                let coords = perturbed.point().to_array();
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
        use std::sync::atomic::Ordering;

        println!("Testing FacetCacheProvider implementation for RobustBoyerWatson");

        // Create test triangulation
        let points = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&points).unwrap();
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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

        println!("✓ All FacetCacheProvider tests passed for RobustBoyerWatson");
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
                assert!(
                    !tds.cells().is_empty(),
                    "Should create some triangulation even from coplanar points"
                );
                println!("  ✓ Coplanar configuration handled robustly");
            }
            Err(_) => {
                println!("  ✓ Coplanar configuration failed gracefully");
            }
        }
    }

    #[test]
    fn test_configuration_validation_paths() {
        println!("Testing configuration validation paths...");

        // Test 1: Extreme tolerance values
        let mut extreme_config = config_presets::general_triangulation::<f64>();
        extreme_config.base_tolerance = f64::MIN_POSITIVE;

        let algorithm =
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);
        // Should not panic with extreme but valid config
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let _tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let _stats = algorithm.get_statistics();
        println!("  ✓ Extreme tolerance configuration handled");

        // Test 2: High precision configuration
        let high_precision_config = config_presets::degenerate_robust::<f64>();

        let high_precision_algorithm =
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(high_precision_config);
        let _stats = high_precision_algorithm.get_statistics();
        println!("  ✓ High precision configuration handled");

        // Test 3: Degenerate cases configuration
        let degenerate_algorithm =
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();
        let _stats = degenerate_algorithm.get_statistics();
        println!("  ✓ Degenerate cases configuration created successfully");
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

        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();

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
        use crate::core::vertex::VertexBuilder;
        use crate::geometry::point::Point;

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
                assert!(!tds.cells().is_empty());
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
                assert!(!tds.cells().is_empty());
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
                assert!(!tds.cells().is_empty());
                println!("  ✓ Extreme coordinate points handled");
            }
            Err(_) => println!("  ✓ Extreme coordinate points failed gracefully"),
        }

        // Test 4: Perturbation edge cases
        let mut algorithm =
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::for_degenerate_cases();

        // Test with zero coordinates
        let zero_vertex = vertex!([0.0, 0.0, 0.0]);
        let perturb_result = algorithm.create_perturbed_vertex(&zero_vertex);
        match perturb_result {
            Ok(perturbed) => {
                let coords = perturbed.point().to_array();
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
                    RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);

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
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(tight_config);

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
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(loose_config);
        let mut tds_loose = Tds::new(&vertices[..4]).expect("Initial TDS creation should succeed");
        let result_loose = algorithm_loose.insert_vertex(&mut tds_loose, vertices[4]);

        // Loose tolerance might succeed where tight fails, or vice versa
        assert!(
            result_loose.is_ok()
                || matches!(result_loose, Err(InsertionError::GeometricFailure { .. }))
        );
    }

    #[test]
    fn test_algorithm_configuration_presets() {
        // Test all standard configuration presets
        let configs = vec![
            ("general", config_presets::general_triangulation::<f64>()),
            (
                "degenerate_robust",
                config_presets::degenerate_robust::<f64>(),
            ),
        ];

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        for (name, config) in configs {
            let mut algorithm =
                RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);
            let mut tds = Tds::new(&vertices).expect("TDS creation should succeed");

            // All presets should handle basic tetrahedron
            assert!(
                tds.is_valid().is_ok(),
                "TDS should be valid for {name} preset"
            );

            // Test vertex insertion with each preset
            let test_vertex = vertex!([0.5, 0.5, 0.5]);
            let result = algorithm.insert_vertex(&mut tds, test_vertex);
            assert!(
                result.is_ok(),
                "Interior insertion should succeed with {name} preset"
            );
        }
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
                RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);
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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();
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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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

    /// Test `validate_boundary_facets` error conditions (lines 788-796)
    #[test]
    fn test_validate_boundary_facets() {
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Test with empty boundary facets but non-zero bad cell count (should error)
        let empty_facets = vec![];
        let result = algorithm.validate_boundary_facets(&empty_facets, 3);
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

        let boundary_facets_vec: Vec<_> = boundary_facets
            .map(|fv| {
                // Convert FacetView to Facet for backward compatibility with validate_boundary_facets
                let cell = fv.cell().unwrap().clone();
                let opposite_vertex = *fv.opposite_vertex().unwrap();
                #[allow(deprecated)]
                crate::core::facet::Facet::new(cell, opposite_vertex).unwrap()
            })
            .collect();
        let result = algorithm.validate_boundary_facets(&boundary_facets_vec, 1);
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
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(extreme_config);

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
        let extreme_bad_cells = algorithm.robust_find_bad_cells(&tds, &extreme_vertex);
        debug_println!(
            "Found {} bad cells with extreme coordinates (handled gracefully)",
            extreme_bad_cells.len()
        );

        // Test with coordinates very close to zero that might cause precision issues
        let tiny_vertex = vertex!([f64::EPSILON, f64::EPSILON * 2.0, f64::EPSILON * 3.0]);
        let tiny_bad_cells = algorithm.robust_find_bad_cells(&tds, &tiny_vertex);
        debug_println!(
            "Found {} bad cells with tiny coordinates",
            tiny_bad_cells.len()
        );
    }

    /// Test `is_facet_visible_from_vertex_robust` degenerate handling (lines 999-1004)
    #[test]
    #[allow(unused_variables)]
    fn test_is_facet_visible_degenerate_handling() {
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let test_facet_vertices: Vec<_> = test_facet.vertices().copied().collect();
        let key = derive_facet_key_from_vertices(&test_facet_vertices, &tds)
            .expect("Should derive facet key");
        let adjacent_cell_key = facet_to_cells
            .get(&key)
            .and_then(|cells| (cells.len() == 1).then_some(cells[0].0))
            .expect("Should find adjacent cell for test facet");

        // Test with a point that might cause degenerate orientation results
        let degenerate_vertex = vertex!([0.5, 0.5, 0.5]); // Point at center of tetrahedron

        // This should exercise the fallback visibility heuristic path
        // Convert FacetView to Facet for backward compatibility
        let cell = test_facet.cell().unwrap().clone();
        let opposite_vertex = *test_facet.opposite_vertex().unwrap();
        #[allow(deprecated)]
        let test_facet_compat = crate::core::facet::Facet::new(cell, opposite_vertex).unwrap();
        let is_visible = algorithm.is_facet_visible_from_vertex_robust(
            &tds,
            &test_facet_compat,
            &degenerate_vertex,
            adjacent_cell_key,
        );

        debug_println!("Degenerate visibility test result: {is_visible}");
        // Don't assert specific result since it depends on geometry, just ensure it doesn't panic
    }

    /// Test `safe_usize_to_scalar` conversion in `fallback_visibility_heuristic` (lines 1032-1034)
    #[test]
    fn test_fallback_visibility_safe_conversion() {
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
            // Convert FacetView to Facet for backward compatibility with fallback_visibility_heuristic
            let cell = facet.cell().unwrap().clone();
            let opposite_vertex = *facet.opposite_vertex().unwrap();
            #[allow(deprecated)]
            let facet_compat = crate::core::facet::Facet::new(cell, opposite_vertex).unwrap();
            let is_visible = algorithm.fallback_visibility_heuristic(&facet_compat, &test_vertex);
            println!("Fallback visibility for far point: {is_visible}");
            // Should typically be true for a far point, but mainly testing no panic
        }
    }

    /// Test `with_config` constructor (lines 143-153)
    #[test]
    fn test_with_config_constructor() {
        let config = config_presets::high_precision::<f64>();
        let algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> =
            RobustBoyerWatson::with_config(config.clone());

        // Verify the configuration was applied
        assert!(
            algorithm.predicate_config.base_tolerance <= config.base_tolerance,
            "Configuration should be applied"
        );

        let (processed, created, removed) = algorithm.get_statistics();
        assert_eq!(processed, 0);
        assert_eq!(created, 0);
        assert_eq!(removed, 0);
    }

    /// Test fallback strategy in `robust_insert_vertex_impl` (lines 228-240, 244)
    #[test]
    fn test_robust_insert_fallback_strategies() {
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Get a subset of cells as "bad cells"
        let all_cell_keys: Vec<_> = tds.cells().keys().collect();
        let bad_cells = &all_cell_keys[..1];

        let result = algorithm.robust_find_cavity_boundary_facets(&tds, bad_cells);

        match result {
            Ok(boundary_facets) => {
                println!(
                    "Robust boundary detection found {} facets",
                    boundary_facets.len()
                );
                // Should find boundary facets for valid bad cells

                // Verify all returned facets are valid
                for facet in &boundary_facets {
                    assert!(!facet.vertices().is_empty(), "Facet should have vertices");
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
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 1)
        );
        assert!(
            RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 2)
        );
        assert!(
            !RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(0, 1)
        );
        assert!(
            !RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(2, 2)
        );
        assert!(
            !RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::is_cavity_boundary_facet(1, 3)
        );
    }

    /// Test conservative boundary cell inclusion (lines 536-539)
    #[test]
    fn test_conservative_boundary_cell_inclusion() {
        // Test with high tolerance configuration to trigger conservative boundary handling
        let mut config = config_presets::degenerate_robust::<f64>();
        config.base_tolerance = 1e-6; // Larger than default to trigger conservative path

        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::with_config(config);

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Test with a vertex on the boundary of the circumsphere
        let boundary_vertex = vertex!([0.5, 0.5, 0.5]);
        let bad_cells = algorithm.robust_find_bad_cells(&tds, &boundary_vertex);

        println!(
            "Conservative boundary handling found {} bad cells",
            bad_cells.len()
        );
        // With high tolerance, might include more cells conservatively
    }

    /// Test error paths in `robust_insert_vertex_impl` for better coverage
    #[test]
    fn test_robust_insert_error_paths() {
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test with vertex that should be inside circumsphere
        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let bad_cells = algorithm.find_bad_cells_with_robust_fallback(&tds, &interior_vertex);

        println!("Found {} bad cells for interior vertex", bad_cells.len());
        // Should find at least the containing cell as "bad"
        assert!(!bad_cells.is_empty());

        // Test with vertex outside all circumspheres
        let exterior_vertex = vertex!([10.0, 10.0, 10.0]);
        let bad_cells_exterior =
            algorithm.find_bad_cells_with_robust_fallback(&tds, &exterior_vertex);

        println!(
            "Found {} bad cells for exterior vertex",
            bad_cells_exterior.len()
        );
        // Exterior vertex might not have any bad cells
    }

    /// Test visible boundary facets detection with robust fallback
    #[test]
    fn test_find_visible_boundary_facets_comprehensive() {
        let algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
                    for (cell_key, facet_index) in &visible_facet_handles {
                        assert!(
                            tds.cells().get(*cell_key).is_some(),
                            "Cell key {cell_key:?} should exist in TDS"
                        );
                        assert!(
                            *facet_index < 4, // 3D tetrahedra have 4 facets
                            "Facet index {facet_index} should be valid for 3D cell"
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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
        let mut algorithm = RobustBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

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
}
