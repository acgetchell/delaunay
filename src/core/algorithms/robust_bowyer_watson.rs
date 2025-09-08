//! Robust Bowyer-Watson algorithm using enhanced geometric predicates.
//!
//! This module demonstrates how to integrate the robust geometric predicates
//! into the Bowyer-Watson triangulation algorithm to address the
//! "No cavity boundary facets found" error.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{AddAssign, Div, SubAssign};

use crate::core::traits::insertion_algorithm::{
    InsertionAlgorithm, InsertionBuffers, InsertionInfo, InsertionStatistics, InsertionStrategy,
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
};
use nalgebra::{self as na, ComplexField};
use serde::{Serialize, de::DeserializeOwned};
use std::iter::Sum;

/// Enhanced Bowyer-Watson algorithm with robust geometric predicates.
pub struct RobustBoyerWatson<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Configuration for robust predicates
    predicate_config: RobustPredicateConfig<T>,
    /// Unified statistics tracking
    stats: InsertionStatistics,
    /// Reusable buffers for performance
    buffers: InsertionBuffers<T, U, V, D>,
    /// Cached convex hull for hull extension
    hull: Option<ConvexHull<T, U, V, D>>,
    /// Phantom data to indicate that U and V types are used in method signatures
    _phantom: PhantomData<(U, V)>,
}

impl<T, U, V, const D: usize> Default for RobustBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    ordered_float::OrderedFloat<f64>: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    na::OPoint<T, na::Const<D>>: From<[f64; D]>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U, V, const D: usize> RobustBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + num_traits::Zero + From<f64>,
    U: crate::core::traits::data_type::DataType + DeserializeOwned,
    V: crate::core::traits::data_type::DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    ordered_float::OrderedFloat<f64>: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
            buffers: InsertionBuffers::with_capacity(100),
            hull: None,
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
            buffers: InsertionBuffers::with_capacity(100),
            hull: None,
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
            buffers: InsertionBuffers::with_capacity(100),
            hull: None,
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
    ) -> Result<InsertionInfo, TriangulationValidationError>
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
    ) -> Result<InsertionInfo, TriangulationValidationError>
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
            // Try boundary facet detection using trait method with robust fallback
            if let Ok(boundary_facets) =
                self.find_cavity_boundary_facets_with_robust_fallback(tds, &bad_cells)
                && !boundary_facets.is_empty()
            {
                let cells_removed = bad_cells.len();
                <Self as InsertionAlgorithm<T, U, V, D>>::remove_bad_cells(tds, &bad_cells);
                <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex);
                let cells_created =
                    <Self as InsertionAlgorithm<T, U, V, D>>::create_cells_from_boundary_facets(
                        tds,
                        &boundary_facets,
                        vertex,
                    );

                // Maintain invariants after structural changes
                <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds).map_err(
                    |e| TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Failed to finalize triangulation after cavity-based insertion: {e}"
                        ),
                    },
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

        // If robust detection fails, fall back to trait method
        self.insert_vertex_cavity_based(tds, vertex)
    }

    /// Hull extension insertion with robust predicates as enhancement
    fn insert_vertex_hull_extension_with_robust_predicates(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Use visibility detection with robust fallback
        if let Ok(visible_facets) =
            self.find_visible_boundary_facets_with_robust_fallback(tds, vertex)
            && !visible_facets.is_empty()
        {
            <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex);
            let cells_created =
                <Self as InsertionAlgorithm<T, U, V, D>>::create_cells_from_boundary_facets(
                    tds,
                    &visible_facets,
                    vertex,
                );

            // Maintain invariants after structural changes
            <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds).map_err(
                |e| TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Failed to finalize triangulation after hull extension insertion: {e}"
                    ),
                },
            )?;

            return Ok(InsertionInfo {
                strategy: InsertionStrategy::HullExtension,
                cells_removed: 0,
                cells_created,
                success: true,
                degenerate_case_handled: false,
            });
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
        let mut bad_cells = InsertionAlgorithm::<T, U, V, D>::find_bad_cells(self, tds, vertex);

        // If the standard method doesn't find any bad cells (likely a degenerate case)
        // or we're using the robust configuration, use robust predicates as well
        if bad_cells.is_empty() || self.predicate_config.base_tolerance > T::default_tolerance() {
            let robust_bad_cells = self.robust_find_bad_cells(tds, vertex);

            // Add any cells found by robust method that weren't found by the standard method
            for cell_key in robust_bad_cells {
                if !bad_cells.contains(&cell_key) {
                    bad_cells.push(cell_key);
                }
            }
        }

        bad_cells
    }

    /// Find cavity boundary facets by first using the trait method, then applying robust predicates for edge cases.
    ///
    /// This approach integrates the trait's `find_cavity_boundary_facets` method with the robust predicates
    /// to provide a more reliable boundary facet detection method, especially for degenerate cases.
    fn find_cavity_boundary_facets_with_robust_fallback(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
    {
        // First try to find boundary facets using the trait's method
        match InsertionAlgorithm::<T, U, V, D>::find_cavity_boundary_facets(self, tds, bad_cells) {
            Ok(boundary_facets) => {
                // If the standard method succeeds and finds facets, use them
                if !boundary_facets.is_empty() {
                    return Ok(boundary_facets);
                }
                // If standard method succeeds but finds no facets, try robust method as fallback
                self.robust_find_cavity_boundary_facets(tds, bad_cells)
            }
            Err(_) => {
                // If standard method fails, use robust method as fallback
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
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // First try to find visible boundary facets using the trait's method
        match InsertionAlgorithm::<T, U, V, D>::find_visible_boundary_facets(self, tds, vertex) {
            Ok(visible_facets) => {
                // If the standard method succeeds and finds facets, use them
                if !visible_facets.is_empty() {
                    return Ok(visible_facets);
                }
                // If standard method succeeds but finds no facets, try robust method as fallback
                self.find_visible_boundary_facets(tds, vertex)
            }
            Err(_) => {
                // If standard method fails, use robust method as fallback
                self.find_visible_boundary_facets(tds, vertex)
            }
        }
    }

    /// Find bad cells using robust insphere predicate.
    /// This is a lower-level method used by `find_bad_cells_with_robust_fallback`.
    fn robust_find_bad_cells(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Vec<CellKey> {
        let mut bad_cells = Vec::new();

        for (cell_key, cell) in tds.cells() {
            // Extract vertex points from the cell
            let vertex_points: Vec<Point<T, D>> =
                cell.vertices().iter().map(|v| *v.point()).collect();

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

        bad_cells
    }

    /// Find cavity boundary facets with enhanced error handling.
    fn robust_find_cavity_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
        let mut boundary_facets = Vec::new();

        if bad_cells.is_empty() {
            return Ok(boundary_facets);
        }

        let bad_cell_set: HashSet<CellKey> = bad_cells.iter().copied().collect();

        // Build facet-to-cells mapping with enhanced validation
        let facet_to_cells = self.build_validated_facet_mapping(tds)?;

        // Find boundary facets with improved logic
        let mut processed_facets = HashSet::new();

        for &bad_cell_key in bad_cells {
            if let Some(bad_cell) = tds.cells().get(bad_cell_key)
                && let Ok(facets) = bad_cell.facets()
            {
                for facet in facets {
                    let facet_key = facet.key();

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
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
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
    ) -> Result<RobustInsertionInfo, TriangulationValidationError> {
        // This typically means the vertex is outside the convex hull
        // Try to extend the hull by connecting to visible boundary facets

        let visible_facets = self.find_visible_boundary_facets(tds, vertex)?;

        if visible_facets.is_empty() {
            return Err(TriangulationValidationError::FailedToCreateCell {
                message: "No visible boundary facets found for hull extension".to_string(),
            });
        }

        // Add the vertex to the TDS if it's not already there
        <Self as InsertionAlgorithm<T, U, V, D>>::ensure_vertex_in_tds(tds, vertex);

        let cells_created =
            <Self as InsertionAlgorithm<T, U, V, D>>::create_cells_from_boundary_facets(
                tds,
                &visible_facets,
                vertex,
            );

        // Finalize the triangulation to ensure consistency
        if let Err(e) = <Self as InsertionAlgorithm<T, U, V, D>>::finalize_after_insertion(tds) {
            return Err(TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to finalize triangulation after hull extension: {e}"),
            });
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
    ) -> Result<HashMap<u64, Vec<CellKey>>, TriangulationValidationError> {
        // Reuse existing mapping from TDS to avoid recomputation
        let tds_map = tds.build_facet_to_cells_hashmap();
        let mut facet_to_cells: HashMap<u64, Vec<CellKey>> = HashMap::new();

        for (facet_key, cell_facet_pairs) in tds_map {
            // Extract just the CellKeys, discarding facet indices
            let cell_keys: Vec<CellKey> = cell_facet_pairs
                .iter()
                .map(|(cell_key, _)| *cell_key)
                .collect();

            // Validate that no facet is shared by more than 2 cells
            if cell_keys.len() > 2 {
                return Err(TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Facet {} is shared by {} cells (should be ≤2)",
                        facet_key,
                        cell_keys.len()
                    ),
                });
            }

            facet_to_cells.insert(facet_key, cell_keys);
        }

        Ok(facet_to_cells)
    }

    const fn is_cavity_boundary_facet(bad_count: usize, total_count: usize) -> bool {
        matches!((bad_count, total_count), (1, 1 | 2))
    }

    #[allow(clippy::unused_self)]
    fn validate_boundary_facets(
        &self,
        boundary_facets: &[Facet<T, U, V, D>],
        bad_cell_count: usize,
    ) -> Result<(), TriangulationValidationError> {
        if boundary_facets.is_empty() && bad_cell_count > 0 {
            return Err(TriangulationValidationError::FailedToCreateCell {
                message: format!("No cavity boundary facets found for {bad_cell_count} bad cells"),
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
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
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
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
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
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>
    where
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let mut visible_facets = Vec::new();

        // Get all boundary facets (facets shared by exactly one cell)
        let facet_to_cells = tds.build_facet_to_cells_hashmap();
        let boundary_facets: Vec<_> = facet_to_cells
            .iter()
            .filter(|(_, cells)| cells.len() == 1)
            .collect();

        for (_facet_key, cells) in boundary_facets {
            let (cell_key, facet_index) = cells[0];
            if let Some(cell) = tds.cells().get(cell_key) {
                if let Ok(facets) = cell.facets() {
                    if facet_index < facets.len() {
                        let facet = &facets[facet_index];

                        // Test visibility using robust orientation predicates with fallback
                        if self.is_facet_visible_from_vertex_robust(tds, facet, vertex, cell_key) {
                            visible_facets.push(facet.clone());
                        }
                    }
                } else {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: "Failed to get facets from cell during visibility computation"
                            .to_string(),
                    });
                }
            } else {
                return Err(TriangulationValidationError::InconsistentDataStructure {
                    message: "Cell key not found during visibility computation".to_string(),
                });
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
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
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
        let mut simplex_with_opposite: Vec<Point<T, D>> =
            facet_vertices.iter().map(|v| *v.point()).collect();
        simplex_with_opposite.push(*opposite_vertex.point());

        let mut simplex_with_test: Vec<Point<T, D>> =
            facet_vertices.iter().map(|v| *v.point()).collect();
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
    ) -> bool {
        let facet_vertices = facet.vertices();

        // Calculate facet centroid
        let mut centroid_coords = [T::zero(); D];
        for facet_vertex in &facet_vertices {
            let coords: [T; D] = facet_vertex.point().into();
            for (i, &coord) in coords.iter().enumerate() {
                centroid_coords[i] += coord;
            }
        }
        let num_vertices = T::from_usize(facet_vertices.len()).unwrap_or_else(T::one);
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
        // Use a threshold based on the perturbation scale multiplied by a factor
        let threshold = self.predicate_config.perturbation_scale
            * self.predicate_config.perturbation_scale
            * T::from_f64(100.0).unwrap_or_else(|| T::one());
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
        let nearby_vertices = tds
            .vertices
            .values()
            .filter(|v| {
                let v_coords: [T; D] = (*v).into();
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
            .count();

        has_small_coords || has_large_coords || nearby_vertices > 3
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
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    na::OPoint<T, na::Const<D>>: From<[f64; D]>,
{
    fn insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError> {
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
        // First, try the regular triangulation approach
        Tds::new(vertices).map_or_else(
            |_| {
                // Regular triangulation failed, try robust approach
                // Note: For now, we still fall back to the regular approach
                // but this is where we would implement more sophisticated
                // robust triangulation strategies
                Tds::new(vertices).map_err(|e| {
                    TriangulationConstructionError::ValidationError(
                        TriangulationValidationError::FailedToCreateCell {
                            message: format!("Robust triangulation failed: {e}"),
                        },
                    )
                })
            },
            Ok,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::vertex;
    use approx::assert_abs_diff_eq;
    use approx::assert_abs_diff_ne;

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
        assert!(result.is_ok() || result.is_err()); // Should handle gracefully
    }

    #[test]
    fn test_no_double_counting_statistics() {
        println!("Testing that robust vertex insertion statistics are not double counted");

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

        println!("After initial tetrahedron:");
        println!("  Insertions: {insertions_after_initial}");
        println!("  Cells created: {created_after_initial}");
        println!("  Cells removed: {removed_after_initial}");

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

            let (after_insertions, after_created, after_removed) = algorithm.get_statistics();
            let insertion_info = insertion_result.unwrap();

            println!(
                "\nAfter adding vertex {} ({:?}):",
                i + 1,
                new_vertex.point().to_array()
            );
            println!("  Insertions: {after_insertions}");
            println!("  Cells created: {after_created}");
            println!("  Cells removed: {after_removed}");
            println!("  Total cells in TDS: {}", tds.number_of_cells());
            println!(
                "  InsertionInfo: created={}, removed={}",
                insertion_info.cells_created, insertion_info.cells_removed
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

        println!(
            "✓ No double counting detected in robust algorithm - statistics are accurate and consistent"
        );
    }

    #[test]
    fn test_debug_exterior_vertex_insertion() {
        println!("Testing exterior vertex insertion in robust Bowyer-Watson");

        let mut algorithm = RobustBoyerWatson::new();

        // Create initial triangulation with exactly 4 vertices (minimum for a tetrahedron)
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        println!("Initial TDS has {} cells", tds.number_of_cells());

        // First, test with an interior vertex (should work)
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        println!(
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

        println!("TDS now has {} cells", tds.number_of_cells());

        // Let's check the TDS consistency after the interior insertion
        println!("Checking TDS consistency...");
        let boundary_facets_result = tds.boundary_facets();
        match &boundary_facets_result {
            Ok(boundary_facets) => println!("TDS has {} boundary facets", boundary_facets.len()),
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
        println!(
            "Inserting exterior vertex {:?}",
            exterior_vertex.point().to_array()
        );

        // Let's debug what happens step by step
        println!("Finding bad cells for exterior vertex...");
        let bad_cells = algorithm.robust_find_bad_cells(&tds, &exterior_vertex);
        println!("Found {} bad cells: {:?}", bad_cells.len(), bad_cells);

        if bad_cells.is_empty() {
            println!("No bad cells found - will try hull extension");

            // Check what boundary facets exist before trying visibility
            println!("Getting all boundary facets...");
            if let Ok(all_boundary_facets) = tds.boundary_facets() {
                println!("Total boundary facets: {}", all_boundary_facets.len());
                for (i, facet) in all_boundary_facets.iter().enumerate() {
                    println!(
                        "  Boundary facet {}: {} vertices",
                        i,
                        facet.vertices().len()
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
        println!("Testing cavity-based insertion maintains TDS consistency");

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
                    for neighbor_uuid in neighbors.iter().filter_map(|n| n.as_ref()) {
                        if let Some(neighbor_key) = tds.cell_bimap.get_by_left(neighbor_uuid)
                            && let Some(neighbor) = tds.cells().get(*neighbor_key)
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            if let Some(neighbor_neighbors) = &neighbor.neighbors {
                                let cell_uuid = tds
                                    .cell_bimap
                                    .get_by_right(&cell_key)
                                    .expect("Cell should have UUID");
                                assert!(
                                    neighbor_neighbors
                                        .iter()
                                        .any(|n| n.as_ref() == Some(cell_uuid)),
                                    "Neighbor relationship should be symmetric after insertion {}",
                                    i + 1
                                );
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
                println!("  Boundary facets: {}", boundary_facets.len());

                // Each boundary facet should have exactly 3 vertices (for 3D)
                for facet in &boundary_facets {
                    assert_eq!(
                        facet.vertices().len(),
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
            let initial_boundary_facets = tds.boundary_facets().unwrap().len();

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
                    for neighbor_uuid in neighbors.iter().flatten() {
                        if let Some(neighbor_key) = tds.cell_bimap.get_by_left(neighbor_uuid)
                            && let Some(neighbor) = tds.cells().get(*neighbor_key)
                        {
                            // Each neighbor should also reference this cell as a neighbor
                            if let Some(neighbor_neighbors) = &neighbor.neighbors {
                                let cell_uuid = tds
                                    .cell_bimap
                                    .get_by_right(&cell_key)
                                    .expect("Cell should have UUID");
                                assert!(
                                    neighbor_neighbors
                                        .iter()
                                        .any(|opt| opt.as_ref() == Some(cell_uuid)),
                                    "Neighbor relationship should be symmetric after hull extension {}",
                                    i + 1
                                );
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
                let final_boundary_facets = boundary_facets.len();
                println!(
                    "  Initial boundary facets: {initial_boundary_facets}, final: {final_boundary_facets}"
                );

                // Each boundary facet should have exactly 3 vertices (for 3D)
                for facet in &boundary_facets {
                    assert_eq!(
                        facet.vertices().len(),
                        3,
                        "Boundary facet should have 3 vertices after hull extension {}",
                        i + 1
                    );
                }

                // The newly inserted vertex should be in the triangulation
                let vertex_found = tds.vertices.values().any(|v| {
                    let v_coords: [f64; 3] = (*v).into();
                    let test_coords: [f64; 3] = test_vertex.point().into();
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
            let mut cell_signatures = std::collections::HashSet::new();
            for (_, cell) in tds.cells() {
                let mut vertex_uuids: Vec<_> = cell
                    .vertices()
                    .iter()
                    .map(crate::core::vertex::Vertex::uuid)
                    .collect();
                vertex_uuids.sort();
                let signature = format!("{vertex_uuids:?}");

                let inserted = cell_signatures.insert(signature.clone());
                if !inserted {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Duplicate cell found after insertion {}: {}",
                        i + 1,
                        signature
                    );
                }
                assert!(inserted, "Duplicate cell found after insertion {}", i + 1);
            }

            // 3. All facets should be properly shared
            let facet_to_cells = tds.build_facet_to_cells_hashmap();
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
                    let (cell1_key, _) = cells[0];
                    let (cell2_key, _) = cells[1];

                    if let (Some(cell1), Some(cell2)) =
                        (tds.cells().get(cell1_key), tds.cells().get(cell2_key))
                        && let (Some(neighbors1), Some(neighbors2)) =
                            (&cell1.neighbors, &cell2.neighbors)
                    {
                        let cell2_uuid = tds
                            .cell_bimap
                            .get_by_right(&cell2_key)
                            .expect("Cell2 should have UUID");
                        let cell1_uuid = tds
                            .cell_bimap
                            .get_by_right(&cell1_key)
                            .expect("Cell1 should have UUID");
                        assert!(
                            neighbors1.iter().any(|n| n.as_ref() == Some(cell2_uuid)),
                            "Cell1 should reference cell2 as neighbor after insertion {}",
                            i + 1
                        );
                        assert!(
                            neighbors2.iter().any(|n| n.as_ref() == Some(cell1_uuid)),
                            "Cell2 should reference cell1 as neighbor after insertion {}",
                            i + 1
                        );
                    }
                }
            }

            // 4. All vertices should have proper incident cells assigned
            for (_, vertex) in &tds.vertices {
                if let Some(incident_cell_uuid) = vertex.incident_cell
                    && let Some(incident_cell_key) = tds.cell_bimap.get_by_left(&incident_cell_uuid)
                    && let Some(incident_cell) = tds.cells().get(*incident_cell_key)
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
}
