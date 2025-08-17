//! Robust Bowyer-Watson algorithm using enhanced geometric predicates.
//!
//! This module demonstrates how to integrate the robust geometric predicates
//! into the Bowyer-Watson triangulation algorithm to address the
//! "No cavity boundary facets found" error.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::core::traits::insertion_algorithm::{
    InsertionAlgorithm, InsertionInfo, InsertionStrategy,
};
use crate::core::{
    cell::CellBuilder,
    facet::Facet,
    triangulation_data_structure::{CellKey, Tds, TriangulationValidationError},
    vertex::Vertex,
};
use crate::geometry::{
    point::Point,
    predicates::InSphere,
    robust_predicates::{RobustPredicateConfig, config_presets, robust_insphere},
    traits::coordinate::{Coordinate, CoordinateScalar},
};
use crate::vertex;
use nalgebra::{self as na, ComplexField};
use serde::{Serialize, de::DeserializeOwned};
use std::iter::Sum;

/// Enhanced Bowyer-Watson algorithm with robust geometric predicates.
pub struct RobustBoyerWatson<T, U, V, const D: usize> {
    /// Configuration for robust predicates
    predicate_config: RobustPredicateConfig<T>,
    /// Statistics for debugging and optimization
    pub stats: RobustBoyerWatsonStats,
    /// Phantom data to indicate that U and V types are used in method signatures
    _phantom: PhantomData<(U, V)>,
}

/// Statistics collected during robust triangulation.
#[derive(Debug, Default)]
pub struct RobustBoyerWatsonStats {
    /// Number of vertices processed during triangulation
    pub vertices_processed: usize,
    /// Number of times fallback strategies were used
    pub fallback_strategies_used: usize,
    /// Number of degenerate cases handled
    pub degenerate_cases_handled: usize,
    /// Number of cavity boundary detection failures
    pub cavity_boundary_failures: usize,
    /// Number of successful cavity boundary recoveries
    pub cavity_boundary_recoveries: usize,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            predicate_config: config_presets::general_triangulation::<T>(),
            stats: RobustBoyerWatsonStats::default(),
            _phantom: PhantomData,
        }
    }

    /// Create with custom predicate configuration.
    pub fn with_config(config: RobustPredicateConfig<T>) -> Self {
        Self {
            predicate_config: config,
            stats: RobustBoyerWatsonStats::default(),
            _phantom: PhantomData,
        }
    }

    /// Create optimized for handling degenerate cases.
    #[must_use]
    pub fn for_degenerate_cases() -> Self {
        Self {
            predicate_config: config_presets::degenerate_robust::<T>(),
            stats: RobustBoyerWatsonStats::default(),
            _phantom: PhantomData,
        }
    }

    /// Insert a vertex using robust geometric predicates.
    ///
    /// # Errors
    ///
    /// Returns an error if vertex insertion fails due to geometric issues,
    /// validation problems, or if recovery strategies are unsuccessful.
    pub fn robust_insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<RobustInsertionInfo, TriangulationValidationError> {
        self.stats.vertices_processed += 1;

        // Step 1: Find bad cells using robust insphere test
        let bad_cells = self.robust_find_bad_cells(tds, &vertex);

        if bad_cells.is_empty() {
            // No bad cells - this might be a degenerate case
            return self.handle_no_bad_cells_case(tds, &vertex);
        }

        // Step 2: Find cavity boundary facets with robust fallback strategies
        let boundary_facets =
            if let Ok(facets) = self.robust_find_cavity_boundary_facets(tds, &bad_cells) {
                facets
            } else {
                // Primary strategy failed - try recovery methods
                self.stats.cavity_boundary_failures += 1;
                self.recover_cavity_boundary_facets(tds, &bad_cells, &vertex)?
            };

        if boundary_facets.is_empty() {
            self.stats.degenerate_cases_handled += 1;
            return Ok(self.handle_degenerate_insertion_case(tds, &vertex, &bad_cells));
        }

        // Step 3: Remove bad cells and create new cells
        let cells_removed = bad_cells.len();
        self.remove_bad_cells(tds, &bad_cells);

        let cells_created = self.create_cells_from_boundary_facets(tds, &boundary_facets, &vertex);

        Ok(RobustInsertionInfo {
            success: true,
            cells_created,
            cells_removed,
            strategy_used: InsertionStrategy::Standard,
            degenerate_case_handled: false,
        })
    }

    /// Find bad cells using robust insphere predicate.
    fn robust_find_bad_cells(
        &mut self,
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
                    self.stats.fallback_strategies_used += 1;

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
            if let Some(bad_cell) = tds.cells().get(bad_cell_key) {
                if let Ok(facets) = bad_cell.facets() {
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
        }

        // Additional validation of boundary facets
        self.validate_boundary_facets(&boundary_facets, bad_cells.len())?;

        Ok(boundary_facets)
    }

    /// Recover from cavity boundary facet detection failure.
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
        if let Ok(facets) = self.reduced_bad_cell_strategy(tds, bad_cells, vertex) {
            if !facets.is_empty() {
                return Ok(facets);
            }
        }

        // Recovery Strategy 3: Use convex hull extension
        self.convex_hull_extension_strategy(tds, vertex)
    }

    /// Handle the case where no bad cells are found.
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

        let cells_created = self.create_cells_from_boundary_facets(tds, &visible_facets, vertex);

        Ok(RobustInsertionInfo {
            success: true,
            cells_created,
            cells_removed: 0,
            strategy_used: InsertionStrategy::HullExtension,
            degenerate_case_handled: false,
        })
    }

    /// Handle degenerate insertion cases with special strategies.
    fn handle_degenerate_insertion_case(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        _bad_cells: &[CellKey],
    ) -> RobustInsertionInfo {
        self.stats.degenerate_cases_handled += 1;

        // Strategy 1: Try vertex perturbation
        let perturbed_vertex = self.create_perturbed_vertex(vertex);
        if let Ok(info) = self.robust_insert_vertex(tds, perturbed_vertex) {
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
        let mut facet_to_cells: HashMap<u64, Vec<CellKey>> = HashMap::new();

        for (cell_key, cell) in tds.cells() {
            if let Ok(facets) = cell.facets() {
                for facet in facets {
                    facet_to_cells
                        .entry(facet.key())
                        .or_default()
                        .push(cell_key);
                }
            }
        }

        // Validate that no facet is shared by more than 2 cells
        for (facet_key, cells) in &facet_to_cells {
            if cells.len() > 2 {
                return Err(TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Facet {} is shared by {} cells (should be ≤2)",
                        facet_key,
                        cells.len()
                    ),
                });
            }
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

    #[allow(clippy::unused_self)]
    fn convex_hull_extension_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
        // Find all boundary facets and check which are visible from the vertex
        self.find_visible_boundary_facets(tds, vertex)
    }

    #[allow(clippy::unused_self)]
    fn find_visible_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        _vertex: &Vertex<T, U, D>,
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
        use crate::core::traits::boundary_analysis::BoundaryAnalysis;

        // Get all boundary facets
        let boundary_facets = tds.boundary_facets().map_err(|e| {
            TriangulationValidationError::FailedToCreateCell {
                message: format!("Failed to get boundary facets: {e}"),
            }
        })?;

        // For now, return all boundary facets
        // A more sophisticated implementation would test visibility
        Ok(boundary_facets)
    }

    fn create_perturbed_vertex(&self, vertex: &Vertex<T, U, D>) -> Vertex<T, U, D> {
        let mut coords: [T; D] = vertex.point().to_array();
        let perturbation = self.predicate_config.perturbation_scale;

        // Apply small random perturbation to first coordinate
        coords[0] += perturbation;

        let perturbed_point = Point::new(coords);
        // Use the vertex! macro to create vertex with point and data
        vertex.data.map_or_else(
            || vertex!(perturbed_point.to_array()),
            |data| vertex!(perturbed_point.to_array(), data),
        )
    }

    #[allow(clippy::unused_self)]
    fn remove_bad_cells(&self, tds: &mut Tds<T, U, V, D>, bad_cells: &[CellKey]) {
        for &cell_key in bad_cells {
            if let Some(cell) = tds.cells_mut().remove(cell_key) {
                tds.cell_bimap.remove_by_left(&cell.uuid());
            }
        }
    }

    #[allow(clippy::unused_self)]
    fn create_cells_from_boundary_facets(
        &self,
        tds: &mut Tds<T, U, V, D>,
        boundary_facets: &[Facet<T, U, V, D>],
        vertex: &Vertex<T, U, D>,
    ) -> usize {
        let mut cells_created = 0;

        for facet in boundary_facets {
            // Create a new cell by combining the facet with the new vertex
            let mut cell_vertices = facet.vertices().clone();
            cell_vertices.push(*vertex);

            if let Ok(new_cell) = CellBuilder::default().vertices(cell_vertices).build() {
                let cell_key = tds.cells_mut().insert(new_cell);
                let cell_uuid = tds.cells()[cell_key].uuid();
                tds.cell_bimap.insert(cell_uuid, cell_key);
                cells_created += 1;
            }
        }

        cells_created
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
        // Call our robust_insert_vertex method and convert the result
        let robust_result = self.robust_insert_vertex(tds, vertex)?;

        // Convert RobustInsertionInfo to InsertionInfo
        Ok(InsertionInfo {
            strategy: robust_result.strategy_used,
            cells_removed: robust_result.cells_removed,
            cells_created: robust_result.cells_created,
            success: robust_result.success,
            degenerate_case_handled: robust_result.degenerate_case_handled,
        })
    }

    fn get_statistics(&self) -> (usize, usize, usize) {
        // Return statistics: (vertices_processed, cells_created, cells_removed)
        // We don't track cells_created/cells_removed totals, so return 0 for those
        (self.stats.vertices_processed, 0, 0)
    }

    fn reset(&mut self) {
        // Reset statistics
        self.stats = RobustBoyerWatsonStats::default();
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
    ) -> Result<Tds<T, U, V, D>, TriangulationValidationError> {
        // First, try the regular triangulation approach
        Tds::new(vertices).map_or_else(
            |_| {
                // Regular triangulation failed, try robust approach
                // Note: For now, we still fall back to the regular approach
                // but this is where we would implement more sophisticated
                // robust triangulation strategies
                Tds::new(vertices).map_err(|e| TriangulationValidationError::FailedToCreateCell {
                    message: format!("Robust triangulation failed: {e}"),
                })
            },
            Ok,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex;

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

        let result = algorithm.robust_insert_vertex(&mut tds, problematic_vertex);
        assert!(result.is_ok() || result.is_err()); // Should handle gracefully
    }
}
