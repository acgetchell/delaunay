//! Common trait for vertex insertion algorithms in Delaunay triangulations.
//!
//! This module defines the `InsertionAlgorithm` trait that provides a unified
//! interface for different vertex insertion strategies, including the basic
//! Bowyer-Watson algorithm and robust variants with enhanced numerical stability.

use crate::core::{
    cell::CellBuilder,
    triangulation_data_structure::{
        Tds, TriangulationConstructionError, TriangulationValidationError,
    },
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use nalgebra::{ComplexField, Const, OPoint};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    iter::Sum,
    ops::{AddAssign, Div, SubAssign},
};

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
    ) -> Result<InsertionInfo, TriangulationValidationError>;

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
    ) -> InsertionStrategy;

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
        T: AddAssign<T>
            + ComplexField<RealField = T>
            + SubAssign<T>
            + Sum
            + From<f64>
            + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
            self.insert_vertex(tds, *vertex)
                .map_err(TriangulationConstructionError::ValidationError)?;
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
        T: AddAssign<T>
            + ComplexField<RealField = T>
            + SubAssign<T>
            + Sum
            + From<f64>
            + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
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
            if !tds.vertex_bimap.contains_left(&vertex.uuid()) {
                let vertex_key = tds.vertices.insert(*vertex);
                tds.vertex_bimap.insert(vertex.uuid(), vertex_key);
            }
        }

        let cell = CellBuilder::default()
            .vertices(vertices)
            .build()
            .map_err(|e| TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex: {e}"),
            })?;

        let cell_key = tds.cells_mut().insert(cell);
        let cell_uuid = tds.cells()[cell_key].uuid();
        tds.cell_bimap.insert(cell_uuid, cell_key);

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
        T: AddAssign<T>
            + ComplexField<RealField = T>
            + SubAssign<T>
            + Sum
            + From<f64>
            + DeserializeOwned,
        U: DeserializeOwned,
        V: DeserializeOwned,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
    {
        // Remove duplicate cells
        tds.remove_duplicate_cells();

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
        T: AddAssign<T> + ComplexField<RealField = T> + SubAssign<T> + Sum + From<f64>,
        f64: From<T>,
        for<'a> &'a T: Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Default implementation: use the regular Tds::new constructor
        Tds::new(vertices)
    }
}
