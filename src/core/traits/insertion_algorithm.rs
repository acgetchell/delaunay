//! Common trait for vertex insertion algorithms in Delaunay triangulations.
//!
//! This module defines the `InsertionAlgorithm` trait that provides a unified
//! interface for different vertex insertion strategies, including the basic
//! Bowyer-Watson algorithm and robust variants with enhanced numerical stability.

use crate::core::{
    triangulation_data_structure::{Tds, TriangulationValidationError},
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use serde::{Serialize, de::DeserializeOwned};

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
    /// - Geometric degeneracy
    /// - Numerical precision issues
    /// - Insufficient vertices for the given dimension
    /// - Topological constraints
    fn triangulate(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertices: &[Vertex<T, U, D>],
    ) -> Result<(), TriangulationValidationError> {
        // Default implementation: insert vertices one by one
        for vertex in vertices {
            self.insert_vertex(tds, *vertex)?;
        }
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
    /// # Ok::<(), delaunay::core::triangulation_data_structure::TriangulationValidationError>(())
    /// ```
    fn new_triangulation(
        &mut self,
        vertices: &[Vertex<T, U, D>],
    ) -> Result<Tds<T, U, V, D>, TriangulationValidationError>
    where
        T: std::ops::AddAssign<T>
            + nalgebra::ComplexField<RealField = T>
            + std::ops::SubAssign<T>
            + std::iter::Sum
            + From<f64>
            + serde::de::DeserializeOwned,
        U: serde::de::DeserializeOwned,
        V: serde::de::DeserializeOwned,
        f64: From<T>,
        for<'a> &'a T: std::ops::Div<T>,
        ordered_float::OrderedFloat<f64>: From<T>,
        nalgebra::OPoint<T, nalgebra::Const<D>>: From<[f64; D]>,
        [f64; D]: Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        // Default implementation: use the regular Tds::new constructor
        Tds::new(vertices).map_err(|e| TriangulationValidationError::FailedToCreateCell {
            message: format!("Failed to create triangulation: {e}"),
        })
    }
}
