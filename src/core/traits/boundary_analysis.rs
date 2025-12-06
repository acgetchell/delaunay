//! Boundary analysis trait for triangulation data structures.

use crate::core::{
    facet::{BoundaryFacetsIter, FacetView},
    traits::data_type::DataType,
    triangulation_data_structure::TriangulationValidationError,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use num_traits::NumCast;

/// Trait for boundary analysis operations on triangulations.
///
/// This trait provides methods to identify and analyze boundary facets
/// in d-dimensional triangulations. A boundary facet is a facet that
/// belongs to only one cell, meaning it lies on the convex hull of
/// the triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
///
/// // Create a simple 3D triangulation (single tetrahedron)
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Use the trait methods
/// let boundary_facets = tds.boundary_facets().expect("Failed to get boundary facets");
/// assert_eq!(boundary_facets.count(), 4); // Tetrahedron has 4 boundary faces
///
/// let count = tds.number_of_boundary_facets();
/// assert_eq!(count, Ok(4));
/// ```
pub trait BoundaryAnalysis<T, U, V, const D: usize>
where
    T: CoordinateScalar + NumCast,
    U: DataType,
    V: DataType,
{
    /// Identifies all boundary facets in the triangulation.
    ///
    /// A boundary facet is a facet that belongs to only one cell, meaning it lies on the
    /// boundary of the triangulation (convex hull). These facets are important for
    /// convex hull computation and boundary analysis.
    ///
    /// # Returns
    ///
    /// A `Result<BoundaryFacetsIter<'_, T, U, V, D>, TriangulationValidationError>` containing an iterator over boundary facets.
    /// The iterator yields facets lazily without pre-allocating a vector, providing better performance.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if any boundary facet cannot be created from the cells.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // A single tetrahedron has 4 boundary facets (all facets are on the boundary)
    /// let boundary_facets_iter = tds.boundary_facets().expect("Failed to get boundary facets iterator");
    /// assert_eq!(boundary_facets_iter.count(), 4);
    /// ```
    fn boundary_facets(
        &self,
    ) -> Result<BoundaryFacetsIter<'_, T, U, V, D>, TriangulationValidationError>;

    /// Checks if a specific facet is a boundary facet.
    ///
    /// A boundary facet is a facet that belongs to only one cell in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one cell),
    /// `Ok(false)` if it's an interior facet (belongs to two cells).
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationError)` if:
    /// - Building the facet-to-cells mapping fails due to data structure inconsistencies
    /// - The triangulation contains invalid cells or corrupted vertex mappings
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get a boundary facet using the new iterator API
    /// let boundary_facets = tds.boundary_facets().unwrap();
    /// let first_facet = boundary_facets.clone().next().unwrap();
    /// // In a single tetrahedron, all facets are boundary facets
    /// assert!(tds.is_boundary_facet(&first_facet).unwrap());
    /// ```
    fn is_boundary_facet(
        &self,
        facet: &FacetView<'_, T, U, V, D>,
    ) -> Result<bool, TriangulationValidationError>;

    /// Checks if a specific facet is a boundary facet using a precomputed facet map.
    ///
    /// This is an optimized version of `is_boundary_facet` that accepts a prebuilt
    /// facet-to-cells map to avoid recomputation in tight loops.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    /// * `facet_to_cells` - Precomputed map from facet keys to cells containing them.
    ///   Obtain this by calling [`build_facet_to_cells_map`] on the triangulation.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one cell),
    /// `Ok(false)` if it's an interior facet (belongs to two cells).
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationError)` if:
    /// - The facet's vertices cannot be found in the triangulation (e.g., facet from different TDS)
    /// - The facet key cannot be derived from its vertices
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Build the facet map once for multiple queries (efficient for batch operations)
    /// let facet_to_cells = tds.build_facet_to_cells_map().expect("facet map should build");
    ///
    /// // Check boundary facets efficiently using the iterator API and cached map
    /// let boundary_facets = tds.boundary_facets().unwrap();
    /// for facet in boundary_facets {
    ///     let is_boundary = tds.is_boundary_facet_with_map(&facet, &facet_to_cells).expect("Should not fail for valid facets");
    ///     println!("Facet is boundary: {is_boundary}");
    ///     // In a single tetrahedron, all facets are boundary facets
    ///     assert!(is_boundary);
    /// }
    /// ```
    ///
    /// [`build_facet_to_cells_map`]: crate::core::triangulation_data_structure::Tds::build_facet_to_cells_map
    fn is_boundary_facet_with_map(
        &self,
        facet: &FacetView<'_, T, U, V, D>,
        facet_to_cells: &crate::core::collections::FacetToCellsMap,
    ) -> Result<bool, TriangulationValidationError>;

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This delegates to `boundary_facets()` for consistent error handling.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of boundary facets in the triangulation,
    /// or a `TriangulationValidationError` if the facet map cannot be built.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if the facet-to-cells map cannot be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Direct API call (recommended for single queries)
    /// assert_eq!(tds.number_of_boundary_facets().unwrap(), 4);
    ///
    /// // Alternative: using iterator (useful for additional processing)
    /// let count_via_iter = dt.boundary_facets().count();
    /// assert_eq!(count_via_iter, 4);
    /// ```
    fn number_of_boundary_facets(&self) -> Result<usize, TriangulationValidationError>;
}
