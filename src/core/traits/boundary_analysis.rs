//! Boundary analysis trait for triangulation data structures.

use crate::core::{
    facet::Facet, traits::data_type::DataType,
    triangulation_data_structure::TriangulationValidationError,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use nalgebra::ComplexField;
use serde::{Serialize, de::DeserializeOwned};
use std::iter::Sum;
use std::ops::{AddAssign, Div, SubAssign};

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
/// use delaunay::core::triangulation_data_structure::Tds;
/// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
/// use delaunay::vertex;
///
/// // Create a simple 3D triangulation (single tetrahedron)
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
///
/// // Use the trait methods
/// let boundary_facets = tds.boundary_facets().expect("Failed to get boundary facets");
/// assert_eq!(boundary_facets.len(), 4); // Tetrahedron has 4 boundary faces
///
/// let count = tds.number_of_boundary_facets();
/// assert_eq!(count, Ok(4));
/// ```
pub trait BoundaryAnalysis<T, U, V, const D: usize>
where
    T: CoordinateScalar
        + AddAssign<T>
        + ComplexField<RealField = T>
        + SubAssign<T>
        + Sum
        + DeserializeOwned,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    /// Identifies all boundary facets in the triangulation.
    ///
    /// A boundary facet is a facet that belongs to only one cell, meaning it lies on the
    /// boundary of the triangulation (convex hull). These facets are important for
    /// convex hull computation and boundary analysis.
    ///
    /// # Returns
    ///
    /// A `Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>` containing all boundary facets in the triangulation.
    /// The facets are returned in no particular order.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if any boundary facet cannot be created from the cells.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // A single tetrahedron has 4 boundary facets (all facets are on the boundary)
    /// let boundary_facets = tds.boundary_facets().expect("Failed to get boundary facets");
    /// assert_eq!(boundary_facets.len(), 4);
    /// ```
    fn boundary_facets(&self) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>;

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
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // Get a facet from one of the cells
    /// if let Some(cell) = tds.cells().values().next() {
    ///     let facets = cell.facets().expect("Failed to get facets from cell");
    ///     if let Some(facet) = facets.first() {
    ///         // In a single tetrahedron, all facets are boundary facets
    ///         assert!(tds.is_boundary_facet(facet).unwrap());
    ///     }
    /// }
    /// ```
    fn is_boundary_facet(
        &self,
        facet: &Facet<T, U, V, D>,
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
    /// `true` if the facet is on the boundary (belongs to only one cell), `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // Build the facet map once for multiple queries (efficient for batch operations)
    /// let facet_to_cells = tds.build_facet_to_cells_map().expect("facet map should build");
    ///
    /// // Check multiple facets efficiently using the cached map
    /// if let Some(cell) = tds.cells().values().next() {
    ///     let facets = cell.facets().expect("Failed to get facets from cell");
    ///     for facet in &facets {
    ///         let is_boundary = tds.is_boundary_facet_with_map(facet, &facet_to_cells);
    ///         println!("Facet is boundary: {is_boundary}");
    ///         // In a single tetrahedron, all facets are boundary facets
    ///         assert!(is_boundary);
    ///     }
    /// }
    /// ```
    ///
    /// [`build_facet_to_cells_map`]: crate::core::triangulation_data_structure::Tds::build_facet_to_cells_map
    fn is_boundary_facet_with_map(
        &self,
        facet: &Facet<T, U, V, D>,
        facet_to_cells: &crate::core::collections::FacetToCellsMap,
    ) -> bool;

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
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // A single tetrahedron has 4 boundary facets
    /// assert_eq!(tds.number_of_boundary_facets().unwrap(), 4);
    /// ```
    fn number_of_boundary_facets(&self) -> Result<usize, TriangulationValidationError>;
}
