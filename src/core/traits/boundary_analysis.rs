//! Boundary analysis trait for triangulation data structures.

use crate::core::{facet::Facet, traits::data_type::DataType};
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
/// let boundary_facets = tds.boundary_facets();
/// assert_eq!(boundary_facets.len(), 4); // Tetrahedron has 4 boundary faces
///
/// let count = tds.number_of_boundary_facets();
/// assert_eq!(count, 4);
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
    /// A `Vec<Facet<T, U, V, D>>` containing all boundary facets in the triangulation.
    /// The facets are returned in no particular order.
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
    /// let boundary_facets = tds.boundary_facets();
    /// assert_eq!(boundary_facets.len(), 4);
    /// ```
    fn boundary_facets(&self) -> Vec<Facet<T, U, V, D>>;

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
    /// // Get a facet from one of the cells
    /// if let Some(cell) = tds.cells().values().next() {
    ///     let facets = cell.facets();
    ///     if let Some(facet) = facets.first() {
    ///         // In a single tetrahedron, all facets are boundary facets
    ///         assert!(tds.is_boundary_facet(facet));
    ///     }
    /// }
    /// ```
    fn is_boundary_facet(&self, facet: &Facet<T, U, V, D>) -> bool;

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This is a more efficient way to count boundary facets without creating
    /// the full vector of facets.
    ///
    /// # Returns
    ///
    /// The number of boundary facets in the triangulation.
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
    /// assert_eq!(tds.number_of_boundary_facets(), 4);
    /// ```
    fn number_of_boundary_facets(&self) -> usize;
}
