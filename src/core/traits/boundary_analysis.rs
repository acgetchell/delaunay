//! Boundary analysis trait for triangulation data structures.

use crate::core::{
    facet::{BoundaryFacetsIter, FacetView},
    tds::TdsError,
};

/// Trait for boundary analysis operations on triangulations.
///
/// This trait provides methods to identify and analyze boundary facets
/// in d-dimensional triangulations. A boundary facet is a facet that
/// belongs to only one simplex, meaning it lies on the convex hull of
/// the triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a simple 3D triangulation (single tetrahedron)
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let tds = dt.tds();
///
/// // Use the trait methods
/// let boundary_facets = tds.boundary_facets()?;
/// assert_eq!(boundary_facets.count(), 4); // Tetrahedron has 4 boundary faces
///
/// let count = tds.number_of_boundary_facets()?;
/// assert_eq!(count, 4);
/// # Ok(())
/// # }
/// ```
pub trait BoundaryAnalysis<U, V, const D: usize> {
    /// Identifies all boundary facets in the triangulation.
    ///
    /// A boundary facet is a facet that belongs to only one simplex, meaning it lies on the
    /// boundary of the triangulation (convex hull). These facets are important for
    /// convex hull computation and boundary analysis.
    ///
    /// # Returns
    ///
    /// A `Result<BoundaryFacetsIter<'_, U, V, D>, TdsError>` containing an iterator over boundary facets.
    /// The iterator yields facets lazily without pre-allocating a vector, providing better performance.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any boundary facet cannot be created from the simplices.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // A single tetrahedron has 4 boundary facets (all facets are on the boundary)
    /// let boundary_facets_iter = tds.boundary_facets()?;
    /// assert_eq!(boundary_facets_iter.count(), 4);
    /// # Ok(())
    /// # }
    /// ```
    fn boundary_facets(&self) -> Result<BoundaryFacetsIter<'_, U, V, D>, TdsError>;

    /// Checks if a specific facet is a boundary facet.
    ///
    /// A boundary facet is a facet that belongs to only one simplex in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one simplex),
    /// `Ok(false)` if it's an interior facet (belongs to two simplices).
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if:
    /// - Building the facet-to-simplices mapping fails due to data structure inconsistencies
    /// - The facet cannot be keyed because it references missing vertices
    /// - The facet-to-simplices map contains an invalid multiplicity other than 1 or 2
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get a boundary facet using the new iterator API
    /// let mut boundary_facets = tds.boundary_facets()?;
    /// let Some(first_facet) = boundary_facets.next() else {
    ///     return Ok(());
    /// };
    /// // In a single tetrahedron, all facets are boundary facets
    /// assert!(tds.is_boundary_facet(&first_facet)?);
    /// # Ok(())
    /// # }
    /// ```
    fn is_boundary_facet(&self, facet: &FacetView<'_, U, V, D>) -> Result<bool, TdsError>;

    /// Checks if a specific facet is a boundary facet using a precomputed facet map.
    ///
    /// This is an optimized version of [`Self::is_boundary_facet`] that accepts a prebuilt
    /// facet-to-simplices map to avoid recomputation in tight loops.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    /// * `facet_to_simplices` - Precomputed map from facet keys to simplices containing them.
    ///   Obtain this by calling [`build_facet_to_simplices_map`] on the triangulation.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one simplex),
    /// `Ok(false)` if it is an interior facet (belongs to two simplices) or the
    /// facet key is absent from the supplied map.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if:
    /// - The facet's vertices cannot be found in the triangulation (e.g., facet from different TDS)
    /// - The facet key cannot be derived from its vertices
    /// - The supplied map contains an invalid multiplicity other than 1 or 2 for the facet
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Build the facet map once for multiple queries (efficient for batch operations)
    /// let facet_to_simplices = tds.build_facet_to_simplices_map()?;
    ///
    /// // Check boundary facets efficiently using the iterator API and cached map
    /// let boundary_facets = tds.boundary_facets()?;
    /// for facet in boundary_facets {
    ///     let is_boundary = tds.is_boundary_facet_with_map(&facet, &facet_to_simplices)?;
    ///     println!("Facet is boundary: {is_boundary}");
    ///     // In a single tetrahedron, all facets are boundary facets
    ///     assert!(is_boundary);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`build_facet_to_simplices_map`]: crate::prelude::tds::Tds::build_facet_to_simplices_map
    fn is_boundary_facet_with_map(
        &self,
        facet: &FacetView<'_, U, V, D>,
        facet_to_simplices: &crate::core::collections::FacetToSimplicesMap,
    ) -> Result<bool, TdsError>;

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This counts boundary facets directly from the facet-to-simplices map.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of boundary facets in the triangulation,
    /// or a [`TdsError`] if the facet map cannot be built or contains invalid topology.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet-to-simplices map cannot be built or
    /// any facet has an invalid multiplicity other than 1 or 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Direct API call (recommended for single queries)
    /// assert_eq!(tds.number_of_boundary_facets()?, 4);
    ///
    /// // Alternative: using iterator (useful for additional processing)
    /// let count_via_iter = dt.boundary_facets()?.count();
    /// assert_eq!(count_via_iter, 4);
    /// # Ok(())
    /// # }
    /// ```
    fn number_of_boundary_facets(&self) -> Result<usize, TdsError>;
}
