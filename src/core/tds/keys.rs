//! Slotmap key types used by the TDS storage layer.

use slotmap::new_key_type;

new_key_type! {
    /// Key type for accessing vertices in the storage map.
    ///
    /// This creates a unique, type-safe identifier for vertices stored in the
    /// triangulation's vertex storage. Each VertexKey corresponds to exactly
    /// one vertex and provides efficient, stable access even as vertices are
    /// added or removed from the triangulation.
    ///
    /// # Examples
    ///
        /// ```
        /// use delaunay::prelude::*;
        ///
        /// # #[derive(Debug, thiserror::Error)]
        /// # enum ExampleError {
        /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
        /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
        /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
        /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
        /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
        /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
        /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
        /// #     #[error(transparent)]
        /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
        /// # }
        /// # fn main() -> Result<(), ExampleError> {
        /// let vertices = [
        ///     delaunay::vertex![0.0, 0.0]?,
        ///     delaunay::vertex![1.0, 0.0]?,
        ///     delaunay::vertex![0.0, 1.0]?,
        /// ];
        /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
        /// let Some(key) = dt.tds().vertex_keys().next() else {
        ///     return Ok(());
        /// };
        /// let _ = key;
        /// # Ok(())
        /// # }
        /// ```
    pub struct VertexKey;
}

new_key_type! {
    /// Key type for accessing simplices in the storage map.
    ///
    /// This creates a unique, type-safe identifier for simplices stored in the
    /// triangulation's simplex storage. Each SimplexKey corresponds to exactly
    /// one simplex and provides efficient, stable access even as simplices are
    /// added or removed during triangulation operations.
    ///
    /// # Examples
    ///
        /// ```
        /// use delaunay::prelude::*;
        ///
        /// # #[derive(Debug, thiserror::Error)]
        /// # enum ExampleError {
        /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
        /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
        /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
        /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
        /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
        /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
        /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
        /// #     #[error(transparent)]
        /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
        /// # }
        /// # fn main() -> Result<(), ExampleError> {
        /// let vertices = [
        ///     delaunay::vertex![0.0, 0.0]?,
        ///     delaunay::vertex![1.0, 0.0]?,
        ///     delaunay::vertex![0.0, 1.0]?,
        /// ];
        /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
        /// let Some(key) = dt.tds().simplex_keys().next() else {
        ///     return Ok(());
        /// };
        /// let _ = key;
        /// # Ok(())
        /// # }
        /// ```
    pub struct SimplexKey;
}
