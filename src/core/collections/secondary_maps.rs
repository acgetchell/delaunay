//! Sparse secondary-map aliases for data associated with slotmap keys.
//!
//! These aliases provide type-safe auxiliary storage keyed by simplices or vertices
//! without requiring dense side arrays.

use crate::core::tds::{SimplexKey, VertexKey};
use slotmap::SparseSecondaryMap;

// =============================================================================
// SLOTMAP SECONDARY MAPS FOR AUXILIARY DATA
// =============================================================================

/// Sparse secondary map for tracking auxiliary data associated with simplices.
///
/// This is the idiomatic way to associate temporary data with `SlotMap` keys during algorithms.
/// Only stores entries for simplices that have associated data (sparse representation).
///
/// # Performance Benefits
///
/// - **Memory efficient**: Only allocates for simplices with data (vs dense array)
/// - **Type safe**: Can only use valid `SimplexKey` from the primary `SlotMap`
/// - **Cache friendly**: Better locality than separate `HashMap<SimplexKey, V>`
/// - **`SlotMap` integration**: Designed specifically for this use case
///
/// # Use Cases
///
/// - **Triangulation algorithms**: Conflict region finding, cavity extraction
/// - **Algorithm state**: Marking simplices as "visited", "in conflict", "processed"
/// - **Temporary data**: Associating algorithm-specific data with simplices
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// use delaunay::prelude::collections::SimplexSecondaryMap;
/// let mut in_conflict: SimplexSecondaryMap<bool> = SimplexSecondaryMap::new();
/// for (simplex_key, _) in dt.simplices() {
///     in_conflict.insert(simplex_key, true);
/// }
/// # Ok(())
/// # }
/// ```
pub type SimplexSecondaryMap<V> = SparseSecondaryMap<SimplexKey, V>;

/// Sparse secondary map for tracking auxiliary data associated with vertices.
///
/// This is the idiomatic way to associate temporary data with `SlotMap` keys during algorithms.
/// Only stores entries for vertices that have associated data (sparse representation).
///
/// # Performance Benefits
///
/// - **Memory efficient**: Only allocates for vertices with data (vs dense array)
/// - **Type safe**: Can only use valid `VertexKey` from the primary `SlotMap`
/// - **Cache friendly**: Better locality than separate `HashMap<VertexKey, V>`
/// - **`SlotMap` integration**: Designed specifically for this use case
///
/// # Use Cases
///
/// - **Algorithm state**: Marking vertices as "visited", "processed", "boundary"
/// - **Distance tracking**: Shortest path, geodesic distance computations
/// - **Temporary data**: Associating algorithm-specific data with vertices
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt: DelaunayTriangulation<_, _, _, 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// use delaunay::prelude::collections::VertexSecondaryMap;
/// let mut processing_order: VertexSecondaryMap<usize> = VertexSecondaryMap::new();
/// for (idx, (vertex_key, _)) in dt.vertices().enumerate() {
///     processing_order.insert(vertex_key, idx);
/// }
/// # Ok(())
/// # }
/// ```
pub type VertexSecondaryMap<V> = SparseSecondaryMap<VertexKey, V>;
