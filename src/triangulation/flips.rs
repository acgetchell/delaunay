//! Triangulation editing operations (bistellar flips).
//!
//! This module exposes **high-level** flip methods for explicit triangulation editing.
//! These operations do **not** automatically restore the Delaunay property.
//! For Delaunay construction/removal, use
//! [`crate::core::delaunay_triangulation::DelaunayTriangulation::insert`] and
//! [`crate::core::delaunay_triangulation::DelaunayTriangulation::remove_vertex`].

#![forbid(unsafe_code)]

pub use crate::core::algorithms::flips::{
    BistellarFlipKind, FlipDirection, FlipError, FlipInfo, RidgeHandle, TriangleHandle,
};
pub use crate::core::edge::EdgeKey;
pub use crate::core::facet::FacetHandle;
pub use crate::core::triangulation_data_structure::{CellKey, VertexKey};

use crate::core::algorithms::flips::{
    apply_bistellar_flip_dynamic, apply_bistellar_flip_k1, apply_bistellar_flip_k1_inverse,
    apply_bistellar_flip_k2, apply_bistellar_flip_k3, build_k2_flip_context,
    build_k2_flip_context_from_edge, build_k3_flip_context, build_k3_flip_context_from_triangle,
};
use crate::core::delaunay_triangulation::DelaunayTriangulation;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::traits::coordinate::CoordinateScalar;

/// High-level triangulation editing operations via bistellar flips.
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::triangulation::flips::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let mut dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulation::new_with_topology_guarantee(
///         &vertices,
///         TopologyGuarantee::PLManifold,
///     )
///     .unwrap();
/// let cell_key = dt.cells().next().unwrap().0;
///
/// // Split a cell by inserting a vertex (k=1 move).
/// let _info = dt
///     .flip_k1_insert(cell_key, vertex!([0.1, 0.1, 0.1]))
///     .unwrap();
/// ```
pub trait BistellarFlips<K, U, V, const D: usize>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Apply a forward k=1 move (cell split) by inserting a vertex into a cell.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the cell is missing, the vertex cannot be inserted,
    /// or the flip would create invalid topology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::flips::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new_with_topology_guarantee(
    ///         &vertices,
    ///         TopologyGuarantee::PLManifold,
    ///     )
    ///     .unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    ///
    /// // Insert a vertex into the cell
    /// let info = dt.flip_k1_insert(cell_key, vertex!([0.25, 0.25, 0.25])).unwrap();
    /// assert!(!info.new_cells.is_empty());
    /// ```
    fn flip_k1_insert(
        &mut self,
        cell_key: CellKey,
        vertex: Vertex<K::Scalar, U, D>,
    ) -> Result<FlipInfo<D>, FlipError>;

    /// Apply an inverse k=1 move (vertex collapse).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the vertex star is not collapsible or the flip
    /// would create invalid topology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::flips::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new_with_topology_guarantee(
    ///         &vertices,
    ///         TopologyGuarantee::PLManifold,
    ///     )
    ///     .unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let inserted = dt.flip_k1_insert(cell_key, vertex!([0.25, 0.25, 0.25])).unwrap();
    /// let inserted_vertex = inserted.inserted_face_vertices[0];
    ///
    /// // Remove the inserted vertex
    /// let info = dt.flip_k1_remove(inserted_vertex).unwrap();
    /// assert!(!info.removed_cells.is_empty());
    /// ```
    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError>;

    /// Apply a k=2 facet flip (forward).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the facet is invalid, the flip would be degenerate,
    /// or the resulting topology would be non-manifold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::flips::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.5, 0.5, 0.3]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Find an interior facet and attempt a k=2 flip
    /// // Note: k=2 flips require specific geometric conditions
    /// let cell_key = dt.cells().next().map(|(k, _)| k);
    /// if let Some(key) = cell_key {
    ///     let has_neighbor = dt.tds().get_cell(key)
    ///         .and_then(|cell| cell.neighbors())
    ///         .map(|neighbors| neighbors.iter().any(|n| n.is_some()))
    ///         .unwrap_or(false);
    ///     if has_neighbor {
    ///         let facet = FacetHandle::new(key, 0);
    ///         let _ = dt.flip_k2(facet);  // May succeed or fail depending on configuration
    ///     }
    /// }
    /// ```
    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError>;

    /// Apply a k=3 ridge flip (forward).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the ridge is invalid, the flip would be degenerate,
    /// or the resulting topology would be non-manifold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::flips::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([1.0, 1.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // k=3 flips require specific ridge configurations in 3D and above
    /// // This is an illustrative example; actual ridge selection depends on topology
    /// let _ = dt;  // Use dt to prevent unused variable warning
    /// ```
    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError>;

    /// Apply an inverse k=2 flip from an edge star (D >= 3).
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the edge star is invalid or the inverse flip
    /// would create invalid topology.
    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError>;

    /// Apply an inverse k=3 flip from a triangle star (D >= 4).
    ///
    /// If `D < 4`, this returns [`FlipError::UnsupportedDimension`].
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the triangle star is invalid or the inverse flip
    /// would create invalid topology.
    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError>;
}

impl<K, U, V, const D: usize> BistellarFlips<K, U, V, D> for Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn flip_k1_insert(
        &mut self,
        cell_key: CellKey,
        vertex: Vertex<K::Scalar, U, D>,
    ) -> Result<FlipInfo<D>, FlipError> {
        apply_bistellar_flip_k1(&mut self.tds, &self.kernel, cell_key, vertex)
    }

    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError> {
        apply_bistellar_flip_k1_inverse(&mut self.tds, &self.kernel, vertex_key)
    }

    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k2_flip_context(&self.tds, facet)?;
        apply_bistellar_flip_k2(&mut self.tds, &self.kernel, &context)
    }

    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k3_flip_context(&self.tds, ridge)?;
        apply_bistellar_flip_k3(&mut self.tds, &self.kernel, &context)
    }

    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k2_flip_context_from_edge(&self.tds, edge)?;
        apply_bistellar_flip_dynamic(&mut self.tds, &self.kernel, D, &context)
    }

    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError> {
        if D < 4 {
            return Err(FlipError::UnsupportedDimension { dimension: D });
        }

        let context = build_k3_flip_context_from_triangle(&self.tds, triangle)?;

        // Avoid const-eval underflow for invalid instantiations (e.g. D=0), even though
        // the public contract for this method requires D>=4.
        let k_move = D
            .checked_sub(1)
            .ok_or(FlipError::UnsupportedDimension { dimension: D })?;

        apply_bistellar_flip_dynamic(&mut self.tds, &self.kernel, k_move, &context)
    }
}

impl<K, U, V, const D: usize> BistellarFlips<K, U, V, D> for DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn flip_k1_insert(
        &mut self,
        cell_key: CellKey,
        vertex: Vertex<K::Scalar, U, D>,
    ) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k1_insert(cell_key, vertex)
    }

    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k1_remove(vertex_key)
    }

    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k2(facet)
    }

    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k3(ridge)
    }

    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k2_inverse_from_edge(edge)
    }

    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError> {
        self.tri.flip_k3_inverse_from_triangle(triangle)
    }
}
