//! Triangulation editing operations (bistellar flips).
//!
//! This module exposes **high-level** flip methods for explicit triangulation editing.
//! These operations do **not** automatically restore the Delaunay property.
//! For Delaunay construction/removal, use
//! [`crate::DelaunayTriangulation::insert`] and
//! [`crate::DelaunayTriangulation::remove_vertex`].

#![forbid(unsafe_code)]

pub use crate::core::algorithms::flips::{
    BistellarFlipKind, BistellarMove, ConstK, DelaunayRepairDiagnostics, DelaunayRepairError,
    DelaunayRepairStats, DelaunayRepairVerificationContext, FlipContextError, FlipDirection,
    FlipEdgeAdjacencyError, FlipError, FlipInfo, FlipMutationError, FlipNeighborWiringError,
    FlipOrientationCheckStage, FlipPredicateError, FlipPredicateOperation,
    FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder, RidgeHandle,
    TriangleHandle, TriangleHandleError, verify_delaunay_for_triangulation,
    verify_delaunay_via_flip_predicates,
};
pub use crate::tds::{EdgeKey, FacetHandle, SimplexKey, VertexKey};

use crate::core::algorithms::flips::{
    apply_bistellar_flip_dynamic, apply_bistellar_flip_k1, apply_bistellar_flip_k1_inverse,
    apply_bistellar_flip_k2, apply_bistellar_flip_k3, build_k2_flip_context,
    build_k2_flip_context_from_edge, build_k3_flip_context, build_k3_flip_context_from_triangle,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::triangulation::DelaunayTriangulation;
/// High-level triangulation editing operations via bistellar flips.
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, TopologyGuarantee};
/// use delaunay::prelude::flips::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Flip(#[from] delaunay::flips::FlipError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
/// ];
/// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// // Split a simplex by inserting a vertex (k=1 move).
/// let _info = dt.flip_k1_insert(simplex_key, delaunay::prelude::Vertex::<(), _>::try_new([0.1, 0.1, 0.1])?)?;
/// # Ok(())
/// # }
/// ```
pub trait BistellarFlips<const D: usize> {
    /// User data type stored on vertices inserted through k=1 flips.
    type VertexData;

    /// Apply a forward k=1 move (simplex split) by inserting a vertex into a simplex.
    ///
    /// # Errors
    ///
    /// Returns [`FlipError`] if the simplex is missing, the vertex cannot be inserted,
    /// or the flip would create invalid topology.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, TopologyGuarantee};
    /// use delaunay::prelude::flips::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Flip(#[from] delaunay::flips::FlipError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Insert a vertex into the simplex
    /// let info = dt.flip_k1_insert(simplex_key, delaunay::prelude::Vertex::<(), _>::try_new([0.25, 0.25, 0.25])?)?;
    /// assert!(!info.new_simplices.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    fn flip_k1_insert(
        &mut self,
        simplex_key: SimplexKey,
        vertex: Vertex<Self::VertexData, D>,
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
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, TopologyGuarantee};
    /// use delaunay::prelude::flips::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Flip(#[from] delaunay::flips::FlipError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let inserted = dt.flip_k1_insert(simplex_key, delaunay::prelude::Vertex::<(), _>::try_new([0.25, 0.25, 0.25])?)?;
    /// let inserted_vertex = inserted.inserted_face_vertices[0];
    ///
    /// // Remove the inserted vertex
    /// let info = dt.flip_k1_remove(inserted_vertex)?;
    /// assert!(!info.removed_simplices.is_empty());
    /// # Ok(())
    /// # }
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
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::flips::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5, 0.3])?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Find an interior facet and attempt a k=2 flip
    /// // Note: k=2 flips require specific geometric conditions
    /// let simplex_key = dt.simplices().next().map(|(k, _)| k);
    /// if let Some(key) = simplex_key {
    ///     let has_neighbor = dt.tds().simplex(key)
    ///         .and_then(|simplex| simplex.neighbors())
    ///         .is_some_and(|mut neighbors| neighbors.any(|n| n.is_some()));
    ///     if has_neighbor {
    ///         let facet = FacetHandle::try_new(dt.tds(), key, 0)?;
    ///         let _ = dt.flip_k2(facet);  // May succeed or fail depending on configuration
    ///     }
    /// }
    /// # Ok(())
    /// # }
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
    /// use delaunay::prelude::construction::DelaunayTriangulationBuilder;
    /// use delaunay::prelude::flips::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // k=3 flips require specific ridge configurations in 3D and above
    /// // This is an illustrative example; actual ridge selection depends on topology
    /// let _ = dt;  // Use dt to prevent unused variable warning
    /// # Ok(())
    /// # }
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

impl<K, U, V, const D: usize> BistellarFlips<D> for Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    type VertexData = U;

    fn flip_k1_insert(
        &mut self,
        simplex_key: SimplexKey,
        vertex: Vertex<U, D>,
    ) -> Result<FlipInfo<D>, FlipError> {
        apply_bistellar_flip_k1(&mut self.tds, simplex_key, vertex)
    }

    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError> {
        apply_bistellar_flip_k1_inverse(&mut self.tds, vertex_key)
    }

    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k2_flip_context(&self.tds, facet)?;
        apply_bistellar_flip_k2(&mut self.tds, &context)
    }

    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k3_flip_context(&self.tds, ridge)?;
        apply_bistellar_flip_k3(&mut self.tds, &context)
    }

    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError> {
        let context = build_k2_flip_context_from_edge(&self.tds, edge)?;
        apply_bistellar_flip_dynamic(&mut self.tds, D, &context)
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

        apply_bistellar_flip_dynamic(&mut self.tds, k_move, &context)
    }
}

impl<K, U, V, const D: usize> BistellarFlips<D> for DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    type VertexData = U;

    fn flip_k1_insert(
        &mut self,
        simplex_key: SimplexKey,
        vertex: Vertex<U, D>,
    ) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k1_insert(simplex_key, vertex);
        if result.is_ok() {
            self.invalidate_repair_caches();
        }
        result
    }

    fn flip_k1_remove(&mut self, vertex_key: VertexKey) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k1_remove(vertex_key);
        if result.is_ok() {
            self.invalidate_repair_caches();
        }
        result
    }

    fn flip_k2(&mut self, facet: FacetHandle) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k2(facet);
        if result.is_ok() {
            self.invalidate_locate_hint_cache();
        }
        result
    }

    fn flip_k3(&mut self, ridge: RidgeHandle) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k3(ridge);
        if result.is_ok() {
            self.invalidate_locate_hint_cache();
        }
        result
    }

    fn flip_k2_inverse_from_edge(&mut self, edge: EdgeKey) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k2_inverse_from_edge(edge);
        if result.is_ok() {
            self.invalidate_locate_hint_cache();
        }
        result
    }

    fn flip_k3_inverse_from_triangle(
        &mut self,
        triangle: TriangleHandle,
    ) -> Result<FlipInfo<D>, FlipError> {
        let result = self.tri.flip_k3_inverse_from_triangle(triangle);
        if result.is_ok() {
            self.invalidate_locate_hint_cache();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    use crate::TopologyGuarantee;
    use crate::core::collections::spatial_hash_grid::HashGridIndex;
    use crate::geometry::kernel::{AdaptiveKernel, FastKernel};
    use slotmap::KeyData;

    #[test]
    fn triangulation_flip_k1_insert_and_remove_roundtrip() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new_with_topology_guarantee(
                &vertices,
                TopologyGuarantee::PLManifold,
            )
            .unwrap();
        let mut tri = dt.as_triangulation().clone();
        let simplex_key = tri.simplices().next().unwrap().0;

        let inserted = tri
            .flip_k1_insert(
                simplex_key,
                crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25, 0.25]).unwrap(),
            )
            .unwrap();
        let inserted_vertex = inserted.inserted_face_vertices[0];
        assert!(!inserted.new_simplices.is_empty());
        assert!(tri.validate().is_ok());

        let removed = tri.flip_k1_remove(inserted_vertex).unwrap();
        assert!(!removed.removed_simplices.is_empty());
        assert!(tri.validate().is_ok());
    }

    #[test]
    fn flip_k1_insert_invalidates_caches() {
        let vertices: Vec<Vertex<(), 3>> = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let simplex_key = dt.simplices().next().unwrap().0;
        dt.insertion_state.last_inserted_simplex = Some(simplex_key);
        let mut spatial_index = HashGridIndex::<3>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);

        dt.flip_k1_insert(
            simplex_key,
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2]).unwrap(),
        )
        .unwrap();

        assert!(dt.insertion_state.last_inserted_simplex.is_none());
        assert!(dt.spatial_index.is_none());
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn triangulation_flip_k2_rejects_invalid_facet_index() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new_with_topology_guarantee(
                &vertices,
                TopologyGuarantee::PLManifold,
            )
            .unwrap();
        let tri = dt.as_triangulation().clone();
        let simplex_key = tri.simplices().next().unwrap().0;

        let err = FacetHandle::try_new(&tri.tds, simplex_key, u8::MAX).unwrap_err();

        assert_matches!(
            err,
            crate::prelude::tds::FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count: 4,
            }
        );
    }

    #[test]
    fn delaunay_flip_k3_inverse_rejects_unsupported_dimension() {
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel(AdaptiveKernel::new());
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let err = dt
            .flip_k3_inverse_from_triangle(TriangleHandle::try_new(a, b, c).unwrap())
            .unwrap_err();

        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 3 });
    }

    #[test]
    fn triangulation_flip_k3_inverse_rejects_zero_dimension_without_underflow() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 0> =
            Triangulation::new_empty(FastKernel::new());
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let err = tri
            .flip_k3_inverse_from_triangle(TriangleHandle::try_new(a, b, c).unwrap())
            .unwrap_err();

        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 0 });
    }
}
